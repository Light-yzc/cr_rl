"""PPO updates for the explicit-memory architecture-v3 policy.

The legacy PPO path feeds sliding-window architecture-v2 tensors.  This module
keeps v3's recurrent memory explicit: every sequence is prefixed with a fresh
state and log-probs are recomputed in chronological order before the PPO loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Iterable

import torch


@dataclass(frozen=True)
class StatePPOConfig:
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    precision: str = "fp16"
    max_initial_logp_error: float = 0.05


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("State PPO fp16/bf16 requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _sequence_forward(model, sequence, device, precision):
    """Recompute one complete chronological sequence from a fresh memory."""
    required = ("header", "entities", "entity_mask", "legal_action_mask",
                "hand_card_ids", "actions", "old_logp", "advantages", "returns")
    missing = [name for name in required if name not in sequence]
    if missing:
        raise KeyError("State PPO sequence is missing: " + ", ".join(missing))
    tensor = lambda key: torch.as_tensor(sequence[key], device=device)
    header, entities, entity_mask = (tensor("header"), tensor("entities"),
                                     tensor("entity_mask"))
    legal = tensor("legal_action_mask").bool()
    hand = tensor("hand_card_ids").long()
    # A sequence is [T,...]; StateActorCritic returns [T,A], [T] and next memory.
    with _autocast(device, precision):
        # Recompute from the rollout-time h0 snapshot.  This keeps old_logp and
        # new_logp conditioned on the same recurrent prefix, as required by
        # recurrent PPO (and avoids silently replacing h0 with zeros).
        state = sequence.get("initial_state")
        if state is None:
            state = model.initial_state(1, device=device)
        else:
            state = model.detach_state(state)
        logp, values, _ = model.forward_state(
            header, entities, entity_mask, legal_action_mask=legal,
            state=state, hand_card_ids=hand, compute_value=True,
        )
    actions = tensor("actions").long()
    chosen = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    # Joint v3 log-probs contain sentinel values for illegal flat actions.
    # Entropy must be evaluated only on the legal categorical support; using
    # exp(-1e9) or an all-masked tile conditional can otherwise create a huge
    # finite scalar and dominate PPO.
    legal_logp = logp.masked_fill(~legal, float("-inf"))
    probabilities = legal_logp.exp()
    entropy = -(probabilities * legal_logp.nan_to_num(0.0, neginf=0.0)).sum(-1)
    return chosen, values, entropy, tensor("old_logp").float(), tensor("advantages").float(), tensor("returns").float()


def _average_distributed_gradients(model):
    """Synchronize v3 PPO gradients when launched with torchrun.

    Rollouts stay independent per rank; averaging before the optimizer step makes
    the resulting update equivalent to one larger dual-GPU batch.  Single-GPU
    runs take the zero-cost fast path.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    world = torch.distributed.get_world_size()
    if world <= 1:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad.div_(world)


def state_ppo_update(model, optimizer, sequences: Iterable[dict], *, config=None, scaler=None):
    """Run one PPO update over chronological v3 sequences.

    ``old_logp`` must be captured by the rollout policy for the same state reset
    and action sequence.  The function deliberately rejects non-finite ratios,
    which catches stale state or a mismatched checkpoint before optimizer steps.
    """
    config = config or StatePPOConfig()
    device = next(model.parameters()).device
    sequences = list(sequences)
    if not sequences:
        raise ValueError("State PPO needs at least one sequence")
    all_adv = torch.cat([torch.as_tensor(s["advantages"], dtype=torch.float32, device=device)
                         for s in sequences])
    if not torch.isfinite(all_adv).all():
        raise FloatingPointError("Non-finite state PPO advantages")
    all_adv = (all_adv - all_adv.mean()) / all_adv.std(unbiased=False).clamp_min(1e-8)
    offset = 0
    policy_losses, value_losses, entropies, ratios = [], [], [], []
    optimizer.zero_grad(set_to_none=True)
    model.train(True)
    for sequence in sequences:
        chosen, values, entropy, old_logp, _, returns = _sequence_forward(
            model, sequence, device, config.precision)
        count = chosen.numel()
        advantages = all_adv[offset:offset + count]
        offset += count
        if not (chosen.shape == old_logp.shape == advantages.shape == returns.shape == values.shape):
            raise ValueError("State PPO sequence tensors have inconsistent lengths")
        initial_error = (chosen.detach().float() - old_logp).abs().max()
        if float(initial_error) > config.max_initial_logp_error:
            raise ValueError(
                "Collected old log-prob differs from the initial v3 Actor: "
                f"{float(initial_error):.6f} > {config.max_initial_logp_error:.6f}"
            )
        log_ratio = chosen.float() - old_logp
        ratio = log_ratio.exp()
        if not torch.isfinite(ratio).all():
            raise FloatingPointError("Non-finite PPO ratio; rollout state/checkpoint mismatch")
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef) * advantages
        policy_losses.append(-torch.minimum(unclipped, clipped).mean())
        value_losses.append((values.float() - returns).square().mean())
        entropies.append(entropy.float().mean())
        ratios.append(ratio.detach().mean())
    if offset != all_adv.numel():
        raise ValueError("State PPO advantage count mismatch")
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy = torch.stack(entropies).mean()
    total = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite state PPO loss")
    if scaler is None:
        total.backward()
        _average_distributed_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if torch.isfinite(torch.as_tensor(grad_norm, device=device)):
            optimizer.step()
        else:
            # Keep a single numerically bad batch from poisoning the recurrent
            # policy (and from cascading into a NCCL failure on the next step).
            optimizer.zero_grad(set_to_none=True)
    else:
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        _average_distributed_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if torch.isfinite(torch.as_tensor(grad_norm, device=device)):
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad(set_to_none=True)
            # unscale_ marks the scaler as consumed; advance it even when the
            # update is discarded so the next PPO batch can unscale normally.
            scaler.update()
    if hasattr(model, "mark_weights_updated"):
        model.mark_weights_updated()
    return {"loss": float(total.detach().cpu()), "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()), "entropy": float(entropy.detach().cpu()),
            "mean_ratio": float(torch.stack(ratios).mean().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu())}
