"""Independent actor/value updates with fixed rollout targets and critic warmup."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

from cr_rl.training.ppo import ppo_loss_with_reference, stop_for_kl


def complete_episode_returns(rewards, terminated, truncated, valid, gamma):
    """MC targets only for complete, fault-free episodes contained in this rollout.

    Partial tails and truncated/faulted games are excluded, never assigned a fake
    terminal zero. The simulator is reset at each rollout collection boundary.
    """
    targets = np.zeros_like(rewards, dtype=np.float32)
    selected = np.zeros_like(valid, dtype=bool)
    games = 0
    for env in range(rewards.shape[1]):
        left = 0
        for step in range(len(rewards)):
            if terminated[step, env] or truncated[step, env]:
                if terminated[step, env] and valid[left : step + 1, env].all():
                    total = 0.0
                    for row in range(step, left - 1, -1):
                        total = float(rewards[row, env]) + gamma * total
                        targets[row, env] = total
                    selected[left : step + 1, env] = True
                    games += 1
                left = step + 1
    return targets, selected, games


def forward_inputs(model, tensors):
    return model(
        *tensors[:5],
        hand_card_ids=tensors[5],
        deck_card_ids=tensors[6],
        enemy_seen_card_ids=tensors[7],
    )


def autocast_for(device):
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _optimizer_step(model, optimizer, samples, max_grad_norm):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(samples)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach().cpu())


def optimizer_steps(optimizer):
    return max((int(s["step"].item()) for s in optimizer.state.values()), default=0)


def optimize_separate(
    actor,
    critic,
    behavior,
    actor_optimizer,
    critic_optimizer,
    make_batch,
    actor_indices,
    critic_indices,
    old_logp,
    actions,
    advantages,
    value_targets,
    *,
    rng,
    device,
    batch_size,
    accumulation,
    actor_epochs,
    critic_epochs,
    clip_coef,
    entropy_coef,
    reference_kl_coef,
    target_kl,
    max_grad_norm,
    actor_enabled=True,
    cache_reference=False,
    defer_scalar_metrics=False,
    actor_batch_size=None,
    critic_batch_size=None,
    emit=lambda **event: None,
):
    """Critic fits independently, even when actor is frozen or fails its KL guard.

    make_batch maps flat rollout indices to raw-observation history tensors.
    Targets/log probabilities are snapshots, never recomputed after value fitting.
    """
    targets = value_targets.detach().clone()
    effective_batch = batch_size * accumulation
    actor_batch = batch_size if actor_batch_size is None else actor_batch_size
    critic_batch = batch_size if critic_batch_size is None else critic_batch_size
    if min(effective_batch, actor_batch, critic_batch) < 1 or any(
        effective_batch % value for value in (actor_batch, critic_batch)
    ):
        raise ValueError("Actor/Critic microbatches must divide the fixed effective batch")
    actor_accumulation = effective_batch // actor_batch
    critic_accumulation = effective_batch // critic_batch
    advantages = advantages.detach().clone()
    old_logp = old_logp.detach().clone()
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    actor_steps = value_steps = 0
    value_losses, policy_metrics = [], []
    initial_logp_errors = []
    reference_cache = None
    if actor_enabled and cache_reference:
        from cr_rl.training.reference_cache import ReferenceDistributionCache

        reference_cache = ReferenceDistributionCache(len(old_logp))
    critic.train()
    for epoch in range(critic_epochs):
        permutation = rng.permutation(critic_indices)
        accumulated = samples = 0
        for start in range(0, len(permutation), critic_batch):
            indices = permutation[start : start + critic_batch]
            with autocast_for(device):
                predictions = forward_inputs(critic, make_batch(indices)).float()
                loss = 0.5 * (predictions - targets[indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite independent value loss")
            (loss * len(indices)).backward()
            samples += len(indices)
            accumulated += 1
            value_losses.append(
                loss.detach() if defer_scalar_metrics else float(loss.detach().cpu())
            )
            if accumulated == critic_accumulation or start + critic_batch >= len(permutation):
                norm = _optimizer_step(critic, critic_optimizer, samples, max_grad_norm)
                value_steps += 1
                emit(
                    phase="critic_optimization",
                    epoch=epoch + 1,
                    critic_optimizer_steps=optimizer_steps(critic_optimizer),
                    value_loss=float(value_losses[-1].cpu())
                    if defer_scalar_metrics
                    else value_losses[-1],
                    grad_norm=norm,
                    learning_rate=float(critic_optimizer.param_groups[0]["lr"]),
                )
                accumulated = samples = 0
    critic.eval()
    actor.train()
    stopped = False
    if actor_enabled:
        for epoch in range(actor_epochs):
            permutation = rng.permutation(actor_indices)
            accumulated = samples = 0
            group_kl = group_approx_kl = 0.0
            for start in range(0, len(permutation), actor_batch):
                indices = permutation[start : start + actor_batch]
                tensors = make_batch(indices)
                with torch.no_grad(), autocast_for(device):
                    if reference_cache is None:
                        reference, _ = forward_inputs(behavior, tensors)
                    else:
                        reference = reference_cache.evaluate(
                            indices, lambda tensors=tensors: forward_inputs(behavior, tensors)[0]
                        )
                with autocast_for(device):
                    logp_all, _ = forward_inputs(actor, tensors)
                distribution = torch.distributions.Categorical(logits=logp_all.float())
                logp = distribution.log_prob(actions[indices])
                if actor_steps == 0:
                    initial_logp_errors.append(
                        (logp.detach() - old_logp[indices]).abs().max()
                        if defer_scalar_metrics
                        else float((logp.detach() - old_logp[indices]).abs().max())
                    )
                loss, metrics = ppo_loss_with_reference(
                    logp_all,
                    reference,
                    logp,
                    old_logp[indices],
                    advantages[indices],
                    torch.zeros_like(logp),
                    torch.zeros_like(logp),
                    distribution.entropy(),
                    clip_coef=clip_coef,
                    value_coef=0.0,
                    entropy_coef=entropy_coef,
                    reference_kl_coef=reference_kl_coef,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite independent policy loss")
                row = {
                    k: v.detach() if defer_scalar_metrics else float(v.cpu())
                    for k, v in metrics.items()
                }
                # Value fitting has its own real loss/axis; policy updates have no value term.
                row.pop("value_loss")
                policy_metrics.append(row)
                group_kl += (
                    row["reference_kl"].double() if defer_scalar_metrics else row["reference_kl"]
                ) * len(indices)
                group_approx_kl += (
                    row["approx_kl"].double() if defer_scalar_metrics else row["approx_kl"]
                ) * len(indices)
                (loss * len(indices)).backward()
                samples += len(indices)
                accumulated += 1
                if accumulated == actor_accumulation or start + actor_batch >= len(permutation):
                    # Judge the complete proposed optimizer batch, never a single microbatch.
                    average_kl = group_kl / samples
                    average_approx_kl = group_approx_kl / samples
                    if defer_scalar_metrics:
                        names = list(row)
                        scalars = (
                            torch.stack(
                                [row[name].double() for name in names]
                                + [average_kl, average_approx_kl, loss.detach().double()]
                            )
                            .cpu()
                            .tolist()
                        )
                        row = dict(zip(names, scalars[: len(names)], strict=True))
                        average_kl, average_approx_kl, current_loss = scalars[-3:]
                    if stop_for_kl(actor_optimizer, average_kl, target_kl):
                        emit(
                            phase="kl_stop",
                            epoch=epoch + 1,
                            approx_kl=average_approx_kl,
                            reference_kl=average_kl,
                            target_kl=target_kl,
                            discarded_microbatches=accumulated,
                            kl_samples=samples,
                            optimizer_steps=actor_steps,
                        )
                        stopped = True
                        break
                    norm = _optimizer_step(actor, actor_optimizer, samples, max_grad_norm)
                    actor_steps += 1
                    emit(
                        phase="optimization",
                        optimizer_steps=actor_steps,
                        optimizer_steps_total=optimizer_steps(actor_optimizer),
                        loss=current_loss if defer_scalar_metrics else float(loss.detach().cpu()),
                        grad_norm=norm,
                        learning_rate=float(actor_optimizer.param_groups[0]["lr"]),
                        **{**row, "reference_kl": average_kl, "approx_kl": average_approx_kl},
                    )
                    accumulated = samples = 0
                    group_kl = group_approx_kl = 0.0
            if stopped:
                break
    actor.eval()

    def mean_metric(values):
        if defer_scalar_metrics:
            return float(torch.stack(values).double().mean().cpu())
        return float(np.mean(values))

    return {
        "actor_steps": actor_steps,
        "critic_steps": value_steps,
        "actor_kl_stopped": stopped,
        "initial_logp_errors": (
            torch.stack(initial_logp_errors).cpu().tolist()
            if defer_scalar_metrics and initial_logp_errors
            else initial_logp_errors
        ),
        "reference_cache": {
            "enabled": reference_cache is not None,
            "forward_calls": reference_cache.forward_calls if reference_cache else 0,
            "hit_rows": reference_cache.hit_rows if reference_cache else 0,
        },
        "metrics": {
            **(
                {k: mean_metric([r[k] for r in policy_metrics]) for k in policy_metrics[0]}
                if policy_metrics
                else {}
            ),
            "value_loss": mean_metric(value_losses) if value_losses else None,
        },
    }
