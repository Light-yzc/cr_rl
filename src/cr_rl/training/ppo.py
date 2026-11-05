from __future__ import annotations

import math

import torch


def factorized_reverse_kl(new_log_probs, reference_log_probs):
    """Per-state reverse KL split by WAIT/card/ability and conditional tile.

    Placement KL is weighted by the new policy's card probability, as required
    by the chain rule. Cardinality alone does not determine either component.
    """
    if new_log_probs.shape != reference_log_probs.shape or new_log_probs.shape[-1] != 2307:
        raise ValueError("Factorized KL requires matching 2307-action distributions")
    new = torch.log_softmax(new_log_probs.float(), dim=-1)
    reference = torch.log_softmax(reference_log_probs.float(), dim=-1)

    def card_logp(joint):
        return torch.cat(
            (
                joint[..., :1],
                joint[..., 1:2305].reshape(*joint.shape[:-1], 4, 576).logsumexp(-1),
                joint[..., 2305:2307],
            ),
            dim=-1,
        )

    card, reference_card = card_logp(new), card_logp(reference)
    card_difference = torch.where(card.exp() > 0, card - reference_card, 0)
    card_kl = (card.exp() * card_difference).sum(-1)
    tiles = new[..., 1:2305].reshape(*new.shape[:-1], 4, 576)
    reference_tiles = reference[..., 1:2305].reshape(*new.shape[:-1], 4, 576)
    conditional_difference = (
        tiles - card[..., 1:5, None] - reference_tiles + reference_card[..., 1:5, None]
    )
    conditional_difference = torch.where(tiles.exp() > 0, conditional_difference, 0)
    tile_kl = (tiles.exp() * conditional_difference).sum((-1, -2))
    return card_kl, tile_kl


def stop_for_factorized_kl(optimizer, card_kl, tile_kl, target_card_kl, target_tile_kl):
    if not all(math.isfinite(x) and x > 0 for x in (target_card_kl, target_tile_kl)):
        raise ValueError("Both factorized KL limits must be finite and positive")
    if card_kl <= target_card_kl and tile_kl <= target_tile_kl:
        return False
    optimizer.zero_grad(set_to_none=True)
    return True


def stop_for_kl(optimizer: torch.optim.Optimizer, approx_kl: float, target_kl: float) -> bool:
    """Reject an over-limit microbatch and discard any pending accumulated gradients.

    Call before backward/step: applying one last update after detecting drift defeats
    the early-stop guard. Previously completed optimizer steps are retained.
    """
    if approx_kl <= target_kl:
        return False
    optimizer.zero_grad(set_to_none=True)
    return True


def ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_coef: float = 0.15,
    value_coef: float = 0.5,
    entropy_coef: float = 0.003,
):
    """Clipped PPO objective for already-recomputed sequence-window log probabilities."""
    ratio = (new_log_prob - old_log_prob).exp()
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    policy = torch.maximum(unclipped, clipped).mean()
    value = 0.5 * (new_value - returns).square().mean()
    entropy_mean = entropy.mean()
    total = policy + value_coef * value - entropy_coef * entropy_mean
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - (new_log_prob - old_log_prob)).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return total, {
        "policy_loss": policy.detach(),
        "value_loss": value.detach(),
        "entropy": entropy_mean.detach(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def ppo_loss_with_reference(
    new_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    new_action_log_prob: torch.Tensor,
    old_action_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_coef: float = 0.15,
    value_coef: float = 0.5,
    entropy_coef: float = 0.003,
    reference_kl_coef: float = 0.01,
):
    """PPO with an explicit reverse-KL trust region to the rollout policy."""
    loss, metrics = ppo_loss(
        new_action_log_prob,
        old_action_log_prob,
        advantages,
        new_value,
        returns,
        entropy,
        clip_coef,
        value_coef,
        entropy_coef,
    )
    if new_log_probs.shape != reference_log_probs.shape:
        raise ValueError("new and reference action distributions must have identical shapes")
    new_log_probs = new_log_probs.float()
    reference_log_probs = reference_log_probs.float()
    new_probabilities = new_log_probs.exp()
    reference_kl = (new_probabilities * (new_log_probs - reference_log_probs)).sum(dim=-1).mean()
    loss = loss + reference_kl_coef * reference_kl
    metrics["reference_kl"] = reference_kl.detach()
    return loss, metrics
