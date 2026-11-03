from __future__ import annotations

import torch


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
