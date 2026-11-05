from __future__ import annotations

import math

import torch


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    next_values: torch.Tensor | None = None,
    episode_ends: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over [time, env] with separate bootstrap and trace-boundary masks.

    ``dones`` means true terminal states and suppresses value bootstrapping.
    ``episode_ends`` also includes time-limit truncations and prevents a trace from
    crossing into the next autoreset episode. ``next_values`` may carry the value of
    the final pre-reset observation for a truncated transition.
    """
    if next_values is None:
        next_values = torch.cat((values[1:], last_value.unsqueeze(0)), dim=0)
    if episode_ends is None:
        episode_ends = dones
    if next_values.shape != rewards.shape or episode_ends.shape != rewards.shape:
        raise ValueError("next_values and episode_ends must match rewards shape [time, env]")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        bootstrap = 1.0 - dones[t].float()
        continuation = 1.0 - episode_ends[t].float()
        delta = rewards[t] + gamma * next_values[t] * bootstrap - values[t]
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[t] = gae
    return advantages, advantages + values


@torch.no_grad()
def compute_actor_critic_targets(
    rewards,
    values,
    dones,
    last_value,
    *,
    gamma,
    gae_lambda,
    critic_gae_lambda=None,
    next_values=None,
    episode_ends=None,
):
    """Freeze Actor advantages and optional independent Critic TD(lambda) targets.

    The default exactly preserves the shared GAE target. Critic lambda=1 uses
    realized discounted rewards up to terminals, with the stored bootstrap for
    unfinished rollout tails or time limits. It does not invent terminal zeros.
    Both targets use collection-time values, before either optimizer runs.
    """
    if critic_gae_lambda is not None and (
        not math.isfinite(critic_gae_lambda) or not 0 <= critic_gae_lambda <= 1
    ):
        raise ValueError("Critic GAE lambda must be finite and in [0, 1]")
    arguments = dict(
        gamma=gamma, next_values=next_values, episode_ends=episode_ends
    )
    advantages, targets = compute_gae(
        rewards, values, dones, last_value, gae_lambda=gae_lambda, **arguments
    )
    if critic_gae_lambda is not None and critic_gae_lambda != gae_lambda:
        _, targets = compute_gae(
            rewards, values, dones, last_value, gae_lambda=critic_gae_lambda, **arguments
        )
    return advantages, targets
