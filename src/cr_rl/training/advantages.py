from __future__ import annotations

import torch


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
