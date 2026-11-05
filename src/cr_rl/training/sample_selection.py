"""Keep all valid learner states for values, and optional real choices for PPO."""

import numpy as np
import torch


def actor_critic_rows(eligible, legal_masks, *, choice_only=False):
    eligible = np.asarray(eligible, dtype=bool)
    legal = np.asarray(legal_masks, dtype=bool)
    if legal.shape[:-1] != eligible.shape:
        raise ValueError("One legal action mask is required for every eligibility row")
    critic = np.flatnonzero(eligible.reshape(-1))
    actor = eligible & (legal.sum(-1) > 1) if choice_only else eligible
    return np.flatnonzero(actor.reshape(-1)), critic


@torch.no_grad()
def normalize_actor_advantages(advantages, actor_indices):
    if not len(actor_indices):
        return torch.zeros_like(advantages)
    selected = advantages[torch.as_tensor(actor_indices, device=advantages.device)]
    return (advantages - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
