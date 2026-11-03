"""Episode-owned history storage across rollout/update boundaries."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeTrace:
    episode_id: int
    seed: int
    observations: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    learners: list = field(default_factory=list)
    valid: bool = True
    completed: bool = False

    def append(self, observations, masks, rewards, learners=None):
        self.observations.append(np.array(observations, copy=True))
        self.masks.append(np.array(masks, copy=True))
        self.rewards.append(np.array(rewards, dtype=np.float32, copy=True))
        self.learners.append(
            np.ones(2, dtype=np.uint8)
            if learners is None
            else np.array(learners, dtype=np.uint8, copy=True)
        )
        return len(self.observations) - 1

    def window(self, index, seat, context_len):
        if not 0 <= index < len(self.observations):
            raise IndexError("Transition is outside its episode")
        sequence = self.observations[max(0, index - context_len + 1) : index + 1]
        window = np.zeros((context_len, sequence[0].shape[-1]), dtype=sequence[0].dtype)
        window[-len(sequence) :] = np.stack(sequence)[:, seat]
        time_mask = np.arange(context_len) >= context_len - len(sequence)
        return window, time_mask

    def finish(self, *, terminated, faulted=False):
        self.valid &= not faulted
        self.completed = bool(terminated and self.valid)

    def monte_carlo_returns(self, gamma):
        if not self.completed:
            raise ValueError("Value warmup requires an entire valid terminal episode")
        result = np.zeros_like(self.rewards, dtype=np.float32)
        total = np.zeros_like(result[0])
        for index in reversed(range(len(result))):
            total = self.rewards[index] + gamma * total
            result[index] = total
        return result


def on_policy_seats(policy_versions, learner_version, designated_seats):
    versions = np.asarray(policy_versions)
    mask = versions == learner_version
    if not mask[np.arange(len(versions)), designated_seats].all():
        raise ValueError("Designated learner is not the current behavior policy")
    return mask.astype(np.uint8)
