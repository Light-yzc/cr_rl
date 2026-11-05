"""Explicit current / frozen PPO history / BC anchor opponent buckets."""

from dataclasses import asdict, dataclass

import numpy as np

FROZEN_VERSION_BASE = 1 << 30


def historical_policy_id(generation):
    if not 0 <= generation < FROZEN_VERSION_BASE:
        raise ValueError("Generation is outside the reserved version namespace")
    return FROZEN_VERSION_BASE + int(generation)


def frozen_policy_version(entry):
    """Use retained identity; restarted branches may share a generation number."""
    if entry.policy_id >= FROZEN_VERSION_BASE:
        if entry.policy_id >= 2 * FROZEN_VERSION_BASE:
            raise ValueError("Frozen policy identity does not fit signed trajectory versions")
        return entry.policy_id
    return historical_policy_id(entry.generation)


def next_history_snapshot_id(entries, generation):
    """Keep legacy generation IDs when free, avoid overwriting seeded snapshots."""
    proposed = historical_policy_id(generation)
    used = {frozen_policy_version(entry) for entry in entries}
    if proposed in used:
        proposed = max(used, default=FROZEN_VERSION_BASE) + 1
    if proposed >= 2 * FROZEN_VERSION_BASE:
        raise ValueError("Frozen policy identity namespace exhausted")
    return proposed


@dataclass(frozen=True)
class RoleMixture:
    current: float = 0.30
    historical: float = 0.50
    anchor: float = 0.20

    def __post_init__(self):
        values = (self.current, self.historical, self.anchor)
        if not all(np.isfinite(v) and v >= 0 for v in values) or not np.isclose(sum(values), 1):
            raise ValueError("Role fractions must be finite, nonnegative and sum to one")

    def state_dict(self):
        return asdict(self)

    def sample(self, pool, rng, own_deck, enemy_deck, learner_version):
        """Return opponent and requested/actual bucket; missing history falls back to BC."""
        draw = rng.random()
        requested = (
            "current"
            if draw < self.current
            else "historical"
            if draw < self.current + self.historical
            else "anchor"
        )
        if requested == "current":
            return None, requested, requested
        candidates = [e for e in pool.entries if e.role == requested]
        actual = requested
        if not candidates:
            actual = "anchor"
            candidates = [e for e in pool.entries if e.role == actual]
        if not candidates:
            raise ValueError("Role league requires a frozen BC anchor")
        scores = np.array(
            [pool.score_for(e, own_deck, enemy_deck, learner_version) for e in candidates]
        )
        weights = np.maximum(1 - scores, 1e-3)
        weights = (1 - pool.uniform_mix) * weights / weights.sum() + pool.uniform_mix / len(weights)
        return candidates[int(rng.choice(len(candidates), p=weights))], requested, actual
