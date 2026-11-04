"""Explicit current / frozen PPO history / BC anchor opponent buckets."""

from dataclasses import asdict, dataclass

import numpy as np

FROZEN_VERSION_BASE = 1 << 30


def historical_policy_id(generation):
    if not 0 <= generation < FROZEN_VERSION_BASE:
        raise ValueError("Generation is outside the reserved version namespace")
    return FROZEN_VERSION_BASE + int(generation)


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
