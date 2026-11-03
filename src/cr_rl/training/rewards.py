"""Discount-consistent tower potentials, computed before simulator autoreset."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    princess: float = 0.15
    king: float = 0.20
    crown: float = 0.15
    overflow: float = 0.0
    enabled: bool = False

    def __post_init__(self):
        weights = (self.princess, self.king, self.crown, self.overflow)
        if not all(np.isfinite(w) and w >= 0 for w in weights):
            raise ValueError("Reward weights must be finite and nonnegative")

    def metadata(self):
        return {"version": 1, "formula": "gamma_phi_next_minus_phi_current", **asdict(self)}


def tower_potentials(observations, schema, config):
    """Return separate HP and crown potentials for each seat's raw observation."""

    def field(name):
        return observations[..., schema.header[name][0]].astype(np.float64)

    hp = np.zeros(observations.shape[:-1], dtype=np.float64)
    for tower, weight in (
        ("left", config.princess),
        ("right", config.princess),
        ("king", config.king),
    ):
        for side, sign in (("own", 1), ("enemy", -1)):
            maximum = field(f"{side}_{tower}_max_hp")
            fraction = np.divide(
                field(f"{side}_{tower}_hp"), maximum, out=np.zeros_like(maximum), where=maximum > 0
            )
            hp += sign * weight * np.clip(fraction, 0, 1)
    crown = config.crown * (field("crowns") - field("crowns_enemy"))
    return hp, crown


def shaped_rewards(
    current,
    following,
    actions,
    terminal_rewards,
    terminated,
    schema,
    gamma,
    config,
    *,
    faulted=None,
):
    """True terminals have zero next potential; time limits retain their potential.

    Faulted transitions get no shaping. Call before reset, using the final
    observation on time limits so that the shaped value can bootstrap correctly.
    Overflow is optional, deliberately non-invariant and not zero-sum.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    raw = np.asarray(terminal_rewards, dtype=np.float32)
    components = {name: np.zeros_like(raw) for name in ("hp", "crown", "overflow")}
    if config.enabled:
        before = tower_potentials(current, schema, config)
        after = tower_potentials(following, schema, config)
        terminal = np.asarray(terminated, dtype=bool)[..., None]
        for name, previous, subsequent in zip(("hp", "crown"), before, after, strict=True):
            components[name] = (gamma * np.where(terminal, 0, subsequent) - previous).astype(
                np.float32
            )
        elixir = current[..., schema.header["elixir"][0]]
        components["overflow"] = (-config.overflow * ((elixir >= 99000) & (actions == 0))).astype(
            np.float32
        )
        if faulted is not None:
            for value in components.values():
                value[np.asarray(faulted, dtype=bool)] = 0
    total = raw.copy()
    for value in components.values():
        total += value
    return total, components
