"""Discount-consistent battle potentials, computed before simulator autoreset."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    princess: float = 0.15
    king: float = 0.20
    crown: float = 0.15
    overflow: float = 0.0
    enabled: bool = False
    army_potential_weight: float = 0.0
    army_reward_mode: str = 'uniform'
    army_elixir_beta: float = .004
    army_elixir_values: dict | None = None
    army_elixir_bar_factor: float = 0.0

    def __post_init__(self):
        weights = (self.princess, self.king, self.crown, self.overflow, self.army_potential_weight)
        if not all(np.isfinite(w) and w >= 0 for w in weights):
            raise ValueError("Reward weights must be finite and nonnegative")
        if self.army_reward_mode not in ('uniform','elixir_potential'):
            raise ValueError('Unknown army reward mode')
        if not np.isfinite(self.army_elixir_beta) or self.army_elixir_beta<0:
            raise ValueError('Invalid army elixir beta')
        if not np.isfinite(self.army_elixir_bar_factor) or self.army_elixir_bar_factor < 0:
            raise ValueError('Invalid army elixir bar factor')
        if self.army_elixir_bar_factor and self.army_reward_mode != 'elixir_potential':
            raise ValueError('Army elixir bar shaping requires elixir_potential')
        if self.army_reward_mode=='elixir_potential' and not self.army_elixir_values:
            raise ValueError('Elixir mode requires a versioned value configuration')

    def metadata(self):
        fields=asdict(self)
        if not self.army_elixir_bar_factor:
            del fields['army_elixir_bar_factor']
        if self.army_reward_mode=='uniform':
            for key in ('army_reward_mode','army_elixir_beta','army_elixir_values'): fields.pop(key)
            return {"version": 1, "formula": "gamma_phi_next_minus_phi_current", **fields}
        version = 3 if self.army_elixir_bar_factor else 2
        return {"version": version, "formula": "gamma_phi_next_minus_phi_current", **fields,
                'ledger_lifecycle':'native_world_reset_on_resume',
                'lineage':'successful_native_play_direct_units_only'}


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


def army_potentials(observations, schema, config):
    """Uniform unit weight: never charge a full card cost for every swarm member."""
    rows = observations[..., schema.header_ints:].reshape(*observations.shape[:-1], schema.max_entities, schema.entity_ints)
    e = schema.entity
    count = observations[..., schema.header["entity_count"][0]]
    valid = np.arange(schema.max_entities) < count[..., None]
    hp = rows[..., e["hp"]].astype(np.float64)
    maximum = np.maximum(rows[..., e["max_hp"]], 1)
    # KIND_CHARACTER=1 includes deployed troops and buildings; towers are kind 2.
    alive = valid & (hp > 0) & (rows[..., e["kind"]] == 1)
    health = 0.3 + 0.7*np.clip(hp/maximum, 0, 1)
    side = rows[..., e["side"]]
    balance = np.where(side == 0, 1, np.where(side == 1, -1, 0))
    return config.army_potential_weight * (alive*health*balance).sum(-1)/10.0


def _elixir_bar_difference(observations, schema, *, paired):
    def available(elixir, reserved):
        amount = observations[..., schema.header[elixir][0]].astype(np.float64)
        if reserved in schema.header:
            amount -= np.maximum(observations[..., schema.header[reserved][0]], 0)
        return np.maximum(amount, 0)

    own = available('elixir', 'elixir_reserved')
    if paired:
        difference = own - own[..., ::-1]
    else:
        # Single-seat observations only expose the simulator's tracked enemy bar.
        difference = own - available('enemy_elixir', 'enemy_elixir_reserved')
    return difference / schema.scales['elixir_scale']


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
    army_values=None,
):
    """True terminals have zero next potential; time limits retain their potential.

    Faulted transitions get no shaping. Call before reset, using the final
    observation on time limits so that the shaped value can bootstrap correctly.
    For paired observations, supply one terminal flag per lane. For independent
    single-seat observations, match the terminal flags to the raw reward shape.
    Overflow is optional, deliberately non-invariant and not zero-sum.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    raw = np.asarray(terminal_rewards, dtype=np.float32)
    components = {name: np.zeros_like(raw) for name in ("hp", "crown", "army", "overflow")}
    if config.enabled:
        before = tower_potentials(current, schema, config)
        after = tower_potentials(following, schema, config)
        terminal = np.asarray(terminated, dtype=bool)
        paired = raw.ndim > terminal.ndim and raw.shape[-1] == 2
        if terminal.ndim < raw.ndim:
            terminal = terminal.reshape(terminal.shape + (1,) * (raw.ndim - terminal.ndim))
        for name, previous, subsequent in zip(("hp", "crown"), before, after, strict=True):
            components[name] = (gamma * np.where(terminal, 0, subsequent) - previous).astype(
                np.float32
            )
        if config.army_reward_mode=='elixir_potential':
            if army_values is None: raise ValueError('Missing pre/post native army values')
            previous, subsequent = (np.asarray(a, dtype=np.float64) for a in army_values)
            if previous.shape!=raw.shape or subsequent.shape!=raw.shape:
                raise ValueError('Native army values have wrong shape')
            if config.army_elixir_bar_factor:
                previous = previous + config.army_elixir_bar_factor * _elixir_bar_difference(
                    current, schema, paired=paired)
                subsequent = subsequent + config.army_elixir_bar_factor * _elixir_bar_difference(
                    following, schema, paired=paired)
            previous = previous * config.army_elixir_beta
            subsequent = subsequent * config.army_elixir_beta
            if not np.isfinite(previous).all() or not np.isfinite(subsequent).all():
                raise ValueError('Nonfinite native army value')
            components['army']=(gamma*np.where(terminal,0,subsequent)-previous).astype(np.float32)
        elif config.army_potential_weight:
            previous = army_potentials(current, schema, config)
            subsequent = army_potentials(following, schema, config)
            components["army"] = (gamma*np.where(terminal, 0, subsequent)-previous).astype(np.float32)
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
