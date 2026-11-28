"""Test-time rollout lookahead search plugin for architecture-v3 policies.

At high-leverage decision frames (available elixir >= 6.0 or contested defensive
frames), forks K macro-distinct candidate actions (WAIT + top distinct hand
slots/abilities) across K * R parallel CrForge simulator lanes for H steps
(~4.0s), evaluates leaf states with the independent StateValueNetwork Critic
(or zero-sum tower/crown/elixir potential when actor-only), and picks the
highest-value action using Common Random Numbers (CRN) across candidates.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np
import torch

from cr_rl.models.state_policy import (
    PolicyMemory,
    StateActorCritic,
    StateValueNetwork,
    reserve_step_memory,
    slice_memory,
    stack_memories,
)


DECISION_SECONDS = 0.4


@dataclass(frozen=True)
class LookaheadConfig:
    max_candidates: int = 4
    rollouts_per_candidate: int = 3
    horizon_steps: int = 10  # 10 * 0.4s = 4.0 seconds
    trigger_elixir: float = 5.5
    defense_elixir: float = 4.0
    bank_min_elixir: float = 3.8
    bank_min_card_cost: float = 5.0
    bank_max_deficit: float = 1.5
    max_bank_steps: int = 8  # 8 * 0.4s = 3.2 seconds max commitment hold
    entropy_threshold: float = 0.50
    min_search_interval_steps: int = 5  # 2.0s cooldown unless a new integer elixir tier unlocks
    max_searches_per_game: int = 40
    gamma: float = 0.9995
    prior_weight: float = 0.008
    max_prior_penalty_nats: float = 2.5
    override_margin: float = 0.004
    threads: int = 6


def encode_bank_candidate(slot: int) -> int:
    """Encode 'hold WAIT until hand `slot` unlocks, then deploy `slot`' as -(slot + 1)."""
    if not 0 <= int(slot) < 4:
        raise ValueError(f"Expected hand slot in 0..3, got {slot}")
    return -(int(slot) + 1)


def decode_bank_candidate(candidate: int) -> int | None:
    """Return target hand slot in 0..3 for BANK_FOR_SLOT candidates (-1..-4), else None."""
    c = int(candidate)
    if not -4 <= c <= -1:
        return None
    return (-c) - 1


def encode_hold_full_candidate(slot: int) -> int:
    """Encode 'hold WAIT on quiet board until >=9.4 elixir, then deploy `slot`' as -(slot + 5)."""
    if not 0 <= int(slot) < 4:
        raise ValueError(f"Expected hand slot in 0..3, got {slot}")
    return -(int(slot) + 5)


def decode_hold_full_candidate(candidate: int) -> int | None:
    """Return target hand slot in 0..3 for HOLD_TO_FULL candidates (-5..-8), else None."""
    c = int(candidate)
    if not -8 <= c <= -5:
        return None
    return (-c) - 5


def _hand_costs_bars(raw_obs: np.ndarray, schema) -> np.ndarray | None:
    cost_idx = schema.header.get("hand_cost")
    if not cost_idx:
        return None
    vals = np.asarray([float(raw_obs[i]) for i in cost_idx], dtype=np.float64)
    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
    if scale > 1.0 and np.any(vals > 20.0):
        vals = np.where(vals >= 0.0, vals / scale, vals)
    return vals


def _emergency_tower_threat(raw_obs: np.ndarray, schema) -> bool:
    """Return True if an enemy troop is at/past the bridge or an enemy projectile/spell is incoming."""
    if not hasattr(schema, "entities"):
        return False
    e = schema.entity
    rows = schema.entities(raw_obs)
    enemy = rows[rows[:, e["side"]] == 1]
    if not len(enemy):
        return False
    kinds = enemy[:, e["kind"]]
    bridge_y = int(getattr(schema, "height_units", 32000)) // 2 + 2 * int(getattr(schema, "tile_units", 1000))
    near_troop = (
        (kinds == schema.kinds["character"])
        & ((enemy[:, e["flags"]] & schema.flags["alive"]) != 0)
        & (enemy[:, e["hp"]] > 0)
        & (enemy[:, e["y"]] <= bridge_y)
    )
    incoming_spell = (kinds == schema.kinds["projectile"]) & (enemy[:, e["y"]] <= bridge_y - 3000)
    return bool((near_troop | incoming_spell).any())


def _critical_low_own_tower(raw_obs: np.ndarray, schema, hp_limit: int = 400) -> bool:
    for name in ("own_left_hp", "own_right_hp", "own_king_hp"):
        idx = schema.header.get(name)
        if idx:
            hp = int(raw_obs[idx[0]])
            if 0 < hp <= hp_limit:
                return True
    return False


def _unaffordable_bank_slots(
    raw_obs: np.ndarray,
    legal_mask: np.ndarray,
    schema,
    *,
    min_card_cost: float = 5.0,
    max_deficit: float = 2.2,
) -> list[int]:
    """Return unaffordable hand slots in [min_card_cost, avail + eff_deficit] worth banking for."""
    legal = np.asarray(legal_mask, dtype=bool).reshape(-1)
    if not legal[0]:
        return []
    costs = _hand_costs_bars(raw_obs, schema)
    if costs is None:
        return []
    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
    elix = float(raw_obs[schema.header["elixir"][0]])
    res = float(raw_obs[schema.header["elixir_reserved"][0]]) if "elixir_reserved" in schema.header else 0.0
    avail = (elix - res) / scale
    if avail >= 9.8:
        return []
    emergency = _emergency_tower_threat(raw_obs, schema)
    if emergency:
        if _critical_low_own_tower(raw_obs, schema):
            return []
        eff_max_deficit = min(float(max_deficit), 1.2)
        eff_min_cost = min(float(min_card_cost), 4.0)
    else:
        eff_max_deficit = float(max_deficit)
        eff_min_cost = float(min_card_cost)

    cols = int(getattr(schema, "cols", 18))
    rows = int(getattr(schema, "rows", 32))
    tiles = cols * rows
    hand_idx = schema.header.get("hand_card")

    scored: list[tuple[float, float, int]] = []
    for slot in range(min(len(costs), int(getattr(schema, "hand_slots", 4)))):
        if hand_idx is not None and int(raw_obs[hand_idx[slot]]) < 0:
            continue
        lo = 1 + slot * tiles
        hi = lo + tiles
        if legal[lo:hi].any():
            continue
        c = float(costs[slot])
        deficit = c - avail
        if c >= eff_min_cost and 0.0 < deficit <= eff_max_deficit:
            # Prefer higher-cost win-condition/pump/anti-air cards first, then smaller deficit
            scored.append((-c, deficit, slot))
    scored.sort()
    return [slot for _neg_c, _def, slot in scored]


def _hold_to_full_slots(
    raw_obs: np.ndarray,
    legal_mask: np.ndarray,
    schema,
    *,
    min_card_cost: float = 5.0,
    min_elixir: float = 5.5,
    full_elixir: float = 9.2,
) -> list[int]:
    """Return already-legal >=5-cost non-global-spell slots worth holding to ~9.4 elixir on quiet boards."""
    legal = np.asarray(legal_mask, dtype=bool).reshape(-1)
    if not legal[0] or _emergency_tower_threat(raw_obs, schema):
        return []
    costs = _hand_costs_bars(raw_obs, schema)
    if costs is None:
        return []
    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
    elix = float(raw_obs[schema.header["elixir"][0]])
    res = float(raw_obs[schema.header["elixir_reserved"][0]]) if "elixir_reserved" in schema.header else 0.0
    avail = (elix - res) / scale
    if not (min_elixir <= avail <= full_elixir):
        return []

    cols = int(getattr(schema, "cols", 18))
    rows = int(getattr(schema, "rows", 32))
    tiles = cols * rows
    hand_idx = schema.header.get("hand_card")

    scored: list[tuple[float, int]] = []
    for slot in range(min(len(costs), int(getattr(schema, "hand_slots", 4)))):
        if hand_idx is not None and int(raw_obs[hand_idx[slot]]) < 0:
            continue
        lo = 1 + slot * tiles
        hi = lo + tiles
        n_leg = int(legal[lo:hi].sum())
        if not (0 < n_leg < GLOBAL_SPELL_MIN_TILES):
            continue
        c = float(costs[slot])
        if c >= min_card_cost:
            scored.append((-c, slot))
    scored.sort()
    return [slot for _neg_c, slot in scored]


GLOBAL_SPELL_MIN_TILES = 500


def _hierarchical_default_action(lp: np.ndarray, legal: np.ndarray, raw_obs: np.ndarray, schema) -> int:
    """Match live `--action-mode argmax`: compare WAIT vs slot total mass vs ability mass, then pick slot peak tile."""
    legal_b = np.asarray(legal, dtype=bool).reshape(-1)
    lp_1d = np.asarray(lp, dtype=np.float64).reshape(-1)
    masked = np.where(legal_b, lp_1d, -np.inf)
    max_lp = float(np.max(masked))
    if not np.isfinite(max_lp):
        return 0
    probs = np.where(legal_b, np.exp(masked - max_lp), 0.0)
    tot = float(probs.sum())
    if tot <= 0.0:
        return int(np.argmax(masked))
    probs /= tot

    cols = int(getattr(schema, "cols", 18))
    rows_n = int(getattr(schema, "rows", 32))
    tiles = cols * rows_n
    hand_slots = int(getattr(schema, "hand_slots", 4))
    first_ab = 1 + hand_slots * tiles

    wait_mass = float(probs[0]) if legal_b[0] else -1.0
    slot_masses = probs[1:first_ab].reshape(hand_slots, tiles).sum(axis=-1)
    skill_masses = probs[first_ab:] if first_ab < len(probs) else np.zeros(0, dtype=np.float64)

    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
    elix_idx = (getattr(schema, "header", None) or {}).get("elixir")
    avail_bars = (float(raw_obs[elix_idx[0]]) / scale) if elix_idx else 0.0
    non_wait_total = (float(slot_masses.sum()) if len(slot_masses) else 0.0) + (
        float(skill_masses.sum()) if len(skill_masses) else 0.0
    )
    near_full = avail_bars >= 9.5 and non_wait_total >= 0.05

    if (not near_full) and legal_b[0] and wait_mass >= non_wait_total:
        return 0
    if len(skill_masses):
        sk = int(np.argmax(skill_masses))
        if skill_masses[sk] >= float(slot_masses.max()) and skill_masses[sk] > 0.0 and legal_b[first_ab + sk]:
            return first_ab + sk
    best_slot = int(np.argmax(slot_masses))
    lo = 1 + best_slot * tiles
    hi = lo + tiles
    if slot_masses[best_slot] > 0.0 and legal_b[lo:hi].any():
        return lo + int(np.argmax(masked[lo:hi]))
    return int(np.argmax(masked))


def _is_engaged_global_spell(
    action: int,
    raw_obs: np.ndarray,
    legal_mask: np.ndarray,
    schema,
    *,
    rage_card_ids: Sequence[int] = (),
) -> bool:
    """Return False if `action` casts a global spell (especially Rage) onto empty/non-combat space."""
    if not hasattr(schema, "entities"):
        return True
    cols = int(getattr(schema, "cols", 18))
    rows_n = int(getattr(schema, "rows", 32))
    tiles = cols * rows_n
    hand_slots = int(getattr(schema, "hand_slots", 4))
    if not (0 < int(action) < 1 + hand_slots * tiles):
        return True
    legal = np.asarray(legal_mask, dtype=bool).reshape(-1)
    slot, rel = divmod(int(action) - 1, tiles)
    lo = 1 + slot * tiles
    hi = lo + tiles
    if int(legal[lo:hi].sum()) < GLOBAL_SPELL_MIN_TILES:
        return True

    row_t, col_t = divmod(rel, cols)
    tu = float(getattr(schema, "tile_units", 1000))
    tx = (col_t + 0.5) * tu
    ty = (row_t + 0.5) * tu

    e = schema.entity
    ents = schema.entities(raw_obs)
    if not len(ents):
        return False
    alive_flag = schema.flags.get("alive", 0)
    bldg_flag = schema.flags.get("building", 0)
    chars = (ents[:, e["kind"]] == schema.kinds["character"]) & (ents[:, e["hp"]] > 0)
    if alive_flag:
        chars &= (ents[:, e["flags"]] & alive_flag) != 0
    towers = (ents[:, e["kind"]] == schema.kinds["tower"]) & (ents[:, e["hp"]] > 0)

    enemy_c = ents[chars & (ents[:, e["side"]] == 1)]
    enemy_t = ents[towers & (ents[:, e["side"]] == 1)]
    own_troop_mask = chars & (ents[:, e["side"]] == 0)
    if bldg_flag:
        own_troop_mask &= (ents[:, e["flags"]] & bldg_flag) == 0
    own_c = ents[own_troop_mask]

    hand_idx = schema.header.get("hand_card")
    card_id = int(raw_obs[hand_idx[slot]]) if hand_idx is not None else -1
    is_rage = bool(rage_card_ids) and (card_id in rage_card_ids)

    if is_rage:
        # 1. Allow finishing off a critically low enemy tower (<= 170 HP) within Rage splash radius
        if len(enemy_t):
            d2_et = (enemy_t[:, e["x"]] - tx) ** 2 + (enemy_t[:, e["y"]] - ty) ** 2
            if bool(((d2_et <= (2.8 * tu) ** 2) & (enemy_t[:, e["hp"]] <= 170)).any()):
                return True
        # 2. Avoid double-stacking onto an already-active friendly area spell (e.g. Rage)
        area_kind = schema.kinds.get("area", -1)
        if area_kind >= 0:
            own_areas = ents[(ents[:, e["side"]] == 0) & (ents[:, e["kind"]] == area_kind)]
            if len(own_areas):
                d2_a = (own_areas[:, e["x"]] - tx) ** 2 + (own_areas[:, e["y"]] - ty) ** 2
                if bool((d2_a <= (3.0 * tu) ** 2).any()):
                    return False
        # 3. Swarm clear: >=2 barrel/swarm units on own side (y<=12.0, hp<=340) OR >=5 lethal-splash swarm anywhere (y<=26.0, hp<=205)
        if ty <= 26.0 * tu and len(enemy_c):
            d2_ec = (enemy_c[:, e["x"]] - tx) ** 2 + (enemy_c[:, e["y"]] - ty) ** 2
            hp_cap = 340 if ty <= 12.0 * tu else 205
            swarm_cnt = int(((d2_ec <= (3.2 * tu) ** 2) & (enemy_c[:, e["hp"]] <= hp_cap)).sum())
            if swarm_cnt >= (2 if ty <= 12.0 * tu else 5):
                return True
        # 4. Engaged friendly troops (any elixir >= 2.0, offense or defense) OR proactive heavy bridge push
        if not len(own_c):
            return False
        d2_own = (own_c[:, e["x"]] - tx) ** 2 + (own_c[:, e["y"]] - ty) ** 2
        buffed = own_c[d2_own <= (4.2 * tu) ** 2]
        buffed_hp = int(buffed[:, e["hp"]].sum()) if len(buffed) else 0
        if buffed_hp < 600:
            return False
        # Proactive heavy bridge push (>=1800 HP at/past bridge marching toward alive enemy tower)
        if buffed_hp >= 1800 and float(buffed[:, e["y"]].max()) >= 13.5 * tu and len(enemy_t):
            return True
        targets = np.concatenate([enemy_c, enemy_t], axis=0) if (len(enemy_c) or len(enemy_t)) else np.empty((0, ents.shape[1]))
        if not len(targets):
            return False
        dx = buffed[:, None, e["x"]] - targets[None, :, e["x"]]
        dy = buffed[:, None, e["y"]] - targets[None, :, e["y"]]
        return bool(((dx * dx + dy * dy) <= (7.0 * tu) ** 2).any())

    if len(enemy_c):
        d2_ec = (enemy_c[:, e["x"]] - tx) ** 2 + (enemy_c[:, e["y"]] - ty) ** 2
        if bool((d2_ec <= (5.2 * tu) ** 2).any()):
            return True
    if len(enemy_t):
        d2_et = (enemy_t[:, e["x"]] - tx) ** 2 + (enemy_t[:, e["y"]] - ty) ** 2
        if bool(((d2_et <= (5.2 * tu) ** 2) & (enemy_t[:, e["hp"]] <= 650)).any()):
            return True
    if len(own_c) and (len(enemy_c) or len(enemy_t)):
        d2_own = (own_c[:, e["x"]] - tx) ** 2 + (own_c[:, e["y"]] - ty) ** 2
        if bool((d2_own <= (4.5 * tu) ** 2).any()):
            targets = np.concatenate([enemy_c, enemy_t], axis=0)
            d2_any = (targets[:, e["x"]] - tx) ** 2 + (targets[:, e["y"]] - ty) ** 2
            if bool((d2_any <= (6.5 * tu) ** 2).any()):
                return True
    return False


def _front_tank_candidates(
    raw_obs: np.ndarray,
    legal_mask: np.ndarray,
    masked_lp: np.ndarray,
    schema,
    *,
    giant_card_ids: Sequence[int] = (),
    non_tank_card_ids: Sequence[int] = (),
) -> tuple[list[int], list[int]]:
    """Return `(immediate_front_tank_actions, bank_for_giant_slots)` ahead of advancing friendly support units."""
    if not hasattr(schema, "entities"):
        return [], []
    e = schema.entity
    ents = schema.entities(raw_obs)
    if not len(ents):
        return [], []
    tu = int(getattr(schema, "tile_units", 1000))
    width_u = int(getattr(schema, "width_units", 18000))
    alive_flag = schema.flags.get("alive", 0)
    bldg_flag = schema.flags.get("building", 0)

    own_mask = (
        (ents[:, e["side"]] == 0)
        & (ents[:, e["kind"]] == schema.kinds["character"])
        & (ents[:, e["hp"]] > 0)
    )
    if alive_flag:
        own_mask &= (ents[:, e["flags"]] & alive_flag) != 0
    if bldg_flag:
        own_mask &= (ents[:, e["flags"]] & bldg_flag) == 0
    own_troops = ents[own_mask]
    if not len(own_troops):
        return [], []

    # Advancing units in own mid-field / bridge zone [3.5, 18.0] tiles that lack a >=2400 HP tank ahead in lane
    advancing = own_troops[(own_troops[:, e["y"]] >= int(3.5 * tu)) & (own_troops[:, e["y"]] <= int(18.0 * tu))]
    if not len(advancing):
        return [], []

    unshielded = []
    for unit in advancing:
        ux, uy = int(unit[e["x"]]), int(unit[e["y"]])
        same_lane = np.abs(own_troops[:, e["x"]] - ux) <= int(4.5 * tu)
        ahead_tank = (
            same_lane
            & (own_troops[:, e["y"]] > uy + tu)
            & (own_troops[:, e["y"]] <= uy + int(6.5 * tu))
            & (own_troops[:, e["max_hp"]] >= 2600)
            & (own_troops[:, e["hp"]] >= 1400)
        )
        if not bool(ahead_tank.any()):
            unshielded.append((int(unit[e["max_hp"]] >= 1000), int(uy), ux, uy))
    if not unshielded:
        return [], []
    unshielded.sort(reverse=True)

    cols = int(getattr(schema, "cols", 18))
    rows_n = int(getattr(schema, "rows", 32))
    tiles = cols * rows_n
    hand_slots = int(getattr(schema, "hand_slots", 4))
    legal = np.asarray(legal_mask, dtype=bool).reshape(-1)
    costs = _hand_costs_bars(raw_obs, schema)
    hand_idx = schema.header.get("hand_card")
    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
    elix = float(raw_obs[schema.header["elixir"][0]])
    res = float(raw_obs[schema.header["elixir_reserved"][0]]) if "elixir_reserved" in schema.header else 0.0
    avail = (elix - res) / scale

    # Check if an unaffordable Giant (deficit <= 1.5 bars) is in hand while backliner hasn't crossed bridge yet!
    bank_giant_slots: list[int] = []
    if legal[0] and not _critical_low_own_tower(raw_obs, schema) and any(uy <= int(14.5 * tu) for _p, _ys, _ux, uy in unshielded):
        for slot in range(hand_slots):
            cid = int(raw_obs[hand_idx[slot]]) if (hand_idx is not None and slot < len(hand_idx)) else -1
            lo = 1 + slot * tiles
            hi = lo + tiles
            if legal[lo:hi].any():
                continue
            c_val = float(costs[slot]) if (costs is not None and slot < len(costs)) else -1.0
            is_giant = (cid in giant_card_ids) if giant_card_ids else (abs(c_val - 5.0) < 0.2 and cid not in non_tank_card_ids)
            if is_giant and 0.0 < (c_val - avail) <= 1.5:
                bank_giant_slots.append(slot)

    # Identify legal non-global-spell frontliner slots, prioritizing Giant over 4-cost mini-tanks & skipping squishy backliners
    slot_order: list[tuple[int, int, float, int]] = []
    for slot in range(hand_slots):
        cid = int(raw_obs[hand_idx[slot]]) if (hand_idx is not None and slot < len(hand_idx)) else -1
        if non_tank_card_ids and cid in non_tank_card_ids:
            continue
        lo = 1 + slot * tiles
        hi = lo + tiles
        n_leg = int(legal[lo:hi].sum())
        if 0 < n_leg < GLOBAL_SPELL_MIN_TILES:
            c_val = float(costs[slot]) if (costs is not None and slot < len(costs)) else 3.0
            is_giant = int(cid in giant_card_ids) if giant_card_ids else 0
            slot_order.append((is_giant, int(c_val >= 4.0), c_val, slot))
    if not slot_order:
        return [], bank_giant_slots
    slot_order.sort(reverse=True)

    grid_idx = np.arange(tiles, dtype=np.int32)
    grid_c = grid_idx % cols
    grid_r = grid_idx // cols

    out_acts: list[int] = []
    for _prio, _y_sort, ux, uy in unshielded[:2]:
        lane_bridge_c = 3.5 if ux < width_u // 2 else (cols - 4.5)
        t_col = int(np.clip(round(0.5 * (ux / tu) + 0.5 * lane_bridge_c), 2, cols - 3))
        t_row = int(np.clip((uy // tu) + 3, 9, 14))
        for _is_g, _ge4, _cval, slot in slot_order[:2]:
            lo = 1 + slot * tiles
            hi = lo + tiles
            near_front = legal[lo:hi] & ((np.abs(grid_c - t_col) + np.abs(grid_r - t_row)) <= 3) & (grid_r >= max(8, (uy // tu) + 1))
            if not near_front.any():
                continue
            scores = np.where(near_front, masked_lp[lo:hi], -np.inf)
            act = lo + int(np.argmax(scores))
            if legal[act] and act not in out_acts:
                out_acts.append(act)
    return out_acts, bank_giant_slots


def _urgent_defense_threat(raw_obs: np.ndarray, schema) -> bool:
    """Return True when an enemy projectile/swarm threatens our tower or >=700 HP crossed the river."""
    if not hasattr(schema, "entities"):
        return False
    e = schema.entity
    ents = schema.entities(raw_obs)
    if not len(ents):
        return False
    enemy = ents[ents[:, e["side"]] == 1]
    if not len(enemy):
        return False
    tu = int(getattr(schema, "tile_units", 1000))
    kinds = enemy[:, e["kind"]]
    alive_flag = schema.flags.get("alive", 0)
    proj_threat = (kinds == schema.kinds.get("projectile", -1)) & (enemy[:, e["y"]] <= int(13.5 * tu))
    if bool(proj_threat.any()):
        return True
    chars = (kinds == schema.kinds["character"]) & (enemy[:, e["hp"]] > 0)
    if alive_flag:
        chars &= (enemy[:, e["flags"]] & alive_flag) != 0
    if not bool(chars.any()):
        return False
    ec = enemy[chars]
    ey = ec[:, e["y"]]
    ehp = ec[:, e["hp"]]
    tower_zone = ey <= int(11.5 * tu)
    if int(tower_zone.sum()) >= 2 or bool((tower_zone & (ehp >= 450)).any()):
        return True
    deep = ey <= int(15.5 * tu)
    return bool(deep.any() and int(ehp[deep].sum()) >= 700)


def _tower_swarm_spell_action(
    slot: int,
    raw_obs: np.ndarray,
    legal: np.ndarray,
    schema,
    *,
    rage_card_ids: Sequence[int] = (),
) -> int | None:
    """When >=2 swarm units are on our tower zone (y<=11.5) or >=5 up to river (y<=15.5), propose centered spell."""
    if not hasattr(schema, "entities"):
        return None
    cols = int(getattr(schema, "cols", 18))
    rows_n = int(getattr(schema, "rows", 32))
    tiles = cols * rows_n
    lo = 1 + slot * tiles
    hi = lo + tiles
    if int(legal[lo:hi].sum()) < GLOBAL_SPELL_MIN_TILES:
        return None
    e = schema.entity
    ents = schema.entities(raw_obs)
    if not len(ents):
        return None
    tu = float(getattr(schema, "tile_units", 1000))
    alive_flag = schema.flags.get("alive", 0)
    chars = (ents[:, e["side"]] == 1) & (ents[:, e["kind"]] == schema.kinds["character"]) & (ents[:, e["hp"]] > 0)
    if alive_flag:
        chars &= (ents[:, e["flags"]] & alive_flag) != 0
    sw = ents[chars & (((ents[:, e["y"]] <= 10.5 * tu) & (ents[:, e["hp"]] <= 340)) | ((ents[:, e["y"]] <= 15.5 * tu) & (ents[:, e["hp"]] <= 205)))]
    if len(sw) < 2:
        return None
    width_u = float(getattr(schema, "width_units", 18000))
    left_sw = sw[sw[:, e["x"]] < width_u * 0.5]
    right_sw = sw[sw[:, e["x"]] >= width_u * 0.5]
    cluster = left_sw if len(left_sw) >= len(right_sw) else right_sw
    if len(cluster) < 2:
        return None
    mean_y = float(cluster[:, e["y"]].mean())
    if mean_y > 11.5 * tu and len(cluster) < 5:
        return None
    sc = int(np.clip(round(float(cluster[:, e["x"]].mean()) / tu - 0.5), 0, cols - 1))
    sr = int(np.clip(round(mean_y / tu - 0.5), 0, 15))
    act = lo + sr * cols + sc
    if legal[act] and _is_engaged_global_spell(act, raw_obs, legal, schema, rage_card_ids=rage_card_ids):
        return int(act)
    return None


def _safe_collector_candidate(
    raw_obs: np.ndarray,
    legal: np.ndarray,
    masked_lp: np.ndarray,
    schema,
    *,
    collector_card_ids: Sequence[int] = (),
) -> int | None:
    """Propose safe behind-princess-tower corner Elixir Collector on calm non-overtime boards."""
    if not collector_card_ids or not hasattr(schema, "entities"):
        return None
    ot_idx = schema.header.get("overtime")
    if ot_idx and int(raw_obs[ot_idx[0]]) != 0:
        return None
    if _emergency_tower_threat(raw_obs, schema):
        return None
    hand_idx = schema.header.get("hand_card")
    if hand_idx is None:
        return None
    e = schema.entity
    ents = schema.entities(raw_obs)
    bldg_flag = schema.flags.get("building", 0)
    if len(ents) and bldg_flag:
        own_bldg = (
            (ents[:, e["side"]] == 0)
            & (ents[:, e["kind"]] == schema.kinds["character"])
            & (ents[:, e["hp"]] > 0)
            & ((ents[:, e["flags"]] & bldg_flag) != 0)
        )
        if bool(own_bldg.any()):
            return None

    cols = int(getattr(schema, "cols", 18))
    rows_n = int(getattr(schema, "rows", 32))
    tiles = cols * rows_n
    hand_slots = int(getattr(schema, "hand_slots", 4))
    l_hp = int(raw_obs[schema.header["own_left_hp"][0]]) if "own_left_hp" in schema.header else 3600
    r_hp = int(raw_obs[schema.header["own_right_hp"][0]]) if "own_right_hp" in schema.header else 3600
    if max(l_hp, r_hp) < 1400:
        return None

    pref_spots = (
        [(1, 1), (2, 1), (1, 2), (8, 5)]
        if l_hp >= r_hp
        else [(cols - 2, 1), (cols - 3, 1), (cols - 2, 2), (9, 5)]
    )
    for slot in range(min(hand_slots, len(hand_idx))):
        if int(raw_obs[hand_idx[slot]]) not in collector_card_ids:
            continue
        lo = 1 + slot * tiles
        hi = lo + tiles
        if not legal[lo:hi].any():
            continue
        best_spot = None
        best_lp = -np.inf
        for c_s, r_s in pref_spots:
            if 0 <= c_s < cols and 0 <= r_s < rows_n:
                cand = lo + r_s * cols + c_s
                if legal[cand] and float(masked_lp[cand]) > best_lp:
                    best_lp = float(masked_lp[cand])
                    best_spot = cand
        if best_spot is not None:
            return int(best_spot)
    return None


def _engaged_push_rage_action(
    slot: int,
    raw_obs: np.ndarray,
    legal: np.ndarray,
    schema,
    *,
    rage_card_ids: Sequence[int] = (),
    min_push_hp: int = 800,
) -> int | None:
    """When our push/counterpush engages enemies OR >=1800 HP push crosses the bridge, propose Rage (with forward lead)."""
    if not rage_card_ids or not hasattr(schema, "entities"):
        return None
    hand_idx = schema.header.get("hand_card")
    if hand_idx is None or slot >= len(hand_idx):
        return None
    if int(raw_obs[hand_idx[slot]]) not in rage_card_ids:
        return None
    cols = int(getattr(schema, "cols", 18))
    rows_n = int(getattr(schema, "rows", 32))
    tiles = cols * rows_n
    lo = 1 + slot * tiles
    hi = lo + tiles
    if int(legal[lo:hi].sum()) < GLOBAL_SPELL_MIN_TILES:
        return None

    e = schema.entity
    ents = schema.entities(raw_obs)
    if not len(ents):
        return None
    tu = float(getattr(schema, "tile_units", 1000))
    width_u = float(getattr(schema, "width_units", 18000))
    alive_flag = schema.flags.get("alive", 0)
    bldg_flag = schema.flags.get("building", 0)

    chars = (ents[:, e["kind"]] == schema.kinds["character"]) & (ents[:, e["hp"]] > 0)
    if alive_flag:
        chars &= (ents[:, e["flags"]] & alive_flag) != 0
    own_mask = chars & (ents[:, e["side"]] == 0)
    if bldg_flag:
        own_mask &= (ents[:, e["flags"]] & bldg_flag) == 0
    own_c = ents[own_mask]
    if not len(own_c):
        return None

    enemy_c = ents[chars & (ents[:, e["side"]] == 1)]
    towers = (ents[:, e["kind"]] == schema.kinds["tower"]) & (ents[:, e["hp"]] > 0) & (ents[:, e["side"]] == 1)
    enemy_t = ents[towers]
    targets = np.concatenate([enemy_c, enemy_t], axis=0) if (len(enemy_c) or len(enemy_t)) else np.empty((0, ents.shape[1]))
    if not len(targets):
        return None

    dx = own_c[:, None, e["x"]] - targets[None, :, e["x"]]
    dy = own_c[:, None, e["y"]] - targets[None, :, e["y"]]
    in_combat = ((dx * dx + dy * dy) <= (6.5 * tu) ** 2).any(axis=1)
    engaged = own_c[in_combat]

    proactive_lead = False
    cluster = np.empty((0, ents.shape[1]), dtype=ents.dtype)
    if len(engaged):
        left_eng = engaged[engaged[:, e["x"]] < width_u * 0.5]
        right_eng = engaged[engaged[:, e["x"]] >= width_u * 0.5]
        cand_cl = left_eng if int(left_eng[:, e["hp"]].sum()) >= int(right_eng[:, e["hp"]].sum()) else right_eng
        tot_hp = int(cand_cl[:, e["hp"]].sum())
        if tot_hp >= min(int(min_push_hp), 650):
            cluster = cand_cl

    # Proactive bridge-push path: heavy push (>=1800 HP) already at/past bridge (y>=13.5) marching toward alive enemy tower
    if not len(cluster) and len(enemy_t):
        adv_push = own_c[own_c[:, e["y"]] >= 13.0 * tu]
        if len(adv_push):
            l_adv = adv_push[adv_push[:, e["x"]] < width_u * 0.5]
            r_adv = adv_push[adv_push[:, e["x"]] >= width_u * 0.5]
            cand_cl = l_adv if int(l_adv[:, e["hp"]].sum()) >= int(r_adv[:, e["hp"]].sum()) else r_adv
            if (
                len(cand_cl)
                and int(cand_cl[:, e["hp"]].sum()) >= max(int(min_push_hp), 1800)
                and bool((cand_cl[:, e["y"]] >= 13.5 * tu).any())
            ):
                cluster = cand_cl
                proactive_lead = True

    if not len(cluster):
        return None

    w = np.maximum(cluster[:, e["max_hp"]], cluster[:, e["hp"]]).astype(np.float64)
    cx = float((cluster[:, e["x"]] * w).sum() / w.sum())
    cy = float((cluster[:, e["y"]] * w).sum() / w.sum())
    # Account for Rage's 1.0s deploy delay by leading advancing pushes forward toward enemy side
    if proactive_lead:
        cy = min(cy + 1.6 * tu, 25.0 * tu)
    elif cy >= 11.0 * tu:
        cy = min(cy + 1.2 * tu, 25.0 * tu)

    # Skip if an active friendly area spell (e.g. Rage) is already ticking on this push
    area_kind = schema.kinds.get("area", -1)
    if area_kind >= 0:
        own_areas = ents[(ents[:, e["side"]] == 0) & (ents[:, e["kind"]] == area_kind)]
        if len(own_areas):
            d2_a = (own_areas[:, e["x"]] - cx) ** 2 + (own_areas[:, e["y"]] - cy) ** 2
            if bool((d2_a <= (3.2 * tu) ** 2).any()):
                return None

    # Shift slightly toward nearby enemy distraction swarm (Skeletons/Bats/Goblins) so Rage both buffs our push & clears swarm
    if len(enemy_c):
        d2_sw = (enemy_c[:, e["x"]] - cx) ** 2 + (enemy_c[:, e["y"]] - cy) ** 2
        near_sw = enemy_c[(d2_sw <= (4.2 * tu) ** 2) & (enemy_c[:, e["hp"]] <= 380)]
        if len(near_sw):
            sx, sy = float(near_sw[:, e["x"]].mean()), float(near_sw[:, e["y"]].mean())
            vx, vy = 0.35 * (sx - cx), 0.35 * (sy - cy)
            v_norm = float(np.hypot(vx, vy))
            if v_norm > 2.2 * tu:
                scale_v = (2.2 * tu) / v_norm
                vx, vy = vx * scale_v, vy * scale_v
            cx, cy = cx + vx, cy + vy

    rc = int(np.clip(round(cx / tu - 0.5), 0, cols - 1))
    rr = int(np.clip(round(cy / tu - 0.5), 0, rows_n - 1))
    act = lo + rr * cols + rc
    if legal[act] and _is_engaged_global_spell(act, raw_obs, legal, schema, rage_card_ids=rage_card_ids):
        return int(act)
    return None


def _tactical_spatial_candidates(
    raw_obs: np.ndarray,
    legal: np.ndarray,
    masked_lp: np.ndarray,
    group_best: Sequence[tuple[float, int, int]],
    schema,
    *,
    rage_card_ids: Sequence[int] = (),
    collector_card_ids: Sequence[int] = (),
) -> list[int]:
    """Propose threat-anchored defense & alive-tower finisher candidates for MCTS rollout verification."""
    if not hasattr(schema, "entities"):
        return []
    e = schema.entity
    ents = schema.entities(raw_obs)
    cols = int(getattr(schema, "cols", 18))
    rows_n = int(getattr(schema, "rows", 32))
    tiles = cols * rows_n
    tu = float(getattr(schema, "tile_units", 1000))
    width_u = float(getattr(schema, "width_units", 18000))
    mid_c = cols // 2
    grid_idx = np.arange(tiles, dtype=np.int32)
    grid_c = grid_idx % cols
    grid_r = grid_idx // cols

    out: list[int] = []

    # 1. Tower-swarm spell reflex candidate (ensures MCTS tests centered spell even if raw argmax hit a stray target)
    hand_slots = int(getattr(schema, "hand_slots", 4))
    for slot in range(hand_slots):
        sw_act = _tower_swarm_spell_action(slot, raw_obs, legal, schema, rage_card_ids=rage_card_ids)
        if sw_act is not None and sw_act not in out:
            out.append(sw_act)

    # 2. Threat-anchored troop defense: point-blank surround vs ranged/kite standoff
    deep_hp = 0
    if len(ents):
        alive_flag = schema.flags.get("alive", 0)
        chars = (ents[:, e["side"]] == 1) & (ents[:, e["kind"]] == schema.kinds["character"]) & (ents[:, e["hp"]] > 0)
        if alive_flag:
            chars &= (ents[:, e["flags"]] & alive_flag) != 0
        ec = ents[chars & (ents[:, e["y"]] <= 15.5 * tu)]
        if len(ec):
            deep_hp = int(ec[:, e["hp"]].sum())
            if deep_hp >= 650:
                left_ec = ec[ec[:, e["x"]] < width_u * 0.5]
                right_ec = ec[ec[:, e["x"]] >= width_u * 0.5]
                focus = left_ec if int(left_ec[:, e["hp"]].sum()) >= int(right_ec[:, e["hp"]].sum()) else right_ec
                if len(focus):
                    w = np.maximum(focus[:, e["hp"]], 100).astype(np.float64)
                    tc = float((focus[:, e["x"]] * w).sum() / w.sum()) / tu - 0.5
                    tr = float((focus[:, e["y"]] * w).sum() / w.sum()) / tu - 0.5
                    dist = np.hypot(grid_c - tc, grid_r - tr)
                    same_half = (grid_c < mid_c) if (tc < mid_c) else (grid_c >= mid_c)
                    troop_slots = [
                        slot
                        for _sc, _pa, slot in group_best
                        if slot >= 0 and 0 < int(legal[1 + slot * tiles : 1 + (slot + 1) * tiles].sum()) < GLOBAL_SPELL_MIN_TILES
                    ]
                    for slot in troop_slots[:2]:
                        lo = 1 + slot * tiles
                        hi = lo + tiles
                        s_leg = legal[lo:hi]
                        s_lp = masked_lp[lo:hi]
                        # (a) Point-blank engage (e.g. MinionHorde / MiniPekka / DarkPrince directly on MegaKnight/Hog)
                        engage = s_leg & same_half & (dist <= 3.2) & (grid_r <= min(14, int(round(tr)) + 2))
                        if engage.any():
                            a_eng = lo + int(np.argmax(np.where(engage, s_lp - 0.05 * dist, -np.inf)))
                            if a_eng not in out:
                                out.append(a_eng)
                        # (b) Safe standoff / center kite (e.g. Wizard 3.5..6.2 tiles away so MK doesn't splash tower+Wizard)
                        standoff = s_leg & same_half & (dist >= 3.4) & (dist <= 6.2) & (grid_r <= min(13, int(round(tr)) + 1))
                        if standoff.any():
                            a_std = lo + int(np.argmax(np.where(standoff, s_lp, -np.inf)))
                            if a_std not in out:
                                out.append(a_std)

    # 2.5 Same-lane engaged tank support (e.g. put MinionHorde/Wizard behind engaged Giant instead of naked opposite-bridge throw)
    engaged_tank_left: bool | None = None
    if len(ents):
        alive_flag = schema.flags.get("alive", 0)
        bldg_flag = schema.flags.get("building", 0)
        own_m = (ents[:, e["side"]] == 0) & (ents[:, e["kind"]] == schema.kinds["character"]) & (ents[:, e["hp"]] >= 1000)
        if alive_flag:
            own_m &= (ents[:, e["flags"]] & alive_flag) != 0
        if bldg_flag:
            own_m &= (ents[:, e["flags"]] & bldg_flag) == 0
        own_tanks = ents[own_m & (ents[:, e["max_hp"]] >= 2400) & (ents[:, e["y"]] >= 11.5 * tu)]
        if len(own_tanks):
            ene_m = (ents[:, e["side"]] == 1) & (ents[:, e["kind"]] == schema.kinds["character"]) & (ents[:, e["hp"]] > 0)
            if alive_flag:
                ene_m &= (ents[:, e["flags"]] & alive_flag) != 0
            ene_c = ents[ene_m]
            best_tank = own_tanks[int(np.argmax(own_tanks[:, e["hp"]]))]
            tx_u, ty_u = float(best_tank[e["x"]]), float(best_tank[e["y"]])
            tank_left = tx_u < width_u * 0.5
            has_contest = ty_u >= 13.5 * tu
            if not has_contest and len(ene_c):
                d2_t = (ene_c[:, e["x"]] - tx_u) ** 2 + (ene_c[:, e["y"]] - ty_u) ** 2
                has_contest = bool((d2_t <= (7.5 * tu) ** 2).any())
            if has_contest:
                engaged_tank_left = tank_left
                tc_t = tx_u / tu - 0.5
                tr_t = ty_u / tu - 0.5
                same_tank_half = (grid_c < mid_c) if tank_left else (grid_c >= mid_c)
                r_lo = min(12, max(7, int(round(tr_t)) - 8))
                r_hi = min(14, max(12, int(round(tr_t))))
                support_zone = same_tank_half & (grid_r >= r_lo) & (grid_r <= r_hi) & (np.abs(grid_c - tc_t) <= 4.5)
                hand_idx_sup = schema.header.get("hand_card")
                for _sc, pa, slot in group_best[:3]:
                    if slot < 0:
                        continue
                    cid_sup = int(raw_obs[hand_idx_sup[slot]]) if (hand_idx_sup is not None and slot < len(hand_idx_sup)) else -1
                    if collector_card_ids and cid_sup in collector_card_ids:
                        continue
                    lo = 1 + slot * tiles
                    hi = lo + tiles
                    if not (0 < int(legal[lo:hi].sum()) < GLOBAL_SPELL_MIN_TILES):
                        continue
                    cand_sup = legal[lo:hi] & support_zone
                    if cand_sup.any():
                        a_sup = lo + int(np.argmax(np.where(cand_sup, masked_lp[lo:hi], -np.inf)))
                        if a_sup not in out and a_sup != pa:
                            out.append(a_sup)

    # 3. Alive / critically-low enemy Princess Tower focus (prevents attacking a dead lane while other tower sits at 225 HP)
    el_idx = schema.header.get("enemy_left_hp")
    er_idx = schema.header.get("enemy_right_hp")
    if el_idx and er_idx:
        el_hp = int(raw_obs[el_idx[0]])
        er_hp = int(raw_obs[er_idx[0]])
        focus_left: bool | None = None
        low_tower_hp = 99999
        if 0 < el_hp <= 1200 and (er_hp == 0 or el_hp <= er_hp):
            focus_left, low_tower_hp = True, el_hp
        elif 0 < er_hp <= 1200 and (el_hp == 0 or er_hp < el_hp):
            focus_left, low_tower_hp = False, er_hp
        elif el_hp > 0 and er_hp == 0 and el_hp <= 3600:
            focus_left, low_tower_hp = True, el_hp
        elif er_hp > 0 and el_hp == 0 and er_hp <= 3600:
            focus_left, low_tower_hp = False, er_hp

        if focus_left is not None:
            tc_tow = 3 if focus_left else (cols - 4)
            tr_tow = 25
            hand_idx = schema.header.get("hand_card")
            for slot in range(hand_slots):
                lo = 1 + slot * tiles
                hi = lo + tiles
                n_leg = int(legal[lo:hi].sum())
                if n_leg >= GLOBAL_SPELL_MIN_TILES and low_tower_hp <= 550:
                    cid = int(raw_obs[hand_idx[slot]]) if hand_idx is not None else -1
                    if cid in rage_card_ids and low_tower_hp > 170:
                        continue
                    tow_act = lo + tr_tow * cols + tc_tow
                    if legal[tow_act] and _is_engaged_global_spell(
                        tow_act, raw_obs, legal, schema, rage_card_ids=rage_card_ids
                    ):
                        if tow_act not in out:
                            out.append(tow_act)
            if deep_hp < 900 and (engaged_tank_left is None or engaged_tank_left == focus_left or low_tower_hp <= 700):
                lane_mask = (grid_c < mid_c) if focus_left else (grid_c >= mid_c)
                bridge_mask = lane_mask & (grid_r >= 11) & (grid_r <= 14)
                for _sc, pa, slot in group_best[:3]:
                    if slot < 0:
                        continue
                    cid_s = int(raw_obs[hand_idx[slot]]) if (hand_idx is not None and slot < len(hand_idx)) else -1
                    if collector_card_ids and cid_s in collector_card_ids:
                        continue
                    lo = 1 + slot * tiles
                    hi = lo + tiles
                    if not (0 < int(legal[lo:hi].sum()) < GLOBAL_SPELL_MIN_TILES):
                        continue
                    pa_c = (pa - lo) % cols
                    pa_r = (pa - lo) // cols
                    if (pa_c < mid_c) != focus_left:
                        cand_m = legal[lo:hi] & lane_mask
                        if cand_m.any():
                            a_lane = lo + int(np.argmax(np.where(cand_m, masked_lp[lo:hi], -np.inf)))
                            if a_lane not in out:
                                out.append(a_lane)
                    if low_tower_hp <= 700 and ((pa_c < mid_c) != focus_left or pa_r < 10):
                        cand_br = legal[lo:hi] & bridge_mask
                        if cand_br.any():
                            a_br = lo + int(np.argmax(np.where(cand_br, masked_lp[lo:hi], -np.inf)))
                            if a_br not in out:
                                out.append(a_br)
    return out


def select_macro_candidates(
    logp: torch.Tensor,
    legal_mask: np.ndarray,
    raw_obs: np.ndarray,
    schema,
    *,
    max_candidates: int = 4,
    spatial_min_l1: int = 6,
    include_bank_macros: bool = False,
    bank_min_card_cost: float = 5.0,
    bank_max_deficit: float = 2.2,
    rage_card_ids: Sequence[int] = (),
    collector_card_ids: Sequence[int] = (),
    giant_card_ids: Sequence[int] = (),
    non_tank_card_ids: Sequence[int] = (),
) -> list[int]:
    """Select up to `max_candidates` macro-distinct actions (WAIT + legal slots + front-tank + collector + bank + tactical)."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    legal = np.asarray(legal_mask, dtype=bool).reshape(-1)
    lp = logp.detach().float().cpu().numpy().reshape(-1)
    if lp.shape != legal.shape:
        raise ValueError(f"logp shape {lp.shape} != legal_mask shape {legal.shape}")

    masked_lp = np.where(legal, lp, -np.inf)
    default_action = _hierarchical_default_action(lp, legal, raw_obs, schema)
    if not legal[default_action]:
        return [0]

    cols = int(getattr(schema, "cols", 18))
    rows = int(getattr(schema, "rows", 32))
    tiles = cols * rows
    hand_slots = int(getattr(schema, "hand_slots", 4))
    first_ability = 1 + hand_slots * tiles
    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
    elixir_idx = schema.header["elixir"][0]
    elixir_bars = float(raw_obs[elixir_idx]) / scale

    # Gather best action per macro group: group 0..3 = hand slots, 4..5 = abilities
    group_best: list[tuple[float, int, int]] = []  # (score, action, slot_or_-1)
    for slot in range(hand_slots):
        lo = 1 + slot * tiles
        hi = lo + tiles
        if legal[lo:hi].any():
            rel = int(np.argmax(masked_lp[lo:hi]))
            act = lo + rel
            if not _is_engaged_global_spell(act, raw_obs, legal, schema, rage_card_ids=rage_card_ids):
                # Rescue engaged push Rage or tower-swarm spell counters when raw argmax missed the cluster!
                rescued = _engaged_push_rage_action(
                    slot, raw_obs, legal, schema, rage_card_ids=rage_card_ids, min_push_hp=700
                )
                if rescued is None:
                    rescued = _tower_swarm_spell_action(slot, raw_obs, legal, schema, rage_card_ids=rage_card_ids)
                if rescued is None:
                    continue
                act = rescued
            # Rank by total slot probability mass so spread-out tile distributions aren't penalized
            slot_lse = float(np.logaddexp.reduce(masked_lp[lo:hi][legal[lo:hi]]))
            group_best.append((slot_lse, act, slot))

    for ab_act in range(first_ability, len(legal)):
        if legal[ab_act]:
            group_best.append((float(masked_lp[ab_act]), ab_act, -1))

    group_best.sort(key=lambda item: item[0], reverse=True)

    # Always keep default_action at index 0 as baseline so search explicitly measures override gain
    candidates: list[int] = [default_action]
    # Always include WAIT when legal and not leaking at >=9.8 elixir so search tests banking/holding!
    if legal[0] and 0 not in candidates and elixir_bars < 9.8 and len(candidates) < max_candidates:
        candidates.append(0)

    # Pass 1.2: Front-Tank Synergy (prioritize BANK_FOR_SLOT(Giant) when <1.5 bars short, then Giant/front-tank ahead of Sparky)
    ft_acts, ft_bank_giant = _front_tank_candidates(
        raw_obs,
        legal,
        masked_lp,
        schema,
        giant_card_ids=giant_card_ids,
        non_tank_card_ids=non_tank_card_ids,
    )
    if include_bank_macros:
        for g_slot in ft_bank_giant:
            if len(candidates) >= max_candidates:
                break
            g_macro = encode_bank_candidate(g_slot)
            if g_macro not in candidates:
                candidates.append(g_macro)
    for ft_act in ft_acts:
        if len(candidates) >= max_candidates:
            break
        if ft_act not in candidates:
            candidates.append(ft_act)

    # Pass 1.25: Engaged-Push / Proactive Bridge-Push Rage (2-cost steamroll multiplier on Giant/Sparky waves!)
    if rage_card_ids and len(candidates) < max_candidates:
        for slot in range(hand_slots):
            if len(candidates) >= max_candidates:
                break
            pr_act = _engaged_push_rage_action(
                slot, raw_obs, legal, schema, rage_card_ids=rage_card_ids, min_push_hp=1200
            )
            if pr_act is not None and pr_act not in candidates:
                candidates.append(pr_act)

    # Pass 1.3: Safe behind-tower corner Elixir Collector on calm non-overtime boards (evaluated by 38-step deep MCTS!)
    if collector_card_ids and len(candidates) < max_candidates:
        col_act = _safe_collector_candidate(
            raw_obs, legal, masked_lp, schema, collector_card_ids=collector_card_ids
        )
        if col_act is not None and col_act not in candidates:
            candidates.append(col_act)

    for _score, act, _slot in group_best:
        if len(candidates) >= max_candidates:
            break
        if act not in candidates:
            candidates.append(act)

    # Pass 1.5: add multi-step BANK_FOR_SLOT only for genuinely unaffordable >=5-cost cards (never HOLD_TO_FULL on playable hands!)
    if include_bank_macros and len(candidates) < max_candidates:
        bank_slots = _unaffordable_bank_slots(
            raw_obs,
            legal,
            schema,
            min_card_cost=bank_min_card_cost,
            max_deficit=bank_max_deficit,
        )
        for slot in bank_slots[:3]:
            if len(candidates) >= max_candidates:
                break
            macro_act = encode_bank_candidate(slot)
            if macro_act not in candidates:
                candidates.append(macro_act)

    # Pass 1.8: add tactical defense (point-blank vs standoff) & alive/low-HP tower finisher proposals for simulator rollout evaluation
    if len(candidates) < max_candidates:
        for tac_act in _tactical_spatial_candidates(
            raw_obs,
            legal,
            masked_lp,
            group_best,
            schema,
            rage_card_ids=rage_card_ids,
            collector_card_ids=collector_card_ids,
        ):
            if len(candidates) >= max_candidates:
                break
            if tac_act not in candidates:
                candidates.append(tac_act)

    # Pass 2: fill remaining budget with spatially distinct placements (opposite lane / backline-vs-bridge)
    if len(candidates) < max_candidates and spatial_min_l1 > 0:
        grid_idx = np.arange(tiles, dtype=np.int32)
        grid_c = grid_idx % cols
        grid_r = grid_idx // cols
        mid_c = cols // 2
        for _score, primary_act, slot in group_best:
            if len(candidates) >= max_candidates:
                break
            if slot < 0:
                continue
            lo = 1 + slot * tiles
            hi = lo + tiles
            slot_legal = legal[lo:hi]
            if int(slot_legal.sum()) < 2:
                continue
            rel0 = primary_act - lo
            c0, r0 = int(rel0 % cols), int(rel0 // cols)
            far_mask = slot_legal & ((np.abs(grid_c - c0) + np.abs(grid_r - r0)) >= spatial_min_l1)
            if not far_mask.any():
                continue
            # Slight tie-break preference (+0.35 nats) for opposite-lane or deep depth shift (>=6 rows)
            opp_lane = (grid_c < mid_c) != (c0 < mid_c)
            depth_shift = np.abs(grid_r - r0) >= spatial_min_l1
            alt_scores = np.where(
                far_mask,
                masked_lp[lo:hi] + 0.35 * (opp_lane | depth_shift).astype(np.float64),
                -np.inf,
            )
            alt_rel = int(np.argmax(alt_scores))
            alt_act = lo + alt_rel
            if (
                legal[alt_act]
                and alt_act not in candidates
                and _is_engaged_global_spell(alt_act, raw_obs, legal, schema, rage_card_ids=rage_card_ids)
            ):
                candidates.append(alt_act)

    return candidates


def should_trigger_search(
    raw_obs: np.ndarray,
    legal_mask: np.ndarray,
    logp: torch.Tensor,
    schema,
    cfg: LookaheadConfig,
    *,
    rage_card_ids: Sequence[int] = (),
    giant_card_ids: Sequence[int] = (),
    non_tank_card_ids: Sequence[int] = (),
) -> bool:
    """Return True on high-elixir macro turns, front-tank/push-Rage synergy turns, unengaged spells, or urgent defense."""
    legal = np.asarray(legal_mask, dtype=bool).reshape(-1)
    if not legal[1:].any():
        return False
    pending_idx = schema.header.get("pending_count")
    if pending_idx and int(raw_obs[pending_idx[0]]) > 0:
        return False

    lp_np = logp.detach().float().cpu().numpy().reshape(-1)
    masked_lp = np.where(legal, lp_np, -np.inf)
    default_action = _hierarchical_default_action(lp_np, legal, raw_obs, schema)
    if default_action > 0 and not _is_engaged_global_spell(
        default_action, raw_obs, legal, schema, rage_card_ids=rage_card_ids
    ):
        return True

    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
    elix = float(raw_obs[schema.header["elixir"][0]])
    res = float(raw_obs[schema.header["elixir_reserved"][0]]) if "elixir_reserved" in schema.header else 0.0
    avail = (elix - res) / scale

    # Unconditional urgent defense trigger at >=2.0 elixir (catches Goblin Barrel / SkeletonArmy / tower rushes!)
    if avail >= 2.0 and _urgent_defense_threat(raw_obs, schema):
        return True

    # Unconditional low-HP enemy tower finisher trigger at >=2.0 elixir (never miss <=700 HP crown closeouts!)
    if avail >= 2.0:
        for t_key in ("enemy_left_hp", "enemy_right_hp", "enemy_king_hp"):
            t_idx = schema.header.get(t_key)
            if t_idx and 0 < int(raw_obs[t_idx[0]]) <= 700:
                return True

    # Unconditional engaged / bridge-push Rage trigger at >=2.0 elixir
    if avail >= 2.0 and rage_card_ids:
        hand_slots = int(getattr(schema, "hand_slots", 4))
        for s_r in range(hand_slots):
            if (
                _engaged_push_rage_action(
                    s_r, raw_obs, legal, schema, rage_card_ids=rage_card_ids, min_push_hp=1000
                )
                is not None
            ):
                return True

    if avail >= cfg.trigger_elixir:
        return True

    if avail >= 3.5:
        ft_acts, ft_bank_giant = _front_tank_candidates(
            raw_obs,
            legal,
            masked_lp,
            schema,
            giant_card_ids=giant_card_ids,
            non_tank_card_ids=non_tank_card_ids,
        )
        if ft_bank_giant or (avail >= 4.0 and ft_acts):
            return True

    if avail >= cfg.bank_min_elixir and _unaffordable_bank_slots(
        raw_obs,
        legal,
        schema,
        min_card_cost=cfg.bank_min_card_cost,
        max_deficit=cfg.bank_max_deficit,
    ):
        return True

    if avail >= cfg.defense_elixir:
        probs = torch.where(
            torch.as_tensor(legal, device=logp.device),
            logp.detach().float().reshape(-1).exp(),
            torch.zeros(1, device=logp.device),
        )
        probs = probs / probs.sum().clamp_min(1e-8)
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
        if entropy >= cfg.entropy_threshold and hasattr(schema, "entities"):
            e = schema.entity
            rows = schema.entities(raw_obs)
            enemy_near = (
                (rows[:, e["side"]] == 1)
                & (rows[:, e["kind"]] == schema.kinds["character"])
                & ((rows[:, e["flags"]] & schema.flags["alive"]) != 0)
                & (rows[:, e["hp"]] > 0)
                & (rows[:, e["y"]] <= schema.height_units // 2 + 2 * schema.tile_units)
            )
            if bool(enemy_near.any()):
                return True
    return False


def _potential_value(raw_obs: np.ndarray, schema) -> float:
    """Zero-sum conserved battle potential (crowns + tower HP + HP-scaled board army + leak penalty)."""
    own_hp = sum(float(raw_obs[schema.header[f"own_{t}_hp"][0]]) for t in ("left", "right", "king"))
    opp_hp = sum(float(raw_obs[schema.header[f"enemy_{t}_hp"][0]]) for t in ("left", "right", "king"))
    crowns = float(raw_obs[schema.header["crowns"][0]] - raw_obs[schema.header["crowns_enemy"][0]])
    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
    own_elix = float(raw_obs[schema.header["elixir"][0]]) / scale

    army_term = 0.0
    if hasattr(schema, "entities") and hasattr(schema, "entity") and hasattr(schema, "kinds"):
        ents = schema.entities(raw_obs)
        if len(ents):
            e = schema.entity
            alive_flag = getattr(schema, "flags", {}).get("alive", 0)
            chars = (ents[:, e["kind"]] == schema.kinds["character"]) & (ents[:, e["hp"]] > 0)
            if alive_flag:
                chars &= (ents[:, e["flags"]] & alive_flag) != 0
            if bool(chars.any()):
                ce = ents[chars]
                hp = ce[:, e["hp"]].astype(np.float64)
                mhp = np.maximum(ce[:, e["max_hp"]].astype(np.float64), 1.0)
                health = 0.3 + 0.7 * np.clip(hp / mhp, 0.0, 1.0)
                body_weight = np.clip(mhp / 1200.0, 0.25, 2.5)
                side_sign = np.where(ce[:, e["side"]] == 0, 1.0, np.where(ce[:, e["side"]] == 1, -1.0, 0.0))
                army_term = 0.0045 * float((health * body_weight * side_sign).sum())

    elix_diff_term = 0.0
    enemy_elix_idx = schema.header.get("enemy_elixir")
    if enemy_elix_idx:
        opp_elix = float(raw_obs[enemy_elix_idx[0]]) / scale
        elix_diff_term = 0.0008 * (own_elix - opp_elix)
    leak_pen = -0.005 * max(0.0, own_elix - 9.5)
    return 0.15 * crowns + 0.06 * ((own_hp - opp_hp) / 3000.0) + army_term + elix_diff_term + leak_pen


def _eval_critic_sequence_leaf(
    critic: StateValueNetwork,
    crit_mem: PolicyMemory,
    h_steps: list[torch.Tensor],
    e_steps: list[torch.Tensor],
    m_steps: list[torch.Tensor],
    leaf_h: np.ndarray,
    leaf_e: np.ndarray,
    leaf_m: np.ndarray,
    live_rows: np.ndarray,
    critic_device: torch.device,
    critic_ctx,
) -> np.ndarray:
    """Evaluate leaf `V(s_{t+H})` for `live_rows` in ONE fused chunked-KDA/GQA prefill instead of H sequential steps."""
    idx_t = torch.as_tensor(live_rows, dtype=torch.long, device=critic_device)
    all_h = [t.index_select(0, idx_t) for t in h_steps] + [torch.as_tensor(leaf_h, device=critic_device)]
    all_e = [t.index_select(0, idx_t) for t in e_steps] + [torch.as_tensor(leaf_e, device=critic_device)]
    all_m = [t.index_select(0, idx_t) for t in m_steps] + [
        torch.as_tensor(leaf_m, dtype=torch.bool, device=critic_device)
    ]
    seq_h = torch.stack(all_h, dim=1)
    max_ent = max(int(t.shape[1]) for t in all_e)
    if all(int(t.shape[1]) == max_ent for t in all_e):
        seq_e = torch.stack(all_e, dim=1)
        seq_m = torch.stack(all_m, dim=1)
    else:
        n_live = len(live_rows)
        t_len = len(all_e)
        feat_d = int(all_e[0].shape[-1])
        seq_e = all_e[0].new_zeros((n_live, t_len, max_ent, feat_d))
        seq_m = torch.zeros((n_live, t_len, max_ent), dtype=torch.bool, device=critic_device)
        for s_i, (et, mt) in enumerate(zip(all_e, all_m)):
            ec = int(et.shape[1])
            seq_e[:, s_i, :ec, :] = et
            seq_m[:, s_i, :ec] = mt
    crit_live = slice_memory(
        crit_mem, live_rows, trim_gqa=True, position_bound=crit_mem.temporal.length_bound
    )
    with critic_ctx:
        vals_seq, _ = critic.forward_state(seq_h, seq_e, seq_m, state=crit_live)
    return vals_seq[:, -1].float().cpu().numpy()


class LookaheadPolicyStream:
    """Drop-in PolicyStream wrapper with forked CrForge rollout lookahead + Critic leaf evaluation."""

    def __init__(
        self,
        actor: StateActorCritic,
        adapter,
        device,
        *,
        critic: StateValueNetwork | None = None,
        search_env=None,
        config: LookaheadConfig | None = None,
        precision: str = "fp16",
        seed: int = 20261122,
        rage_card_ids: Sequence[int] = (),
        collector_card_ids: Sequence[int] = (),
        giant_card_ids: Sequence[int] = (),
        non_tank_card_ids: Sequence[int] = (),
    ):
        self.actor = actor.eval()
        self.critic = critic.eval() if critic is not None else None
        self.adapter = adapter
        self.device = torch.device(device)
        if self.critic is not None:
            try:
                self.critic_device = next(self.critic.parameters()).device
            except StopIteration:
                self.critic_device = self.device
        else:
            self.critic_device = self.device
        self.precision = precision
        self.cfg = config or LookaheadConfig()
        self.search_env = search_env
        self.rng = np.random.default_rng(seed)
        self.rage_card_ids = tuple(int(x) for x in rage_card_ids)
        self.collector_card_ids = tuple(int(x) for x in collector_card_ids)
        self.giant_card_ids = tuple(int(x) for x in giant_card_ids)
        self.non_tank_card_ids = tuple(int(x) for x in non_tank_card_ids)
        self.hand_indices = list(adapter.schema.header["hand_card"])

        # Match-level replay context for exact simulator forking
        self.match_seed: int | None = None
        self.deck0: Sequence[str] | None = None
        self.deck1: Sequence[str] | None = None
        self.slots0: Sequence[int] | None = None
        self.slots1: Sequence[int] | None = None
        self.seat: int = 0
        self.prefix_actions: list[tuple[int, int]] = []

        self.actor_state: PolicyMemory | None = None
        self.opp_actor_state: PolicyMemory | None = None
        self.critic_state: PolicyMemory | None = None
        self._step_idx: int = 0
        self._match_searches: int = 0
        self._last_search_step: int = -999
        self._last_search_elixir_tier: int = -1
        self._active_bank_slot: int | None = None
        self._active_bank_card_id: int = -1
        self._active_bank_deadline: int = -1
        self._active_bank_hold_full: bool = False
        self.last_step_info: dict = {}

        self.stats = dict(
            decisions=0,
            searches_triggered=0,
            action_overrides=0,
            wait_overrides=0,
            bank_commitments=0,
            bank_deployments=0,
            total_search_seconds=0.0,
        )

    def bind_match(
        self,
        *,
        seed: int,
        deck0: Sequence[str],
        deck1: Sequence[str],
        slots0: Sequence[int],
        slots1: Sequence[int],
        seat: int,
    ) -> None:
        self.reset()
        self.match_seed = int(seed)
        self.deck0 = list(deck0)
        self.deck1 = list(deck1)
        self.slots0 = list(slots0)
        self.slots1 = list(slots1)
        self.seat = int(seat)

    def record_step_actions(self, action0: int, action1: int) -> None:
        self.prefix_actions.append((int(action0), int(action1)))

    def reset(self) -> None:
        self.actor_state = None
        self.opp_actor_state = None
        self.critic_state = None
        self._step_idx = 0
        self._match_searches = 0
        self._last_search_step = -999
        self._last_search_elixir_tier = -1
        self._active_bank_slot = None
        self._active_bank_card_id = -1
        self._active_bank_deadline = -1
        self._active_bank_hold_full = False
        self.prefix_actions.clear()

    def _autocast(self, dev: torch.device | None = None, *, for_actor: bool = False):
        if for_actor and getattr(self.actor, "ppo_actor_fp32", False):
            return nullcontext()
        target = dev or self.device
        if self.precision == "fp16" and target.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def observe_opponent(self, opp_raw: np.ndarray, opp_legal: np.ndarray) -> None:
        """Advance opponent's self-play mirror memory so lookahead rollouts start warm."""
        h, e, m = self.adapter(np.asarray(opp_raw)[None])
        tensors = [torch.as_tensor(v, device=self.device) for v in (h, e, m)]
        hand = torch.as_tensor(np.asarray(opp_raw)[self.hand_indices][None], device=self.device)
        legal_t = torch.as_tensor(np.asarray(opp_legal, dtype=bool)[None], device=self.device)
        with self._autocast(for_actor=True):
            _, _, self.opp_actor_state = self.actor.step(
                *tensors, legal_t, state=self.opp_actor_state, hand_card_ids=hand, compute_value=False
            )

    def prepare_step_from_logp(
        self, raw: np.ndarray, legal: np.ndarray, logp_1d: torch.Tensor
    ) -> tuple[int, list[int] | None, dict]:
        """Evaluate active bank/hold commitments and search triggers from pre-computed `logp_1d`."""
        mask_1d = torch.as_tensor(legal, device=logp_1d.device, dtype=torch.bool)
        masked_lp = logp_1d.masked_fill(~mask_1d, -torch.inf)
        schema = self.adapter.schema
        cols = int(getattr(schema, "cols", 18))
        rows = int(getattr(schema, "rows", 32))
        tiles = cols * rows
        legal_np = np.asarray(legal, dtype=bool).reshape(-1)
        default_action = _hierarchical_default_action(
            logp_1d.detach().float().cpu().numpy(), legal_np, raw, schema
        )
        chosen_action = default_action
        bank_wait = False
        bank_slot_used: int | None = None
        unlocked_bank_slot: int | None = None

        scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
        elix = float(raw[schema.header["elixir"][0]])
        res = float(raw[schema.header["elixir_reserved"][0]]) if "elixir_reserved" in schema.header else 0.0
        avail_bars = (elix - res) / scale
        hold_full_used = False

        # Follow through on active BANK_FOR_SLOT or HOLD_TO_FULL commitment across intermediate frames
        if self._active_bank_slot is not None:
            bs = int(self._active_bank_slot)
            lo = 1 + bs * tiles
            hi = lo + tiles
            same_card = int(raw[self.hand_indices[bs]]) == self._active_bank_card_id
            emergency = _emergency_tower_threat(raw, schema)
            is_hold_full = bool(self._active_bank_hold_full)
            costs_cur = _hand_costs_bars(raw, schema)
            rem_def = (float(costs_cur[bs]) - avail_bars) if (costs_cur is not None and bs < len(costs_cur)) else 9.9
            abort_emergency = emergency and (
                is_hold_full or _critical_low_own_tower(raw, schema) or rem_def > 1.2
            )
            if self._step_idx > self._active_bank_deadline or not same_card or abort_emergency:
                self._active_bank_slot = None
                self._active_bank_hold_full = False
            elif is_hold_full:
                if avail_bars < 9.4 and legal_np[0]:
                    if default_action != 0:
                        self.stats["action_overrides"] += 1
                        self.stats["wait_overrides"] += 1
                    self.last_step_info = dict(
                        default_action=default_action,
                        chosen_action=0,
                        searched=False,
                        overridden=(default_action != 0),
                        bank_wait=True,
                        hold_full=True,
                        bank_slot=bs,
                        score_gain=0.0,
                    )
                    self._step_idx += 1
                    self.stats["decisions"] += 1
                    return 0, None, {}
                elif legal_np[lo:hi].any():
                    unlocked_bank_slot = bs
                    self._active_bank_slot = None
                    self._active_bank_hold_full = False
                    slot_rel = int(masked_lp[lo:hi].argmax(dim=-1).item())
                    chosen_action = lo + slot_rel
                    self.stats["bank_deployments"] += 1
                else:
                    self._active_bank_slot = None
                    self._active_bank_hold_full = False
            elif not legal_np[lo:hi].any() and legal_np[0]:
                if default_action != 0:
                    self.stats["action_overrides"] += 1
                    self.stats["wait_overrides"] += 1
                self.last_step_info = dict(
                    default_action=default_action,
                    chosen_action=0,
                    searched=False,
                    overridden=(default_action != 0),
                    bank_wait=True,
                    hold_full=False,
                    bank_slot=bs,
                    score_gain=0.0,
                )
                self._step_idx += 1
                self.stats["decisions"] += 1
                return 0, None, {}
            elif legal_np[lo:hi].any():
                unlocked_bank_slot = bs
                self._active_bank_slot = None
                self._active_bank_hold_full = False
                slot_rel = int(masked_lp[lo:hi].argmax(dim=-1).item())
                chosen_action = lo + slot_rel
                self.stats["bank_deployments"] += 1
            else:
                self._active_bank_slot = None
                self._active_bank_hold_full = False

        cur_tier = int(avail_bars)
        cooldown_ok = (
            unlocked_bank_slot is not None
            or (self._step_idx - self._last_search_step) >= self.cfg.min_search_interval_steps
            or cur_tier > self._last_search_elixir_tier
        )
        if (
            self.search_env is not None
            and self.match_seed is not None
            and cooldown_ok
            and self._match_searches < self.cfg.max_searches_per_game
            and should_trigger_search(
                raw,
                legal,
                logp_1d,
                schema,
                self.cfg,
                rage_card_ids=self.rage_card_ids,
                giant_card_ids=self.giant_card_ids,
                non_tank_card_ids=self.non_tank_card_ids,
            )
        ):
            candidates = select_macro_candidates(
                logp_1d,
                legal,
                raw,
                schema,
                max_candidates=self.cfg.max_candidates,
                include_bank_macros=True,
                bank_min_card_cost=self.cfg.bank_min_card_cost,
                bank_max_deficit=self.cfg.bank_max_deficit,
                rage_card_ids=self.rage_card_ids,
                collector_card_ids=self.collector_card_ids,
                giant_card_ids=self.giant_card_ids,
                non_tank_card_ids=self.non_tank_card_ids,
            )
            if chosen_action != default_action and chosen_action >= 0 and chosen_action not in candidates:
                candidates[0] = chosen_action
            if len(candidates) >= 2:
                return chosen_action, candidates, dict(
                    default_action=default_action,
                    cur_tier=cur_tier,
                    unlocked_bank_slot=unlocked_bank_slot,
                )

        if unlocked_bank_slot is not None and chosen_action != default_action:
            self.stats["action_overrides"] += 1

        self.last_step_info = dict(
            default_action=default_action,
            chosen_action=chosen_action,
            searched=False,
            overridden=(chosen_action != default_action),
            bank_wait=bank_wait,
            hold_full=hold_full_used,
            bank_slot=bank_slot_used,
            unlocked_bank_slot=unlocked_bank_slot,
            score_gain=0.0,
        )
        self._step_idx += 1
        self.stats["decisions"] += 1
        return chosen_action, None, {}

    def apply_search_result(
        self,
        raw: np.ndarray,
        raw_choice: int,
        score_gain: float,
        pending: dict,
        search_elapsed: float,
    ) -> int:
        """Commit lookahead search outcome, banking latches, and telemetry."""
        default_action = int(pending["default_action"])
        cur_tier = int(pending["cur_tier"])
        unlocked_bank_slot = pending.get("unlocked_bank_slot")

        self._last_search_step = self._step_idx
        self._last_search_elixir_tier = cur_tier
        self._match_searches += 1
        self.stats["searches_triggered"] += 1
        self.stats["total_search_seconds"] += float(search_elapsed)

        bank_wait = False
        hold_full_used = False
        bank_slot_used: int | None = None
        bank_target = decode_bank_candidate(raw_choice)
        hold_target = decode_hold_full_candidate(raw_choice)
        if bank_target is not None or hold_target is not None:
            t_slot = bank_target if bank_target is not None else int(hold_target)
            self._active_bank_slot = t_slot
            self._active_bank_card_id = int(raw[self.hand_indices[t_slot]])
            self._active_bank_deadline = self._step_idx + self.cfg.max_bank_steps
            self._active_bank_hold_full = hold_target is not None
            chosen_action = 0
            bank_wait = True
            hold_full_used = hold_target is not None
            bank_slot_used = t_slot
            self.stats["bank_commitments"] += 1
        else:
            chosen_action = int(raw_choice)

        if chosen_action != default_action:
            self.stats["action_overrides"] += 1
            if chosen_action == 0:
                self.stats["wait_overrides"] += 1

        self.last_step_info = dict(
            default_action=default_action,
            chosen_action=chosen_action,
            searched=True,
            overridden=(chosen_action != default_action),
            bank_wait=bank_wait,
            hold_full=hold_full_used,
            bank_slot=bank_slot_used,
            unlocked_bank_slot=unlocked_bank_slot,
            score_gain=float(score_gain),
        )
        self._step_idx += 1
        self.stats["decisions"] += 1
        return chosen_action

    @torch.inference_mode()
    def act(self, raw: np.ndarray, legal: np.ndarray, *, opp_raw: np.ndarray | None = None, opp_legal: np.ndarray | None = None) -> tuple[int, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()

        if opp_raw is not None and opp_legal is not None:
            self.observe_opponent(opp_raw, opp_legal)

        h, e, m = self.adapter(np.asarray(raw)[None])
        tensors = [torch.as_tensor(v, device=self.device) for v in (h, e, m)]
        hand = torch.as_tensor(np.asarray(raw)[self.hand_indices][None], device=self.device)
        legal_t = torch.as_tensor(np.asarray(legal, dtype=bool)[None], device=self.device)

        with self._autocast(for_actor=True):
            logp, _, self.actor_state = self.actor.step(
                *tensors, legal_t, state=self.actor_state, hand_card_ids=hand, compute_value=False
            )
        if self.critic is not None:
            c_tensors = (
                tensors
                if self.critic_device == self.device
                else [t.to(self.critic_device) for t in tensors]
            )
            with self._autocast(self.critic_device):
                _, self.critic_state = self.critic.step(*c_tensors, state=self.critic_state)

        chosen_action, candidates, pending = self.prepare_step_from_logp(raw, legal, logp[0])
        if candidates is not None:
            t_s = time.perf_counter()
            raw_choice, score_gain = self._run_lookahead(
                candidates, logp[0], raw, chosen_action, return_gain=True, root_legal=legal
            )
            chosen_action = self.apply_search_result(
                raw, raw_choice, score_gain, pending, time.perf_counter() - t_s
            )
        return chosen_action, time.perf_counter() - started

    @staticmethod
    def _sample_legal_crn(lp: torch.Tensor, mask_np: np.ndarray, u_1d: np.ndarray) -> np.ndarray:
        """CRN inverse-CDF sampling guaranteed to return strictly legal actions in [0, A-1]."""
        mask_b = np.asarray(mask_np, dtype=bool)
        mask_t = torch.as_tensor(mask_b, dtype=torch.bool, device=lp.device)
        probs = torch.where(mask_t, lp.float().exp(), torch.zeros_like(lp, dtype=torch.float32))
        cdf = probs.cumsum(-1)
        u_t = torch.as_tensor(np.clip(u_1d, 1e-6, 1.0 - 1e-6), dtype=torch.float32, device=lp.device)[:, None]
        target = (u_t * cdf[:, -1:]).clamp_min(1e-12)
        acts = (
            torch.searchsorted(cdf, target, right=False)
            .squeeze(1)
            .clamp(0, lp.shape[-1] - 1)
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        rows = np.arange(len(acts))
        bad = ~mask_b[rows, acts]
        if bad.any():
            fallback = mask_b.argmax(axis=-1).astype(np.int32)
            acts = np.where(bad, fallback, acts)
        return acts

    def _fast_forward_prefix(self, vector, lanes_used: int) -> None:
        """Reset and advance `lanes_used` lanes through `self.prefix_actions` with one sync at the end."""
        prefix = self.prefix_actions
        if hasattr(vector, "lanes") and hasattr(vector, "_sync"):
            def _ff_one(idx: int) -> None:
                lane = vector.lanes[idx]
                lane.reset([self.match_seed], self.deck0, self.deck1, self.slots0, self.slots1)
                act_buf = lane.actions
                for a0, a1 in prefix:
                    act_buf[0, 0] = a0
                    act_buf[0, 1] = a1
                    lane.step()
                vector._sync(idx)

            executor = getattr(vector, "_executor", None)
            if executor is not None and lanes_used > 1:
                list(executor.map(_ff_one, range(lanes_used)))
            else:
                for idx in range(lanes_used):
                    _ff_one(idx)
            return

        lane_ids = np.arange(lanes_used)
        for lane in range(lanes_used):
            vector.reset_env(lane, self.match_seed, self.deck0, self.deck1, self.slots0, self.slots1)
        for a0, a1 in prefix:
            vector.actions[:lanes_used, 0] = a0
            vector.actions[:lanes_used, 1] = a1
            vector.step_lanes(lane_ids)

    @torch.inference_mode()
    def _run_lookahead(
        self,
        candidates: list[int],
        root_logp: torch.Tensor,
        root_raw: np.ndarray,
        default_action: int,
        *,
        return_gain: bool = False,
        root_legal: np.ndarray | None = None,
    ):
        k = len(candidates)
        r = self.cfg.rollouts_per_candidate
        lanes_used = k * r
        vector = self.search_env.vector
        schema = self.adapter.schema
        seat = self.seat
        opp_seat = 1 - seat

        # 1. Reset and fast-forward native simulator lanes through exact match prefix (no NN calls!)
        self._fast_forward_prefix(vector, lanes_used)
        saved_root_legal = (
            np.asarray(root_legal, dtype=bool)
            if root_legal is not None
            else self.search_env.masks[0, seat].copy()
        )

        # 2. Broadcast warm recurrent memories across all K*R lanes (2*L actor streams: own + opp)
        own_mem = slice_memory(self.actor_state, [0] * lanes_used)
        if self.opp_actor_state is not None:
            opp_mem = slice_memory(self.opp_actor_state, [0] * lanes_used)
        else:
            with self._autocast(for_actor=True):
                opp_mem = self.actor.initial_state(lanes_used, device=self.device)
        crit_mem = slice_memory(self.critic_state, [0] * lanes_used) if self.critic_state is not None else None

        root_pot = _potential_value(root_raw, schema)
        returns = np.zeros(lanes_used, dtype=np.float64)
        discounts = np.ones(lanes_used, dtype=np.float64)
        prev_pot = np.full(lanes_used, root_pot, dtype=np.float64)
        live = np.ones(lanes_used, dtype=bool)

        # Pre-generate CRN uniform draws [horizon_steps, R, 2] shared across all K candidates!
        crn_draws = self.rng.random((self.cfg.horizon_steps, r, 2), dtype=np.float32)
        tiles = int(getattr(schema, "cols", 18)) * int(getattr(schema, "rows", 32))

        # Per-lane multi-step BANK_FOR_SLOT / HOLD_TO_FULL commitment tracking
        scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))
        lane_bank_slot = np.full(lanes_used, -1, dtype=np.int32)
        lane_bank_card = np.full(lanes_used, -1, dtype=np.int32)
        lane_hold_full = np.zeros(lanes_used, dtype=bool)
        for c_idx, cand in enumerate(candidates):
            bs = decode_bank_candidate(cand)
            hs = decode_hold_full_candidate(cand)
            if bs is not None or hs is not None:
                ts = bs if bs is not None else int(hs)
                lane_bank_slot[c_idx * r : (c_idx + 1) * r] = ts
                lane_bank_card[c_idx * r : (c_idx + 1) * r] = int(root_raw[self.hand_indices[ts]])
                if hs is not None:
                    lane_hold_full[c_idx * r : (c_idx + 1) * r] = True

        crit_h_steps: list[torch.Tensor] = []
        crit_e_steps: list[torch.Tensor] = []
        crit_m_steps: list[torch.Tensor] = []

        for h_step in range(self.cfg.horizon_steps):
            if not live.any():
                break
            raw_all = self.search_env.observations[:lanes_used]
            mask_all = self.search_env.masks[:lanes_used]
            active = np.flatnonzero(live)

            # Opponent actions on step 0..H-1 (and own actions on step 1..H-1)
            opp_raw_b = raw_all[:, opp_seat]
            opp_mask_b = mask_all[:, opp_seat]
            if not live.all():
                opp_mask_b = opp_mask_b.copy()
                opp_mask_b[~live, 1:] = False
                opp_mask_b[~live, 0] = True
            oh, oe, om = self.adapter(opp_raw_b)
            opp_mem = reserve_step_memory(opp_mem, 128)
            with self._autocast(for_actor=True):
                if hasattr(self.actor, "step_sample_crn"):
                    opp_acts, opp_mem = self.actor.step_sample_crn(
                        torch.as_tensor(oh, device=self.device),
                        torch.as_tensor(oe, device=self.device),
                        torch.as_tensor(om, device=self.device),
                        opp_mask_b,
                        np.tile(crn_draws[h_step, :, 1], k),
                        state=opp_mem,
                        hand_card_ids=torch.as_tensor(opp_raw_b[:, self.hand_indices], device=self.device),
                    )
                else:
                    opp_lp, _, opp_mem = self.actor.step(
                        torch.as_tensor(oh, device=self.device),
                        torch.as_tensor(oe, device=self.device),
                        torch.as_tensor(om, device=self.device),
                        torch.as_tensor(opp_mask_b, dtype=torch.bool, device=self.device),
                        state=opp_mem,
                        hand_card_ids=torch.as_tensor(opp_raw_b[:, self.hand_indices], device=self.device),
                        compute_value=False,
                    )
                    opp_acts = self._sample_legal_crn(opp_lp, opp_mask_b, np.tile(crn_draws[h_step, :, 1], k))

            if h_step == 0:
                # Root action was already stepped into own_mem on the real frame; bank macros start with WAIT (0)
                root_exec = [0 if int(c) < 0 else int(c) for c in candidates]
                own_acts = np.repeat(np.asarray(root_exec, dtype=np.int32), r)
            else:
                own_raw_b = raw_all[:, seat]
                own_mask_b = mask_all[:, seat]
                if not live.all():
                    own_mask_b = own_mask_b.copy()
                    own_mask_b[~live, 1:] = False
                    own_mask_b[~live, 0] = True
                fw_own: np.ndarray | None = None
                fs_own: np.ndarray | None = None
                if np.any(lane_bank_slot >= 0):
                    fw_own = np.zeros(lanes_used, dtype=bool)
                    fs_own = np.full(lanes_used, -1, dtype=np.int32)
                    for lane in active:
                        bs = int(lane_bank_slot[lane])
                        if bs < 0:
                            continue
                        lo = 1 + bs * tiles
                        hi = lo + tiles
                        l_raw = own_raw_b[lane]
                        l_elx = float(l_raw[schema.header["elixir"][0]])
                        l_res = float(l_raw[schema.header["elixir_reserved"][0]]) if "elixir_reserved" in schema.header else 0.0
                        l_avail = (l_elx - l_res) / scale
                        l_emerg = _emergency_tower_threat(l_raw, schema)
                        is_hf = bool(lane_hold_full[lane])
                        l_costs = _hand_costs_bars(l_raw, schema)
                        l_def = (float(l_costs[bs]) - l_avail) if (l_costs is not None and bs < len(l_costs)) else 9.9
                        abort_emerg = l_emerg and (is_hf or _critical_low_own_tower(l_raw, schema) or l_def > 1.2)
                        if (
                            h_step > self.cfg.max_bank_steps
                            or int(l_raw[self.hand_indices[bs]]) != int(lane_bank_card[lane])
                            or abort_emerg
                        ):
                            lane_bank_slot[lane] = -1
                        elif is_hf:
                            if l_avail < 9.4 and own_mask_b[lane, 0]:
                                fw_own[lane] = True
                            elif own_mask_b[lane, lo:hi].any():
                                fs_own[lane] = bs
                                lane_bank_slot[lane] = -1
                            else:
                                lane_bank_slot[lane] = -1
                        elif own_mask_b[lane, lo:hi].any():
                            fs_own[lane] = bs
                            lane_bank_slot[lane] = -1
                        elif own_mask_b[lane, 0]:
                            fw_own[lane] = True

                sh, se, sm = self.adapter(own_raw_b)
                sh_t = torch.as_tensor(sh, device=self.device)
                se_t = torch.as_tensor(se, device=self.device)
                sm_t = torch.as_tensor(sm, device=self.device)
                own_mem = reserve_step_memory(own_mem, 128)
                with self._autocast(for_actor=True):
                    if hasattr(self.actor, "step_sample_crn"):
                        own_acts, own_mem = self.actor.step_sample_crn(
                            sh_t,
                            se_t,
                            sm_t,
                            own_mask_b,
                            np.tile(crn_draws[h_step, :, 0], k),
                            state=own_mem,
                            hand_card_ids=torch.as_tensor(own_raw_b[:, self.hand_indices], device=self.device),
                            force_wait=fw_own,
                            force_slot_argmax=fs_own,
                        )
                    else:
                        own_lp, _, own_mem = self.actor.step(
                            sh_t,
                            se_t,
                            sm_t,
                            torch.as_tensor(own_mask_b, dtype=torch.bool, device=self.device),
                            state=own_mem,
                            hand_card_ids=torch.as_tensor(own_raw_b[:, self.hand_indices], device=self.device),
                            compute_value=False,
                        )
                        own_acts = self._sample_legal_crn(own_lp, own_mask_b, np.tile(crn_draws[h_step, :, 0], k))
                        if fw_own is not None and fs_own is not None:
                            own_lp_np = own_lp.detach().float().cpu().numpy()
                            for lane in active:
                                if fw_own[lane]:
                                    own_acts[lane] = 0
                                elif fs_own[lane] >= 0:
                                    lo = 1 + int(fs_own[lane]) * tiles
                                    hi = lo + tiles
                                    slot_masked = np.where(own_mask_b[lane, lo:hi], own_lp_np[lane, lo:hi], -np.inf)
                                    own_acts[lane] = lo + int(np.argmax(slot_masked))
                if self.critic is not None and crit_mem is not None:
                    crit_h_steps.append(sh_t if self.critic_device == self.device else sh_t.to(self.critic_device))
                    crit_e_steps.append(se_t if self.critic_device == self.device else se_t.to(self.critic_device))
                    crit_m_steps.append(
                        sm_t.bool() if self.critic_device == self.device else sm_t.to(self.critic_device, dtype=torch.bool)
                    )

            vector.actions[:lanes_used, seat] = own_acts
            vector.actions[:lanes_used, opp_seat] = opp_acts
            vector.step_lanes(active)

            for lane in active:
                cur_pot = _potential_value(self.search_env.observations[lane, seat], schema)
                step_shaping = self.cfg.gamma * cur_pot - prev_pot[lane]
                prev_pot[lane] = cur_pot
                if self.search_env.dones[lane]:
                    info = vector.info(lane)
                    if info.get("fault"):
                        term = 0.0
                    elif info["winner"] == seat:
                        term = 1.0
                    elif info["winner"] == opp_seat:
                        term = -1.0
                    else:
                        term = 0.0
                    returns[lane] += discounts[lane] * (term + step_shaping)
                    live[lane] = False
                else:
                    returns[lane] += discounts[lane] * step_shaping
                    discounts[lane] *= self.cfg.gamma

        # 3. Bootstrap surviving lanes at t + H using 1-shot fused Critic sequence prefill (or potential fallback)
        if live.any():
            live_rows = np.flatnonzero(live)
            leaf_obs = self.search_env.observations[live_rows, seat]
            if self.critic is not None and crit_mem is not None:
                lh, le, lm = self.adapter(leaf_obs)
                leaf_vals = _eval_critic_sequence_leaf(
                    self.critic,
                    crit_mem,
                    crit_h_steps,
                    crit_e_steps,
                    crit_m_steps,
                    lh,
                    le,
                    lm,
                    live_rows,
                    self.critic_device,
                    self._autocast(self.critic_device),
                )
                for pos_l, lane in enumerate(live_rows):
                    returns[lane] += discounts[lane] * float(leaf_vals[pos_l])
            else:
                for pos_l, lane in enumerate(live_rows):
                    returns[lane] += discounts[lane] * (_potential_value(leaf_obs[pos_l], schema) - root_pot)

        cand_returns = returns.reshape(k, r).mean(axis=1)
        lp_np = root_logp.detach().float().cpu().numpy().reshape(-1)
        def_lp = float(lp_np[default_action]) if 0 <= default_action < len(lp_np) else float(np.max(lp_np))
        bank_prior = max(float(lp_np[0]), def_lp - 0.5)
        raw_priors = []
        for a in candidates:
            if int(a) < 0:
                raw_priors.append(bank_prior)
            else:
                p_val = float(lp_np[a])
                if int(a) > 0 and not _is_engaged_global_spell(
                    int(a), root_raw, saved_root_legal, schema, rage_card_ids=self.rage_card_ids
                ):
                    p_val -= 4.0
                raw_priors.append(p_val)
        cand_priors = np.asarray(raw_priors, dtype=np.float64)
        cand_priors = np.maximum(cand_priors - cand_priors.max(), -float(self.cfg.max_prior_penalty_nats))
        total_scores = cand_returns + self.cfg.prior_weight * cand_priors

        best_idx = int(np.argmax(total_scores))
        default_idx = candidates.index(default_action) if default_action in candidates else 0
        gain = float(total_scores[best_idx] - total_scores[default_idx])
        chosen = int(candidates[best_idx]) if (best_idx != default_idx and gain >= self.cfg.override_margin) else int(default_action)
        if return_gain:
            return chosen, (gain if chosen != default_action else 0.0)
        return chosen


@torch.inference_mode()
def run_batched_lookahead(
    tasks: Sequence[dict],
    *,
    search_env,
    actor: StateActorCritic,
    critic: StateValueNetwork | None,
    adapter,
    actor_mem: PolicyMemory,
    opp_mem_offset: int,
    critic_mem: PolicyMemory | None,
    cfg: LookaheadConfig,
    device: torch.device,
    critic_device: torch.device,
    precision: str = "fp16",
) -> list[tuple[int, float]]:
    """Run short-horizon CRN lookahead rollouts for multiple matches simultaneously in one batch."""
    if not tasks:
        return []
    r = int(cfg.rollouts_per_candidate)
    schema = adapter.schema
    hand_indices = list(schema.header["hand_card"])
    tiles = int(getattr(schema, "cols", 18)) * int(getattr(schema, "rows", 32))
    scale = float((getattr(schema, "scales", None) or {}).get("elixir_scale", 10000))

    spans: list[tuple[int, int, int]] = []
    lane_task_idx: list[int] = []
    own_indices: list[int] = []
    opp_indices: list[int] = []
    total_lanes = 0
    for b_idx, task in enumerate(tasks):
        k_b = len(task["candidates"])
        lanes_b = k_b * r
        spans.append((total_lanes, total_lanes + lanes_b, k_b))
        total_lanes += lanes_b
        ms = int(task["mem_slot"])
        lane_task_idx.extend([b_idx] * lanes_b)
        own_indices.extend([ms] * lanes_b)
        opp_indices.extend([int(opp_mem_offset) + ms] * lanes_b)

    if total_lanes > int(search_env.count):
        raise ValueError(f"Batched lookahead requested {total_lanes} lanes > search_env capacity {search_env.count}")

    vector = search_env.vector
    can_native_fork = (
        hasattr(vector, "fork_tasks_parallel")
        and getattr(vector, "lanes", None)
        and hasattr(vector.lanes[0], "fork_from")
        and all(
            t.get("source_vector") is not None
            and getattr(t["source_vector"], "lanes", None)
            and hasattr(t["source_vector"].lanes[0], "fork_from")
            for t in tasks
        )
    )
    # 1. Zero-replay native JVM fork_from when live source battles are provided; else parallel prefix replay
    if can_native_fork:
        vector.fork_tasks_parallel(
            [
                (tasks[b_idx]["source_vector"], int(tasks[b_idx]["source_lane"]), range(spans[b_idx][0], spans[b_idx][1]))
                for b_idx in range(len(tasks))
            ]
        )
    elif hasattr(vector, "lanes") and hasattr(vector, "_sync"):
        def _ff_lane(lane_idx: int) -> None:
            st = tasks[lane_task_idx[lane_idx]]["stream"]
            lane = vector.lanes[lane_idx]
            lane.reset([st.match_seed], st.deck0, st.deck1, st.slots0, st.slots1)
            act_buf = lane.actions
            for a0, a1 in st.prefix_actions:
                act_buf[0, 0] = a0
                act_buf[0, 1] = a1
                lane.step()
            vector._sync(lane_idx)

        executor = getattr(vector, "_executor", None)
        if executor is not None and total_lanes > 1:
            list(executor.map(_ff_lane, range(total_lanes)))
        else:
            for lane_idx in range(total_lanes):
                _ff_lane(lane_idx)
    else:
        max_pref = 0
        for lane_idx in range(total_lanes):
            st = tasks[lane_task_idx[lane_idx]]["stream"]
            vector.reset_env(lane_idx, st.match_seed, st.deck0, st.deck1, st.slots0, st.slots1)
            max_pref = max(max_pref, len(st.prefix_actions))
        for p_step in range(max_pref):
            need = []
            for lane_idx in range(total_lanes):
                pref = tasks[lane_task_idx[lane_idx]]["stream"].prefix_actions
                if p_step < len(pref):
                    a0, a1 = pref[p_step]
                    vector.actions[lane_idx, 0] = a0
                    vector.actions[lane_idx, 1] = a1
                    need.append(lane_idx)
            if need:
                vector.step_lanes(need)

    # 2. Gather warm recurrent memories for all `total_lanes` in one indexed slice per network
    own_mem = slice_memory(
        actor_mem, own_indices, trim_gqa=True, position_bound=actor_mem.temporal.length_bound
    )
    opp_mem = slice_memory(
        actor_mem, opp_indices, trim_gqa=True, position_bound=actor_mem.temporal.length_bound
    )
    crit_mem = (
        slice_memory(critic_mem, own_indices, trim_gqa=True, position_bound=critic_mem.temporal.length_bound)
        if (critic is not None and critic_mem is not None)
        else None
    )

    lane_rows = np.arange(total_lanes, dtype=np.int32)
    lane_seat = np.empty(total_lanes, dtype=np.int32)
    returns = np.zeros(total_lanes, dtype=np.float64)
    discounts = np.ones(total_lanes, dtype=np.float64)
    prev_pot = np.empty(total_lanes, dtype=np.float64)
    root_pots = np.empty(total_lanes, dtype=np.float64)
    live = np.ones(total_lanes, dtype=bool)
    lane_bank_slot = np.full(total_lanes, -1, dtype=np.int32)
    lane_bank_card = np.full(total_lanes, -1, dtype=np.int32)
    lane_hold_full = np.zeros(total_lanes, dtype=bool)
    root_own_acts = np.empty(total_lanes, dtype=np.int32)
    u_draws = np.empty((cfg.horizon_steps, total_lanes, 2), dtype=np.float32)

    for b_idx, task in enumerate(tasks):
        lo_l, hi_l, k_b = spans[b_idx]
        st: LookaheadPolicyStream = task["stream"]
        cands = task["candidates"]
        r_raw = task["root_raw"]
        lane_seat[lo_l:hi_l] = int(st.seat)
        rp = _potential_value(r_raw, schema)
        prev_pot[lo_l:hi_l] = rp
        root_pots[lo_l:hi_l] = rp
        crn_b = st.rng.random((cfg.horizon_steps, r, 2), dtype=np.float32)
        u_draws[:, lo_l:hi_l, :] = np.tile(crn_b, (1, k_b, 1))
        root_exec = [0 if int(c) < 0 else int(c) for c in cands]
        root_own_acts[lo_l:hi_l] = np.repeat(np.asarray(root_exec, dtype=np.int32), r)
        for c_idx, cand in enumerate(cands):
            bs = decode_bank_candidate(cand)
            hs = decode_hold_full_candidate(cand)
            if bs is not None or hs is not None:
                ts = bs if bs is not None else int(hs)
                c_lo = lo_l + c_idx * r
                c_hi = c_lo + r
                lane_bank_slot[c_lo:c_hi] = ts
                lane_bank_card[c_lo:c_hi] = int(r_raw[hand_indices[ts]])
                if hs is not None:
                    lane_hold_full[c_lo:c_hi] = True

    lane_opp_seat = 1 - lane_seat
    actor_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if (precision == "fp16" and device.type == "cuda" and not getattr(actor, "ppo_actor_fp32", False))
        else nullcontext()
    )
    critic_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if (precision == "fp16" and critic_device.type == "cuda")
        else nullcontext()
    )

    both_mem: PolicyMemory | None = None
    crit_h_steps: list[torch.Tensor] = []
    crit_e_steps: list[torch.Tensor] = []
    crit_m_steps: list[torch.Tensor] = []

    for h_step in range(cfg.horizon_steps):
        if not live.any():
            break
        raw_all = search_env.observations[:total_lanes]
        mask_all = search_env.masks[:total_lanes]
        active = np.flatnonzero(live)

        opp_raw_b = raw_all[lane_rows, lane_opp_seat]
        opp_mask_b = mask_all[lane_rows, lane_opp_seat]
        if not live.all():
            opp_mask_b = opp_mask_b.copy()
            opp_mask_b[~live, 1:] = False
            opp_mask_b[~live, 0] = True

        if h_step == 0:
            oh, oe, om = adapter(opp_raw_b)
            opp_mem = reserve_step_memory(opp_mem, 128)
            with actor_ctx:
                if hasattr(actor, "step_sample_crn"):
                    opp_acts, opp_mem = actor.step_sample_crn(
                        torch.as_tensor(oh, device=device),
                        torch.as_tensor(oe, device=device),
                        torch.as_tensor(om, device=device),
                        opp_mask_b,
                        u_draws[h_step, :, 1],
                        state=opp_mem,
                        hand_card_ids=torch.as_tensor(opp_raw_b[:, hand_indices], device=device),
                    )
                else:
                    opp_lp, _, opp_mem = actor.step(
                        torch.as_tensor(oh, device=device),
                        torch.as_tensor(oe, device=device),
                        torch.as_tensor(om, device=device),
                        torch.as_tensor(opp_mask_b, dtype=torch.bool, device=device),
                        state=opp_mem,
                        hand_card_ids=torch.as_tensor(opp_raw_b[:, hand_indices], device=device),
                        compute_value=False,
                    )
                    opp_acts = LookaheadPolicyStream._sample_legal_crn(opp_lp, opp_mask_b, u_draws[h_step, :, 1])
            own_acts = root_own_acts.copy()
            both_mem = stack_memories([own_mem, opp_mem], device=device)
        else:
            own_raw_b = raw_all[lane_rows, lane_seat]
            own_mask_b = mask_all[lane_rows, lane_seat]
            if not live.all():
                own_mask_b = own_mask_b.copy()
                own_mask_b[~live, 1:] = False
                own_mask_b[~live, 0] = True

            fw_both: np.ndarray | None = None
            fs_both: np.ndarray | None = None
            if np.any(lane_bank_slot >= 0):
                fw_both = np.zeros(2 * total_lanes, dtype=bool)
                fs_both = np.full(2 * total_lanes, -1, dtype=np.int32)
                for lane in active:
                    bs = int(lane_bank_slot[lane])
                    if bs < 0:
                        continue
                    lo = 1 + bs * tiles
                    hi = lo + tiles
                    l_raw = own_raw_b[lane]
                    l_elx = float(l_raw[schema.header["elixir"][0]])
                    l_res = float(l_raw[schema.header["elixir_reserved"][0]]) if "elixir_reserved" in schema.header else 0.0
                    l_avail = (l_elx - l_res) / scale
                    l_emerg = _emergency_tower_threat(l_raw, schema)
                    is_hf = bool(lane_hold_full[lane])
                    l_costs = _hand_costs_bars(l_raw, schema)
                    l_def = (float(l_costs[bs]) - l_avail) if (l_costs is not None and bs < len(l_costs)) else 9.9
                    abort_emerg = l_emerg and (is_hf or _critical_low_own_tower(l_raw, schema) or l_def > 1.2)
                    if (
                        h_step > cfg.max_bank_steps
                        or int(l_raw[hand_indices[bs]]) != int(lane_bank_card[lane])
                        or abort_emerg
                    ):
                        lane_bank_slot[lane] = -1
                    elif is_hf:
                        if l_avail < 9.4 and own_mask_b[lane, 0]:
                            fw_both[lane] = True
                        elif own_mask_b[lane, lo:hi].any():
                            fs_both[lane] = bs
                            lane_bank_slot[lane] = -1
                        else:
                            lane_bank_slot[lane] = -1
                    elif own_mask_b[lane, lo:hi].any():
                        fs_both[lane] = bs
                        lane_bank_slot[lane] = -1
                    elif own_mask_b[lane, 0]:
                        fw_both[lane] = True

            both_raw = np.concatenate([own_raw_b, opp_raw_b], axis=0)
            both_mask = np.concatenate([own_mask_b, opp_mask_b], axis=0)
            u_both = np.concatenate([u_draws[h_step, :, 0], u_draws[h_step, :, 1]], axis=0)
            bh, be, bm = adapter(both_raw)
            bh_t = torch.as_tensor(bh, device=device)
            be_t = torch.as_tensor(be, device=device)
            bm_t = torch.as_tensor(bm, device=device)
            both_mem = reserve_step_memory(both_mem, 128)
            with actor_ctx:
                if hasattr(actor, "step_sample_crn"):
                    both_acts, both_mem = actor.step_sample_crn(
                        bh_t,
                        be_t,
                        bm_t,
                        both_mask,
                        u_both,
                        state=both_mem,
                        hand_card_ids=torch.as_tensor(both_raw[:, hand_indices], device=device),
                        force_wait=fw_both,
                        force_slot_argmax=fs_both,
                    )
                    own_acts = both_acts[:total_lanes]
                    opp_acts = both_acts[total_lanes:]
                else:
                    both_lp, _, both_mem = actor.step(
                        bh_t,
                        be_t,
                        bm_t,
                        torch.as_tensor(both_mask, dtype=torch.bool, device=device),
                        state=both_mem,
                        hand_card_ids=torch.as_tensor(both_raw[:, hand_indices], device=device),
                        compute_value=False,
                    )
                    own_lp = both_lp[:total_lanes]
                    opp_lp = both_lp[total_lanes:]
                    opp_acts = LookaheadPolicyStream._sample_legal_crn(opp_lp, opp_mask_b, u_draws[h_step, :, 1])
                    own_acts = LookaheadPolicyStream._sample_legal_crn(own_lp, own_mask_b, u_draws[h_step, :, 0])
                    if fw_both is not None and fs_both is not None:
                        own_lp_np = own_lp.detach().float().cpu().numpy()
                        for lane in active:
                            if fw_both[lane]:
                                own_acts[lane] = 0
                            elif fs_both[lane] >= 0:
                                lo = 1 + int(fs_both[lane]) * tiles
                                hi = lo + tiles
                                slot_masked = np.where(own_mask_b[lane, lo:hi], own_lp_np[lane, lo:hi], -np.inf)
                                own_acts[lane] = lo + int(np.argmax(slot_masked))
            if critic is not None and crit_mem is not None:
                crit_h_steps.append(bh_t[:total_lanes] if critic_device == device else bh_t[:total_lanes].to(critic_device))
                crit_e_steps.append(be_t[:total_lanes] if critic_device == device else be_t[:total_lanes].to(critic_device))
                crit_m_steps.append(
                    bm_t[:total_lanes].bool()
                    if critic_device == device
                    else bm_t[:total_lanes].to(critic_device, dtype=torch.bool)
                )

        vector.actions[lane_rows, lane_seat] = own_acts
        vector.actions[lane_rows, lane_opp_seat] = opp_acts
        vector.step_lanes(active)

        for lane in active:
            s_l = int(lane_seat[lane])
            cur_pot = _potential_value(search_env.observations[lane, s_l], schema)
            step_shaping = cfg.gamma * cur_pot - prev_pot[lane]
            prev_pot[lane] = cur_pot
            if search_env.dones[lane]:
                info = vector.info(lane)
                if info.get("fault"):
                    term = 0.0
                elif info["winner"] == s_l:
                    term = 1.0
                elif info["winner"] == (1 - s_l):
                    term = -1.0
                else:
                    term = 0.0
                returns[lane] += discounts[lane] * (term + step_shaping)
                live[lane] = False
            else:
                returns[lane] += discounts[lane] * step_shaping
                discounts[lane] *= cfg.gamma

    # 3. Batched 1-shot fused leaf Critic bootstrap across surviving lanes of all tasks
    if live.any():
        live_rows = np.flatnonzero(live)
        leaf_obs = search_env.observations[live_rows, lane_seat[live_rows]]
        if critic is not None and crit_mem is not None:
            lh, le, lm = adapter(leaf_obs)
            leaf_vals = _eval_critic_sequence_leaf(
                critic,
                crit_mem,
                crit_h_steps,
                crit_e_steps,
                crit_m_steps,
                lh,
                le,
                lm,
                live_rows,
                critic_device,
                critic_ctx,
            )
            for pos_l, lane in enumerate(live_rows):
                returns[lane] += discounts[lane] * float(leaf_vals[pos_l])
        else:
            for pos_l, lane in enumerate(live_rows):
                returns[lane] += discounts[lane] * (_potential_value(leaf_obs[pos_l], schema) - root_pots[lane])

    # 4. Per-match candidate aggregation and override margin gating
    outputs: list[tuple[int, float]] = []
    for b_idx, task in enumerate(tasks):
        lo_l, hi_l, k_b = spans[b_idx]
        st = task["stream"]
        cands = task["candidates"]
        r_raw = task["root_raw"]
        r_legal = np.asarray(task["root_legal"], dtype=bool)
        def_act = int(task["baseline_action"])
        cand_returns = returns[lo_l:hi_l].reshape(k_b, r).mean(axis=1)
        lp_np = task["root_logp"].detach().float().cpu().numpy().reshape(-1)
        def_lp = float(lp_np[def_act]) if 0 <= def_act < len(lp_np) else float(np.max(lp_np))
        bank_prior = max(float(lp_np[0]), def_lp - 0.5)
        raw_priors = []
        for a in cands:
            if int(a) < 0:
                raw_priors.append(bank_prior)
            else:
                p_val = float(lp_np[a])
                if int(a) > 0 and not _is_engaged_global_spell(
                    int(a), r_raw, r_legal, schema, rage_card_ids=st.rage_card_ids
                ):
                    p_val -= 4.0
                raw_priors.append(p_val)
        cand_priors = np.asarray(raw_priors, dtype=np.float64)
        cand_priors = np.maximum(cand_priors - cand_priors.max(), -float(cfg.max_prior_penalty_nats))
        total_scores = cand_returns + cfg.prior_weight * cand_priors

        best_idx = int(np.argmax(total_scores))
        default_idx = cands.index(def_act) if def_act in cands else 0
        gain = float(total_scores[best_idx] - total_scores[default_idx])
        chosen = int(cands[best_idx]) if (best_idx != default_idx and gain >= cfg.override_margin) else def_act
        outputs.append((chosen, gain if chosen != def_act else 0.0))
    return outputs
