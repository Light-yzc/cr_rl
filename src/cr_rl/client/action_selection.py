"""Deployment action selection: pure model probabilities + physical empty-spell veto."""
from __future__ import annotations

import numpy as np

from .schema_v4 import SchemaV4 as S


DAMAGE_SPELL_RADIUS = {
    "Zap": 4.0, "Snowball": 4.0, "Arrows": 5.5, "Fireball": 4.0,
    "Poison": 5.0, "Rocket": 3.5, "Lightning": 5.0, "Log": 4.5,
    "BarbLog": 4.5, "Tornado": 5.0, "Earthquake": 5.0, "Freeze": 4.5,
    "Vines": 4.5, "RoyalDelivery": 4.5,
}


def _rage_has_engagement(x: float, y: float, obs: np.ndarray) -> bool:
    """Reject casting Rage onto empty grass; require engaged friendly troops, low tower, or tower swarm."""
    count = int(obs[S.H_ENTITY_COUNT])
    if count <= 0:
        return False
    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    alive = rows[:, S.E_HP] > 0
    chars = alive & (rows[:, S.E_KIND] == S.KIND_CHARACTER)
    towers = alive & (rows[:, S.E_KIND] == S.KIND_TOWER)
    enemy_c = rows[chars & (rows[:, S.E_SIDE] == 1)]
    enemy_t = rows[towers & (rows[:, S.E_SIDE] == 1)]

    if len(enemy_t):
        d2_et = (enemy_t[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_t[:, S.E_Y] / 1000.0 - y) ** 2
        if bool(((d2_et <= 2.8 ** 2) & (enemy_t[:, S.E_HP] <= 170)).any()):
            return True

    if y <= 11.5 and len(enemy_c):
        d2_ec = (enemy_c[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_c[:, S.E_Y] / 1000.0 - y) ** 2
        if int(((d2_ec <= 3.2 ** 2) & (enemy_c[:, S.E_HP] <= 380)).sum()) >= 2:
            return True

    own_troop = chars & (rows[:, S.E_SIDE] == 0) & ((rows[:, S.E_FLAGS] & S.F_BUILDING) == 0)
    own_c = rows[own_troop]
    if not len(own_c):
        return False
    d2_own = (own_c[:, S.E_X] / 1000.0 - x) ** 2 + (own_c[:, S.E_Y] / 1000.0 - y) ** 2
    buffed = own_c[d2_own <= 3.8 ** 2]
    if not len(buffed) or int(buffed[:, S.E_HP].sum()) < 300:
        return False
    targets = np.concatenate([enemy_c, enemy_t], axis=0) if (len(enemy_c) or len(enemy_t)) else np.empty((0, S.ENTITY_INTS))
    if not len(targets):
        return False
    dx = buffed[:, None, S.E_X] / 1000.0 - targets[None, :, S.E_X] / 1000.0
    dy = buffed[:, None, S.E_Y] / 1000.0 - targets[None, :, S.E_Y] / 1000.0
    return bool(((dx * dx + dy * dy) <= 6.5 ** 2).any())


def _best_cluster_tile(slot_probs: np.ndarray, slot_legal: np.ndarray) -> int:
    """Pick the legal tile maximizing local 3x3 policy probability mass."""
    valid = slot_legal & (slot_probs > 0)
    if not valid.any():
        return int(np.argmax(np.where(slot_legal, slot_probs, -np.inf)))
    grid = slot_probs.reshape(S.ROWS, S.COLS)
    pad = np.pad(grid, 1, mode="constant")
    cluster = sum(pad[dr : dr + S.ROWS, dc : dc + S.COLS] for dr in range(3) for dc in range(3)).reshape(-1)
    scores = np.where(valid, cluster + 0.25 * slot_probs, -np.inf)
    return int(np.argmax(scores))


def spell_has_target(action, card_name, obs):
    decoded = S.decode_action(action)
    if decoded["kind"] != "play":
        return True
    x, y = decoded["col"] + 0.5, decoded["row"] + 0.5
    if card_name == "Rage":
        return _rage_has_engagement(x, y, obs)
    radius = DAMAGE_SPELL_RADIUS.get(card_name)
    if radius is None:
        return True
    count = int(obs[S.H_ENTITY_COUNT])
    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    enemy = (rows[:, S.E_SIDE] == 1) & (rows[:, S.E_HP] > 0)
    in_range = ((rows[:, S.E_X] / 1000 - x) ** 2 + (rows[:, S.E_Y] / 1000 - y) ** 2) <= radius ** 2
    tower = rows[:, S.E_KIND] == S.KIND_TOWER
    body = rows[:, S.E_KIND] == S.KIND_CHARACTER
    if card_name == "Log":
        dx = np.abs(rows[:, S.E_X] / 1000 - x)
        dy = rows[:, S.E_Y] / 1000 - y
        in_range = (dx <= 1.95) & (dy >= -0.6) & (dy <= 10.1 + 0.6)
        enemy &= (rows[:, S.E_FLAGS] & S.F_AIR) == 0
    return bool((enemy & in_range & (body | (tower & (rows[:, S.E_HP] <= 450)))).any())


def select_live_action(
    probabilities,
    legal_mask,
    obs,
    hand_cards,
    *,
    play_threshold=0.45,
    slot_threshold=0.28,
    tile_temperature=0.0,
    rng=None,
    ability_threshold=0.28,
    selection_mode="confidence",
):
    """Select an action directly from model probabilities with empty-spell sanity veto."""
    if selection_mode not in ("policy_sample", "confidence", "argmax"):
        raise ValueError("Unknown live selection mode")
    p = np.asarray(probabilities, dtype=np.float64).copy()
    legal = np.asarray(legal_mask, dtype=bool)
    if p.shape != (S.ACTIONS,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected finite nonnegative 2307-action probabilities")
    p[~legal] = 0
    if p.sum() <= 0:
        return 0, {"spell_vetoed": False, "selection_reason": "no_legal_mass"}
    p /= p.sum()

    if selection_mode == "policy_sample":
        rng = np.random.default_rng() if rng is None else rng
        proposed = int(rng.choice(S.ACTIONS, p=p))
        decoded = S.decode_action(proposed)
        info = {
            "selection_mode": selection_mode,
            "proposed_action": proposed,
            "proposed_action_probability": float(p[proposed]),
            "spell_vetoed": False,
            "selection_reason": "policy_wait",
        }
        if decoded["kind"] == "play":
            if not spell_has_target(proposed, hand_cards[decoded["slot"]], obs):
                info.update(spell_vetoed=True, selection_reason="spell_guard_veto", vetoed_action=proposed)
                return 0, info
            info["selection_reason"] = "play"
        elif decoded["kind"] == "ability":
            info["selection_reason"] = "ability"
        return proposed, info

    if selection_mode == "argmax":
        wait_mass = float(p[0])
        tiles = p[1 : S.FIRST_ABILITY].reshape(4, S.TILES)
        slots = tiles.sum(-1)
        skills = p[S.FIRST_ABILITY :]
        non_wait_max = max(float(slots.max()), float(skills.max()))
        near_full = int(obs[S.H_ELIXIR]) >= 97000 and (slots.sum() + skills.sum()) >= 0.05

        if (not near_full) and legal[0] and wait_mass >= non_wait_max:
            return 0, {
                "selection_mode": selection_mode,
                "proposed_action": 0,
                "proposed_action_probability": wait_mass,
                "spell_vetoed": False,
                "selection_reason": "policy_wait",
            }

        skill = int(skills.argmax())
        if skills[skill] >= float(slots.max()) and skills[skill] > 0 and legal[S.FIRST_ABILITY + skill]:
            ab_act = S.FIRST_ABILITY + skill
            return ab_act, {
                "selection_mode": selection_mode,
                "proposed_action": ab_act,
                "proposed_action_probability": float(p[ab_act]),
                "spell_vetoed": False,
                "selection_reason": "ability",
            }

        first_vetoed = None
        for slot in np.argsort(-slots):
            slot = int(slot)
            if slots[slot] <= 0:
                break
            if first_vetoed is not None and not near_full and slots[slot] < max(wait_mass, 0.22):
                break
            slot_legal = legal[1 + slot * S.TILES : 1 + (slot + 1) * S.TILES]
            if not slot_legal.any():
                continue
            card_name = hand_cards[slot] if slot < len(hand_cards) else None
            tile = _best_cluster_tile(tiles[slot], slot_legal)
            proposed = S.play_tile(slot, tile)
            if not spell_has_target(proposed, card_name, obs):
                if first_vetoed is None:
                    first_vetoed = proposed
                continue
            reason = "leak_guard_play" if (near_full and wait_mass >= non_wait_max) else "play"
            return proposed, {
                "selection_mode": selection_mode,
                "proposed_action": proposed,
                "proposed_action_probability": float(p[proposed]),
                "selected_slot_probability": float(slots[slot]),
                "spell_vetoed": first_vetoed is not None,
                "selection_reason": reason,
            }

        if first_vetoed is not None:
            return 0, {
                "selection_mode": selection_mode,
                "proposed_action": first_vetoed,
                "proposed_action_probability": float(p[first_vetoed]),
                "spell_vetoed": True,
                "selection_reason": "spell_guard_veto",
                "vetoed_action": first_vetoed,
            }
        return 0, {
            "selection_mode": selection_mode,
            "proposed_action": 0,
            "proposed_action_probability": wait_mass,
            "spell_vetoed": False,
            "selection_reason": "policy_wait",
        }

    tiles = p[1 : S.FIRST_ABILITY].reshape(4, S.TILES)
    slots = tiles.sum(-1)
    best = int(slots.argmax())
    near_full = obs[S.H_ELIXIR] >= 95000
    play_limit = 0.15 if near_full else play_threshold
    slot_limit = 0.10 if near_full else slot_threshold
    info = {
        "spell_vetoed": False,
        "selection_reason": "confidence_wait",
        "effective_play_threshold": play_limit,
        "effective_slot_threshold": slot_limit,
    }
    skills = p[S.FIRST_ABILITY :]
    skill = int(skills.argmax())
    if (
        skills[skill] >= ability_threshold
        and skills[skill] >= max(float(p[0]), float(slots.max()))
        and legal[S.FIRST_ABILITY + skill]
    ):
        info["selection_reason"] = "ability"
        return S.FIRST_ABILITY + skill, info
    if slots.sum() < play_limit or slots[best] < slot_limit:
        return 0, info
    if tile_temperature <= 1e-4:
        tile = int(tiles[best].argmax())
    else:
        rng = np.random.default_rng() if rng is None else rng
        valid = legal[1 + best * S.TILES : 1 + (best + 1) * S.TILES] & (tiles[best] > 0)
        logp = np.full(S.TILES, -np.inf)
        logp[valid] = np.log(tiles[best, valid]) / tile_temperature
        weights = np.exp(logp - logp.max())
        tile = int(rng.choice(S.TILES, p=weights / weights.sum()))
    action = S.play_tile(best, tile)
    if not spell_has_target(action, hand_cards[best], obs):
        info.update(spell_vetoed=True, selection_reason="spell_guard_veto", vetoed_action=action)
        return 0, info
    info["selection_reason"] = "play"
    return action, info
