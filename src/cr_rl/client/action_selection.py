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

# Conservative guaranteed 1-cast Crown Tower damage at Tournament Level 11.
LETHAL_SPELL_TOWER_DMG_L11 = {
    "Rage": 55,
    "Zap": 46,
    "Snowball": 46,
    "Log": 56,
    "BarbLog": 56,
    "Tornado": 40,
    "Freeze": 28,
    "Arrows": 90,
    "Fireball": 205,
    "Poison": 180,
    "Earthquake": 180,
    "Lightning": 310,
    "Rocket": 365,
}


def _rage_has_engagement(x: float, y: float, obs: np.ndarray) -> bool:
    """Reject casting Rage onto empty grass or starving low-elixir defense; require real engagement."""
    count = int(obs[S.H_ENTITY_COUNT])
    if count <= 0:
        return False
    elix_raw = int(obs[S.H_ELIXIR]) if len(obs) > S.H_ELIXIR else 0
    res_raw = max(0, int(obs[S.H_ELIXIR_RESERVED])) if len(obs) > S.H_ELIXIR_RESERVED else 0
    avail_bars = float(elix_raw - res_raw) / float(S.ELIXIR_SCALE)
    low_elixir = 1.9 <= avail_bars < 4.2

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

    if y <= 15.0 and len(enemy_c):
        d2_ec = (enemy_c[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_c[:, S.E_Y] / 1000.0 - y) ** 2
        hp_cap = 340 if y <= 10.5 else 205
        swarm_cnt = int(((d2_ec <= 3.2 ** 2) & (enemy_c[:, S.E_HP] <= hp_cap)).sum())
        if swarm_cnt >= (2 if y <= 10.5 else 6):
            return True

    if low_elixir and len(enemy_c):
        unbuffed_threat = (enemy_c[:, S.E_Y] / 1000.0 <= 16.5) & (
            ((enemy_c[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_c[:, S.E_Y] / 1000.0 - y) ** 2) > 5.5 ** 2
        )
        if int(enemy_c[unbuffed_threat, S.E_HP].sum()) >= 600:
            return False

    own_troop = chars & (rows[:, S.E_SIDE] == 0) & ((rows[:, S.E_FLAGS] & S.F_BUILDING) == 0)
    own_c = rows[own_troop]
    if not len(own_c):
        return False
    d2_own = (own_c[:, S.E_X] / 1000.0 - x) ** 2 + (own_c[:, S.E_Y] / 1000.0 - y) ** 2
    buffed = own_c[d2_own <= 3.8 ** 2]
    min_hp = 1800 if low_elixir else 450
    if not len(buffed) or int(buffed[:, S.E_HP].sum()) < min_hp:
        return False
    if low_elixir and float(buffed[:, S.E_Y].max()) / 1000.0 < 13.5:
        return False
    if int(buffed[:, S.E_HP].sum()) >= 1200 and float(buffed[:, S.E_Y].max()) / 1000.0 >= 13.5 and len(enemy_t):
        return True
    targets = np.concatenate([enemy_c, enemy_t], axis=0) if (len(enemy_c) or len(enemy_t)) else np.empty((0, S.ENTITY_INTS))
    if not len(targets):
        return False
    dx = buffed[:, None, S.E_X] / 1000.0 - targets[None, :, S.E_X] / 1000.0
    dy = buffed[:, None, S.E_Y] / 1000.0 - targets[None, :, S.E_Y] / 1000.0
    return bool(((dx * dx + dy * dy) <= 7.0 ** 2).any())


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


def _orient_center_seam_tile(tile: int, slot_probs: np.ndarray, slot_legal: np.ndarray, obs: np.ndarray) -> int:
    """Break center-line (col 8 vs 9, row <= 9) ties toward the weaker/alive enemy Princess Tower."""
    n_leg = int(slot_legal.sum())
    if not (0 < n_leg < 500) or len(obs) < S.H_TOWERS + 12:
        return tile
    row, col = divmod(int(tile), S.COLS)
    if row > 9 or col not in (8, 9):
        return tile
    ene_l = int(obs[S.H_TOWERS + 8])
    ene_r = int(obs[S.H_TOWERS + 10])
    if ene_l > 0 and ene_r > 0 and abs(ene_l - ene_r) >= 1200:
        target_col = 8 if ene_l < ene_r else 9
    elif ene_l == 0 and 0 < ene_r <= 3600:
        target_col = 9
    elif ene_r == 0 and 0 < ene_l <= 3600:
        target_col = 8
    else:
        return tile
    if col == target_col:
        return tile
    alt_tile = row * S.COLS + target_col
    if slot_legal[alt_tile] and float(slot_probs[alt_tile]) >= 0.25 * float(slot_probs[tile]):
        return int(alt_tile)
    return tile


def _lethal_tower_spell_action(
    hand_cards,
    legal: np.ndarray,
    tiles: np.ndarray,
    slots: np.ndarray,
    obs: np.ndarray,
) -> tuple[int, int] | None:
    """Return `(action, slot)` when a legal spell in hand can 1-cast finish an alive enemy tower."""
    if len(obs) < S.HEADER_INTS:
        return None
    count = int(obs[S.H_ENTITY_COUNT])
    if count <= 0:
        return None
    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    enemy_t = rows[(rows[:, S.E_KIND] == S.KIND_TOWER) & (rows[:, S.E_SIDE] == 1) & (rows[:, S.E_HP] > 0)]
    if not len(enemy_t):
        return None
    overtime = int(obs[S.H_OVERTIME]) > 0
    sec_left = int(obs[S.H_SECTION_LEFT_MS])
    urgent_clock = overtime or (0 <= sec_left <= 20000)

    deck_ids = obs[S.H_DECK : S.H_DECK + S.DECK_SIZE]
    deck_lvls = obs[S.H_DECK_LEVEL : S.H_DECK_LEVEL + S.DECK_SIZE]
    hand_ids = obs[S.H_HAND_CARD : S.H_HAND_CARD + S.HAND_SLOTS]

    for slot in range(min(4, len(hand_cards))):
        card_name = hand_cards[slot]
        base_dmg = LETHAL_SPELL_TOWER_DMG_L11.get(card_name)
        if base_dmg is None:
            continue
        if not urgent_clock and float(slots[slot]) < 0.05:
            continue
        slot_legal = legal[1 + slot * S.TILES : 1 + (slot + 1) * S.TILES]
        if not slot_legal.any():
            continue
        lvl = 11
        cid = int(hand_ids[slot])
        for d_i in range(S.DECK_SIZE):
            if int(deck_ids[d_i]) == cid and int(deck_lvls[d_i]) > 0:
                lvl = int(deck_lvls[d_i])
                break
        lethal_hp = int(base_dmg * (1.1 ** max(0, min(lvl, 16) - 11)))
        order = sorted(range(len(enemy_t)), key=lambda i: (0 if (int(enemy_t[i, S.E_FLAGS]) & S.F_KING) else 1, int(enemy_t[i, S.E_HP])))
        for i_t in order:
            tow = enemy_t[i_t]
            if int(tow[S.E_HP]) <= lethal_hp:
                tc = int(np.clip(round(float(tow[S.E_X]) / 1000.0 - 0.5), 0, S.COLS - 1))
                tr = int(np.clip(round(float(tow[S.E_Y]) / 1000.0 - 0.5), 0, S.ROWS - 1))
                if card_name in ("Log", "BarbLog"):
                    tr = min(tr, 14)
                t_idx = tr * S.COLS + tc
                if slot_legal[t_idx]:
                    act = S.play_tile(slot, t_idx)
                    if spell_has_target(act, card_name, obs):
                        return act, slot
    return None


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
        valid_non_wait = float(slots.sum()) + float(skills.sum())
        near_full = int(obs[S.H_ELIXIR]) >= 95000 and valid_non_wait >= 0.05

        lethal = _lethal_tower_spell_action(hand_cards, legal, tiles, slots, obs)
        if lethal is not None:
            l_act, l_slot = lethal
            return l_act, {
                "selection_mode": selection_mode,
                "proposed_action": l_act,
                "proposed_action_probability": float(p[l_act]),
                "selected_slot_probability": float(slots[l_slot]),
                "spell_vetoed": False,
                "selection_reason": "lethal_tower_spell",
            }

        if (not near_full) and legal[0] and wait_mass >= valid_non_wait:
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
            if first_vetoed is not None and (not near_full) and legal[0] and wait_mass >= valid_non_wait:
                break
            slot_legal = legal[1 + slot * S.TILES : 1 + (slot + 1) * S.TILES]
            if not slot_legal.any():
                valid_non_wait = max(0.0, valid_non_wait - float(slots[slot]))
                continue
            card_name = hand_cards[slot] if slot < len(hand_cards) else None
            tile = _best_cluster_tile(tiles[slot], slot_legal)
            tile = _orient_center_seam_tile(tile, tiles[slot], slot_legal, obs)
            proposed = S.play_tile(slot, tile)
            if not spell_has_target(proposed, card_name, obs):
                if first_vetoed is None:
                    first_vetoed = proposed
                valid_non_wait = max(0.0, valid_non_wait - float(slots[slot]))
                continue
            reason = "leak_guard_play" if (near_full and wait_mass >= valid_non_wait) else "play"
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
    near_full = int(obs[S.H_ELIXIR]) >= 95000
    play_limit = 0.15 if near_full else play_threshold
    slot_limit = 0.10 if near_full else slot_threshold
    info = {
        "selection_mode": selection_mode,
        "spell_vetoed": False,
        "selection_reason": "confidence_wait",
        "effective_play_threshold": play_limit,
        "effective_slot_threshold": slot_limit,
    }
    lethal = _lethal_tower_spell_action(hand_cards, legal, tiles, slots, obs)
    if lethal is not None:
        l_act, l_slot = lethal
        info.update(
            proposed_action=l_act,
            proposed_action_probability=float(p[l_act]),
            selected_slot_probability=float(slots[l_slot]),
            selection_reason="lethal_tower_spell",
        )
        return l_act, info
    skills = p[S.FIRST_ABILITY :]
    skill = int(skills.argmax())
    if (
        skills[skill] >= ability_threshold
        and skills[skill] >= max(float(p[0]), float(slots.max()))
        and legal[S.FIRST_ABILITY + skill]
    ):
        ab_act = S.FIRST_ABILITY + skill
        info.update(
            proposed_action=ab_act,
            proposed_action_probability=float(p[ab_act]),
            selection_reason="ability",
        )
        return ab_act, info
    if (slots.sum() + skills.sum()) < play_limit:
        info.update(proposed_action=0, proposed_action_probability=float(p[0]))
        return 0, info
    first_vetoed = None
    for slot in np.argsort(-slots):
        slot = int(slot)
        if slots[slot] < slot_limit or slots[slot] <= 0:
            break
        slot_legal = legal[1 + slot * S.TILES : 1 + (slot + 1) * S.TILES]
        if not slot_legal.any():
            continue
        if tile_temperature <= 1e-4:
            tile = _best_cluster_tile(tiles[slot], slot_legal)
            tile = _orient_center_seam_tile(tile, tiles[slot], slot_legal, obs)
        else:
            rng = np.random.default_rng() if rng is None else rng
            valid = slot_legal & (tiles[slot] > 0)
            logp = np.full(S.TILES, -np.inf)
            logp[valid] = np.log(tiles[slot, valid]) / tile_temperature
            weights = np.exp(logp - logp.max())
            tile = int(rng.choice(S.TILES, p=weights / weights.sum()))
        action = S.play_tile(slot, tile)
        card_name = hand_cards[slot] if slot < len(hand_cards) else None
        if not spell_has_target(action, card_name, obs):
            if first_vetoed is None:
                first_vetoed = action
            continue
        info.update(
            proposed_action=action,
            proposed_action_probability=float(p[action]),
            selected_slot_probability=float(slots[slot]),
            spell_vetoed=first_vetoed is not None,
            selection_reason="play",
        )
        return action, info
    if first_vetoed is not None:
        info.update(
            proposed_action=first_vetoed,
            proposed_action_probability=float(p[first_vetoed]),
            spell_vetoed=True,
            selection_reason="spell_guard_veto",
            vetoed_action=first_vetoed,
        )
        return 0, info
    info.update(proposed_action=0, proposed_action_probability=float(p[0]))
    return 0, info
