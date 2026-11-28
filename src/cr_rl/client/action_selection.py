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
    """Reject empty-grass or duplicate-stacked Rage; allow real combat, swarm splash, and heavy bridge pushes."""
    count = int(obs[S.H_ENTITY_COUNT])
    if count <= 0:
        return False

    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    alive = rows[:, S.E_HP] > 0
    chars = alive & (rows[:, S.E_KIND] == S.KIND_CHARACTER)
    towers = alive & (rows[:, S.E_KIND] == S.KIND_TOWER)
    enemy_c = rows[chars & (rows[:, S.E_SIDE] == 1)]
    enemy_t = rows[towers & (rows[:, S.E_SIDE] == 1)]

    # 1. Critically low enemy tower finish within Rage splash radius
    if len(enemy_t):
        d2_et = (enemy_t[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_t[:, S.E_Y] / 1000.0 - y) ** 2
        if bool(((d2_et <= 2.8 ** 2) & (enemy_t[:, S.E_HP] <= 170)).any()):
            return True

    # 2. Avoid double-stacking Rage onto an already-active friendly area spell
    own_areas = rows[(rows[:, S.E_SIDE] == 0) & (rows[:, S.E_KIND] == S.KIND_AREA)]
    if len(own_areas):
        d2_a = (own_areas[:, S.E_X] / 1000.0 - x) ** 2 + (own_areas[:, S.E_Y] / 1000.0 - y) ** 2
        if bool((d2_a <= 3.0 ** 2).any()):
            return False

    # 3. Swarm clear: >=2 barrel/swarm units on own side (y<=12.0, hp<=340) OR >=5 lethal-splash swarm anywhere (y<=26.0, hp<=205)
    if y <= 26.0 and len(enemy_c):
        d2_ec = (enemy_c[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_c[:, S.E_Y] / 1000.0 - y) ** 2
        hp_cap = 340 if y <= 12.0 else 205
        swarm_cnt = int(((d2_ec <= 3.2 ** 2) & (enemy_c[:, S.E_HP] <= hp_cap)).sum())
        if swarm_cnt >= (2 if y <= 12.0 else 5):
            return True

    # 4. Friendly troop engagement (any elixir >= 2.0, offense or defense) OR proactive heavy bridge push
    own_troop = chars & (rows[:, S.E_SIDE] == 0) & ((rows[:, S.E_FLAGS] & S.F_BUILDING) == 0)
    own_c = rows[own_troop]
    if not len(own_c):
        return False
    d2_own = (own_c[:, S.E_X] / 1000.0 - x) ** 2 + (own_c[:, S.E_Y] / 1000.0 - y) ** 2
    buffed = own_c[d2_own <= 3.8 ** 2]
    buffed_hp = int(buffed[:, S.E_HP].sum()) if len(buffed) else 0
    if buffed_hp < 600:
        return False
    # Proactive heavy bridge push (>=1800 HP already at/past bridge marching toward alive enemy tower)
    if buffed_hp >= 1800 and float(buffed[:, S.E_Y].max()) / 1000.0 >= 13.5 and len(enemy_t):
        return True
    targets = np.concatenate([enemy_c, enemy_t], axis=0) if (len(enemy_c) or len(enemy_t)) else np.empty((0, S.ENTITY_INTS))
    if not len(targets):
        return False
    dx = buffed[:, None, S.E_X] / 1000.0 - targets[None, :, S.E_X] / 1000.0
    dy = buffed[:, None, S.E_Y] / 1000.0 - targets[None, :, S.E_Y] / 1000.0
    return bool(((dx * dx + dy * dy) <= 7.0 ** 2).any())


class HazardTempoState:
    """Stateful cumulative-intent integrator for balanced `hazard` deployment tempo.

    Bridges the binary cliff of single-frame `argmax` (ultra-passive below 50%, snap-play above 50%)
    and the noisy single-frame randomness of `policy_sample`.
    """

    def __init__(
        self,
        *,
        instant_play: float = 0.55,
        instant_slot: float = 0.45,
        min_step_play: float = 0.22,
        hazard_trigger: float = 0.65,
    ):
        self.instant_play = float(instant_play)
        self.instant_slot = float(instant_slot)
        self.min_step_play = float(min_step_play)
        self.hazard_trigger = float(hazard_trigger)
        self.accum_hazard: float = 0.0
        self.sticky_slot: int | None = None
        self.sticky_frames: int = 0

    def reset(self) -> None:
        self.accum_hazard = 0.0
        self.sticky_slot = None
        self.sticky_frames = 0

    def step(self, play_mass: float, top_slot: int | None, top_slot_mass: float, elixir_bars: float) -> tuple[bool, str]:
        """Update cumulative play hazard and return `(should_play, reason_hint)`."""
        # Smoothly ease thresholds as elixir approaches 10.0 so there is no abrupt 9.5-bar cliff
        if elixir_bars >= 9.0:
            frac = min(1.0, max(0.0, (elixir_bars - 9.0) / 0.5))
            inst_p = self.instant_play - 0.20 * frac
            inst_s = self.instant_slot - 0.15 * frac
            trig = self.hazard_trigger - 0.25 * frac
            min_step = self.min_step_play * (1.0 - 0.5 * frac)
        else:
            inst_p = self.instant_play
            inst_s = self.instant_slot
            trig = self.hazard_trigger
            min_step = self.min_step_play

        if play_mass >= inst_p or top_slot_mass >= inst_s:
            return True, "hazard_instant"

        if play_mass < min_step or top_slot is None:
            self.accum_hazard *= 0.40
            if self.accum_hazard < 0.05:
                self.reset()
            return False, "hazard_wait"

        if self.sticky_slot is not None and top_slot != self.sticky_slot:
            self.accum_hazard *= 0.50
            self.sticky_frames = 0
        self.sticky_slot = top_slot
        self.sticky_frames += 1
        self.accum_hazard = 1.0 - (1.0 - self.accum_hazard) * (1.0 - min(0.95, play_mass))
        if self.accum_hazard >= trig:
            return True, "hazard_accum"
        return False, "hazard_wait"


def _refine_rage_tile(tile: int, slot_probs: np.ndarray, slot_legal: np.ndarray, obs: np.ndarray) -> int:
    """Nudge Rage within local high-probability neighborhood (+1..2 lead / swarm cover) without overriding intent."""
    count = int(obs[S.H_ENTITY_COUNT]) if len(obs) > S.H_ENTITY_COUNT else 0
    if count <= 0:
        return tile
    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    chars = (rows[:, S.E_HP] > 0) & (rows[:, S.E_KIND] == S.KIND_CHARACTER)
    own_c = rows[chars & (rows[:, S.E_SIDE] == 0) & ((rows[:, S.E_FLAGS] & S.F_BUILDING) == 0)]
    ene_sw = rows[chars & (rows[:, S.E_SIDE] == 1) & (rows[:, S.E_HP] <= 340)]
    if not len(own_c) and not len(ene_sw):
        return tile
    base_r, base_c = divmod(int(tile), S.COLS)
    base_p = float(slot_probs[tile])
    if base_p <= 0:
        return tile
    best_tile = int(tile)
    best_score = -1e9
    for dr in (-1, 0, 1, 2):
        for dc in (-1, 0, 1):
            if abs(dr) + abs(dc) > 2:
                continue
            nr, nc = base_r + dr, base_c + dc
            if not (0 <= nr < S.ROWS and 0 <= nc < S.COLS):
                continue
            cand = nr * S.COLS + nc
            if not slot_legal[cand] or float(slot_probs[cand]) < 0.25 * base_p:
                continue
            cx, cy = nc + 0.5, nr + 0.5
            score = 0.0
            if len(own_c):
                # Reward leading advancing troops (+1.2 tiles ahead in y when moving toward enemy)
                lead_y = np.where(own_c[:, S.E_Y] >= 11000, np.minimum(own_c[:, S.E_Y] / 1000.0 + 1.2, 25.0), own_c[:, S.E_Y] / 1000.0)
                d2_o = (own_c[:, S.E_X] / 1000.0 - cx) ** 2 + (lead_y - cy) ** 2
                covered_hp = float(own_c[d2_o <= 3.0 ** 2, S.E_HP].sum())
                score += covered_hp / 1000.0
            if len(ene_sw):
                d2_s = (ene_sw[:, S.E_X] / 1000.0 - cx) ** 2 + (ene_sw[:, S.E_Y] / 1000.0 - cy) ** 2
                score += 0.45 * float((d2_s <= 3.0 ** 2).sum())
            score += 0.20 * (float(slot_probs[cand]) / base_p)
            if score > best_score:
                best_score = score
                best_tile = cand
    return best_tile


def _avoid_endgame_backline_tile(
    tile: int,
    slot_probs: np.ndarray,
    slot_legal: np.ndarray,
    obs: np.ndarray,
    card_name: str | None,
) -> int:
    """In the final 12s when behind/tied and an enemy tower is <=500 HP, avoid slow backline (row<=5) drops."""
    if card_name in ("Elixir Collector", None) or len(obs) < S.H_TOWERS + 12:
        return tile
    n_leg = int(slot_legal.sum())
    if not (0 < n_leg < 500):
        return tile
    overtime = int(obs[S.H_OVERTIME]) > 0
    sec_left = int(obs[S.H_SECTION_LEFT_MS])
    if overtime or not (0 < sec_left <= 12000):
        return tile
    own_crowns = int(obs[S.H_CROWNS_OWN]) if hasattr(S, "H_CROWNS_OWN") else int(obs[4])
    ene_crowns = int(obs[S.H_CROWNS_ENEMY]) if hasattr(S, "H_CROWNS_ENEMY") else int(obs[5])
    if own_crowns > ene_crowns:
        return tile
    row, _col = divmod(int(tile), S.COLS)
    if row > 5:
        return tile
    ene_l = int(obs[S.H_TOWERS + 8])
    ene_r = int(obs[S.H_TOWERS + 10])
    if 0 < ene_l <= 500 and (ene_r == 0 or ene_l <= ene_r):
        want_left = True
    elif 0 < ene_r <= 500 and (ene_l == 0 or ene_r < ene_l):
        want_left = False
    else:
        return tile
    grid_idx = np.arange(S.TILES, dtype=np.int32)
    grid_r = grid_idx // S.COLS
    grid_c = grid_idx % S.COLS
    lane = (grid_c < S.COLS // 2) if want_left else (grid_c >= S.COLS // 2)
    bridge = slot_legal & lane & (grid_r >= 11) & (grid_r <= 14)
    if not bridge.any():
        return tile
    cand = int(np.argmax(np.where(bridge, slot_probs, -np.inf)))
    if float(slot_probs[cand]) >= 0.02 * max(float(slot_probs[tile]), 1e-9):
        return cand
    return tile


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
    tempo_state: HazardTempoState | None = None,
):
    """Select an action directly from model probabilities with physical empty-spell veto."""
    if selection_mode not in ("policy_sample", "confidence", "argmax", "hazard"):
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

    if selection_mode in ("argmax", "hazard"):
        wait_mass = float(p[0])
        tiles = p[1 : S.FIRST_ABILITY].reshape(4, S.TILES)
        slots = tiles.sum(-1)
        skills = p[S.FIRST_ABILITY :]
        valid_non_wait = float(slots.sum()) + float(skills.sum())
        elixir_bars = float(obs[S.H_ELIXIR]) / float(S.ELIXIR_SCALE) if len(obs) > S.H_ELIXIR else 0.0
        near_full = int(obs[S.H_ELIXIR]) >= 95000 and valid_non_wait >= 0.05

        lethal = _lethal_tower_spell_action(hand_cards, legal, tiles, slots, obs)
        if lethal is not None:
            l_act, l_slot = lethal
            if tempo_state is not None:
                tempo_state.reset()
            return l_act, {
                "selection_mode": selection_mode,
                "proposed_action": l_act,
                "proposed_action_probability": float(p[l_act]),
                "selected_slot_probability": float(slots[l_slot]),
                "spell_vetoed": False,
                "selection_reason": "lethal_tower_spell",
            }

        hazard_hint = None
        if selection_mode == "hazard":
            st = tempo_state if tempo_state is not None else HazardTempoState()
            top_slot = int(np.argmax(slots)) if float(slots.max()) > 0 else None
            top_slot_mass = float(slots[top_slot]) if top_slot is not None else 0.0
            should_play, hazard_hint = st.step(valid_non_wait, top_slot, top_slot_mass, elixir_bars)
            if (not near_full) and (not should_play) and legal[0]:
                return 0, {
                    "selection_mode": selection_mode,
                    "proposed_action": 0,
                    "proposed_action_probability": wait_mass,
                    "spell_vetoed": False,
                    "selection_reason": "hazard_wait",
                    "hazard_accum": round(float(st.accum_hazard), 4),
                    "hazard_sticky_frames": int(st.sticky_frames),
                }
        elif (not near_full) and legal[0] and wait_mass >= valid_non_wait:
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
            if tempo_state is not None:
                tempo_state.reset()
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
            if first_vetoed is not None and (not near_full) and legal[0]:
                if selection_mode == "argmax" and wait_mass >= valid_non_wait:
                    break
                if selection_mode == "hazard" and valid_non_wait < 0.22:
                    break
            slot_legal = legal[1 + slot * S.TILES : 1 + (slot + 1) * S.TILES]
            if not slot_legal.any():
                valid_non_wait = max(0.0, valid_non_wait - float(slots[slot]))
                continue
            card_name = hand_cards[slot] if slot < len(hand_cards) else None
            tile = _best_cluster_tile(tiles[slot], slot_legal)
            if card_name == "Rage":
                tile = _refine_rage_tile(tile, tiles[slot], slot_legal, obs)
            else:
                tile = _orient_center_seam_tile(tile, tiles[slot], slot_legal, obs)
                tile = _avoid_endgame_backline_tile(tile, tiles[slot], slot_legal, obs, card_name)
            proposed = S.play_tile(slot, tile)
            if not spell_has_target(proposed, card_name, obs):
                if first_vetoed is None:
                    first_vetoed = proposed
                valid_non_wait = max(0.0, valid_non_wait - float(slots[slot]))
                continue
            if tempo_state is not None:
                tempo_state.reset()
            if near_full and ((selection_mode == "argmax" and wait_mass >= valid_non_wait) or hazard_hint == "hazard_wait"):
                reason = "leak_guard_play"
            elif selection_mode == "hazard" and hazard_hint == "hazard_accum":
                reason = "hazard_play"
            else:
                reason = "play"
            return proposed, {
                "selection_mode": selection_mode,
                "proposed_action": proposed,
                "proposed_action_probability": float(p[proposed]),
                "selected_slot_probability": float(slots[slot]),
                "spell_vetoed": first_vetoed is not None,
                "selection_reason": reason,
            }

        if first_vetoed is not None:
            if tempo_state is not None:
                tempo_state.reset()
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
            "selection_reason": "hazard_wait" if selection_mode == "hazard" else "policy_wait",
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
        card_name = hand_cards[slot] if slot < len(hand_cards) else None
        if tile_temperature <= 1e-4:
            tile = _best_cluster_tile(tiles[slot], slot_legal)
            if card_name == "Rage":
                tile = _refine_rage_tile(tile, tiles[slot], slot_legal, obs)
            else:
                tile = _orient_center_seam_tile(tile, tiles[slot], slot_legal, obs)
                tile = _avoid_endgame_backline_tile(tile, tiles[slot], slot_legal, obs, card_name)
        else:
            rng = np.random.default_rng() if rng is None else rng
            valid = slot_legal & (tiles[slot] > 0)
            logp = np.full(S.TILES, -np.inf)
            logp[valid] = np.log(tiles[slot, valid]) / tile_temperature
            weights = np.exp(logp - logp.max())
            tile = int(rng.choice(S.TILES, p=weights / weights.sum()))
        action = S.play_tile(slot, tile)
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
