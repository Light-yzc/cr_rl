"""Deployment-only confidence gates; PPO continues to sample its own policy."""
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
    """Allow Rage only on engaged friendly pushes, low-HP tower finishers, or >=2 swarm units ON our tower."""
    count = int(obs[S.H_ENTITY_COUNT])
    if count <= 0:
        return False
    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    alive = rows[:, S.E_HP] > 0
    chars = alive & (rows[:, S.E_KIND] == S.KIND_CHARACTER)
    towers = alive & (rows[:, S.E_KIND] == S.KIND_TOWER)
    enemy_c = rows[chars & (rows[:, S.E_SIDE] == 1)]
    enemy_t = rows[towers & (rows[:, S.E_SIDE] == 1)]

    # Allow finishing a critically low enemy tower (<= 170 HP) within Rage radius
    if len(enemy_t):
        d2_et = (enemy_t[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_t[:, S.E_Y] / 1000.0 - y) ** 2
        if bool(((d2_et <= 2.8 ** 2) & (enemy_t[:, S.E_HP] <= 170)).any()):
            return True

    # Defensive naked Rage ONLY on our own tower zone (y <= 10.0) hitting >=2 low-HP swarm units (e.g. Goblin Barrel)
    if y <= 10.0 and len(enemy_c):
        d2_ec = (enemy_c[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_c[:, S.E_Y] / 1000.0 - y) ** 2
        if int(((d2_ec <= 3.0 ** 2) & (enemy_c[:, S.E_HP] <= 380)).sum()) >= 2:
            return True

    own_troop = chars & (rows[:, S.E_SIDE] == 0) & ((rows[:, S.E_FLAGS] & S.F_BUILDING) == 0)
    own_c = rows[own_troop]
    if not len(own_c):
        return False
    d2_own = (own_c[:, S.E_X] / 1000.0 - x) ** 2 + (own_c[:, S.E_Y] / 1000.0 - y) ** 2
    buffed = own_c[d2_own <= 3.8 ** 2]
    # Require meaningful friendly combat HP (>= 350 total HP, not a single dying 99-HP Minion)
    if not len(buffed) or int(buffed[:, S.E_HP].sum()) < 350:
        return False
    targets = np.concatenate([enemy_c, enemy_t], axis=0) if (len(enemy_c) or len(enemy_t)) else np.empty((0, S.ENTITY_INTS))
    if not len(targets):
        return False
    dx = buffed[:, None, S.E_X] / 1000.0 - targets[None, :, S.E_X] / 1000.0
    dy = buffed[:, None, S.E_Y] / 1000.0 - targets[None, :, S.E_Y] / 1000.0
    return bool(((dx * dx + dy * dy) <= 6.5 ** 2).any())


BUILDING_ONLY_CARDS = frozenset({
    "Giant", "RoyalGiant", "GoblinGiant", "ElectroGiant", "Golem",
    "ElixirGolem", "LavaHound", "Balloon", "HogRider", "RamRider",
    "BattleRam", "WallBreakers", "SuspiciousBush", "Elixir Collector",
})
PASSIVE_INVEST_CARDS = frozenset({"Elixir Collector"})
HIGH_COST_BANK_CARDS = frozenset({"ZapMachine", "Elixir Collector", "Golem", "LavaHound", "Pekka", "MegaKnight", "ThreeMusketeers"})
AOE_TROOP_OR_DAMAGE_SPELL = frozenset({
    "Wizard", "DarkPrince", "Valkyrie", "Executioner", "Bowler", "BabyDragon",
    "Witch", "MotherWitch", "Firecracker", "ElectroDragon", "Bomber", "Princess",
    "IceWizard", "Arrows", "Zap", "Snowball", "Log", "BarbLog",
    "Fireball", "Poison", "Tornado", "RoyalDelivery",
})
TOWER_SWARM_CHEAP_SPELLS = frozenset({"Rage", "Zap", "Snowball", "Log", "BarbLog", "Arrows"})
RANGED_SQUISHY_DEFENDERS = frozenset({
    "Wizard", "Musketeer", "Archers", "IceWizard", "Princess", "DartGoblin",
    "Firecracker", "Executioner", "Witch", "MotherWitch", "ElectroWizard",
})
SLOW_SINGLE_TARGET_ON_SWARM = frozenset({
    "ZapMachine", "MiniPekka", "Pekka", "Prince", "Giant", "Golem",
    "LavaHound", "Elixir Collector",
})


def _analyze_board_pressure(obs: np.ndarray) -> dict:
    """Inspect live Schema v4 entities & match context in seat frame (own side: y < 16000, left: x < 9000)."""
    tick = int(obs[S.H_TICK])
    overtime = int(obs[S.H_OVERTIME]) > 0
    behind_crowns = bool(
        obs[S.H_CROWNS] >= 0
        and obs[S.H_CROWNS_ENEMY] >= 0
        and int(obs[S.H_CROWNS]) < int(obs[S.H_CROWNS_ENEMY])
    )

    own_left_hp = int(obs[S.H_TOWERS + 2])
    own_right_hp = int(obs[S.H_TOWERS + 4])
    own_king_hp = int(obs[S.H_TOWERS])
    low_own_tower = any(0 < hp <= 2200 for hp in (own_left_hp, own_right_hp, own_king_hp))
    critical_own_tower = any(0 < hp <= 1800 for hp in (own_left_hp, own_right_hp, own_king_hp))
    # Never plant a 70s Elixir Collector after the first 75s (tick >= 1500), in overtime, behind in crowns, or with a low tower
    no_pump_window = bool(overtime or tick >= 1500 or behind_crowns or critical_own_tower)

    enemy_left_hp = int(obs[S.H_TOWERS + 8])
    enemy_right_hp = int(obs[S.H_TOWERS + 10])
    finisher_tower_lane = (
        "left" if 0 < enemy_left_hp <= 1200 else ("right" if 0 < enemy_right_hp <= 1200 else None)
    )
    # If one enemy Princess Tower is already destroyed and the other is alive, focus offense on the alive lane
    alive_attack_lane = None
    if enemy_left_hp > 0 and enemy_right_hp == 0 and (obs[S.H_TOWERS + 11] > 0 or enemy_left_hp <= 2500):
        alive_attack_lane = "left"
    elif enemy_right_hp > 0 and enemy_left_hp == 0 and (obs[S.H_TOWERS + 9] > 0 or enemy_right_hp <= 2500):
        alive_attack_lane = "right"

    base_info = {
        "quiet_board": True,
        "deep_enemy_hp": 0,
        "left_threat_hp": 0,
        "right_threat_hp": 0,
        "defend_target": None,
        "tower_under_attack": False,
        "swarm_on_tower": False,
        "tower_swarm_tile": None,
        "swarm_emergency": False,
        "swarm_lane": None,
        "low_own_tower": low_own_tower,
        "own_left_hp": own_left_hp,
        "own_right_hp": own_right_hp,
        "no_pump_window": no_pump_window,
        "has_own_pump": False,
        "own_tank_lanes": set(),
        "own_push_support_lane": None,
        "own_push_support_row": -1,
        "finisher_tower_lane": finisher_tower_lane,
        "alive_attack_lane": alive_attack_lane,
    }

    count = int(obs[S.H_ENTITY_COUNT])
    if count <= 0:
        return base_info

    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    alive = rows[:, S.E_HP] > 0

    own_chars = rows[alive & (rows[:, S.E_KIND] == S.KIND_CHARACTER) & (rows[:, S.E_SIDE] == 0)]
    if len(own_chars):
        own_bldg = (own_chars[:, S.E_FLAGS] & S.F_BUILDING) != 0
        if bool((own_bldg & (own_chars[:, S.E_Y] <= 8500)).any()):
            base_info["has_own_pump"] = True
        own_troops = own_chars[~own_bldg]
        if len(own_troops):
            eff_max = np.maximum(own_troops[:, S.E_MAX_HP], own_troops[:, S.E_HP])
            tanks = own_troops[(eff_max >= 3200) & (own_troops[:, S.E_HP] >= 900)]
            for t_row in tanks:
                base_info["own_tank_lanes"].add("left" if int(t_row[S.E_X]) < 9000 else "right")
            supports = own_troops[
                (eff_max >= 900)
                & (eff_max < 3200)
                & (own_troops[:, S.E_HP] >= 450)
                & (own_troops[:, S.E_Y] <= 16500)
            ]
            if len(supports):
                sup_max = np.maximum(supports[:, S.E_MAX_HP], supports[:, S.E_HP])
                for idx in np.argsort(-sup_max):
                    sx = int(supports[idx, S.E_X])
                    sy = int(supports[idx, S.E_Y])
                    s_left = sx < 9000
                    has_front_tank = False
                    for t_row in tanks:
                        tx, ty = int(t_row[S.E_X]), int(t_row[S.E_Y])
                        if (tx < 9000) == s_left and ty >= sy - 1500:
                            has_front_tank = True
                            break
                    if not has_front_tank:
                        base_info["own_push_support_lane"] = "left" if s_left else "right"
                        base_info["own_push_support_row"] = int(sy // 1000)
                        break

    enemy_chars = rows[alive & (rows[:, S.E_KIND] == S.KIND_CHARACTER) & (rows[:, S.E_SIDE] == 1)]
    enemy_proj = rows[(rows[:, S.E_KIND] == S.KIND_PROJECTILE) & (rows[:, S.E_SIDE] == 1)]

    if not len(enemy_chars):
        base_info["quiet_board"] = not bool(len(enemy_proj) and (enemy_proj[:, S.E_Y] <= 15000).any())
        return base_info

    ey = enemy_chars[:, S.E_Y]
    ex = enemy_chars[:, S.E_X]
    ehp = enemy_chars[:, S.E_HP]
    quiet = not bool((ey <= 18000).any() or (len(enemy_proj) and (enemy_proj[:, S.E_Y] <= 15000).any()))
    deep_mask = ey <= 15500
    deep_hp = int(ehp[deep_mask].sum()) if deep_mask.any() else 0
    left_hp = int(ehp[deep_mask & (ex < 9000)].sum()) if deep_mask.any() else 0
    right_hp = int(ehp[deep_mask & (ex >= 9000)].sum()) if deep_mask.any() else 0

    defend_target = None
    if deep_hp >= 1100 and (left_hp >= 1100 or right_hp >= 1100):
        lane_mask = deep_mask & ((ex < 9000) if left_hp >= right_hp else (ex >= 9000))
        if lane_mask.any():
            sub_x = ex[lane_mask]
            sub_y = ey[lane_mask]
            sub_w = np.maximum(ehp[lane_mask], 100).astype(np.float64)
            tc = int(np.clip(round(float((sub_x * sub_w).sum() / sub_w.sum()) / 1000.0 - 0.5), 0, S.COLS - 1))
            tr = int(np.clip(round(float((sub_y * sub_w).sum() / sub_w.sum()) / 1000.0 - 0.5), 0, 14))
            defend_target = (tc, tr)

    near_tower_mask = ey <= 11500
    tower_attack = bool(
        (near_tower_mask & (ehp >= 300)).any()
        or (int(ehp[near_tower_mask].sum()) >= 350 if near_tower_mask.any() else False)
    )
    tower_swarm_mask = (ey <= 10000) & (ehp <= 380)
    swarm_on_tower = bool(int(tower_swarm_mask.sum()) >= 2)
    tower_swarm_tile = None
    if swarm_on_tower:
        sc = int(np.clip(round(float(ex[tower_swarm_mask].mean()) / 1000.0 - 0.5), 0, S.COLS - 1))
        sr = int(np.clip(round(float(ey[tower_swarm_mask].mean()) / 1000.0 - 0.5), 0, 10))
        tower_swarm_tile = (sc, sr)

    swarm_mask = (ey <= 18500) & (ehp <= 450)
    swarm_emergency = bool(swarm_on_tower or int(swarm_mask.sum()) >= 4)
    swarm_lane = None
    if swarm_emergency and swarm_mask.any():
        sw_left = int((swarm_mask & (ex < 9000)).sum())
        sw_right = int((swarm_mask & (ex >= 9000)).sum())
        if sw_left >= 2 and sw_left >= sw_right:
            swarm_lane = "left"
        elif sw_right >= 2:
            swarm_lane = "right"

    base_info.update(
        quiet_board=quiet,
        deep_enemy_hp=deep_hp,
        left_threat_hp=left_hp,
        right_threat_hp=right_hp,
        defend_target=defend_target,
        tower_under_attack=tower_attack,
        swarm_on_tower=swarm_on_tower,
        tower_swarm_tile=tower_swarm_tile,
        swarm_emergency=swarm_emergency,
        swarm_lane=swarm_lane,
    )
    return base_info


def _best_cluster_tile(
    slot_probs: np.ndarray,
    slot_legal: np.ndarray,
    *,
    prefer_backline: bool = False,
    defend_lane: str | None = None,
    defend_target: tuple[int, int] | None = None,
    ranged_defender: bool = False,
    front_of_push: tuple[str, int] | None = None,
    attack_lane: str | None = None,
) -> int:
    """Pick the legal tile maximizing 3x3 cluster probability mass + tactical anchor tie-break."""
    restricted = slot_legal.copy()
    prox_bonus = np.zeros(S.TILES, dtype=np.float64)

    if front_of_push is not None:
        push_lane, sup_row = front_of_push
        grid_mask = restricted.reshape(S.ROWS, S.COLS).copy()
        min_r = min(14, max(6, int(sup_row) + 1))
        grid_mask[:min_r, :] = False
        grid_mask[16:, :] = False
        if push_lane == "left":
            grid_mask[:, 9:] = False
        else:
            grid_mask[:, :9] = False
        cand_mask = grid_mask.reshape(-1)
        if cand_mask.any():
            restricted = cand_mask
    elif defend_target is not None:
        tc, tr = defend_target
        rr, cc = np.indices((S.ROWS, S.COLS))
        dist = np.hypot(cc - tc, rr - tr)
        grid_mask = restricted.reshape(S.ROWS, S.COLS).copy()
        grid_mask[15:, :] = False
        if tc < 9:
            grid_mask[:, 9:] = False
        else:
            grid_mask[:, :9] = False
        if ranged_defender:
            # Keep ranged splashers 3..6.5 tiles away from MegaKnight/tanks so they don't get jumped/splashed on tower
            standoff = grid_mask & (dist >= 2.8) & (dist <= 6.8) & (rr <= min(13, tr + 1))
            if standoff.any():
                grid_mask = standoff
            prox_bonus = (-0.015 * np.abs(dist - 4.5)).reshape(-1)
        else:
            # Drop melee / MinionHorde directly onto / just in front of the threat (never 7 tiles away at col=10!)
            engage = grid_mask & (dist <= 3.6) & (rr <= min(14, tr + 2))
            if engage.any():
                grid_mask = engage
            prox_bonus = (-0.025 * dist).reshape(-1)
        cand_mask = grid_mask.reshape(-1)
        if cand_mask.any():
            restricted = cand_mask
    elif defend_lane in ("left", "right"):
        grid_mask = restricted.reshape(S.ROWS, S.COLS).copy()
        grid_mask[15:, :] = False
        if defend_lane == "left":
            grid_mask[:, 9:] = False
        else:
            grid_mask[:, :9] = False
        cand_mask = grid_mask.reshape(-1)
        if cand_mask.any():
            restricted = cand_mask
    else:
        grid_mask = restricted.reshape(S.ROWS, S.COLS).copy()
        changed = False
        if prefer_backline:
            grid_mask[9:, :] = False
            changed = True
        if attack_lane == "left":
            grid_mask[:, 9:] = False
            changed = True
        elif attack_lane == "right":
            grid_mask[:, :9] = False
            changed = True
        if changed:
            cand_mask = grid_mask.reshape(-1)
            if cand_mask.any():
                restricted = cand_mask

    valid = restricted & (slot_probs > 0)
    if not valid.any():
        valid = restricted
    if not valid.any():
        valid = slot_legal & (slot_probs > 0)
    if not valid.any():
        return int(np.argmax(np.where(slot_legal, slot_probs, -np.inf)))
    grid = slot_probs.reshape(S.ROWS, S.COLS)
    pad = np.pad(grid, 1, mode="constant")
    cluster = sum(pad[dr : dr + S.ROWS, dc : dc + S.COLS] for dr in range(3) for dc in range(3)).reshape(-1)
    scores = np.where(valid, cluster + 0.25 * slot_probs + prox_bonus, -np.inf)
    return int(np.argmax(scores))


def spell_has_target(action, card_name, obs):
    decoded = S.decode_action(action)
    if decoded["kind"] != "play":
        return True
    x, y = decoded["col"] + .5, decoded["row"] + .5
    if card_name == "Rage":
        return _rage_has_engagement(x, y, obs)
    radius = DAMAGE_SPELL_RADIUS.get(card_name)
    if radius is None:
        return True
    count = int(obs[S.H_ENTITY_COUNT])
    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    enemy = (rows[:, S.E_SIDE] == 1) & (rows[:, S.E_HP] > 0)
    in_range = ((rows[:, S.E_X]/1000-x)**2 + (rows[:, S.E_Y]/1000-y)**2) <= radius**2
    tower = rows[:, S.E_KIND] == S.KIND_TOWER
    body = rows[:, S.E_KIND] == S.KIND_CHARACTER
    if card_name=='Log':
        dx=np.abs(rows[:,S.E_X]/1000-x)
        dy=rows[:,S.E_Y]/1000-y
        in_range=(dx<=1.95)&(dy>=-.6)&(dy<=10.1+.6)
        enemy &= (rows[:,S.E_FLAGS]&S.F_AIR)==0
    return bool((enemy & in_range & (body | (tower & (rows[:, S.E_HP] <= 450)))).any())


def select_live_action(probabilities, legal_mask, obs, hand_cards, *, play_threshold=.45,
                       slot_threshold=.28, tile_temperature=0.0, rng=None,
                       ability_threshold=.28, selection_mode='confidence'):
    """Compare calibrated legal skills, WAIT and card mass, then choose a tile."""
    if selection_mode not in ('policy_sample', 'confidence', 'argmax'):
        raise ValueError('Unknown live selection mode')
    p = np.asarray(probabilities, dtype=np.float64).copy()
    legal = np.asarray(legal_mask, dtype=bool)
    if p.shape != (S.ACTIONS,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected finite nonnegative 2307-action probabilities")
    p[~legal] = 0
    if p.sum() <= 0:
        return 0, {"spell_vetoed": False, "selection_reason": "no_legal_mass"}
    p /= p.sum()
    if selection_mode == 'policy_sample':
        rng = np.random.default_rng() if rng is None else rng
        proposed = int(rng.choice(S.ACTIONS, p=p))
        decoded = S.decode_action(proposed)
        info = {'selection_mode': selection_mode, 'proposed_action': proposed,
                'proposed_action_probability': float(p[proposed]),
                'spell_vetoed': False, 'selection_reason': 'policy_wait'}
        if decoded['kind'] == 'play':
            if not spell_has_target(proposed, hand_cards[decoded['slot']], obs):
                info.update(spell_vetoed=True, selection_reason='spell_guard_veto',
                            vetoed_action=proposed)
                return 0, info
            info['selection_reason'] = 'play'
        elif decoded['kind'] == 'ability':
            info['selection_reason'] = 'ability'
        return proposed, info
    if selection_mode == 'argmax':
        wait_mass = float(p[0])
        tiles = p[1:S.FIRST_ABILITY].reshape(4, S.TILES)
        slots = tiles.sum(-1)
        skills = p[S.FIRST_ABILITY:]
        non_wait_max = max(float(slots.max()), float(skills.max()))
        elixir_raw = int(obs[S.H_ELIXIR])
        board = _analyze_board_pressure(obs)
        near_full = elixir_raw >= 97000 and (slots.sum() + skills.sum()) >= 0.05

        # 0) Immediate Goblin-Barrel / Tower-Swarm spell reflex:
        # Never sit in WAIT=0.97 eating 800 damage from a Goblin Barrel while holding a legal Rage/Zap/Log/Arrows!
        if board["swarm_on_tower"] and board["tower_swarm_tile"] is not None:
            sc, sr = board["tower_swarm_tile"]
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname not in TOWER_SWARM_CHEAP_SPELLS:
                    continue
                s_leg = legal[1 + s_idx * S.TILES : 1 + (s_idx + 1) * S.TILES]
                if not s_leg.any():
                    continue
                center_t = sr * S.COLS + sc
                if not s_leg[center_t]:
                    center_t = _best_cluster_tile(tiles[s_idx], s_leg, defend_target=(sc, sr))
                act_sw = S.play_tile(s_idx, center_t)
                if spell_has_target(act_sw, cname, obs):
                    return act_sw, {
                        'selection_mode': selection_mode,
                        'proposed_action': act_sw,
                        'proposed_action_probability': float(p[act_sw]),
                        'selected_slot_probability': float(slots[s_idx]),
                        'spell_vetoed': False,
                        'selection_reason': 'tower_swarm_spell_reflex',
                    }

        # 1) Quiet-board 6-cost banking guard
        if (
            board["quiet_board"]
            and not near_full
            and legal[0]
            and 40000 <= elixir_raw < 58500
            and non_wait_max < 0.72
        ):
            has_unaffordable_core = False
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname == "Elixir Collector" and (board["no_pump_window"] or board["has_own_pump"]):
                    continue
                if cname in HIGH_COST_BANK_CARDS and not legal[1 + s_idx * S.TILES : 1 + (s_idx + 1) * S.TILES].any():
                    has_unaffordable_core = True
                    break
            if has_unaffordable_core:
                return 0, {
                    'selection_mode': selection_mode,
                    'proposed_action': 0,
                    'proposed_action_probability': wait_mass,
                    'spell_vetoed': False,
                    'selection_reason': 'quiet_bank_wait',
                }

        # 2) Emergency tower-under-attack & push-front-tank unfreeze
        defender_mass = 0.0
        giant_legal_mass = 0.0
        for s_idx in range(min(4, len(hand_cards))):
            cname = hand_cards[s_idx]
            slot_leg = legal[1 + s_idx * S.TILES : 1 + (s_idx + 1) * S.TILES]
            if not slot_leg.any():
                continue
            if cname not in BUILDING_ONLY_CARDS and cname != "Rage" and slots[s_idx] > defender_mass:
                defender_mass = float(slots[s_idx])
            if cname == "Giant" and slots[s_idx] > giant_legal_mass:
                giant_legal_mass = float(slots[s_idx])

        unfreeze_elixir = (
            30000 if board["swarm_on_tower"] else (45000 if board["low_own_tower"] else 62000)
        )
        emergency_unfreeze = (
            board["tower_under_attack"]
            and elixir_raw >= unfreeze_elixir
            and defender_mass >= (0.04 if board["swarm_on_tower"] else 0.12)
        )
        push_tank_unfreeze = (
            board["own_push_support_lane"] is not None
            and board["own_push_support_row"] >= 8
            and not board["tower_under_attack"]
            and elixir_raw >= 55000
            and giant_legal_mass >= 0.05
        )

        # 2b) Suppress naked bridge-spam MinionHorde when no enemy tank on our side & no friendly tank on board:
        # If MinionHorde is the ONLY reason non_wait_max > wait_mass at < 9.5 elixir while holding Giant/ZapMachine/Wizard, wait to play the core!
        naked_horde_board = (board["deep_enemy_hp"] < 1100 and not board["own_tank_lanes"] and not board["tower_under_attack"])
        if naked_horde_board and not near_full and legal[0]:
            non_horde_max = float(skills.max()) if len(skills) else 0.0
            has_core_in_hand = False
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname in ("Giant", "ZapMachine", "Wizard"):
                    has_core_in_hand = True
                if cname != "MinionHorde" and cname != "Elixir Collector" and slots[s_idx] > non_horde_max:
                    non_horde_max = float(slots[s_idx])
            if has_core_in_hand and wait_mass >= non_horde_max and elixir_raw < 95000:
                return 0, {
                    'selection_mode': selection_mode,
                    'proposed_action': 0,
                    'proposed_action_probability': wait_mass,
                    'spell_vetoed': False,
                    'selection_reason': 'suppress_naked_horde_wait',
                }

        if (
            (not near_full)
            and (not emergency_unfreeze)
            and (not push_tank_unfreeze)
            and legal[0]
            and wait_mass >= non_wait_max
        ):
            return 0, {'selection_mode': selection_mode, 'proposed_action': 0,
                       'proposed_action_probability': wait_mass, 'spell_vetoed': False,
                       'selection_reason': 'policy_wait'}
        skill = int(skills.argmax())
        if skills[skill] >= float(slots.max()) and skills[skill] > 0 and legal[S.FIRST_ABILITY + skill]:
            ab_act = S.FIRST_ABILITY + skill
            return ab_act, {'selection_mode': selection_mode, 'proposed_action': ab_act,
                            'proposed_action_probability': float(p[ab_act]), 'spell_vetoed': False,
                            'selection_reason': 'ability'}

        # Rank slots with live tactical guards
        eff_slots = slots.copy()
        forbid_pump = bool(
            board["no_pump_window"]
            or board["has_own_pump"]
            or board["own_push_support_lane"] is not None
            or board["deep_enemy_hp"] >= 700
            or board["tower_under_attack"]
        )
        if forbid_pump:
            for s_idx in range(min(4, len(hand_cards))):
                if hand_cards[s_idx] in PASSIVE_INVEST_CARDS:
                    eff_slots[s_idx] *= 0.02

        if board["swarm_emergency"]:
            live_aoe_slots: list[int] = []
            allowed_aoe = AOE_TROOP_OR_DAMAGE_SPELL | ({"Rage"} if board["swarm_on_tower"] else set())
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname not in allowed_aoe or slots[s_idx] <= 0:
                    continue
                s_leg = legal[1 + s_idx * S.TILES : 1 + (s_idx + 1) * S.TILES]
                if not s_leg.any():
                    continue
                if cname in DAMAGE_SPELL_RADIUS or cname == "Rage":
                    cand_t = _best_cluster_tile(tiles[s_idx], s_leg)
                    if not spell_has_target(S.play_tile(s_idx, cand_t), cname, obs):
                        continue
                live_aoe_slots.append(s_idx)
            if live_aoe_slots:
                for s_idx in live_aoe_slots:
                    eff_slots[s_idx] += 0.22
                for s_idx in range(min(4, len(hand_cards))):
                    if hand_cards[s_idx] in SLOW_SINGLE_TARGET_ON_SWARM:
                        eff_slots[s_idx] *= 0.15
            elif not board["tower_under_attack"] and board["finisher_tower_lane"] is not None and giant_legal_mass > 0:
                for s_idx in range(min(4, len(hand_cards))):
                    if hand_cards[s_idx] == "MiniPekka":
                        eff_slots[s_idx] *= 0.25
                    elif hand_cards[s_idx] == "Giant" and eff_slots[s_idx] > 0:
                        eff_slots[s_idx] += 0.18

        if board["deep_enemy_hp"] >= 900 and defender_mass >= 0.05:
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname in PASSIVE_INVEST_CARDS:
                    eff_slots[s_idx] *= 0.02
                elif cname in BUILDING_ONLY_CARDS and board["tower_under_attack"]:
                    eff_slots[s_idx] *= 0.15

        if board["own_push_support_lane"] is not None and not board["tower_under_attack"]:
            for s_idx in range(min(4, len(hand_cards))):
                if hand_cards[s_idx] == "Giant" and eff_slots[s_idx] > 0:
                    eff_slots[s_idx] += 0.25

        # Strongly penalize naked MinionHorde whenever there is no enemy tank to shred on our side and no friendly tank
        if naked_horde_board:
            has_other_playable = any(
                s_idx < len(hand_cards)
                and hand_cards[s_idx] not in ("MinionHorde", "Elixir Collector", "Rage")
                and slots[s_idx] > 0
                and legal[1 + s_idx * S.TILES : 1 + (s_idx + 1) * S.TILES].any()
                for s_idx in range(4)
            )
            if has_other_playable:
                for s_idx in range(min(4, len(hand_cards))):
                    if hand_cards[s_idx] == "MinionHorde":
                        eff_slots[s_idx] *= 0.10

        if near_full and board["quiet_board"] and wait_mass >= non_wait_max:
            own_alive_princess = int(obs[S.H_TOWERS + 2] > 0) + int(obs[S.H_TOWERS + 4] > 0)
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname == "Elixir Collector" and own_alive_princess >= 1 and not forbid_pump and eff_slots[s_idx] > 0:
                    eff_slots[s_idx] += 0.25
                elif cname in ("ZapMachine", "Giant") and eff_slots[s_idx] > 0:
                    eff_slots[s_idx] += 0.18

        defend_lane = None
        if board["left_threat_hp"] >= 1100 and board["left_threat_hp"] >= board["right_threat_hp"]:
            defend_lane = "left"
        elif board["right_threat_hp"] >= 1100 and board["right_threat_hp"] > board["left_threat_hp"]:
            defend_lane = "right"
        elif board["swarm_emergency"] and board["swarm_lane"] in ("left", "right"):
            defend_lane = board["swarm_lane"]

        first_vetoed = None
        for slot in np.argsort(-eff_slots):
            slot = int(slot)
            if slots[slot] <= 0:
                break
            if (
                first_vetoed is not None
                and not near_full
                and not emergency_unfreeze
                and not push_tank_unfreeze
                and slots[slot] < max(wait_mass, 0.22)
            ):
                break
            slot_legal = legal[1 + slot * S.TILES : 1 + (slot + 1) * S.TILES]
            if not slot_legal.any():
                continue
            card_name = hand_cards[slot] if slot < len(hand_cards) else None
            is_spell = card_name in DAMAGE_SPELL_RADIUS or card_name == "Rage"
            use_front_push = (
                (board["own_push_support_lane"], board["own_push_support_row"])
                if (card_name == "Giant" and board["own_push_support_lane"] is not None and not board["tower_under_attack"])
                else None
            )
            use_def_target = (
                board["defend_target"]
                if (board["defend_target"] is not None and use_front_push is None and card_name not in BUILDING_ONLY_CARDS and not is_spell)
                else None
            )
            use_def_lane = (
                defend_lane
                if (
                    defend_lane
                    and use_front_push is None
                    and use_def_target is None
                    and card_name not in BUILDING_ONLY_CARDS
                    and not is_spell
                )
                else None
            )
            use_attack_lane = None
            if use_front_push is None and use_def_target is None and use_def_lane is None and not is_spell:
                if card_name == "Elixir Collector":
                    use_attack_lane = "left" if board["own_left_hp"] >= board["own_right_hp"] else "right"
                elif board["alive_attack_lane"] is not None and board["deep_enemy_hp"] < 900:
                    use_attack_lane = board["alive_attack_lane"]
            use_backline = bool(
                (card_name == "Elixir Collector")
                or (
                    near_full
                    and board["quiet_board"]
                    and use_front_push is None
                    and wait_mass >= non_wait_max
                    and not is_spell
                )
            )
            tile = _best_cluster_tile(
                tiles[slot],
                slot_legal,
                prefer_backline=use_backline,
                defend_lane=use_def_lane,
                defend_target=use_def_target,
                ranged_defender=(card_name in RANGED_SQUISHY_DEFENDERS),
                front_of_push=use_front_push,
                attack_lane=use_attack_lane,
            )
            proposed = S.play_tile(slot, tile)
            if not spell_has_target(proposed, card_name, obs):
                if first_vetoed is None:
                    first_vetoed = proposed
                continue
            reason = (
                'emergency_defense'
                if emergency_unfreeze
                else ('push_tank_synergy' if push_tank_unfreeze else ('leak_guard_play' if (near_full and wait_mass >= non_wait_max) else 'play'))
            )
            return proposed, {'selection_mode': selection_mode, 'proposed_action': proposed,
                              'proposed_action_probability': float(p[proposed]),
                              'selected_slot_probability': float(slots[slot]),
                              'spell_vetoed': first_vetoed is not None,
                              'defend_lane': defend_lane if (use_def_target is not None or use_def_lane is not None) else None,
                              'prefer_backline': use_backline,
                              'selection_reason': reason}
        if first_vetoed is not None:
            return 0, {'selection_mode': selection_mode, 'proposed_action': first_vetoed,
                       'proposed_action_probability': float(p[first_vetoed]),
                       'spell_vetoed': True, 'selection_reason': 'spell_guard_veto',
                       'vetoed_action': first_vetoed}
        return 0, {'selection_mode': selection_mode, 'proposed_action': 0,
                   'proposed_action_probability': wait_mass, 'spell_vetoed': False,
                   'selection_reason': 'policy_wait'}
    tiles = p[1:S.FIRST_ABILITY].reshape(4, S.TILES)
    slots = tiles.sum(-1)
    best = int(slots.argmax())
    near_full = obs[S.H_ELIXIR] >= 95000
    play_limit = .15 if near_full else play_threshold
    slot_limit = .10 if near_full else slot_threshold
    info = {"spell_vetoed": False, "selection_reason": "confidence_wait",
            "effective_play_threshold": play_limit, "effective_slot_threshold": slot_limit}
    skills=p[S.FIRST_ABILITY:]
    skill=int(skills.argmax())
    if (skills[skill]>=ability_threshold and skills[skill]>=max(float(p[0]),float(slots.max()))
            and legal[S.FIRST_ABILITY+skill]):
        info['selection_reason']='ability'
        return S.FIRST_ABILITY+skill,info
    if slots.sum() < play_limit or slots[best] < slot_limit:
        return 0, info
    if tile_temperature <= 1e-4:
        tile = int(tiles[best].argmax())
    else:
        rng = np.random.default_rng() if rng is None else rng
        valid = legal[1+best*S.TILES:1+(best+1)*S.TILES] & (tiles[best] > 0)
        logp = np.full(S.TILES, -np.inf)
        logp[valid] = np.log(tiles[best, valid])/tile_temperature
        weights = np.exp(logp-logp.max())
        tile = int(rng.choice(S.TILES, p=weights/weights.sum()))
    action = S.play_tile(best, tile)
    if not spell_has_target(action, hand_cards[best], obs):
        info.update(spell_vetoed=True, selection_reason="spell_guard_veto", vetoed_action=action)
        return 0, info
    info["selection_reason"] = "play"
    return action, info
