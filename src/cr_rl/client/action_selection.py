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
    """Allow Rage only on engaged friendly pushes or low-HP enemy swarms."""
    count = int(obs[S.H_ENTITY_COUNT])
    if count <= 0:
        return False
    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    alive = rows[:, S.E_HP] > 0
    chars = alive & (rows[:, S.E_KIND] == S.KIND_CHARACTER)
    towers = alive & (rows[:, S.E_KIND] == S.KIND_TOWER)
    enemy_c = rows[chars & (rows[:, S.E_SIDE] == 1)]
    if len(enemy_c):
        d2_ec = (enemy_c[:, S.E_X] / 1000.0 - x) ** 2 + (enemy_c[:, S.E_Y] / 1000.0 - y) ** 2
        if bool(((d2_ec <= 3.2 ** 2) & (enemy_c[:, S.E_HP] <= 400)).any()):
            return True
    own_troop = chars & (rows[:, S.E_SIDE] == 0) & ((rows[:, S.E_FLAGS] & S.F_BUILDING) == 0)
    own_c = rows[own_troop]
    if not len(own_c):
        return False
    d2_own = (own_c[:, S.E_X] / 1000.0 - x) ** 2 + (own_c[:, S.E_Y] / 1000.0 - y) ** 2
    buffed = own_c[d2_own <= 3.8 ** 2]
    if not len(buffed):
        return False
    enemy_t = rows[towers & (rows[:, S.E_SIDE] == 1)]
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
AOE_SWARM_ANSWERS = frozenset({
    "Wizard", "DarkPrince", "Valkyrie", "Executioner", "Bowler", "BabyDragon",
    "Witch", "MotherWitch", "Firecracker", "ElectroDragon", "Bomber", "Princess",
    "IceWizard", "Rage", "Arrows", "Zap", "Snowball", "Log", "BarbLog",
    "Fireball", "Poison", "Tornado", "RoyalDelivery",
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
    no_pump_window = bool(overtime or tick >= 2100 or behind_crowns)

    own_left_hp = int(obs[S.H_TOWERS + 2])
    own_right_hp = int(obs[S.H_TOWERS + 4])
    own_king_hp = int(obs[S.H_TOWERS])
    low_own_tower = any(0 < hp <= 2200 for hp in (own_left_hp, own_right_hp, own_king_hp))

    enemy_left_hp = int(obs[S.H_TOWERS + 8])
    enemy_right_hp = int(obs[S.H_TOWERS + 10])
    finisher_tower_lane = (
        "left" if 0 < enemy_left_hp <= 900 else ("right" if 0 < enemy_right_hp <= 900 else None)
    )

    base_info = {
        "quiet_board": True,
        "deep_enemy_hp": 0,
        "left_threat_hp": 0,
        "right_threat_hp": 0,
        "tower_under_attack": False,
        "swarm_on_tower": False,
        "swarm_emergency": False,
        "swarm_lane": None,
        "low_own_tower": low_own_tower,
        "no_pump_window": no_pump_window,
        "has_own_pump": False,
        "own_push_support_lane": None,
        "own_push_support_row": -1,
        "finisher_tower_lane": finisher_tower_lane,
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
    deep_mask = ey <= 14500
    deep_hp = int(ehp[deep_mask].sum()) if deep_mask.any() else 0
    left_hp = int(ehp[deep_mask & (ex < 8800)].sum()) if deep_mask.any() else 0
    right_hp = int(ehp[deep_mask & (ex > 9200)].sum()) if deep_mask.any() else 0

    near_tower_mask = ey <= 11500
    tower_attack = bool(
        (near_tower_mask & (ehp >= 300)).any()
        or (int(ehp[near_tower_mask].sum()) >= 350 if near_tower_mask.any() else False)
    )
    swarm_on_tower = bool(int((near_tower_mask & (ehp <= 450)).sum()) >= 2)
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
        tower_under_attack=tower_attack,
        swarm_on_tower=swarm_on_tower,
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
    front_of_push: tuple[str, int] | None = None,
) -> int:
    """Pick the legal tile maximizing 3x3 cluster probability mass + local peak tie-break."""
    restricted = slot_legal.copy()
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
        if (cand_mask & (slot_probs > 0)).any():
            restricted = cand_mask
    elif defend_lane in ("left", "right"):
        grid_mask = restricted.reshape(S.ROWS, S.COLS).copy()
        # Keep own-side defensive rows + center pull zone on the threatened lane
        grid_mask[15:, :] = False
        if defend_lane == "left":
            grid_mask[:, 11:] = False
        else:
            grid_mask[:, :7] = False
        cand_mask = grid_mask.reshape(-1)
        if (cand_mask & (slot_probs > 0)).any():
            restricted = cand_mask
    elif prefer_backline:
        grid_mask = restricted.reshape(S.ROWS, S.COLS).copy()
        grid_mask[9:, :] = False
        cand_mask = grid_mask.reshape(-1)
        if (cand_mask & (slot_probs > 0)).any():
            restricted = cand_mask

    valid = restricted & (slot_probs > 0)
    if not valid.any():
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
    x, y = decoded["col"] + .5, decoded["row"] + .5
    if card_name == "Rage":
        return _rage_has_engagement(x, y, obs)
    radius = DAMAGE_SPELL_RADIUS.get(card_name)
    if radius is None:
        return True
    # Action centers and observation entities already share the seat frame.
    count = int(obs[S.H_ENTITY_COUNT])
    rows = np.asarray(obs[S.HEADER_INTS:]).reshape(S.MAX_ENTITIES, S.ENTITY_INTS)[:count]
    enemy = (rows[:, S.E_SIDE] == 1) & (rows[:, S.E_HP] > 0)
    in_range = ((rows[:, S.E_X]/1000-x)**2 + (rows[:, S.E_Y]/1000-y)**2) <= radius**2
    tower = rows[:, S.E_KIND] == S.KIND_TOWER
    body = rows[:, S.E_KIND] == S.KIND_CHARACTER
    if card_name=='Log':
        # Pinned 16.635.3 projectiles.json / LogProjectileRolling:
        # ProjectileRadius=1950, ProjectileRadiusY=600, ProjectileRange=10100.
        # The seat frame always rolls toward increasing Y. A circular check
        # wrongly vetoed enemies directly ahead and accepted enemies behind.
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
        # Sampling the joint is exactly equivalent to sampling the card and
        # then its conditional tile. Do not apply confidence thresholds or
        # temperature: those silently replace the policy trained by PPO.
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
        # Hierarchical argmax over factorized policy P(card/wait/ability) -> 3x3 cluster peak tile,
        # with live adverse-defense & quiet-board banking/leak guards.
        wait_mass = float(p[0])
        tiles = p[1:S.FIRST_ABILITY].reshape(4, S.TILES)
        slots = tiles.sum(-1)
        skills = p[S.FIRST_ABILITY:]
        non_wait_max = max(float(slots.max()), float(skills.max()))
        elixir_raw = int(obs[S.H_ELIXIR])
        board = _analyze_board_pressure(obs)
        near_full = elixir_raw >= 97000 and (slots.sum() + skills.sum()) >= 0.05

        # 1) Quiet-board 6-cost banking guard: avoid impulse-dumping 4-cost cards at 4.0..5.85 elixir
        # when holding an unaffordable 6+ cost core (Sparky / Elixir Collector) with no enemy pressure.
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

        # 2) Emergency tower-under-attack & push-front-tank unfreeze:
        # - Never freeze in WAIT while an enemy unit or Goblin Barrel is hitting our tower.
        # - Do not hold Giant in WAIT until an un-tanked Sparky/Wizard crosses the river naked.
        defender_mass = 0.0
        giant_legal_mass = 0.0
        for s_idx in range(min(4, len(hand_cards))):
            cname = hand_cards[s_idx]
            slot_leg = legal[1 + s_idx * S.TILES : 1 + (s_idx + 1) * S.TILES]
            if not slot_leg.any():
                continue
            if cname not in BUILDING_ONLY_CARDS and slots[s_idx] > defender_mass:
                defender_mass = float(slots[s_idx])
            if cname == "Giant" and slots[s_idx] > giant_legal_mass:
                giant_legal_mass = float(slots[s_idx])

        unfreeze_elixir = (
            30000 if board["swarm_on_tower"] else (45000 if board["low_own_tower"] else 62000)
        )
        emergency_unfreeze = (
            board["tower_under_attack"]
            and elixir_raw >= unfreeze_elixir
            and defender_mass >= (0.08 if board["swarm_on_tower"] else 0.14)
        )
        push_tank_unfreeze = (
            board["own_push_support_lane"] is not None
            and board["own_push_support_row"] >= 8
            and not board["tower_under_attack"]
            and elixir_raw >= 55000
            and giant_legal_mass >= 0.05
        )

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

        # Rank slots with live tactical guards (anti-suicide-pump, anti-swarm, Giant tank-in-front).
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
                    eff_slots[s_idx] *= 0.05

        if board["swarm_emergency"]:
            live_aoe_slots: list[int] = []
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname not in AOE_SWARM_ANSWERS or slots[s_idx] <= 0:
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

        if board["deep_enemy_hp"] >= 900 and defender_mass >= 0.08:
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname in PASSIVE_INVEST_CARDS:
                    eff_slots[s_idx] *= 0.05
                elif cname in BUILDING_ONLY_CARDS and board["tower_under_attack"]:
                    eff_slots[s_idx] *= 0.20

        if board["own_push_support_lane"] is not None and not board["tower_under_attack"]:
            for s_idx in range(min(4, len(hand_cards))):
                if hand_cards[s_idx] == "Giant" and eff_slots[s_idx] > 0:
                    eff_slots[s_idx] += 0.25

        if board["quiet_board"] and board["own_push_support_lane"] is None:
            has_solid_opener = any(
                s_idx < len(hand_cards)
                and hand_cards[s_idx] in ("Giant", "ZapMachine", "DarkPrince", "Wizard")
                and slots[s_idx] >= 0.03
                and legal[1 + s_idx * S.TILES : 1 + (s_idx + 1) * S.TILES].any()
                for s_idx in range(4)
            )
            if has_solid_opener:
                for s_idx in range(min(4, len(hand_cards))):
                    if hand_cards[s_idx] == "MinionHorde":
                        eff_slots[s_idx] *= 0.25

        if near_full and board["quiet_board"] and wait_mass >= non_wait_max:
            own_alive_princess = int(obs[S.H_TOWERS + 2] > 0) + int(obs[S.H_TOWERS + 4] > 0)
            for s_idx in range(min(4, len(hand_cards))):
                cname = hand_cards[s_idx]
                if cname == "Elixir Collector" and own_alive_princess >= 1 and not forbid_pump and eff_slots[s_idx] > 0:
                    eff_slots[s_idx] += 0.25
                elif cname in ("ZapMachine", "Giant") and eff_slots[s_idx] > 0:
                    eff_slots[s_idx] += 0.15

        defend_lane = None
        if board["left_threat_hp"] >= 1400 and board["right_threat_hp"] < 400:
            defend_lane = "left"
        elif board["right_threat_hp"] >= 1400 and board["left_threat_hp"] < 400:
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
            use_front_push = (
                (board["own_push_support_lane"], board["own_push_support_row"])
                if (card_name == "Giant" and board["own_push_support_lane"] is not None and not board["tower_under_attack"])
                else None
            )
            use_def_lane = (
                defend_lane
                if (
                    defend_lane
                    and use_front_push is None
                    and card_name not in BUILDING_ONLY_CARDS
                    and card_name not in DAMAGE_SPELL_RADIUS
                    and card_name != "Rage"
                )
                else None
            )
            use_backline = bool(
                near_full
                and board["quiet_board"]
                and use_front_push is None
                and wait_mass >= non_wait_max
                and card_name not in DAMAGE_SPELL_RADIUS
                and card_name != "Rage"
            )
            tile = _best_cluster_tile(
                tiles[slot],
                slot_legal,
                prefer_backline=use_backline,
                defend_lane=use_def_lane,
                front_of_push=use_front_push,
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
                              'defend_lane': use_def_lane,
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
    # Only executable card mass counts toward the play threshold.
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
