import numpy as np

from cr_rl.client.action_selection import select_live_action, spell_has_target
from cr_rl.client.schema_v4 import SchemaV4 as S


def test_log_reaches_ground_target_ahead_beyond_old_circle():
    obs=np.zeros(S.SEAT_INTS,dtype=np.int32);obs[S.H_ENTITY_COUNT]=1
    row=obs[S.HEADER_INTS:S.HEADER_INTS+S.ENTITY_INTS]
    row[S.E_KIND]=S.KIND_CHARACTER;row[S.E_SIDE]=1;row[S.E_HP]=1153
    row[S.E_X]=13800;row[S.E_Y]=14271
    action=S.play(0,13,9)  # Actual recorded tick1691: 4.77 tiles ahead.
    assert spell_has_target(action,'Log',obs)
    row[S.E_FLAGS]=S.F_AIR
    assert not spell_has_target(action,'Log',obs)
    row[S.E_FLAGS]=0;row[S.E_Y]=8000
    assert not spell_has_target(action,'Log',obs)  # Behind the roll.
    row[S.E_Y]=14271;row[S.E_X]=10000
    assert not spell_has_target(action,'Log',obs)  # Outside the lane width.


def test_log_second_recorded_false_veto():
    obs=np.zeros(S.SEAT_INTS,dtype=np.int32);obs[S.H_ENTITY_COUNT]=1
    row=obs[S.HEADER_INTS:S.HEADER_INTS+S.ENTITY_INTS]
    row[S.E_KIND]=S.KIND_CHARACTER;row[S.E_SIDE]=1;row[S.E_HP]=1963
    row[S.E_X]=3476;row[S.E_Y]=14987
    assert spell_has_target(S.play(0,2,10),'Log',obs)


def test_rage_empty_board_veto_and_hierarchical_argmax_anti_hijack():
    # Match 1 #09 exact failure pattern: Wizard has 67.5% card mass spread across 20 defensive tiles,
    # Rage has 23.0% concentrated on 2 tiles (0.115 > 0.034), own board EMPTY vs Miner (1460 HP).
    obs = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs[S.H_ELIXIR] = 52000
    obs[S.H_ENTITY_COUNT] = 1
    miner = obs[S.HEADER_INTS : S.HEADER_INTS + S.ENTITY_INTS]
    miner[S.E_KIND] = S.KIND_CHARACTER
    miner[S.E_SIDE] = 1
    miner[S.E_HP] = 1460
    miner[S.E_X] = 4500
    miner[S.E_Y] = 6500

    p = np.zeros(S.ACTIONS, dtype=np.float64)
    legal = np.zeros(S.ACTIONS, dtype=bool)
    legal[0] = True
    p[0] = 0.095

    # Slot 0 = Rage concentrated on (4, 14) and (5, 14)
    r1, r2 = S.play(0, 4, 14), S.play(0, 5, 14)
    legal[r1] = legal[r2] = True
    p[r1] = 0.12
    p[r2] = 0.11
    assert not spell_has_target(r1, "Rage", obs)

    # Slot 3 = Wizard spread across a 3x3 cluster centered at (4, 5) + periphery (total = 0.675, max single tile = 0.06 < 0.12!)
    wiz_center = S.play(3, 4, 5)
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            a = S.play(3, 4 + dc, 5 + dr)
            legal[a] = True
            p[a] = 0.06 if (dc == 0 and dr == 0) else 0.055
    stray_edge = S.play(3, 17, 14)
    legal[stray_edge] = True
    p[stray_edge] = 0.075  # isolated single pixel higher than wiz_center's 0.06, but 3x3 cluster at (4,5) = 0.50!
    p /= p.sum()

    chosen, info = select_live_action(p, legal, obs, ("Rage", "Giant", "Elixir Collector", "Wizard"), selection_mode="argmax")
    assert chosen == wiz_center
    assert info["selection_reason"] == "play"

    # When an engaged friendly Giant is near the Miner, Rage IS allowed by spell_has_target:
    obs[S.H_ENTITY_COUNT] = 2
    giant = obs[S.HEADER_INTS + S.ENTITY_INTS : S.HEADER_INTS + 2 * S.ENTITY_INTS]
    giant[S.E_KIND] = S.KIND_CHARACTER
    giant[S.E_SIDE] = 0
    giant[S.E_HP] = 3500
    giant[S.E_X] = 4500
    giant[S.E_Y] = 12500
    assert spell_has_target(S.play(0, 4, 12), "Rage", obs)


def test_live_blunders_020919_goblin_barrel_pump_and_giant_push():
    # 1) Blunder A: 3x BarrelGoblin (hp=323 each) on left tower -> never drop 6-cost ZapMachine over Rage / DarkPrince!
    obs = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs[S.H_ELIXIR] = 98000
    obs[S.H_TOWERS + 2] = 1922
    obs[S.H_ENTITY_COUNT] = 3
    for i, (gx, gy) in enumerate([(3500, 6500), (4200, 7200), (3000, 7200)]):
        ent = obs[S.HEADER_INTS + i * S.ENTITY_INTS : S.HEADER_INTS + (i + 1) * S.ENTITY_INTS]
        ent[S.E_KIND] = S.KIND_CHARACTER
        ent[S.E_SIDE] = 1
        ent[S.E_HP] = 323
        ent[S.E_X] = gx
        ent[S.E_Y] = gy

    p = np.zeros(S.ACTIONS, dtype=np.float64)
    legal = np.zeros(S.ACTIONS, dtype=bool)
    legal[0] = True
    p[0] = 0.25
    # Slot 0 = ZapMachine (0.35), Slot 1 = DarkPrince (0.22), Slot 2 = Rage on tower (0.18)
    zm = S.play(0, 2, 3)
    dp = S.play(1, 3, 6)
    rg = S.play(2, 3, 6)
    legal[zm] = legal[dp] = legal[rg] = True
    p[zm], p[dp], p[rg] = 0.35, 0.22, 0.18
    p /= p.sum()
    assert spell_has_target(rg, "Rage", obs)
    chosen, _ = select_live_action(p, legal, obs, ("ZapMachine", "DarkPrince", "Rage", "Giant"), selection_mode="argmax")
    assert chosen in (dp, rg)

    # 2) Blunder B & C: t=2740 (2x elixir, down 0-1 crowns, lone ZapMachine walking up left lane at row 10)
    # -> NEVER plant Elixir Collector on leak_guard; place Giant in front of ZapMachine on left lane!
    obs2 = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs2[S.H_TICK] = 2740
    obs2[S.H_ELIXIR] = 98000
    obs2[S.H_CROWNS] = 0
    obs2[S.H_CROWNS_ENEMY] = 1
    obs2[S.H_TOWERS + 2] = 0
    obs2[S.H_TOWERS + 4] = 2800
    obs2[S.H_TOWERS + 8] = 390
    obs2[S.H_ENTITY_COUNT] = 1
    own_zm = obs2[S.HEADER_INTS : S.HEADER_INTS + S.ENTITY_INTS]
    own_zm[S.E_KIND] = S.KIND_CHARACTER
    own_zm[S.E_SIDE] = 0
    own_zm[S.E_HP] = 1920
    own_zm[S.E_MAX_HP] = 1920
    own_zm[S.E_X] = 3500
    own_zm[S.E_Y] = 10500

    p2 = np.zeros(S.ACTIONS, dtype=np.float64)
    leg2 = np.zeros(S.ACTIONS, dtype=bool)
    leg2[0] = True
    p2[0] = 0.55
    pump_act = S.play(0, 2, 2)
    horde_act = S.play(1, 3, 14)
    giant_back = S.play(2, 14, 3)
    giant_front = S.play(2, 3, 13)
    leg2[pump_act] = leg2[horde_act] = leg2[giant_back] = leg2[giant_front] = True
    p2[pump_act] = 0.08
    p2[horde_act] = 0.22
    p2[giant_back] = 0.09
    p2[giant_front] = 0.06
    p2 /= p2.sum()

    chosen2, _ = select_live_action(
        p2, leg2, obs2, ("Elixir Collector", "MinionHorde", "Giant", "MiniPekka"), selection_mode="argmax"
    )
    assert chosen2 == giant_front

