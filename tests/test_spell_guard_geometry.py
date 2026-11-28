import numpy as np

from cr_rl.client.action_selection import select_live_action, spell_has_target
from cr_rl.client.schema_v4 import SchemaV4 as S


def test_log_reaches_ground_target_ahead_beyond_old_circle():
    obs = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs[S.H_ENTITY_COUNT] = 1
    row = obs[S.HEADER_INTS : S.HEADER_INTS + S.ENTITY_INTS]
    row[S.E_KIND] = S.KIND_CHARACTER
    row[S.E_SIDE] = 1
    row[S.E_HP] = 1153
    row[S.E_X] = 13800
    row[S.E_Y] = 14271
    action = S.play(0, 13, 9)  # Actual recorded tick1691: 4.77 tiles ahead.
    assert spell_has_target(action, "Log", obs)
    row[S.E_FLAGS] = S.F_AIR
    assert not spell_has_target(action, "Log", obs)
    row[S.E_FLAGS] = 0
    row[S.E_Y] = 8000
    assert not spell_has_target(action, "Log", obs)  # Behind the roll.
    row[S.E_Y] = 14271
    row[S.E_X] = 10000
    assert not spell_has_target(action, "Log", obs)  # Outside the lane width.


def test_log_second_recorded_false_veto():
    obs = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs[S.H_ENTITY_COUNT] = 1
    row = obs[S.HEADER_INTS : S.HEADER_INTS + S.ENTITY_INTS]
    row[S.E_KIND] = S.KIND_CHARACTER
    row[S.E_SIDE] = 1
    row[S.E_HP] = 1963
    row[S.E_X] = 3476
    row[S.E_Y] = 14987
    assert spell_has_target(S.play(0, 2, 10), "Log", obs)


def test_rage_empty_board_veto_and_hierarchical_argmax_anti_hijack():
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

    r1, r2 = S.play(0, 4, 14), S.play(0, 5, 14)
    legal[r1] = legal[r2] = True
    p[r1] = 0.12
    p[r2] = 0.11
    assert not spell_has_target(r1, "Rage", obs)

    wiz_center = S.play(3, 4, 5)
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            a = S.play(3, 4 + dc, 5 + dr)
            legal[a] = True
            p[a] = 0.06 if (dc == 0 and dr == 0) else 0.055
    stray_edge = S.play(3, 17, 14)
    legal[stray_edge] = True
    p[stray_edge] = 0.075
    p /= p.sum()

    chosen, info = select_live_action(p, legal, obs, ("Rage", "Giant", "Elixir Collector", "Wizard"), selection_mode="argmax")
    assert chosen == wiz_center
    assert info["selection_reason"] == "play"

    obs[S.H_ENTITY_COUNT] = 2
    giant = obs[S.HEADER_INTS + S.ENTITY_INTS : S.HEADER_INTS + 2 * S.ENTITY_INTS]
    giant[S.E_KIND] = S.KIND_CHARACTER
    giant[S.E_SIDE] = 0
    giant[S.E_HP] = 3500
    giant[S.E_X] = 4500
    giant[S.E_Y] = 12500
    assert spell_has_target(S.play(0, 4, 12), "Rage", obs)


def test_rage_allows_tower_barrel_swarm_but_vetoes_naked_river_bats():
    # Tower Goblin Barrel (>=2 swarm units at y <= 11.5) IS a valid Rage target
    obs_barrel = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs_barrel[S.H_ENTITY_COUNT] = 3
    for i, (gx, gy) in enumerate([(14500, 5200), (15500, 6800), (13500, 6800)]):
        ent = obs_barrel[S.HEADER_INTS + i * S.ENTITY_INTS : S.HEADER_INTS + (i + 1) * S.ENTITY_INTS]
        ent[S.E_KIND] = S.KIND_CHARACTER
        ent[S.E_SIDE] = 1
        ent[S.E_HP] = 323
        ent[S.E_X] = gx
        ent[S.E_Y] = gy
    assert spell_has_target(S.play(0, 14, 6), "Rage", obs_barrel)

    # Naked Rage at river (3, 14) on Bats + GoldenKnight with empty own board MUST be vetoed
    obs_riv = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs_riv[S.H_ENTITY_COUNT] = 5
    for i in range(4):
        b = obs_riv[S.HEADER_INTS + i * S.ENTITY_INTS : S.HEADER_INTS + (i + 1) * S.ENTITY_INTS]
        b[S.E_KIND] = S.KIND_CHARACTER
        b[S.E_SIDE] = 1
        b[S.E_HP] = 130
        b[S.E_X] = 3500
        b[S.E_Y] = 15500
    gk = obs_riv[S.HEADER_INTS + 4 * S.ENTITY_INTS : S.HEADER_INTS + 5 * S.ENTITY_INTS]
    gk[S.E_KIND] = S.KIND_CHARACTER
    gk[S.E_SIDE] = 1
    gk[S.E_HP] = 2450
    gk[S.E_X] = 4000
    gk[S.E_Y] = 15800
    assert not spell_has_target(S.play(2, 3, 14), "Rage", obs_riv)


def test_rage_allows_engaged_low_elixir_combat_blocks_duplicate_area_and_hazard_tempo():
    from cr_rl.client.action_selection import HazardTempoState

    # 1) Engaged defense/push at 2.3 elixir (1500 HP DarkPrince fighting enemy at y=11.5) MUST be allowed (no low_elixir burden!)
    obs = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs[S.H_ELIXIR] = 23000
    obs[S.H_ENTITY_COUNT] = 3
    rows = obs[S.HEADER_INTS:].reshape(S.MAX_ENTITIES, S.ENTITY_INTS)
    rows[0, S.E_KIND], rows[0, S.E_SIDE], rows[0, S.E_X], rows[0, S.E_Y], rows[0, S.E_HP] = S.KIND_CHARACTER, 0, 3500, 11500, 1500
    rows[1, S.E_KIND], rows[1, S.E_SIDE], rows[1, S.E_X], rows[1, S.E_Y], rows[1, S.E_HP] = S.KIND_CHARACTER, 1, 3800, 13000, 1100
    # Even with an enemy Knight in the opposite lane (x=15000, y=14000, hp=1400), engaged Rage MUST NOT be blocked!
    rows[2, S.E_KIND], rows[2, S.E_SIDE], rows[2, S.E_X], rows[2, S.E_Y], rows[2, S.E_HP] = S.KIND_CHARACTER, 1, 15000, 14000, 1400
    assert spell_has_target(S.play(0, 3, 11), "Rage", obs)

    # 2) Duplicate active friendly Rage area within 3.0 tiles MUST be blocked
    obs[S.H_ENTITY_COUNT] = 4
    rows[3, S.E_KIND], rows[3, S.E_SIDE], rows[3, S.E_X], rows[3, S.E_Y], rows[3, S.E_HP] = S.KIND_AREA, 0, 3500, 12000, 1
    assert not spell_has_target(S.play(0, 3, 11), "Rage", obs)

    # 3) Hazard tempo mode bridges argmax paralysis: persistent 35% play intent fires on frame 3, while 10% noise never fires!
    obs_h = np.zeros(S.SEAT_INTS, dtype=np.int32)
    obs_h[S.H_ELIXIR] = 55000
    legal = np.zeros(S.ACTIONS, dtype=bool)
    legal[0] = True
    t_act = S.play(1, 3, 10)
    legal[t_act] = True
    hand = ("Rage", "Giant", "Elixir Collector", "Wizard")

    st = HazardTempoState()
    # Low-prob noise (10%) for 6 frames -> never fires
    p_noise = np.zeros(S.ACTIONS, dtype=np.float64)
    p_noise[0], p_noise[t_act] = 0.90, 0.10
    for _ in range(6):
        a, info = select_live_action(p_noise, legal, obs_h, hand, selection_mode="hazard", tempo_state=st)
        assert a == 0 and info["selection_reason"] == "hazard_wait"

    # Below 7.0 elixir (5.5 bars), moderate 35% intent MUST wait so we don't dump units instead of banking!
    p_mod = np.zeros(S.ACTIONS, dtype=np.float64)
    p_mod[0], p_mod[t_act] = 0.65, 0.35
    for _ in range(4):
        a_low, _ = select_live_action(p_mod, legal, obs_h, hand, selection_mode="hazard", tempo_state=st)
        assert a_low == 0

    # At >= 7.0 elixir (7.5 bars), persistent 35% intent accumulates and fires cleanly by frame 4!
    obs_h[S.H_ELIXIR] = 75000
    a1, i1 = select_live_action(p_mod, legal, obs_h, hand, selection_mode="hazard", tempo_state=st)
    a2, i2 = select_live_action(p_mod, legal, obs_h, hand, selection_mode="hazard", tempo_state=st)
    a3, i3 = select_live_action(p_mod, legal, obs_h, hand, selection_mode="hazard", tempo_state=st)
    a4, i4 = select_live_action(p_mod, legal, obs_h, hand, selection_mode="hazard", tempo_state=st)
    assert a1 == 0 and a2 == 0 and a3 == 0 and a4 == t_act
    assert i4["selection_reason"] == "hazard_play"
    assert st.accum_hazard == 0.0  # resets after firing

    # Strong intent (60%) -> fires immediately on frame 1!
    p_strong = np.zeros(S.ACTIONS, dtype=np.float64)
    p_strong[0], p_strong[t_act] = 0.40, 0.60
    a_imm, i_imm = select_live_action(p_strong, legal, obs_h, hand, selection_mode="hazard", tempo_state=st)
    assert a_imm == t_act and i_imm["selection_reason"] == "play"
