from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from cr_rl.training.test_time_search import (
    LookaheadConfig,
    LookaheadPolicyStream,
    decode_bank_candidate,
    decode_hold_full_candidate,
    encode_bank_candidate,
    encode_hold_full_candidate,
    run_batched_lookahead,
    select_macro_candidates,
    should_trigger_search,
)


@dataclass
class _DummySchema:
    cols: int = 18
    rows: int = 32
    hand_slots: int = 4
    scales: dict = None
    header: dict = None

    def __post_init__(self):
        self.scales = {"elixir_scale": 10000}
        self.header = {
            "elixir": (2,),
            "elixir_reserved": (3,),
            "pending_count": (53,),
            "hand_card": (7, 8, 9, 10),
            "crowns": (4,),
            "crowns_enemy": (5,),
            "own_left_hp": (36,),
            "own_right_hp": (37,),
            "own_king_hp": (38,),
            "enemy_left_hp": (39,),
            "enemy_right_hp": (40,),
            "enemy_king_hp": (41,),
        }


def test_select_macro_candidates_distinct_slots_and_wait():
    schema = _DummySchema()
    raw = np.zeros(224, dtype=np.int32)
    raw[2] = 65000  # 6.5 elixir (< 9.8, so WAIT should be included!)

    legal = np.zeros(2307, dtype=bool)
    legal[0] = True
    # Slot 0 has 3 adjacent high-scoring tiles (normally raw top-3 would waste all 3 on slot 0!)
    legal[10:15] = True
    # Slot 1 (577..1152) and Slot 2 (1153..1728) also have legal tiles
    legal[600:605] = True
    legal[1200:1205] = True

    logp = torch.full((2307,), -20.0)
    logp[10] = -0.5
    logp[11] = -0.6
    logp[12] = -0.7
    logp[602] = -1.2
    logp[1203] = -1.8
    logp[0] = -2.5

    cands = select_macro_candidates(logp, legal, raw, schema, max_candidates=4)
    assert len(cands) == 4
    assert cands[0] == 10  # top overall action in slot 0
    assert 0 in cands  # WAIT explicitly included for banking comparison!
    assert 602 in cands  # best tile of slot 1
    assert 1203 in cands  # best tile of slot 2

    cfg = LookaheadConfig(trigger_elixir=6.0)
    assert should_trigger_search(raw, legal, logp, schema, cfg) is True

    raw[2] = 20000  # 2.0 elixir -> fast-path skip
    assert should_trigger_search(raw, legal, logp, schema, cfg) is False


def test_select_macro_candidates_spatial_lane_and_depth_splits():
    schema = _DummySchema()
    raw = np.zeros(224, dtype=np.int32)
    raw[2] = 75000  # 7.5 elixir

    legal = np.zeros(2307, dtype=bool)
    legal[0] = True
    # Slot 0: backline left (row=28, col=3 -> rel=28*18+3=507, act=508), adjacent (act=509),
    # and opposite-lane / bridge (row=16, col=14 -> rel=16*18+14=302, act=303)
    legal[508] = True
    legal[509] = True
    legal[303] = True
    # Slots 1..3 each have 1 legal tile
    legal[577 + 100] = True
    legal[1153 + 100] = True
    legal[1729 + 100] = True

    logp = torch.full((2307,), -20.0)
    logp[508] = -0.3  # primary slot 0 backline left
    logp[509] = -0.4  # adjacent slot 0 tile (should NOT be picked!)
    logp[303] = -1.1  # spatially separated opposite-lane/bridge slot 0 tile (SHOULD be picked!)
    logp[677] = -1.5
    logp[1253] = -1.9
    logp[1829] = -2.2
    logp[0] = -2.5

    cands = select_macro_candidates(logp, legal, raw, schema, max_candidates=8)
    # Pass 1 picks 508, 0 (WAIT), 677, 1253, 1829 (5 actions covering all 4 hand cards + WAIT)
    # Pass 2 adds 303 (spatially separated slot 0 placement) and ignores adjacent 509!
    assert len(cands) == 6
    assert 508 in cands and 303 in cands
    assert 509 not in cands
    assert {0, 677, 1253, 1829}.issubset(set(cands))


def test_sample_legal_crn_never_picks_illegal_or_oob():
    mask = np.zeros((6, 2307), dtype=bool)
    mask[:, 0] = True
    mask[:, 500] = True
    mask[:, 1200] = True
    # Leave trailing actions 2305, 2306 illegal (-inf) and interior gaps illegal
    lp = torch.full((6, 2307), -float("inf"))
    lp[:, 0] = -1.2
    lp[:, 500] = -0.6
    lp[:, 1200] = -1.8

    u = np.array([0.0, 1e-9, 0.29, 0.31, 0.9999999, 1.0], dtype=np.float32)
    acts = LookaheadPolicyStream._sample_legal_crn(lp, mask, u)
    assert acts.shape == (6,)
    assert np.all((acts >= 0) & (acts < 2307))
    for i, a in enumerate(acts):
        assert mask[i, a], f"lane {i} sampled illegal action {a}"


def test_bank_for_slot_triggers_at_4_elixir_when_holding_6_cost_cards():
    schema = _DummySchema()
    schema.header["hand_cost"] = (11, 12, 13, 14)
    raw = np.zeros(224, dtype=np.int32)
    raw[2] = 41000  # 4.1 elixir (just like dec=123 in live Sparky match!)
    # Hand: [Elixir Collector(6), ZapMachine(6), DarkPrince(4), Wizard(5)]
    raw[7:11] = [10, 20, 30, 40]
    raw[11:15] = [6, 6, 4, 5]

    legal = np.zeros(2307, dtype=bool)
    legal[0] = True
    # Only slot 2 (DarkPrince, 1153..1728) is currently affordable!
    legal[1153 + 24] = True
    legal[1153 + 300] = True

    logp = torch.full((2307,), -20.0)
    logp[1153 + 24] = -0.1  # Actor strongly wants to dump DarkPrince at 4.1 elixir!
    logp[1153 + 300] = -1.5
    logp[0] = -2.4

    cfg = LookaheadConfig(trigger_elixir=5.0, bank_min_elixir=3.8, bank_min_card_cost=5.0, bank_max_deficit=2.2)
    # Even though 4.1 < trigger_elixir (5.0), banking trigger fires because 6-cost cards have deficit 1.9 <= 2.2!
    assert should_trigger_search(raw, legal, logp, schema, cfg) is True

    cands = select_macro_candidates(
        logp, legal, raw, schema, max_candidates=7, include_bank_macros=True
    )
    # Must include DarkPrince immediate play, 1-step WAIT (0), AND BANK_FOR_SLOT macros for slot 0, slot 1, slot 3!
    assert (1153 + 24) in cands
    assert 0 in cands
    assert encode_bank_candidate(0) in cands and decode_bank_candidate(encode_bank_candidate(0)) == 0
    assert encode_bank_candidate(1) in cands and decode_bank_candidate(encode_bank_candidate(1)) == 1
    assert encode_bank_candidate(3) in cands and decode_bank_candidate(encode_bank_candidate(3)) == 3


def test_front_tank_synergy_and_empty_rage_veto():
    @dataclass
    class _EntitySchema(_DummySchema):
        tile_units: int = 1000
        width_units: int = 18000
        height_units: int = 32000
        entity: dict = None
        kinds: dict = None
        flags: dict = None

        def __post_init__(self):
            super().__post_init__()
            self.header["hand_cost"] = (11, 12, 13, 14)
            self.entity = {"side": 0, "kind": 1, "x": 2, "y": 3, "hp": 4, "max_hp": 5, "flags": 6}
            self.kinds = {"character": 1, "tower": 2, "projectile": 3, "area": 4}
            self.flags = {"alive": 1, "building": 2}
            self._ents = np.zeros((0, 7), dtype=np.int32)

        def entities(self, _raw):
            return self._ents

    schema = _EntitySchema()
    raw = np.zeros(224, dtype=np.int32)
    raw[2] = 56000  # 5.6 elixir
    # Hand: [Rage(2, id=99), MinionHorde(5, id=22), MiniPekka(4, id=18), Giant(5, id=3)]
    raw[7:11] = [99, 22, 18, 3]
    raw[11:15] = [2, 5, 4, 5]

    # Own Sparky advancing at left bridge approach (x=3500, y=11500 -> col=3, row=11), zero enemies near own half
    schema._ents = np.array(
        [
            [0, 1, 3500, 11500, 2319, 2319, 1],  # Own Sparky approaching left bridge!
            [1, 2, 3500, 25500, 3687, 4858, 1],  # Enemy left princess tower
        ],
        dtype=np.int32,
    )

    legal = np.zeros(2307, dtype=bool)
    legal[0] = True
    legal[1:577] = True  # Slot 0 = Rage (all 576 tiles legal!)
    # Slots 1..3 = troops (rows 0..14 legal on own half = 15 * 18 = 270 tiles)
    for s in (1, 2, 3):
        legal[1 + s * 576 : 1 + s * 576 + 15 * 18] = True

    logp = torch.full((2307,), -15.0)
    # Flawed Actor wants to cast empty-combat Rage @(3, 5) in own backfield, and gives Giant its peak at backline @(6, 1)!
    rage_backfield = 1 + 0 * 576 + 5 * 18 + 3
    giant_backline = 1 + 3 * 576 + 1 * 18 + 6
    logp[rage_backfield] = -0.2
    logp[giant_backline] = -2.0
    # Giant bridge tiles near (col=3, row=14) have much lower Actor prior (-6.0), previously never searched!
    giant_bridge = 1 + 3 * 576 + 14 * 18 + 3
    logp[giant_bridge] = -5.5
    logp[0] = -2.5

    cfg = LookaheadConfig(trigger_elixir=6.0)
    # Even though 5.6 < 6.0, both unengaged Rage AND Sparky front-tank synergy trigger search!
    assert should_trigger_search(raw, legal, logp, schema, cfg, rage_card_ids=(99,)) is True

    cands = select_macro_candidates(logp, legal, raw, schema, max_candidates=6, rage_card_ids=(99,))
    # Must inject Giant at bridge in front of Sparky (`giant_bridge`), include WAIT (0), and NOT add another unengaged Rage!
    assert giant_bridge in cands
    assert 0 in cands


def test_mcts_barrel_low_elixir_trigger_river_rage_veto_and_tactical_candidates():
    @dataclass
    class _TacSchema(_DummySchema):
        tile_units: int = 1000
        width_units: int = 18000
        height_units: int = 32000
        entity: dict = None
        kinds: dict = None
        flags: dict = None

        def __post_init__(self):
            super().__post_init__()
            self.header["hand_cost"] = (11, 12, 13, 14)
            self.entity = {"side": 0, "kind": 1, "x": 2, "y": 3, "hp": 4, "max_hp": 5, "flags": 6}
            self.kinds = {"character": 1, "tower": 2, "projectile": 3, "area": 4}
            self.flags = {"alive": 1, "building": 2}
            self._ents = np.zeros((0, 7), dtype=np.int32)

        def entities(self, _raw):
            return self._ents

    schema = _TacSchema()
    raw = np.zeros(224, dtype=np.int32)
    raw[2] = 35000  # 3.5 elixir (< 4.5 old defense_elixir!)
    raw[7:11] = [99, 10, 3, 22]  # [Rage(2), Pump(6), Giant(5), MinionHorde(5)]
    raw[11:15] = [2, 6, 5, 5]
    raw[36:42] = [3600, 1645, 5800, 225, 0, 5800]  # Enemy R=0, Enemy L=225 HP!

    # 1) Goblin Barrel landed on right tower (3 goblins at (14.5, 6.2)), Actor is stuck at WAIT=0.97 and puts Rage argmax in empty corner (0,0)!
    schema._ents = np.array(
        [
            [1, 1, 14500, 5200, 323, 323, 1],
            [1, 1, 15500, 6800, 323, 323, 1],
            [1, 1, 13500, 6600, 323, 323, 1],
        ],
        dtype=np.int32,
    )
    legal = np.zeros(2307, dtype=bool)
    legal[0] = True
    legal[1:577] = True  # Rage legal
    logp = torch.full((2307,), -18.0)
    logp[0] = -0.03  # WAIT ~ 0.97 (very low entropy!)
    logp[1 + 0] = -4.0  # Rage raw argmax at empty corner (0, 0)

    cfg = LookaheadConfig(trigger_elixir=6.0, defense_elixir=4.5, entropy_threshold=0.50)
    assert should_trigger_search(raw, legal, logp, schema, cfg, rage_card_ids=(99,)) is True
    cands_barrel = select_macro_candidates(logp, legal, raw, schema, max_candidates=6, rage_card_ids=(99,))
    expected_rage_barrel = 1 + 6 * 18 + 14
    assert expected_rage_barrel in cands_barrel

    # 2) River Bats + GoldenKnight at y=15500 with empty own board: naked river Rage MUST be rejected by MCTS!
    schema._ents = np.array(
        [
            [1, 1, 3500, 15500, 130, 130, 1],
            [1, 1, 3800, 15500, 130, 130, 1],
            [1, 1, 4000, 15800, 2450, 2450, 1],
        ],
        dtype=np.int32,
    )
    river_rage = 1 + 14 * 18 + 3
    logp_riv = torch.full((2307,), -18.0)
    logp_riv[river_rage] = -0.1
    logp_riv[0] = -2.0
    cands_riv = select_macro_candidates(logp_riv, legal, raw, schema, max_candidates=6, rage_card_ids=(99,))
    assert cands_riv == [river_rage, 0]  # default kept at idx 0 for penalty, no extra naked river Rage added!

    # 3) MegaKnight at (3, 8) on Left Tower while MinionHorde raw heatmap peaks on Right bridge (15, 14):
    # MCTS MUST propose point-blank & standoff left-lane candidates for rollout verification!
    raw[2] = 52000
    schema._ents = np.array([[1, 1, 3500, 8500, 3001, 3300, 1]], dtype=np.int32)
    legal[1 + 3 * 576 : 1 + 3 * 576 + 15 * 18] = True  # MinionHorde legal on own half
    wrong_right_horde = 1 + 3 * 576 + 14 * 18 + 15
    logp_mk = torch.full((2307,), -16.0)
    logp_mk[wrong_right_horde] = -0.2
    logp_mk[0] = -2.5
    cands_mk = select_macro_candidates(logp_mk, legal, raw, schema, max_candidates=7, rage_card_ids=(99,))
    horde_cols = [((a - 1 - 3 * 576) % 18) for a in cands_mk if (1 + 3 * 576) <= a < (1 + 4 * 576)]
    assert any(c <= 6 for c in horde_cols), f"Expected left-tower defense candidate in {horde_cols}"

    # 4) Offensive / Counterpush Rage at 2.3 elixir!
    # Giant (2900 HP) + Sparky (2319 HP) crossed bridge at (3.5, 19.5) engaging enemy Knight + Skeletons,
    # Actor outputs WAIT=0.95 and puts Rage raw argmax in backfield (0, 0) -> MCTS MUST trigger at 2.3 elixir
    # and propose Rage right on the Giant+Sparky+Skeletons combat cluster!
    raw[2] = 23000  # Only 2.3 elixir left after deploying Giant + Sparky!
    schema._ents = np.array(
        [
            [0, 1, 3500, 19500, 2900, 4090, 1],  # Own Giant across bridge
            [0, 1, 3500, 17500, 2319, 2319, 1],  # Own Sparky behind Giant
            [1, 1, 3800, 21000, 1700, 1700, 1],  # Enemy Knight defending
            [1, 1, 3600, 20200, 81, 81, 1],      # Enemy Skeletons distracting Sparky/Giant
            [1, 1, 4100, 20400, 81, 81, 1],
        ],
        dtype=np.int32,
    )
    leg_push = np.zeros(2307, dtype=bool)
    leg_push[0] = True
    leg_push[1:577] = True  # Only 2-cost Rage is affordable at 2.3 elixir!
    logp_push = torch.full((2307,), -18.0)
    logp_push[0] = -0.05  # Actor stubbornly wants WAIT (~0.95) instead of offensive Rage!
    logp_push[1 + 0] = -3.5  # Raw Rage argmax at unengaged corner (0, 0)

    assert should_trigger_search(raw, leg_push, logp_push, schema, cfg, rage_card_ids=(99,)) is True
    cands_push = select_macro_candidates(logp_push, leg_push, raw, schema, max_candidates=5, rage_card_ids=(99,))
    rage_push_tiles = [((a - 1) % 18, (a - 1) // 18) for a in cands_push if 1 <= a <= 576 and a != 1]
    assert any(2 <= c <= 5 and 17 <= r <= 21 for c, r in rage_push_tiles), f"Expected push Rage tile, got {rage_push_tiles}"


def test_hold_to_full_pump_at_6_elixir_and_urgent_balloon_wizard_bank():
    @dataclass
    class _BalloonSchema(_DummySchema):
        tile_units: int = 1000
        width_units: int = 18000
        height_units: int = 32000
        entity: dict = None
        kinds: dict = None
        flags: dict = None

        def __post_init__(self):
            super().__post_init__()
            self.header["hand_cost"] = (11, 12, 13, 14)
            self.entity = {"side": 0, "kind": 1, "x": 2, "y": 3, "hp": 4, "max_hp": 5, "flags": 6}
            self.kinds = {"character": 1, "tower": 2, "projectile": 3, "area": 4}
            self.flags = {"alive": 1, "building": 2, "air": 4}
            self._ents = np.zeros((0, 7), dtype=np.int32)

        def entities(self, _raw):
            return self._ents

    schema = _BalloonSchema()
    raw = np.zeros(224, dtype=np.int32)
    raw[36:39] = [3600, 3600, 5800]  # healthy own towers

    # Part 1: Quiet board at 6.13 elixir with [Rage(2), ZapMachine(6), Elixir Collector(6), Wizard(5)] all legal!
    raw[2] = 61300
    raw[7:11] = [99, 20, 10, 40]
    raw[11:15] = [2, 6, 6, 5]
    legal = np.zeros(2307, dtype=bool)
    legal[0] = True
    legal[1:577] = True  # Rage
    for s in (1, 2, 3):
        legal[1 + s * 576 : 1 + s * 576 + 270] = True
    logp = torch.full((2307,), -15.0)
    logp[1 + 3 * 576 + 20] = -0.3  # Wizard immediate
    logp[1 + 1 * 576 + 18] = -1.1  # ZapMachine
    logp[1 + 2 * 576 + 10] = -1.4  # Pump
    logp[0] = -2.0

    cands_quiet = select_macro_candidates(
        logp, legal, raw, schema, max_candidates=7, include_bank_macros=True, rage_card_ids=(99,)
    )
    assert encode_hold_full_candidate(1) in cands_quiet and decode_hold_full_candidate(encode_hold_full_candidate(1)) == 1
    assert encode_hold_full_candidate(2) in cands_quiet and decode_hold_full_candidate(encode_hold_full_candidate(2)) == 2
    assert decode_bank_candidate(encode_hold_full_candidate(2)) is None

    # Part 2: Enemy Balloon crossing bridge (y=15000 <= 18000 -> emergency!) at 4.10 elixir!
    # Hand: [ZapMachine(6), Wizard(5), DarkPrince(4), Elixir Collector(6)]; only DarkPrince (slot 2) legal!
    raw[2] = 41000
    raw[7:11] = [20, 40, 30, 10]
    raw[11:15] = [6, 5, 4, 6]
    schema._ents = np.array([[1, 1, 3500, 15000, 2200, 2200, 5]], dtype=np.int32)  # Enemy Balloon at bridge!
    legal_def = np.zeros(2307, dtype=bool)
    legal_def[0] = True
    legal_def[1 + 2 * 576 : 1 + 2 * 576 + 270] = True  # Only DarkPrince legal
    logp_def = torch.full((2307,), -15.0)
    logp_def[1 + 2 * 576 + 200] = -0.2  # Actor wants to panic-play ground DarkPrince vs Balloon!
    logp_def[0] = -1.8

    cfg = LookaheadConfig(trigger_elixir=6.0, bank_min_elixir=3.8, bank_min_card_cost=5.0, bank_max_deficit=2.2)
    assert should_trigger_search(raw, legal_def, logp_def, schema, cfg) is True
    cands_def = select_macro_candidates(
        logp_def, legal_def, raw, schema, max_candidates=6, include_bank_macros=True
    )
    # Wizard (slot 1, deficit 0.9 <= 1.2) MUST be offered as BANK->Wizard, while 6-cost cards (deficit 1.9 > 1.2) are excluded!
    assert encode_bank_candidate(1) in cands_def
    assert encode_bank_candidate(0) not in cands_def
    assert encode_hold_full_candidate(2) not in cands_def


def test_batched_lookahead_matches_single_stream_across_mixed_seats_and_banking():
    """Verify `run_batched_lookahead` matches single-stream `act()` across seat=0 & seat=1."""
    from cr_rl.models import ModelConfig
    from cr_rl.models.state_policy import StateActorCritic, StateValueNetwork, stack_memories
    from cr_rl.models.state_temporal import StateTemporalConfig

    schema = _DummySchema()
    schema.header["hand_cost"] = (11, 12, 13, 14)
    header_fields = [f"h{i}" for i in range(64)]
    entity_fields = ["side", "kind", "x", "y", "hp", "max_hp", "flags"]

    class _DummyAdapter:
        def __init__(self, sc):
            self.schema = sc

        def __call__(self, raw_batch):
            arr = np.asarray(raw_batch, dtype=np.float32)
            b = arr.shape[0]
            h = np.tanh(arr[:, :64] / 10000.0).astype(np.float32)
            e = np.zeros((b, 4, 7), dtype=np.float32)
            e[:, 0, 4] = 1.0
            m = np.zeros((b, 4), dtype=bool)
            m[:, 0] = True
            return h, e, m

    class _DeterministicLane:
        def __init__(self):
            self.observations = np.zeros((1, 2, 224), dtype=np.int32)
            self.masks = np.zeros((1, 2, 2307), dtype=bool)
            self.actions = np.zeros((1, 2), dtype=np.int32)
            self.rewards = np.zeros((1, 2), dtype=np.float32)
            self.dones = np.zeros((1,), dtype=bool)
            self.infos = np.zeros((1, 4), dtype=np.int32)
            self._tick = 0
            self._seed = 0

        def reset(self, seeds, deck0, deck1, slots0=None, slots1=None):
            self._seed = int(seeds[0])
            self._tick = 0
            self.dones[0] = False
            self._populate()

        def _populate(self):
            for s in (0, 1):
                obs = self.observations[0, s]
                obs.fill(0)
                obs[2] = min(100000, 45000 + self._tick * 8000)
                obs[7:11] = [10 + s, 20 + s, 30 + s, 40 + s]
                obs[11:15] = [6, 5, 4, 6]
                obs[36:39] = [3600, 3600, 5800]
                obs[39:42] = [3600 - self._tick * 50, 3600, 5800]
                mk = self.masks[0, s]
                mk.fill(False)
                mk[0] = True
                # Slot 2 (4-cost) always legal; Slot 1 (5-cost) unlocks at >=5.0 elixir; Slot 0 (6-cost) at >=6.0
                mk[1 + 2 * 576 : 1 + 2 * 576 + 8] = True
                if obs[2] >= 50000:
                    mk[1 + 1 * 576 : 1 + 1 * 576 + 8] = True
                if obs[2] >= 60000:
                    mk[1 + 0 * 576 : 1 + 0 * 576 + 8] = True

        def step(self):
            a0, a1 = int(self.actions[0, 0]), int(self.actions[0, 1])
            self._tick += 1
            self._populate()
            # Reward deploying higher-slot / unlocked 5-6 cost cards symmetrically per seat!
            if a0 > 0:
                self.observations[0, 0, 39] -= 180 + (a0 % 17) * 20
                self.observations[0, 1, 36] = self.observations[0, 0, 39]
            if a1 > 0:
                self.observations[0, 1, 39] -= 180 + (a1 % 17) * 20
                self.observations[0, 0, 36] = self.observations[0, 1, 39]

        def info(self, _idx=0):
            return {"fault": False, "winner": -1}

    class _MockSearchEnv:
        def __init__(self, count: int):
            self.count = count
            self.lanes = [_DeterministicLane() for _ in range(count)]
            self.observations = np.zeros((count, 2, 224), dtype=np.int32)
            self.masks = np.zeros((count, 2, 2307), dtype=bool)
            self.actions = np.zeros((count, 2), dtype=np.int32)
            self.dones = np.zeros((count,), dtype=bool)
            self._executor = None
            self.vector = self

        def _sync(self, idx: int):
            self.observations[idx] = self.lanes[idx].observations[0]
            self.masks[idx] = self.lanes[idx].masks[0]
            self.dones[idx] = self.lanes[idx].dones[0]

        def step_lanes(self, indices):
            for idx in indices:
                i = int(idx)
                self.lanes[i].actions[0] = self.actions[i]
                self.lanes[i].step()
                self._sync(i)

        def info(self, idx: int):
            return self.lanes[int(idx)].info(0)

    torch.manual_seed(7)
    cfg_m = ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=16,
        temporal_heads=2,
        temporal_layers=2,
        header_dim=64,
        entity_feature_dim=7,
        vocab_size=64,
        reserve_to_wait=False,
    )
    t_cfg = StateTemporalConfig(
        width=16, heads=2, kv_heads=2, head_dim=8, pattern=("kda", "gqa"), ffn_multiplier=2, chunk_size=4, cache_initial_capacity=8
    )
    dev = torch.device("cpu")
    actor = StateActorCritic(cfg_m, t_cfg, frame_microbatch=16).to(dev).eval()
    critic = StateValueNetwork(cfg_m, t_cfg, decision_ticks=8, frame_microbatch=16).to(dev).eval()
    adapter = _DummyAdapter(schema)
    la_cfg = LookaheadConfig(
        max_candidates=5, rollouts_per_candidate=2, horizon_steps=4, trigger_elixir=4.0, override_margin=0.0
    )

    # Single-stream reference runs for match A (seat=0) and match B (seat=1)
    single_choices = []
    single_gains = []
    for m_idx, seat in enumerate((0, 1)):
        s_env = _MockSearchEnv(16)
        play_lane = _DeterministicLane()
        play_lane.reset([100 + m_idx], ["A"] * 8, ["B"] * 8, [0] * 8, [0] * 8)
        st = LookaheadPolicyStream(actor, adapter, dev, critic=critic, search_env=s_env, config=la_cfg, precision="fp32", seed=500 + m_idx)
        st.bind_match(seed=100 + m_idx, deck0=["A"] * 8, deck1=["B"] * 8, slots0=[0] * 8, slots1=[0] * 8, seat=seat)
        # Warm up 1 prefix step without search
        play_lane.actions[0] = [0, 0]
        st.observe_opponent(play_lane.observations[0, 1 - seat], play_lane.masks[0, 1 - seat])
        h0, e0, m0 = adapter(play_lane.observations[0, seat][None])
        _, _, st.actor_state = actor.step(
            torch.as_tensor(h0), torch.as_tensor(e0), torch.as_tensor(m0),
            torch.as_tensor(play_lane.masks[0, seat][None]), state=st.actor_state,
            hand_card_ids=torch.as_tensor(play_lane.observations[0, seat, st.hand_indices][None]),
        )
        _, st.critic_state = critic.step(torch.as_tensor(h0), torch.as_tensor(e0), torch.as_tensor(m0), state=st.critic_state)
        st.record_step_actions(0, 0)
        play_lane.step()

        act_out, _ = st.act(
            play_lane.observations[0, seat],
            play_lane.masks[0, seat],
            opp_raw=play_lane.observations[0, 1 - seat],
            opp_legal=play_lane.masks[0, 1 - seat],
        )
        assert st.last_step_info["searched"] is True
        single_choices.append(act_out)
        single_gains.append(st.last_step_info["score_gain"])

    # Now run BOTH matches simultaneously in ONE fused step + ONE `run_batched_lookahead` call!
    b_env = _MockSearchEnv(32)
    lanes_p = [_DeterministicLane(), _DeterministicLane()]
    streams_b = []
    for m_idx, seat in enumerate((0, 1)):
        lanes_p[m_idx].reset([100 + m_idx], ["A"] * 8, ["B"] * 8, [0] * 8, [0] * 8)
        st = LookaheadPolicyStream(actor, adapter, dev, critic=critic, search_env=b_env, config=la_cfg, precision="fp32", seed=500 + m_idx)
        st.bind_match(seed=100 + m_idx, deck0=["A"] * 8, deck1=["B"] * 8, slots0=[0] * 8, slots1=[0] * 8, seat=seat)
        streams_b.append(st)

    actor_mem = actor.initial_state(4, device=dev)
    crit_mem = critic.initial_state(2, device=dev)
    for step_i in range(2):
        own_r = np.stack([lanes_p[i].observations[0, i] for i in (0, 1)])
        own_m = np.stack([lanes_p[i].masks[0, i] for i in (0, 1)])
        opp_r = np.stack([lanes_p[i].observations[0, 1 - i] for i in (0, 1)])
        opp_m = np.stack([lanes_p[i].masks[0, 1 - i] for i in (0, 1)])
        both_r = np.concatenate([own_r, opp_r], axis=0)
        both_m = np.concatenate([own_m, opp_m], axis=0)
        bh, be, bm = adapter(both_r)
        both_lp, _, actor_mem = actor.step(
            torch.as_tensor(bh), torch.as_tensor(be), torch.as_tensor(bm),
            torch.as_tensor(both_m), state=actor_mem,
            hand_card_ids=torch.as_tensor(both_r[:, streams_b[0].hand_indices]),
        )
        _, crit_mem = critic.step(torch.as_tensor(bh[:2]), torch.as_tensor(be[:2]), torch.as_tensor(bm[:2]), state=crit_mem)
        if step_i == 0:
            for i in (0, 1):
                streams_b[i].record_step_actions(0, 0)
                lanes_p[i].actions[0] = [0, 0]
                lanes_p[i].step()
        else:
            tasks = []
            for i in (0, 1):
                base_a, cands, pend = streams_b[i].prepare_step_from_logp(own_r[i], own_m[i], both_lp[i])
                assert cands is not None
                tasks.append(
                    dict(
                        pos=i,
                        mem_slot=i,
                        stream=streams_b[i],
                        candidates=cands,
                        root_logp=both_lp[i],
                        root_raw=own_r[i],
                        root_legal=own_m[i],
                        baseline_action=base_a,
                        pending=pend,
                    )
                )
            outs = run_batched_lookahead(
                tasks,
                search_env=b_env,
                actor=actor,
                critic=critic,
                adapter=adapter,
                actor_mem=actor_mem,
                opp_mem_offset=2,
                critic_mem=crit_mem,
                cfg=la_cfg,
                device=dev,
                critic_device=dev,
                precision="fp32",
            )
            batched_choices = [
                tasks[i]["stream"].apply_search_result(own_r[i], outs[i][0], outs[i][1], tasks[i]["pending"], 0.01)
                for i in (0, 1)
            ]
            batched_gains = [streams_b[i].last_step_info["score_gain"] for i in (0, 1)]
            assert batched_choices == single_choices
            np.testing.assert_allclose(batched_gains, single_gains, rtol=1e-5, atol=1e-6)


def test_play_distill_batch_handles_staggered_dones_and_chunked_searches(tmp_path):
    """Verify `_play_distill_batch` shrinks `actor_mem`/`crit_mem` cleanly when games finish at different steps."""
    import sys
    import types

    # Stub crforge_env only if absent in current Python env so LookaheadDistiller imports cleanly
    restore_stub = False
    if "crforge_env" not in sys.modules:
        try:
            import crforge_env  # noqa: F401
        except ImportError:
            stub = types.ModuleType("crforge_env")
            stub.LadderConfig = object
            stub.VectorBattle = object
            sys.modules["crforge_env"] = stub
            restore_stub = True

    try:
        from cr_rl.models import ModelConfig
        from cr_rl.models.state_policy import StateActorCritic, StateValueNetwork
        from cr_rl.models.state_temporal import StateTemporalConfig
        from cr_rl.training.lookahead_distiller import LookaheadDistiller

        schema = _DummySchema()
        schema.header["hand_cost"] = (11, 12, 13, 14)

        def _field(obs, name):
            return int(obs[schema.header[name][0]])

        schema.field = _field

        class _DummyAdapter:
            def __init__(self, sc):
                self.schema = sc

            def __call__(self, raw_batch):
                arr = np.asarray(raw_batch, dtype=np.float32)
                b = arr.shape[0]
                h = np.tanh(arr[:, :64] / 10000.0).astype(np.float32)
                e = np.zeros((b, 4, 7), dtype=np.float32)
                e[:, 0, 4] = 1.0
                m = np.zeros((b, 4), dtype=bool)
                m[:, 0] = True
                return h, e, m

        class _StaggeredLane:
            def __init__(self, finish_at: int, winner: int):
                self.finish_at = finish_at
                self.winner = winner
                self.observations = np.zeros((1, 2, 224), dtype=np.int32)
                self.masks = np.zeros((1, 2, 2307), dtype=bool)
                self.actions = np.zeros((1, 2), dtype=np.int32)
                self.dones = np.zeros((1,), dtype=bool)
                self._steps = 0

            def reset(self, seeds, d0, d1, s0=None, s1=None):
                self._steps = 0
                self.dones[0] = False
                self._fill()

            def _fill(self):
                for s in (0, 1):
                    obs = self.observations[0, s]
                    obs.fill(0)
                    obs[2] = 65000
                    obs[7:11] = [0, 1, 2, 3]
                    obs[11:15] = [6, 5, 4, 2]
                    obs[4] = 2 if s == self.winner else 0
                    obs[5] = 0 if s == self.winner else 2
                    obs[36:39] = [3600, 3600, 5800]
                    obs[39:42] = [3000 - self._steps * 100, 3600, 5800]
                    mk = self.masks[0, s]
                    mk.fill(False)
                    mk[0] = True
                    mk[10:14] = True
                    mk[600:604] = True
                    mk[1200:1204] = True

            def step(self):
                self._steps += 1
                self._fill()
                if self._steps >= self.finish_at:
                    self.dones[0] = True

            def info(self, _i=0):
                return {"fault": False, "winner": self.winner if self.dones[0] else -1}

        class _MockVecEnv:
            def __init__(self, lanes_list):
                self.lanes = lanes_list
                self.count = len(lanes_list)
                self.schema = schema
                self.vocabulary = ["ZapMachine", "Wizard", "DarkPrince", "Rage", "Giant", "Elixir Collector", "MiniPekka", "MinionHorde"]
                self.observations = np.zeros((self.count, 2, 224), dtype=np.int32)
                self.masks = np.zeros((self.count, 2, 2307), dtype=bool)
                self.actions = np.zeros((self.count, 2), dtype=np.int32)
                self.dones = np.zeros((self.count,), dtype=bool)
                self._executor = None
                self.vector = self

            def _sync(self, idx: int):
                self.observations[idx] = self.lanes[idx].observations[0]
                self.masks[idx] = self.lanes[idx].masks[0]
                self.dones[idx] = self.lanes[idx].dones[0]

            def reset_env(self, idx, seed, d0, d1, s0, s1):
                self.lanes[idx].reset([seed], d0, d1, s0, s1)
                self._sync(idx)

            def step_lanes(self, indices):
                for idx in indices:
                    i = int(idx)
                    self.lanes[i].actions[0] = self.actions[i]
                    self.lanes[i].step()
                    self._sync(i)

            def info(self, idx: int):
                return self.lanes[int(idx)].info(0)

        torch.manual_seed(19)
        cfg_m = ModelConfig(
            entity_dim=16, entity_heads=2, temporal_dim=16, temporal_heads=2, temporal_layers=2,
            header_dim=64, entity_feature_dim=7, vocab_size=16, reserve_to_wait=False,
        )
        t_cfg = StateTemporalConfig(
            width=16, heads=2, kv_heads=2, head_dim=8, pattern=("kda", "gqa"), ffn_multiplier=2, chunk_size=4, cache_initial_capacity=8
        )
        dev = torch.device("cpu")
        actor = StateActorCritic(cfg_m, t_cfg, frame_microbatch=16).to(dev).eval()
        critic = StateValueNetwork(cfg_m, t_cfg, decision_ticks=8, frame_microbatch=16).to(dev).eval()
        adapter = _DummyAdapter(schema)

        # 3 concurrent games finishing at steps 2, 4, 6; search_env has capacity for only 2 concurrent searches (16 lanes)
        play_env = _MockVecEnv([_StaggeredLane(2, 0), _StaggeredLane(4, 1), _StaggeredLane(6, 0)])
        search_env = _MockVecEnv([_StaggeredLane(99, 0) for _ in range(16)])
        distiller = LookaheadDistiller(tmp_path / "ld_stagger", max_buffer_episodes=8, seed=42)
        la_cfg = LookaheadConfig(
            max_candidates=4, rollouts_per_candidate=2, horizon_steps=3, trigger_elixir=5.0, override_margin=0.0
        )
        decks = [["ZapMachine"] * 8, ["Wizard"] * 8]
        forms = [[0] * 8, [0] * 8]
        deck_names = ["sparky", "opp"]
        specs = [
            dict(seed=11, search_seed=21, opp_rng=np.random.default_rng(31), opp_temp=0.0, seat=0, own_deck_idx=0, opp_deck_idx=1),
            dict(seed=12, search_seed=22, opp_rng=np.random.default_rng(32), opp_temp=0.5, seat=1, own_deck_idx=0, opp_deck_idx=1),
            dict(seed=13, search_seed=23, opp_rng=np.random.default_rng(33), opp_temp=0.0, seat=0, own_deck_idx=0, opp_deck_idx=1),
        ]
        out = distiller._play_distill_batch(
            play_env, search_env, actor, critic, adapter,
            specs=specs, decks=decks, forms=forms, deck_names=deck_names,
            device=dev, precision="fp32", cfg=la_cfg, max_game_steps=10,
        )
        assert len(out) == 3
        assert [rep["steps"] for _ep, rep in out] == [2, 4, 6]
        assert all(rep["won"] and not rep["faulted"] and rep["searches"] >= 1 for _ep, rep in out)
    finally:
        if restore_stub:
            sys.modules.pop("crforge_env", None)


def test_hierarchical_default_naked_rage_skeleton_army_giant_bank_and_corner_pump():
    @dataclass
    class _FullTacSchema(_DummySchema):
        tile_units: int = 1000
        width_units: int = 18000
        height_units: int = 32000
        entity: dict = None
        kinds: dict = None
        flags: dict = None

        def __post_init__(self):
            super().__post_init__()
            self.header["hand_cost"] = (11, 12, 13, 14)
            self.header["overtime"] = (60,)
            self.entity = {"side": 0, "kind": 1, "x": 2, "y": 3, "hp": 4, "max_hp": 5, "flags": 6}
            self.kinds = {"character": 1, "tower": 2, "projectile": 3, "area": 4}
            self.flags = {"alive": 1, "building": 2}
            self._ents = np.zeros((0, 7), dtype=np.int32)

        def entities(self, _raw):
            return self._ents

    schema = _FullTacSchema()
    cfg = LookaheadConfig(trigger_elixir=6.0, defense_elixir=4.5, bank_min_elixir=3.8)

    # 1) Hierarchical default catches diffuse naked Rage (slot sum 0.72 > WAIT 0.28, single tile 0.09 < WAIT 0.28!)
    raw = np.zeros(224, dtype=np.int32)
    raw[2] = 28000  # 2.8 elixir
    raw[7:11] = [99, 10, 3, 20]  # [Rage(2), Pump(6), Giant(5), Sparky(6)]
    raw[11:15] = [2, 6, 5, 6]
    raw[36:42] = [3600, 3600, 5800, 3600, 3600, 5800]
    schema._ents = np.array([[1, 1, 3500, 20500, 970, 1200, 1]], dtype=np.int32)  # Medium enemy at y=20.5, 0 own units!
    legal = np.zeros(2307, dtype=bool)
    legal[0] = True
    legal[1:577] = True
    lp_diff = torch.full((2307,), -25.0)
    lp_diff[0] = float(np.log(0.28))
    for k_t in range(8):
        lp_diff[1 + 20 * 18 + 2 + k_t] = float(np.log(0.72 / 8.0))  # Each tile 0.09 < 0.28, but slot sum = 0.72!
    assert should_trigger_search(raw, legal, lp_diff, schema, cfg, rage_card_ids=(99,)) is True
    cands_diff = select_macro_candidates(lp_diff, legal, raw, schema, max_candidates=5, rage_card_ids=(99,))
    assert cands_diff[0] != 0 and 0 in cands_diff

    # 2) 12-Skeleton Army crossing river/bridge at y=13.5 triggers centered Rage without -4.0 nat penalty!
    schema._ents = np.array(
        [[1, 1, 14500 + (i % 3) * 300, 13000 + (i // 3) * 400, 130, 130, 1] for i in range(12)],
        dtype=np.int32,
    )
    lp_wait = torch.full((2307,), -20.0)
    lp_wait[0] = -0.05
    lp_wait[1] = -3.5
    assert should_trigger_search(raw, legal, lp_wait, schema, cfg, rage_card_ids=(99,)) is True
    cands_skarmy = select_macro_candidates(lp_wait, legal, raw, schema, max_candidates=5, rage_card_ids=(99,))
    skarmy_tiles = [((a - 1) % 18, (a - 1) // 18) for a in cands_skarmy if 1 <= a <= 576]
    assert any(13 <= c <= 16 and 12 <= r <= 15 for c, r in skarmy_tiles), f"Got {skarmy_tiles}"

    # 3) Proactive bridge-push Rage: Giant+Sparky at y=14.5 with enemy tower at y=25.5 (11 tiles away, 0 enemy units)
    raw[2] = 23000
    schema._ents = np.array(
        [
            [0, 1, 3500, 14800, 3200, 4090, 1],
            [0, 1, 3500, 13600, 2319, 2319, 1],
            [1, 2, 3500, 25500, 3600, 4858, 1],
        ],
        dtype=np.int32,
    )
    assert should_trigger_search(raw, legal, lp_wait, schema, cfg, rage_card_ids=(99,)) is True
    cands_bpush = select_macro_candidates(lp_wait, legal, raw, schema, max_candidates=5, rage_card_ids=(99,))
    bpush_tiles = [((a - 1) % 18, (a - 1) // 18) for a in cands_bpush if 1 <= a <= 576]
    assert any(2 <= c <= 5 and 14 <= r <= 17 for c, r in bpush_tiles), f"Got {bpush_tiles}"

    # 4) Sparky advancing at y=10.5 with 4.2 elixir and [Rage(2), Wizard(5), DarkPrince(4), Giant(5)] prioritizes BANK->Giant!
    raw[2] = 42000
    raw[7:11] = [99, 40, 27, 3]
    raw[11:15] = [2, 5, 4, 5]
    schema._ents = np.array(
        [
            [0, 1, 3500, 10500, 2319, 2319, 1],
            [1, 2, 3500, 25500, 3600, 4858, 1],
        ],
        dtype=np.int32,
    )
    leg_ft = np.zeros(2307, dtype=bool)
    leg_ft[0] = True
    leg_ft[1:577] = True
    leg_ft[1 + 2 * 576 : 1 + 2 * 576 + 270] = True  # DarkPrince (slot 2) legal, Giant (slot 3) 0.8 elx short!
    lp_ft = torch.full((2307,), -18.0)
    lp_ft[1 + 2 * 576 + 100] = -0.2
    lp_ft[0] = -2.0
    cands_ft = select_macro_candidates(
        lp_ft,
        leg_ft,
        raw,
        schema,
        max_candidates=6,
        include_bank_macros=True,
        rage_card_ids=(99,),
        giant_card_ids=(3,),
        non_tank_card_ids=(40,),
    )
    assert encode_bank_candidate(3) in cands_ft[:4]

    # 5) Calm non-overtime board at 7.0 elixir injects safe behind-tower corner Elixir Collector!
    raw[2] = 70000
    raw[7:11] = [99, 10, 27, 3]  # slot 1 = Collector (id=10)
    raw[11:15] = [2, 6, 4, 5]
    raw[36:39] = [3600, 2200, 5800]  # Left tower healthier -> prefers (1, 1) / (2, 1)
    schema._ents = np.array([[1, 2, 3500, 25500, 3600, 4858, 1]], dtype=np.int32)
    leg_col = np.zeros(2307, dtype=bool)
    leg_col[0] = True
    for s_i in range(4):
        leg_col[1 + s_i * 576 : 1 + s_i * 576 + 270] = True
    lp_col = torch.full((2307,), -18.0)
    lp_col[1 + 2 * 576 + 180] = -0.3  # Model raw argmax wants DarkPrince at bridge
    lp_col[1 + 1 * 576 + 12 * 18 + 8] = -1.5  # Model raw Collector peak is exposed mid-board (8, 12)!
    lp_col[1 + 1 * 576 + 1 * 18 + 1] = -4.0   # Safe corner (1, 1)
    cands_col = select_macro_candidates(
        lp_col, leg_col, raw, schema, max_candidates=7, include_bank_macros=True, rage_card_ids=(99,), collector_card_ids=(10,)
    )
    safe_left_corner = 1 + 1 * 576 + 1 * 18 + 1
    assert safe_left_corner in cands_col



