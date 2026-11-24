from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct

import numpy as np
import pytest

from cr_rl.client.nulls_reader import (
    AOE_DATA_LIFE_OFF,
    AOE_DATA_MAX_RADIUS_OFF,
    AOE_DATA_RADIUS_OFF,
    AOE_LIFE_OVR_OFF,
    AOE_LVL_OFF,
    AOE_TIME_LEFT_OFF,
    BUFF_ARR_OFF,
    BUFF_CNT_OFF,
    BUFF_DP_HIT_SPEED_MULT_OFF,
    BUFF_DP_INVISIBLE_OFF,
    BUFF_DP_SPEED_MULT_OFF,
    BUFF_INST_DP_OFF,
    BUFF_INST_TIME_OFF,
    CARD_DP_OFF,
    CARD_EVO_CNT_OFF,
    CARD_LVL_OFF,
    CARD_SLOT_FLAGS_OFF,
    CHAMP_CD_OFF,
    CHAMP_CHARGES_OFF,
    CHAMP_CMD_MS_OFF,
    CHAMP_DECK_IDX_OFF,
    CHAMP_DP_OFF,
    CHAMP_STATE_OFF,
    CHAMP_UNITS_VEC_OFF,
    CHAR_ABILITY_TICKS_OFF,
    CHAR_DELAY_MS_OFF,
    CHAR_DEPLOY_MS_OFF,
    CHAR_INVISIBLE_OFF,
    CHAR_STATE_OFF,
    CHAR_UNDERGROUND_OFF,
    CLK_FLAGS_OFF,
    CLK_PHASE_OFF,
    CLK_SEC_EL_OFF,
    CLK_SEC_LEN_OFF,
    CLK_SEC_TYPE_OFF,
    CLK_SECTION_OFF,
    CLK_TICK_OFF,
    CLK_TIME_MS_OFF,
    COMBAT_ATTACK_MS_OFF,
    COMBAT_LOAD_MS_OFF,
    COMBAT_TARGET_PTR_OFF,
    CTX_LB_OFF,
    CTX_V7_OFF,
    DECK_CARDS_ARR_OFF,
    DP_COST_OFF,
    DP_SCID_OFF,
    DP_STR_DATA_OFF,
    DP_STR_LEN_OFF,
    ENT_DP_OFF,
    ENT_ID_OFF,
    ENT_IS_CLONE_OFF,
    ENT_LVL_OFF,
    ENT_SIDE_OFF,
    ENT_SUB18_OFF,
    ENT_TAG1_OFF,
    ENT_VT_OFF,
    ENT_X_OFF,
    ENT_Y_OFF,
    ENT_Z_OFF,
    G_LOCAL_ACC,
    G_STATE,
    GM_CTX_OFF,
    HP_MAX_OFF,
    HP_MAX_SHIELD_OFF,
    HP_SHIELD_OFF,
    HP_VAL_OFF,
    LB_CLOCK_OFF,
    LB_DECKS_OFF,
    LB_END_WRAP_OFF,
    LB_PLAYER_OFFS,
    LB_SUBSTATE_OFF,
    LB_SUMMONERS_OFF,
    PQ_ARR0_OFF,
    PQ_CNT0_OFF,
    PQ_ITEM_CARD_DP_OFF,
    PQ_ITEM_COST_OFF,
    PQ_ITEM_DECK_IDX_OFF,
    PQ_ITEM_IS_ABILITY_OFF,
    PQ_ITEM_SEQ_OFF,
    PQ_ITEM_SIZE,
    PQ_ITEM_TIMEOUT_TICK_OFF,
    PROJ_AIM_X_OFF,
    PROJ_AIM_Y_OFF,
    PROJ_LVL_OFF,
    ST_GM_OFF,
    ST_MODE_OFF,
    SUMMONER_CHAMP_OFFS,
    SUMMONER_ELIXIR_OFF,
    SUMMONER_HAND_PTR_OFF,
    SUMMONER_HAND_REFILL_MS_OFF,
    SUMMONER_PENDING_PTR_OFF,
    SUMMONER_PLAYED_OFF,
    SUMMONER_QUEUE_PTR_OFF,
    SUMMONER_SEEN_EVO_NEED_OFF,
    SUMMONER_SEEN_EVO_OFF,
    SUMMONER_SEEN_OFF,
    TAG_CHARGING,
    TAG_FORCE_AIR,
    TAG_HIDDEN,
    TAG_INACTIVE,
    TAG_NO_ATTACK,
    TAG_NO_MOVE,
    V7_ARR_OFF,
    V7_CNT_OFF,
    VT_AREA_EFFECT,
    VT_BUFF_COMP,
    VT_CHAMPION_CTRL,
    VT_CHARACTER,
    VT_COMBAT_COMP,
    VT_HITPOINT_COMP,
    VT_KING_TOWER,
    VT_MOVE_COMP,
    VT_PROJECTILE,
    NotInBattleError,
    NullsBattleSnapshot,
    NullsBuff,
    NullsCard,
    NullsEntity,
    NullsMemoryReader,
    NullsSummoner,
    ReadBudgetExceededError,
)
from cr_rl.client.schema_adapter import NullsToSchemaV4Adapter
from cr_rl.client.schema_v4 import SchemaV4
from cr_rl.client.live_control import PendingPlay, pending_targets_for


def test_unknown_layouts_and_empty_entities_never_enter_model_or_mask():
    snap=_sample_snapshot()
    template=snap.entities[0]
    bad=(replace(template,eid=0,name='',scid=0),
         replace(template,eid=4000020,vt=0x1A4FFF8,name='TowerPrincessProjectile'),
         replace(template,eid=5999999,vt=0x1234567,name='KingTower'))
    reference=NullsToSchemaV4Adapter(TABLES_DIR).adapt(snap)
    adapter=NullsToSchemaV4Adapter(TABLES_DIR)
    result=adapter.adapt(replace(snap,entities=snap.entities+bad))
    np.testing.assert_array_equal(reference.obs,result.obs)
    np.testing.assert_array_equal(reference.legal_mask,result.legal_mask)
    assert [e['eid'] for e in result.rejected_entities]==[0,4000020,5999999]
    assert all(eid not in adapter._seat_ids for eid in (0,4000020,5999999))
    repeated=adapter.adapt(replace(snap,entities=snap.entities+bad))
    np.testing.assert_array_equal(result.obs,repeated.obs)
    assert result.rejected_entities==repeated.rejected_entities


@pytest.mark.parametrize('seat', [0, 1])
def test_controller_pending_point_reaches_model_and_replays(seat):
    from cr_rl.client.nulls_reader import NullsPendingCommand
    snap=_sample_snapshot(own_seat=seat)
    own=snap.summoners[seat]
    index=own.hand[0]
    card=own.deck[index]
    command=NullsPendingCommand(seq=10,deck_idx=index,card_name=card.name,
                               card_scid=card.scid,cost=card.cost,is_ability=False,
                               timeout_tick=snap.tick+12)
    summoners=list(snap.summoners)
    summoners[seat]=replace(own,pending_commands=(command,),elixir_reserved=card.cost*10000)
    snap=replace(snap,summoners=tuple(summoners))
    point=(SchemaV4.game_x(seat,3),SchemaV4.game_y(seat,8))
    pending=PendingPlay('match',own.played_count,own.hand,0,10.,1,point)
    targets=pending_targets_for(pending,own.hand)
    adapter=NullsToSchemaV4Adapter(TABLES_DIR)
    result=adapter.adapt(snap,pending_slots=(True,False,False,False),pending_targets=targets)
    at=SchemaV4.H_PENDING
    assert tuple(result.obs[at+1:at+3])==(3500,8500)
    context={'pending_slots':[True,False,False,False],
             'pending_targets_battle':[list(x) if x else None for x in targets]}
    fresh=NullsToSchemaV4Adapter(TABLES_DIR)
    restored=fresh.adapt(NullsBattleSnapshot.from_dict(snap.to_dict()),
                         pending_slots=tuple(context['pending_slots']),
                         pending_targets=context['pending_targets_battle'])
    np.testing.assert_array_equal(result.obs,restored.obs)
    np.testing.assert_array_equal(result.legal_mask,restored.legal_mask)
    unknown=fresh.adapt(snap,pending_targets=pending_targets_for(None,own.hand))
    assert tuple(unknown.obs[at+1:at+3])==(-1,-1)

TABLES_DIR = Path.home() / ".crforge/tables/16.635.3"
pytestmark = pytest.mark.skipif(not TABLES_DIR.exists(), reason="16.635.3 tables not found")


class FakeMemBackend:
    """In-memory sparse page backend modeling ARM64 Nulls structs."""

    def __init__(self, max_poll_bytes: int = 1024 * 1024) -> None:
        self.pages: dict[int, bytearray] = {}
        self.max_poll_bytes = max_poll_bytes
        self._poll_bytes = 0
        self._cached_pages: set[int] = set()

    @property
    def poll_bytes_read(self) -> int:
        return self._poll_bytes

    def begin_poll(self) -> None:
        self._poll_bytes = 0
        self._cached_pages.clear()

    def _charge_page(self, page: int) -> None:
        if page not in self._cached_pages:
            if self._poll_bytes + 4096 > self.max_poll_bytes:
                raise ReadBudgetExceededError("Fake read cap exceeded")
            self._poll_bytes += 4096
            self._cached_pages.add(page)

    def prefetch_pages(self, addrs: list[int]) -> None:
        for a in addrs:
            p = (a & 0x00FFFFFFFFFFFFFF) & ~0xFFF
            if p in self.pages:
                self._charge_page(p)

    def read(self, addr: int, size: int) -> bytes:
        clean = addr & 0x00FFFFFFFFFFFFFF
        out = bytearray()
        for off in range(size):
            cur = clean + off
            p = cur & ~0xFFF
            self._charge_page(p)
            buf = self.pages.get(p)
            out.append(buf[cur - p] if buf is not None else 0)
        return bytes(out)

    def write(self, addr: int, data: bytes) -> None:
        for idx, b in enumerate(data):
            cur = addr + idx
            p = cur & ~0xFFF
            if p not in self.pages:
                self.pages[p] = bytearray(4096)
            self.pages[p][cur - p] = b

    def w_u64(self, addr: int, val: int) -> None:
        self.write(addr, struct.pack("<Q", val))

    def w_i32(self, addr: int, val: int) -> None:
        self.write(addr, struct.pack("<i", val))


def _alloc_cursor(start: int = 0x20000000):
    cur = [start]

    def alloc(size: int) -> int:
        addr = cur[0]
        cur[0] += ((size + 63) // 64) * 64
        return addr

    return alloc


def _write_logic_data(mem: FakeMemBackend, alloc, name: str, scid: int, cost: int = 0) -> int:
    dp = alloc(0x160)
    raw = name.encode("utf-8")
    mem.w_i32(dp + DP_STR_LEN_OFF, len(raw))
    mem.w_i32(dp + DP_STR_LEN_OFF + 4, len(raw))
    if len(raw) <= 7:
        mem.write(dp + DP_STR_DATA_OFF, raw + b"\x00")
    else:
        sptr = alloc(len(raw) + 8)
        mem.write(sptr, raw + b"\x00")
        mem.w_u64(dp + DP_STR_DATA_OFF, sptr)
    mem.w_i32(dp + DP_SCID_OFF, scid)
    mem.w_i32(dp + DP_COST_OFF, cost)
    return dp


def _build_fake_memory(
    *, mode: int = 4, clk_flags: int = 65536, max_poll_bytes: int = 1024 * 1024
) -> tuple[FakeMemBackend, int]:
    base = 0x10000000
    mem = FakeMemBackend(max_poll_bytes=max_poll_bytes)
    alloc = _alloc_cursor(0x30000000)

    st = alloc(128)
    acc_obj = alloc(128)
    gm = alloc(256)
    ctx = alloc(256)
    v7 = alloc(64)
    lb = alloc(512)
    p0 = alloc(64)
    p1 = alloc(64)
    clk = alloc(96)
    b1b0 = alloc(64)
    obj18 = alloc(320)

    mem.w_u64(base + G_STATE, st)
    mem.w_u64(base + G_LOCAL_ACC, acc_obj)
    mem.write(acc_obj + 0x28, struct.pack("<ii", 0, 71526360))
    mem.w_i32(st + ST_MODE_OFF, mode)
    mem.w_u64(st + ST_GM_OFF, gm)
    mem.w_u64(gm + GM_CTX_OFF, ctx)
    mem.w_u64(ctx + CTX_V7_OFF, v7)
    mem.w_u64(ctx + CTX_LB_OFF, lb)

    mem.w_i32(lb + LB_SUBSTATE_OFF, 0)
    mem.w_u64(lb + LB_PLAYER_OFFS[0], p0)
    mem.w_u64(lb + LB_PLAYER_OFFS[1], p1)
    mem.write(p0, struct.pack("<ii", 0, 0))
    mem.write(p1, struct.pack("<ii", 0, 71526360))

    mem.w_u64(lb + LB_CLOCK_OFF, clk)
    mem.w_i32(clk + CLK_TICK_OFF, 240)
    mem.w_i32(clk + CLK_TIME_MS_OFF, 12000)
    mem.write(clk + CLK_FLAGS_OFF, struct.pack("<I", clk_flags))
    mem.w_i32(clk + CLK_SEC_EL_OFF, 12000)
    mem.w_i32(clk + CLK_SEC_LEN_OFF, 180000)
    mem.w_i32(clk + CLK_PHASE_OFF, 0)

    mem.w_u64(lb + LB_END_WRAP_OFF, b1b0)
    mem.w_u64(b1b0 + 0x18, obj18)

    d0_cards = [
        ("Knight", 26000000, 3, 10),
        ("Archer", 26000001, 3, 10),
        ("Goblins", 26000002, 2, 10),
        ("Giant", 26000003, 5, 8),
        ("Fireball", 28000000, 4, 8),
        ("Arrows", 28000001, 3, 10),
        ("Musketeer", 26000014, 4, 8),
        ("Wizard", 26000017, 5, 9),
    ]
    d1_cards = [
        ("Wizard", 26000017, 5, 9),
        ("DarkPrince", 26000027, 4, 7),
        ("ZapMachine", 26000033, 6, 3),
        ("Elixir Collector", 27000007, 6, 9),
        ("Fireball", 28000000, 4, 8),
        ("Giant", 26000003, 5, 8),
        ("Musketeer", 26000014, 4, 8),
        ("MiniPekka", 26000018, 4, 9),
    ]

    decks = []
    summs = []
    for seat, specs, elix, played in (
        (0, d0_cards, 50000, 2),
        (1, d1_cards, 70000, 1),
    ):
        d = alloc(64)
        arr = alloc(64)
        mem.w_u64(d + DECK_CARDS_ARR_OFF, arr)
        for idx, (nm, scid, cost, lvl_raw) in enumerate(specs):
            cp = alloc(64)
            cdp = _write_logic_data(mem, alloc, nm, scid, cost)
            mem.w_u64(cp + CARD_DP_OFF, cdp)
            mem.w_i32(cp + CARD_LVL_OFF, lvl_raw)
            mem.w_u64(arr + idx * 8, cp)

        s = alloc(960)
        hand_p = alloc(32)
        queue_p = alloc(32)
        mem.write(hand_p, struct.pack("<4i", 0, 1, 2, 3))
        mem.write(queue_p, struct.pack("<8i", 4, 5, 6, 7, 0, 0, 0, 0))
        mem.w_u64(s + SUMMONER_HAND_PTR_OFF, hand_p)
        mem.w_u64(s + SUMMONER_QUEUE_PTR_OFF, queue_p)
        mem.write(s + SUMMONER_SEEN_OFF, struct.pack("<8i", 26000000, -1, -1, -1, -1, -1, -1, -1))
        mem.w_i32(s + SUMMONER_ELIXIR_OFF, elix)
        mem.w_i32(s + SUMMONER_PLAYED_OFF, played)
        decks.append(d)
        summs.append(s)

    mem.w_u64(lb + LB_DECKS_OFF, decks[0])
    mem.w_u64(lb + LB_DECKS_OFF + 8, decks[1])
    mem.w_u64(lb + LB_SUMMONERS_OFF, summs[0])
    mem.w_u64(lb + LB_SUMMONERS_OFF + 8, summs[1])

    # Build 2 towers + 1 buffed Knight targeting enemy tower (with Combat/HP/Buff)
    king_dp = _write_logic_data(mem, alloc, "KingTower", 35000000)
    ptower_dp = _write_logic_data(mem, alloc, "PrincessTower", 35000001)
    knight_dp = _write_logic_data(mem, alloc, "Knight", 34000000)
    rage_dp = _write_logic_data(mem, alloc, "Rage", 0)

    def write_ent(eid, vt, dp, side, x, y, hp, max_hp, shield=0, target_ep=0, buff_dp=0):
        ep = alloc(0x200)
        sub18 = alloc(0x40)
        c_combat = alloc(0x40)
        c_move = alloc(0x40)
        c_hp = alloc(0x40)
        c_buff = alloc(0x40)

        mem.w_u64(ep + ENT_VT_OFF, base + vt)
        mem.w_i32(ep + ENT_ID_OFF, eid)
        mem.w_u64(ep + ENT_SUB18_OFF, sub18)
        mem.w_u64(ep + ENT_DP_OFF, dp)
        mem.w_i32(ep + ENT_SIDE_OFF, side)
        mem.w_i32(ep + ENT_X_OFF, x)
        mem.w_i32(ep + ENT_Y_OFF, y)
        mem.w_i32(ep + ENT_Z_OFF, 0)
        mem.w_i32(ep + ENT_LVL_OFF, 10)

        mem.w_u64(sub18 + 0x00, c_combat)
        mem.w_u64(sub18 + 0x08, c_move)
        mem.w_u64(sub18 + 0x10, c_hp)
        mem.w_u64(sub18 + 0x18, c_buff)

        mem.w_u64(c_combat, base + VT_COMBAT_COMP)
        if target_ep:
            mem.w_u64(c_combat + COMBAT_TARGET_PTR_OFF, target_ep)

        mem.w_u64(c_move, base + VT_MOVE_COMP)

        mem.w_u64(c_hp, base + VT_HITPOINT_COMP)
        mem.w_i32(c_hp + HP_VAL_OFF, hp)
        mem.w_i32(c_hp + HP_MAX_OFF, max_hp)
        mem.w_i32(c_hp + HP_SHIELD_OFF, shield)
        mem.w_i32(c_hp + HP_MAX_SHIELD_OFF, shield)

        mem.w_u64(c_buff, base + VT_BUFF_COMP)
        if buff_dp:
            barr = alloc(16)
            binst = alloc(64)
            mem.w_u64(binst + BUFF_INST_DP_OFF, buff_dp)
            mem.w_i32(binst + BUFF_INST_TIME_OFF, 2400)
            mem.w_u64(barr, binst)
            mem.w_u64(c_buff + BUFF_ARR_OFF, barr)
            mem.w_i32(c_buff + BUFF_CNT_OFF, 1)
        else:
            mem.w_i32(c_buff + BUFF_CNT_OFF, 0)
        return ep

    e0 = write_ent(5000000, VT_KING_TOWER, king_dp, 0, 9000, 3000, 4824, 4824)
    e1 = write_ent(5000003, VT_KING_TOWER, king_dp, 1, 9000, 29000, 4824, 4824)
    e2 = write_ent(5000020, VT_CHARACTER, knight_dp, 1, 14500, 15000, 1200, 1452, target_ep=e0, buff_dp=rage_dp)
    eptrs = [e0, e1, e2]

    arr_v7 = alloc(64)
    for i, ep in enumerate(eptrs):
        mem.w_u64(arr_v7 + i * 8, ep)
    mem.w_u64(v7 + V7_ARR_OFF, arr_v7)
    mem.w_i32(v7 + V7_CNT_OFF, len(eptrs))
    return mem, base


def _sample_snapshot(own_seat: int = 1) -> NullsBattleSnapshot:
    d0 = (
        NullsCard("Knight", 26000000, 3, 10),
        NullsCard("Archer", 26000001, 3, 10),
        NullsCard("Goblins", 26000002, 2, 10),
        NullsCard("Giant", 26000003, 5, 8),
        NullsCard("Fireball", 28000000, 4, 8),
        NullsCard("Arrows", 28000001, 3, 10),
        NullsCard("Musketeer", 26000014, 4, 8),
        NullsCard("Mirror", 28000006, 0, 5),
    )
    s0 = NullsSummoner(
        seat=0,
        acc_id=(0, 0),
        elixir=45000,
        played_count=1,
        hand=(0, 1, 2, 3),
        queue=(4, 5, 6, 7),
        seen_scids=(26000017, -1, -1, -1, -1, -1, -1, -1),
        deck=d0,
    )
    d1 = (
        NullsCard("Wizard", 26000017, 5, 9),
        NullsCard("DarkPrince", 26000027, 4, 7),
        NullsCard("ZapMachine", 26000033, 6, 3),
        NullsCard("Elixir Collector", 27000007, 6, 9),
        NullsCard("Fireball", 28000000, 4, 8),
        NullsCard("Giant", 26000003, 5, 8),
        NullsCard("Mirror", 28000006, 0, 5),
        NullsCard("MiniPekka", 26000018, 4, 9),
    )
    s1 = NullsSummoner(
        seat=1,
        acc_id=(0, 71526360),
        elixir=62000,
        played_count=1,
        hand=(0, 6, 2, 3),
        queue=(4, 5, 7, 1),
        seen_scids=(26000000, -1, -1, -1, -1, -1, -1, -1),
        deck=d1,
    )
    entities = (
        NullsEntity(5000000, VT_KING_TOWER, "KingTower", 35000000, 0, 9000, 3000, 0, 10, 4824, 4824, 0, 0, None, ()),
        NullsEntity(5000001, VT_CHARACTER, "PrincessTower", 35000001, 0, 3500, 6500, 0, 10, 3052, 3052, 0, 0, None, ()),
        NullsEntity(5000002, VT_CHARACTER, "PrincessTower", 35000001, 0, 14500, 6500, 0, 10, 3052, 3052, 0, 0, None, ()),
        NullsEntity(5000003, VT_KING_TOWER, "KingTower", 35000000, 1, 9000, 29000, 0, 15, 6247, 6247, 0, 0, None, ()),
        NullsEntity(5000004, VT_CHARACTER, "PrincessTower", 35000001, 1, 3500, 25500, 0, 15, 4424, 4424, 0, 0, None, ()),
        NullsEntity(5000005, VT_CHARACTER, "PrincessTower", 35000001, 1, 14500, 25500, 0, 15, 4424, 4424, 0, 0, None, ()),
        NullsEntity(
            5000020,
            VT_CHARACTER,
            "Giant",
            34000003,
            1,
            3500,
            21000,
            0,
            10,
            3500,
            3500,
            0,
            0,
            5000001,
            (NullsBuff("Rage", 0, 1500),),
        ),
        NullsEntity(5000021, VT_PROJECTILE, "Fireball", 0, 0, 4000, 18000, 100, None, None, None, None, None, None, None),
    )
    return NullsBattleSnapshot(
        mode=4,
        battle_substate=0,
        tick=100,
        time_ms=5000,
        sec_elapsed_ms=5000,
        sec_len_ms=180000,
        phase=0,
        clock_running=True,
        clock_ended=False,
        own_seat=own_seat,
        local_acc=(0, 71526360),
        summoners=(s0, s1),
        entities=entities,
        bytes_read=120000,
        monotonic_ts=12.5,
    )


def test_reader_decodes_fake_memory_and_gates_nonbattle():
    mem_lobby, base_l = _build_fake_memory(mode=3)
    with pytest.raises(NotInBattleError):
        NullsMemoryReader(mem_lobby, base_l).read_snapshot()

    mem_ended, base_e = _build_fake_memory(mode=4, clk_flags=256)
    with pytest.raises(NotInBattleError):
        NullsMemoryReader(mem_ended, base_e).read_snapshot()

    mem_ok, base_ok = _build_fake_memory(mode=4, clk_flags=65536)
    snap = NullsMemoryReader(mem_ok, base_ok).read_snapshot()
    assert snap.own_seat == 1
    assert snap.tick == 240
    assert snap.time_ms == 12000
    assert snap.bytes_read < 1024 * 1024
    assert snap.summoners[1].elixir == 70000
    assert snap.summoners[1].deck[0].name == "Knight" or snap.summoners[1].deck[3].name == "Elixir Collector"
    # Verify both inline (Knight, 6 chars) and heap ptr (Elixir Collector, 16 chars)
    assert snap.summoners[0].deck[0].name == "Knight"
    assert snap.summoners[1].deck[3].name == "Elixir Collector"
    knight = [e for e in snap.entities if e.eid == 5000020][0]
    assert knight.hp == 1200
    assert knight.max_hp == 1452
    assert knight.target_eid == 5000000
    assert len(knight.buffs or ()) == 1
    assert knight.buffs[0].name == "Rage"
    assert knight.buffs[0].time_left == 2400

    # Budget ceiling enforcement
    mem_tiny, base_t = _build_fake_memory(mode=4, max_poll_bytes=16384)
    with pytest.raises(ReadBudgetExceededError):
        NullsMemoryReader(mem_tiny, base_t).read_snapshot()


def test_projectile_does_not_follow_character_component_layout():
    mem,base=_build_fake_memory()
    signature=struct.pack('<Q',base+VT_CHARACTER)
    entity=next(page+offset for page,data in mem.pages.items()
                for offset in [bytes(data).find(signature)] if offset>=0)
    forbidden=0x24000000000
    mem.w_u64(entity+ENT_VT_OFF,base+VT_PROJECTILE)
    mem.w_u64(entity+ENT_SUB18_OFF,forbidden)
    original_read,original_prefetch=mem.read,mem.prefetch_pages
    def read(addr,size):
        assert addr!=forbidden
        return original_read(addr,size)
    def prefetch(addrs):
        assert forbidden not in addrs
        return original_prefetch(addrs)
    mem.read,mem.prefetch_pages=read,prefetch
    snap=NullsMemoryReader(mem,base).read_snapshot()
    projectile=next(e for e in snap.entities if e.vt==VT_PROJECTILE)
    assert projectile.hp is None


def test_metadata_cache_preserves_snapshot_and_refreshes_dynamic_hitpoints():
    mem,base=_build_fake_memory()
    cached=NullsMemoryReader(mem,base)
    first=cached.read_snapshot()
    warm=cached.read_snapshot()
    reference=NullsMemoryReader(mem,base,cache_metadata=False).read_snapshot()
    def state(snapshot):
        result=snapshot.to_dict()
        result.pop('bytes_read');result.pop('monotonic_ts')
        return result
    assert state(warm)==state(reference)==state(first)
    assert warm.bytes_read<=reference.bytes_read
    signature=struct.pack('<Q',base+VT_HITPOINT_COMP)
    hp=next(page+offset for page,data in mem.pages.items()
            for offset in [bytes(data).find(signature)] if offset>=0)
    mem.w_i32(hp+HP_VAL_OFF,123)
    updated=cached.read_snapshot()
    assert any(entity.hp==123 for entity in updated.entities)
    assert not any(entity.hp==123 for entity in warm.entities)


def test_adapter_shapes_unknowns_seat_rotation_and_state():
    adapter = NullsToSchemaV4Adapter(TABLES_DIR)
    snap = _sample_snapshot(own_seat=1)
    out = adapter.adapt(snap)

    assert out.obs.shape == (SchemaV4.SEAT_INTS,)
    assert out.obs.dtype == np.int32
    assert out.legal_mask.shape == (SchemaV4.ACTIONS,)
    assert out.legal_mask.dtype == bool
    assert out.own_seat == 1
    assert out.unmatched_names == ()

    # Header verified checks
    assert out.obs[SchemaV4.H_VERSION] if hasattr(SchemaV4, "H_VERSION") else True
    assert out.obs[SchemaV4.H_TICK] == 100
    assert out.obs[SchemaV4.H_ELIXIR] == 62000
    assert out.obs[SchemaV4.H_CROWNS] == 0
    assert out.obs[SchemaV4.H_CROWNS_ENEMY] == 0
    # seat=1 deck: Wizard (Rare +2, raw 9 -> 12), DarkPrince (Epic +5, raw 7 -> 13), ZapMachine (Leg +8, raw 3 -> 12)
    assert out.obs[SchemaV4.H_DECK_LEVEL + 0] == 12
    assert out.obs[SchemaV4.H_DECK_LEVEL + 1] == 13
    assert out.obs[SchemaV4.H_DECK_LEVEL + 2] == 12
    # Next card is Fireball (queue[0]=4)
    assert out.obs[SchemaV4.H_NEXT_CARD] == adapter.vocab.row_id("Fireball")
    # Enemy seen SCID 26000017 -> Wizard
    assert out.obs[SchemaV4.H_ENEMY_SEEN] == adapter.vocab.row_id("Wizard")

    # Unverified optional header fields stay UNKNOWN (-1); play gate & mid-match bar accumulation are initialized cleanly
    assert out.obs[SchemaV4.H_ELIXIR_RESERVED] == SchemaV4.UNKNOWN
    assert out.obs[SchemaV4.H_PLAY_OPEN_IN_TICKS] == 0  # tick=100 >= START_COOLDOWN_TICKS (90) -> already open
    assert out.obs[SchemaV4.H_FULL_BAR_MS] == 0
    assert out.obs[SchemaV4.H_LAST_REFUSAL] == SchemaV4.UNKNOWN
    early_out = NullsToSchemaV4Adapter(TABLES_DIR).adapt(replace(snap, tick=20, sec_elapsed_ms=1000))
    assert early_out.obs[SchemaV4.H_PLAY_OPEN_IN_TICKS] == 70 and not early_out.legal_mask[1:].any()
    pre_start = NullsToSchemaV4Adapter(TABLES_DIR).adapt(replace(snap, tick=-1, sec_elapsed_ms=0))
    assert pre_start.obs[SchemaV4.H_PLAY_OPEN_IN_TICKS] == 90 and not pre_start.legal_mask[1:].any()

    # Seat-1 coordinate transformation + Giant targeting enemy PrincessTower at (3500, 6500)
    # In seat=1 frame, battle (3500, 21000) -> (3500, 32000-21000=11000)
    ent_rows = out.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
    giant_row = [r for r in ent_rows[: out.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ROW] == adapter.vocab.row_id("Giant")][0]
    assert giant_row[SchemaV4.E_X] == 3500
    assert giant_row[SchemaV4.E_Y] == 11000
    assert giant_row[SchemaV4.E_DX] == 0
    assert giant_row[SchemaV4.E_DY] == 0
    assert giant_row[SchemaV4.E_STATE] == 2  # existing unit with a target
    assert giant_row[SchemaV4.E_SPEED_PERCENT] == SchemaV4.UNKNOWN
    assert giant_row[SchemaV4.E_BUFF_COUNT] == 1
    assert giant_row[SchemaV4.E_BUFFS] == adapter.vocab.buff_id("Rage")
    assert giant_row[SchemaV4.E_BUFFS + 1] == 1500
    assert giant_row[SchemaV4.E_TARGET] > 0

    # Projectile row check (unverified level & aim_x must be -1)
    proj_row = ent_rows[out.obs[SchemaV4.H_ENTITY_COUNT] - 1]
    assert proj_row[SchemaV4.E_KIND] == SchemaV4.KIND_PROJECTILE
    assert proj_row[SchemaV4.E_LEVEL] == SchemaV4.UNKNOWN
    assert proj_row[SchemaV4.E_AIM_X] == SchemaV4.UNKNOWN

    # Hand slot 0 is Wizard (cost 5, affordable with 6 elixir); slot 1 is Mirror (blocked without last_played_card);
    # slot 2 is ZapMachine (cost 6, affordable); slot 3 is Elixir Collector (cost 6, affordable)
    assert out.legal_mask[SchemaV4.WAIT]
    assert out.legal_mask[1 + 0 * SchemaV4.TILES : 1 + 1 * SchemaV4.TILES].any()
    assert not out.legal_mask[1 + 1 * SchemaV4.TILES : 1 + 2 * SchemaV4.TILES].any()

    # Step 2: move Giant + drop enemy princess tower (5000001) + simulate playing Wizard -> Knight replaces slot 0
    s1_next = replace(
        snap.summoners[1],
        played_count=2,
        hand=(4, 6, 2, 3),  # slot 0 Wizard -> Fireball, so last_played_card becomes Wizard
        elixir=50000,
    )
    ents_next = tuple(
        e for e in snap.entities if e.eid != 5000001
    )
    ents_next = tuple(
        replace(e, x=3500, y=20000) if e.eid == 5000020 else e for e in ents_next
    )
    snap2 = replace(snap, tick=108, time_ms=5400, summoners=(snap.summoners[0], s1_next), entities=ents_next)
    out2 = adapter.adapt(snap2)

    assert out2.obs[SchemaV4.H_CROWNS] == 1  # enemy lost 1 princess tower
    # Enemy princess tower slot (idx 4 in tower order for seat=1: battle x=3500 -> seat_x=3500 < 9000 -> left!)
    assert out2.obs[SchemaV4.H_TOWERS + 2 * 4] == 0
    assert out2.obs[SchemaV4.H_TOWERS + 2 * 4 + 1] == 3052  # cached max_hp

    # Giant moved in battle y: 21000 -> 20000; in seat=1 y: 11000 -> 12000 => dy = +1000
    ent_rows2 = out2.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
    giant_row2 = [r for r in ent_rows2[: out2.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ROW] == adapter.vocab.row_id("Giant")][0]
    assert giant_row2[SchemaV4.E_ID] == giant_row[SchemaV4.E_ID]
    assert giant_row2[SchemaV4.E_DX] == 0
    assert giant_row2[SchemaV4.E_DY] == 1000

    # Mirror in slot 1 would cost 5+1=6 > 5 elixir; bump elixir to 60000 on tick 116 to verify Mirror unlocks
    s1_next3 = replace(s1_next, elixir=60000)
    out3 = adapter.adapt(replace(snap2, tick=116, summoners=(snap.summoners[0], s1_next3)))
    assert out3.legal_mask[1 + 1 * SchemaV4.TILES : 1 + 2 * SchemaV4.TILES].any()
    # Pocket opened on enemy left side (seat col <= 8, row 17..22 should have legal tiles for Wizard via Mirror!)
    assert out3.legal_mask[SchemaV4.play(1, 3, 19)]

    # Serialization round-trip check
    snap_rt = NullsBattleSnapshot.from_dict(snap.to_dict())
    assert snap_rt == snap


def test_inferred_deployment_motion_and_verified_override():
    adapter = NullsToSchemaV4Adapter(TABLES_DIR)
    snap = _sample_snapshot()
    adapter.adapt(snap)
    giant = next(e for e in snap.entities if e.name == 'Giant')
    spawned = replace(giant,eid=giant.eid+999,target_eid=None)
    next_snap = replace(snap,tick=snap.tick+8,time_ms=snap.time_ms+400,entities=(*snap.entities,spawned))
    def row(out,eid):
        data=out.obs[SchemaV4.HEADER_INTS:].reshape(SchemaV4.MAX_ENTITIES,SchemaV4.ENTITY_INTS)
        return next(r for r in data[:out.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ID]==adapter._seat_id_of(eid))
    out=adapter.adapt(next_snap)
    assert row(out,spawned.eid)[SchemaV4.E_STATE] == 4
    assert row(out,spawned.eid)[SchemaV4.E_DEPLOY_MS] > 0
    moved=replace(spawned,x=spawned.x+100)
    out=adapter.adapt(replace(next_snap,tick=next_snap.tick+8,time_ms=next_snap.time_ms+400,entities=(*snap.entities,moved)))
    assert row(out,spawned.eid)[SchemaV4.E_STATE] == 1
    assert row(out,spawned.eid)[SchemaV4.E_DEPLOY_MS] == 0
    verified=replace(moved,state=3,deploy_ms=0,attack_ms=125)
    out=adapter.adapt(replace(next_snap,tick=next_snap.tick+16,time_ms=next_snap.time_ms+800,entities=(*snap.entities,verified)))
    assert row(out,spawned.eid)[SchemaV4.E_STATE] == 3
    assert row(out,spawned.eid)[SchemaV4.E_ATTACK_MS] == 125
    adapter.reset_match()
    assert not adapter.first_seen_ms and not adapter.recent_enemy_plays


def test_fake_memory_extended_components_and_metadata():
    mem, base = _build_fake_memory()
    alloc = _alloc_cursor(0x40000000)

    # Read pointers wired by _build_fake_memory so we can enrich structs in place
    st = int.from_bytes(mem.read(base + G_STATE, 8), "little")
    gm = int.from_bytes(mem.read(st + ST_GM_OFF, 8), "little")
    ctx = int.from_bytes(mem.read(gm + GM_CTX_OFF, 8), "little")
    lb = int.from_bytes(mem.read(ctx + CTX_LB_OFF, 8), "little")
    v7 = int.from_bytes(mem.read(ctx + CTX_V7_OFF, 8), "little")
    clk = int.from_bytes(mem.read(lb + LB_CLOCK_OFF, 8), "little")
    sum1 = int.from_bytes(mem.read(lb + LB_SUMMONERS_OFF + 8, 8), "little")
    sum0 = int.from_bytes(mem.read(lb + LB_SUMMONERS_OFF, 8), "little")
    deck1 = int.from_bytes(mem.read(lb + LB_DECKS_OFF + 8, 8), "little")
    arr1 = int.from_bytes(mem.read(deck1 + DECK_CARDS_ARR_OFF, 8), "little")

    # Timeline overtime section (section=1, type=1, elapsed=30s of 120s -> 90s left)
    mem.w_i32(clk + CLK_SECTION_OFF, 1)
    mem.w_i32(clk + CLK_SEC_TYPE_OFF, 1)
    mem.w_i32(clk + CLK_SEC_EL_OFF, 30000)
    mem.w_i32(clk + CLK_SEC_LEN_OFF, 120000)

    # Hand refill timer + deck slots (slot 0 Wizard Evo charged, slot 5 Giant Hero)
    mem.w_i32(sum1 + SUMMONER_HAND_REFILL_MS_OFF, 1450)
    card0 = int.from_bytes(mem.read(arr1 + 0 * 8, 8), "little")
    mem.w_i32(card0 + CARD_SLOT_FLAGS_OFF, SchemaV4.SLOT_EVOLUTION)
    mem.w_i32(card0 + CARD_EVO_CNT_OFF, 2)
    card5 = int.from_bytes(mem.read(arr1 + 5 * 8, 8), "little")
    mem.w_i32(card5 + CARD_SLOT_FLAGS_OFF, SchemaV4.SLOT_HERO)
    mem.write(sum1 + SUMMONER_HAND_PTR_OFF, struct.pack("<Q", alloc(32)))
    hp_ptr = int.from_bytes(mem.read(sum1 + SUMMONER_HAND_PTR_OFF, 8), "little")
    mem.write(hp_ptr, struct.pack("<4i", 0, 5, 2, 3))

    # Enemy evo tracking on summoner 0
    mem.write(sum0 + SUMMONER_SEEN_EVO_OFF, struct.pack("<8i", 1, 0, -1, -1, -1, -1, -1, -1))
    mem.write(sum0 + SUMMONER_SEEN_EVO_NEED_OFF, struct.pack("<8i", 2, 0, 0, 0, 0, 0, 0, 0))

    # PendingQueue: 1 pending card (slot 0 Wizard cost=5 exec in 12 ticks -> timeout=240+40+12=292)
    # + 1 pending ability (deck_idx=5 Giant cost=2 timeout=295)
    pq = alloc(64)
    pq_ptrs = alloc(16)
    p0_it = alloc(PQ_ITEM_SIZE)
    p1_it = alloc(PQ_ITEM_SIZE)
    wiz_dp = _write_logic_data(mem, alloc, "Wizard", 26000017, 5)
    giant_dp = _write_logic_data(mem, alloc, "Giant", 26000003, 5)
    mem.w_u64(pq_ptrs + 0, p0_it)
    mem.w_u64(pq_ptrs + 8, p1_it)
    mem.w_u64(pq + PQ_ARR0_OFF, pq_ptrs)
    mem.w_i32(pq + PQ_CNT0_OFF, 2)
    # Item 0: card
    mem.w_i32(p0_it + PQ_ITEM_SEQ_OFF, 10)
    mem.w_i32(p0_it + PQ_ITEM_DECK_IDX_OFF, 0)
    mem.w_u64(p0_it + PQ_ITEM_CARD_DP_OFF, wiz_dp)
    mem.w_i32(p0_it + PQ_ITEM_COST_OFF, 5)
    mem.w_i32(p0_it + PQ_ITEM_IS_ABILITY_OFF, 0)
    mem.w_i32(p0_it + PQ_ITEM_TIMEOUT_TICK_OFF, 292)
    # Item 1: ability
    mem.w_i32(p1_it + PQ_ITEM_SEQ_OFF, 11)
    mem.w_i32(p1_it + PQ_ITEM_DECK_IDX_OFF, 5)
    mem.w_u64(p1_it + PQ_ITEM_CARD_DP_OFF, giant_dp)
    mem.w_i32(p1_it + PQ_ITEM_COST_OFF, 2)
    mem.w_i32(p1_it + PQ_ITEM_IS_ABILITY_OFF, 1)
    mem.w_i32(p1_it + PQ_ITEM_TIMEOUT_TICK_OFF, 295)
    mem.w_u64(sum1 + SUMMONER_PENDING_PTR_OFF, pq)

    # Read existing 3 entities, enrich KingTower inactive, enrich Knight combat + buffs + flags + timers
    arr_v7 = int.from_bytes(mem.read(v7 + V7_ARR_OFF, 8), "little")
    e0, e1, e2 = [int.from_bytes(mem.read(arr_v7 + i * 8, 8), "little") for i in range(3)]
    mem.w_u64(e0 + ENT_TAG1_OFF, TAG_INACTIVE)  # enemy KingTower sleeping
    mem.w_u64(e1 + ENT_TAG1_OFF, 0)  # own KingTower awake

    # Enrich Knight e2: deploy=750, state=4, attack_ms=420, load_ms=85, clone/underground/invis/tags, ability_ticks=7
    mem.w_i32(e2 + ENT_IS_CLONE_OFF, 1)
    mem.w_u64(e2 + ENT_TAG1_OFF, TAG_HIDDEN | TAG_NO_MOVE | TAG_NO_ATTACK | TAG_CHARGING | TAG_FORCE_AIR)
    mem.w_i32(e2 + CHAR_STATE_OFF, 4)
    mem.w_i32(e2 + CHAR_DELAY_MS_OFF, 200)
    mem.w_i32(e2 + CHAR_DEPLOY_MS_OFF, 750)
    mem.write(e2 + CHAR_UNDERGROUND_OFF, bytes([1, 1]))
    mem.w_i32(e2 + CHAR_ABILITY_TICKS_OFF, 7)

    sub18_e2 = int.from_bytes(mem.read(e2 + ENT_SUB18_OFF, 8), "little")
    c_combat_e2 = int.from_bytes(mem.read(sub18_e2 + 0x00, 8), "little")
    c_buff_e2 = int.from_bytes(mem.read(sub18_e2 + 0x18, 8), "little")
    mem.w_i32(c_combat_e2 + COMBAT_ATTACK_MS_OFF, 420)
    mem.w_i32(c_combat_e2 + COMBAT_LOAD_MS_OFF, 85)

    # 4 buffs: Freeze (stop), Slow (-35,-35), Rage (+35,+35, 2400ms), Forever (-1ms) -> test rank + math
    def write_buff(name: str, speed_m: int, hit_m: int, invis: int, t_left: int) -> int:
        bdp = _write_logic_data(mem, alloc, name, 0)
        mem.w_i32(bdp + BUFF_DP_HIT_SPEED_MULT_OFF, hit_m)
        mem.w_i32(bdp + BUFF_DP_SPEED_MULT_OFF, speed_m)
        mem.write(bdp + BUFF_DP_INVISIBLE_OFF, bytes([invis]))
        bi = alloc(64)
        mem.w_u64(bi + BUFF_INST_DP_OFF, bdp)
        mem.w_i32(bi + BUFF_INST_TIME_OFF, t_left)
        return bi

    b_rage = write_buff("Rage", 135, 135, 0, 2400)
    b_slow = write_buff("IceWizardCold", -35, -35, 0, 1500)
    b_forever = write_buff("GhostInvisibility", 0, 0, 1, -1)
    b_freeze = write_buff("Freeze", -100, -100, 0, 900)
    barr4 = alloc(32)
    for idx, bi in enumerate((b_forever, b_rage, b_slow, b_freeze)):
        mem.w_u64(barr4 + idx * 8, bi)
    mem.w_u64(c_buff_e2 + BUFF_ARR_OFF, barr4)
    mem.w_i32(c_buff_e2 + BUFF_CNT_OFF, 4)

    # Hero/Champion controller slot 0 on sum1 targeting GiantHero entity e3
    e3 = alloc(0x200)
    hero_dp = _write_logic_data(mem, alloc, "Giant", 34000003)
    mem.w_u64(e3 + ENT_VT_OFF, base + VT_CHARACTER)
    mem.w_i32(e3 + ENT_ID_OFF, 5000047)
    mem.w_u64(e3 + ENT_DP_OFF, hero_dp)
    mem.w_i32(e3 + ENT_SIDE_OFF, 1)
    mem.w_i32(e3 + ENT_X_OFF, 9000)
    mem.w_i32(e3 + ENT_Y_OFF, 24000)
    mem.w_i32(e3 + ENT_LVL_OFF, 10)
    mem.w_i32(e3 + CHAR_STATE_OFF, 1)

    ctrl0 = alloc(256)
    uvec = alloc(16)
    uarr = alloc(8)
    mem.w_u64(uarr, e3)
    mem.w_u64(uvec + 0, uarr)
    mem.w_i32(uvec + 12, 1)
    mem.w_u64(ctrl0 + 0, base + VT_CHAMPION_CTRL)
    mem.w_i32(ctrl0 + CHAMP_CD_OFF, 3000)
    mem.w_i32(ctrl0 + CHAMP_CHARGES_OFF, -1)
    mem.w_i32(ctrl0 + CHAMP_DECK_IDX_OFF, 5)
    mem.w_u64(ctrl0 + CHAMP_DP_OFF, giant_dp)
    mem.w_i32(ctrl0 + CHAMP_STATE_OFF, 2)  # READY
    mem.w_u64(ctrl0 + CHAMP_UNITS_VEC_OFF, uvec)
    mem.w_i32(ctrl0 + CHAMP_CMD_MS_OFF, 0)
    mem.w_u64(sum1 + SUMMONER_CHAMP_OFFS[0], ctrl0)

    # Projectile e4 + AreaEffect e5
    fb_dp = _write_logic_data(mem, alloc, "Fireball", 0)
    e4 = alloc(0x200)
    mem.w_u64(e4 + ENT_VT_OFF, base + VT_PROJECTILE)
    mem.w_i32(e4 + ENT_ID_OFF, 5000080)
    mem.w_u64(e4 + ENT_DP_OFF, fb_dp)
    mem.w_i32(e4 + ENT_SIDE_OFF, 1)
    mem.w_i32(e4 + ENT_X_OFF, 6000)
    mem.w_i32(e4 + ENT_Y_OFF, 20000)
    mem.w_i32(e4 + ENT_Z_OFF, 100)
    mem.w_i32(e4 + PROJ_AIM_X_OFF, 4000)
    mem.w_i32(e4 + PROJ_AIM_Y_OFF, 6500)
    mem.w_i32(e4 + PROJ_LVL_OFF, 10)

    psn_dp = _write_logic_data(mem, alloc, "Poison", 0)
    mem.w_i32(psn_dp + AOE_DATA_MAX_RADIUS_OFF, 4000)
    mem.w_i32(psn_dp + AOE_DATA_LIFE_OFF, 4000)
    mem.w_i32(psn_dp + AOE_DATA_RADIUS_OFF, 2000)
    e5 = alloc(0x1C0)
    mem.w_u64(e5 + ENT_VT_OFF, base + VT_AREA_EFFECT)
    mem.w_i32(e5 + ENT_ID_OFF, 5000081)
    mem.w_u64(e5 + ENT_DP_OFF, psn_dp)
    mem.w_i32(e5 + ENT_SIDE_OFF, 1)
    mem.w_i32(e5 + ENT_X_OFF, 4000)
    mem.w_i32(e5 + ENT_Y_OFF, 7000)
    mem.w_i32(e5 + AOE_LVL_OFF, 12)
    mem.w_i32(e5 + AOE_TIME_LEFT_OFF, 2000)
    mem.w_i32(e5 + AOE_LIFE_OVR_OFF, 4000)

    all_eptrs = [e0, e1, e2, e3, e4, e5]
    new_arr_v7 = alloc(64)
    for idx, ep in enumerate(all_eptrs):
        mem.w_u64(new_arr_v7 + idx * 8, ep)
    mem.w_u64(v7 + V7_ARR_OFF, new_arr_v7)
    mem.w_i32(v7 + V7_CNT_OFF, len(all_eptrs))

    reader = NullsMemoryReader(mem, libg_base=base)
    snap = reader.read_snapshot()
    assert snap.section == 1 and snap.section_type == 1
    assert snap.summoners[1].hand_refill_ms == 1450
    assert snap.summoners[1].elixir_reserved == 70000
    assert len(snap.summoners[1].pending_commands or ()) == 2
    assert snap.summoners[1].abilities and snap.summoners[1].abilities[0].unit_eid == 5000047

    adapter = NullsToSchemaV4Adapter(TABLES_DIR, allow_abilities=True)
    out = adapter.adapt(snap, pending_targets=((3500, 22000), None, None, None))
    assert out.obs[SchemaV4.H_SECTION] == 1
    assert out.obs[SchemaV4.H_OVERTIME] == 1
    assert out.obs[SchemaV4.H_SECTION_LEFT_MS] == 90000
    assert out.obs[SchemaV4.H_HAND_REFILL_MS] == 1450
    assert out.obs[SchemaV4.H_ELIXIR_RESERVED] == 70000
    assert out.obs[SchemaV4.H_PENDING_COUNT] == 1
    assert out.obs[SchemaV4.H_PENDING + 0] == adapter.vocab.row_id("Wizard")
    assert out.obs[SchemaV4.H_PENDING + 1] == 3500
    assert out.obs[SchemaV4.H_PENDING + 2] == SchemaV4.to_seat_y(1, 22000)
    assert out.obs[SchemaV4.H_PENDING + 3] == 12
    assert out.obs[SchemaV4.H_HAND_PENDING + 0] == 1
    assert out.obs[SchemaV4.H_HAND_FORM + 0] == SchemaV4.FORM_EVOLVED
    assert out.obs[SchemaV4.H_HAND_FORM + 1] == SchemaV4.FORM_HERO
    assert out.obs[SchemaV4.H_ENEMY_SEEN_EVOLUTION + 0] == 1
    assert out.obs[SchemaV4.H_ENEMY_SEEN_EVOLUTION_NEED + 0] == 2
    # GiantHero ability pending command at deck_idx=5 -> cmd_ticks=15, so ABILITY_0 blocked while pending!
    assert out.obs[SchemaV4.H_ABILITY + 0] == adapter.vocab.row_id("Giant")
    assert out.obs[SchemaV4.H_ABILITY + 5] == 15
    assert not out.legal_mask[SchemaV4.ability(0)]

    # Clear pending ability cmd and update Knight buffs to Rage+Slow+Forever -> test 135*65//100 = 87% speed & forever -1ms
    mem.w_i32(pq + PQ_CNT0_OFF, 0)
    mem.w_i32(clk + CLK_TICK_OFF, 241)
    barr2 = alloc(24)
    mem.w_u64(barr2 + 0, b_forever)
    mem.w_u64(barr2 + 8, b_rage)
    mem.w_u64(barr2 + 16, b_slow)
    mem.w_u64(c_buff_e2 + BUFF_ARR_OFF, barr2)
    mem.w_i32(c_buff_e2 + BUFF_CNT_OFF, 3)
    snap2 = reader.read_snapshot()
    out_ab = adapter.adapt(snap2)
    assert out_ab.legal_mask[SchemaV4.ability(0)]
    k2 = next(
        r for r in out_ab.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)[: out_ab.obs[SchemaV4.H_ENTITY_COUNT]]
        if r[SchemaV4.E_ID] == adapter._seat_id_of(5000020)
    )
    assert k2[SchemaV4.E_SPEED_PERCENT] == 87 and k2[SchemaV4.E_HIT_SPEED_PERCENT] == 87
    assert [k2[SchemaV4.E_BUFFS + 2 * b] for b in range(3)] == [
        adapter.vocab.buff_id("IceWizardCold"),
        adapter.vocab.buff_id("Rage"),
        adapter.vocab.buff_id("GhostInvisibility"),
    ]
    assert k2[SchemaV4.E_BUFFS + 2 * 2 + 1] == -1  # forever buff time_left == -1
    # Partial buff (missing dp ptr) -> speed_percent / hit_speed_percent must be UNKNOWN instead of fake 100
    b_partial = alloc(64)
    barr_p = alloc(8)
    mem.w_u64(barr_p, b_partial)
    mem.w_u64(c_buff_e2 + BUFF_ARR_OFF, barr_p)
    mem.w_i32(c_buff_e2 + BUFF_CNT_OFF, 1)
    snap_p = NullsMemoryReader(mem, libg_base=base, cache_metadata=False).read_snapshot()
    mem.w_i32(clk + CLK_TICK_OFF, 242)
    snap_p = NullsMemoryReader(mem, libg_base=base, cache_metadata=False).read_snapshot()
    out_p = adapter.adapt(snap_p)
    k_p = next(
        r for r in out_p.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)[: out_p.obs[SchemaV4.H_ENTITY_COUNT]]
        if r[SchemaV4.E_ID] == adapter._seat_id_of(5000020)
    )
    assert k_p[SchemaV4.E_SPEED_PERCENT] == SchemaV4.UNKNOWN and k_p[SchemaV4.E_HIT_SPEED_PERCENT] == SchemaV4.UNKNOWN
    out_no_ab = NullsToSchemaV4Adapter(TABLES_DIR, allow_abilities=False).adapt(snap2)
    assert not out_no_ab.legal_mask[SchemaV4.ability(0)]

    ent_rows = out.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
    by_eid = {
        eid: ent_rows[i]
        for i, row in enumerate(ent_rows[: out.obs[SchemaV4.H_ENTITY_COUNT]])
        for eid in (5000000, 5000003, 5000020, 5000080, 5000081)
        if row[SchemaV4.E_ID] == adapter._seat_id_of(eid)
    }
    assert (by_eid[5000000][SchemaV4.E_FLAGS] & SchemaV4.F_AWAKE) == 0
    assert (by_eid[5000003][SchemaV4.E_FLAGS] & SchemaV4.F_AWAKE) != 0

    knight = by_eid[5000020]
    for flag_bit in (
        SchemaV4.F_CLONE,
        SchemaV4.F_UNDERGROUND,
        SchemaV4.F_INVISIBLE,
        SchemaV4.F_HIDDEN,
        SchemaV4.F_NO_MOVE,
        SchemaV4.F_NO_ATTACK,
        SchemaV4.F_CHARGED,
        SchemaV4.F_AIR,
    ):
        assert knight[SchemaV4.E_FLAGS] & flag_bit
    assert knight[SchemaV4.E_DEPLOY_MS] == 750
    assert knight[SchemaV4.E_ATTACK_MS] == 420
    assert knight[SchemaV4.E_LOAD_MS] == 85
    assert knight[SchemaV4.E_ABILITY_TICKS] == 7
    assert knight[SchemaV4.E_SPEED_PERCENT] == 0 and knight[SchemaV4.E_HIT_SPEED_PERCENT] == 0
    assert knight[SchemaV4.E_BUFF_COUNT] == 4  # Total buffs; only the first three are listed.
    assert [knight[SchemaV4.E_BUFFS + 2 * b] for b in range(SchemaV4.BUFF_SLOTS)] == [
        adapter.vocab.buff_id("Freeze"),
        adapter.vocab.buff_id("IceWizardCold"),
        adapter.vocab.buff_id("Rage"),
    ]

    # Projectile & AreaEffect rows
    proj = by_eid[5000080]
    assert proj[SchemaV4.E_LEVEL] == 11
    assert proj[SchemaV4.E_AIM_X] == 4000 and proj[SchemaV4.E_AIM_Y] == SchemaV4.to_seat_y(1, 6500)
    assert proj[SchemaV4.E_STATE] == SchemaV4.NO_STATE
    aoe = by_eid[5000081]
    assert aoe[SchemaV4.E_LEVEL] == 13
    assert aoe[SchemaV4.E_TIME_LEFT_MS] == 2000
    assert aoe[SchemaV4.E_RADIUS] == 3000  # halfway between 2000 and 4000
    assert aoe[SchemaV4.E_STATE] == SchemaV4.NO_STATE and aoe[SchemaV4.E_DEPLOY_MS] == 0


def test_semantics_seats_idempotency_enemy_hiding_and_edge_cases():
    adapter = NullsToSchemaV4Adapter(TABLES_DIR)
    base_snap = _sample_snapshot(own_seat=0)

    # Full-bar ms accumulates ONLY after initial observation & when own_sum.elixir >= 100000
    s0_full = replace(base_snap.summoners[0], elixir=100000, elixir_reserved=None)
    s0_snap = replace(base_snap, summoners=(s0_full, base_snap.summoners[1]))
    o0 = adapter.adapt(s0_snap)
    assert o0.obs[SchemaV4.H_FULL_BAR_MS] == 0

    # Duplicate same-tick adapt() must be idempotent (episode_step, full_bar_ms, dx/dy unchanged)
    o0_dup = adapter.adapt(s0_snap)
    assert np.array_equal(o0.obs, o0_dup.obs)

    # Skipped ticks (+16 ticks, +800ms) accumulates full_bar_ms and tracks motion
    giant = next(e for e in s0_snap.entities if e.name == "Giant")
    moved_ents = tuple(replace(e, x=4100, y=20000) if e.eid == giant.eid else e for e in s0_snap.entities)
    o1 = adapter.adapt(replace(s0_snap, tick=116, time_ms=5800, entities=moved_ents))
    assert o1.obs[SchemaV4.H_FULL_BAR_MS] == 800
    rows1 = o1.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
    g1 = next(r for r in rows1[: o1.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ID] == adapter._seat_id_of(giant.eid))
    assert g1[SchemaV4.E_X] == SchemaV4.to_seat_x(0, 4100) and g1[SchemaV4.E_Y] == SchemaV4.to_seat_y(0, 20000)
    assert g1[SchemaV4.E_DX] == -600 and g1[SchemaV4.E_DY] == -1000

    # Immediate ID reuse with different card/name (without gap) resets motion to (0, 0)
    reused_direct = replace(giant, name="Archer", scid=34000001, x=4500, y=19500)
    o_rd = adapter.adapt(
        replace(s0_snap, tick=120, time_ms=6000, entities=(*[e for e in moved_ents if e.eid != giant.eid], reused_direct))
    )
    rows_rd = o_rd.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
    rd_ent = next(r for r in rows_rd[: o_rd.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ID] == adapter._seat_id_of(giant.eid))
    assert rd_ent[SchemaV4.E_DX] == 0 and rd_ent[SchemaV4.E_DY] == 0

    # Entity disappears then reappears -> motion resets to (0, 0)
    no_giant = tuple(e for e in moved_ents if e.eid != giant.eid)
    adapter.adapt(replace(s0_snap, tick=124, time_ms=6200, entities=no_giant))
    reused = replace(giant, name="Knight", scid=34000000, x=4500, y=19500)
    o_reused = adapter.adapt(replace(s0_snap, tick=132, time_ms=6600, entities=(*no_giant, reused)))
    rows_r = o_reused.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
    r_ent = next(r for r in rows_r[: o_reused.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ID] == adapter._seat_id_of(giant.eid))
    assert r_ent[SchemaV4.E_DX] == 0 and r_ent[SchemaV4.E_DY] == 0

    # NO_TARGET outputs (-1, 0, 0); reserved elixir restricts legal_mask
    no_tgt = replace(reused, target_eid=None)
    s0_res = replace(s0_full, elixir=50000, elixir_reserved=20000)  # 3 whole elixir available; Giant slot 3 costs 5
    o_res = adapter.adapt(
        replace(s0_snap, tick=140, time_ms=7000, summoners=(s0_res, base_snap.summoners[1]), entities=(*no_giant, no_tgt))
    )
    rows_res = o_res.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
    r_nt = next(r for r in rows_res[: o_res.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ID] == adapter._seat_id_of(giant.eid))
    assert (r_nt[SchemaV4.E_TARGET], r_nt[SchemaV4.E_TARGET_X], r_nt[SchemaV4.E_TARGET_Y]) == (-1, 0, 0)
    assert not o_res.legal_mask[1 + 3 * SchemaV4.TILES : 1 + 4 * SchemaV4.TILES].any()

    # Enemy plays: Fireball hides AIM while in flight -> reveals PLAY_POINT_SHOWN after landing;
    # Underground enemy Miner hides HP/target and suppresses PLAY_POINT_SHOWN; 3 Goblins deduplicate to 1 cluster
    fb_proj = NullsEntity(5000090, VT_PROJECTILE, "Fireball", 0, 1, 9000, 15000, 100, None, None, None, None, None, None, None, aim_x=3500, aim_y=6500)
    miner = NullsEntity(5000091, VT_CHARACTER, "Miner", 34000032, 1, 6000, 10000, 0, 10, 1000, 1000, 0, 0, None, (), state=6, is_underground=True)
    gobs = tuple(
        NullsEntity(5000100 + i, VT_CHARACTER, "Goblin", 34000001, 1, 8000 + i * 200, 20000, 0, 10, 200, 200, 0, 0, None, ())
        for i in range(3)
    )
    e_sum_play = replace(
        base_snap.summoners[1],
        played_count=base_snap.summoners[1].played_count + 3,
        seen_scids=(26000017, 28000000, 26000032, 26000002, -1, -1, -1, -1),
    )
    o_ep = adapter.adapt(
        replace(
            s0_snap,
            tick=148,
            time_ms=7400,
            summoners=(s0_full, e_sum_play),
            entities=(*no_giant, fb_proj, miner, *gobs),
        )
    )
    rows_ep = o_ep.obs[SchemaV4.HEADER_INTS :].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
    fb_row = next(r for r in rows_ep[: o_ep.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ID] == adapter._seat_id_of(fb_proj.eid))
    assert fb_row[SchemaV4.E_AIM_X] == SchemaV4.UNKNOWN and fb_row[SchemaV4.E_AIM_Y] == SchemaV4.UNKNOWN
    m_row = next(r for r in rows_ep[: o_ep.obs[SchemaV4.H_ENTITY_COUNT]] if r[SchemaV4.E_ID] == adapter._seat_id_of(miner.eid))
    assert m_row[SchemaV4.E_HP] == SchemaV4.UNKNOWN and (m_row[SchemaV4.E_FLAGS] & SchemaV4.F_UNDERGROUND)

    # Check landing reveals Fireball point in H_ENEMY_PLAYS on next tick
    o_land = adapter.adapt(
        replace(s0_snap, tick=156, time_ms=7800, summoners=(s0_full, e_sum_play), entities=(*no_giant, miner, *gobs))
    )
    plays = o_land.obs[SchemaV4.H_ENEMY_PLAYS : SchemaV4.H_QUEUE].reshape(SchemaV4.ENEMY_PLAY_ROWS, SchemaV4.ENEMY_PLAY_INTS)
    fb_play = next(p for p in plays if p[0] == adapter.vocab.row_id("Fireball"))
    assert fb_play[7] & SchemaV4.PLAY_POINT_SHOWN

    # Tick rewind resets episode cleanly
    o_rewind = adapter.adapt(replace(s0_snap, tick=10, time_ms=500))
    assert o_rewind.obs[SchemaV4.H_FULL_BAR_MS] == 0 and o_rewind.obs[SchemaV4.H_EPISODE_STEP] == 0


def test_legacy_snapshot_from_dict_backward_compat():
    snap = _sample_snapshot()
    raw = snap.to_dict()
    for k in ("section", "sec_type", "rate_idx"):
        raw.pop(k, None)
    for s in raw["summoners"]:
        for k in ("elixir_reserved", "hand_refill_ms", "pending_commands", "abilities", "seen_evo", "seen_evo_need"):
            s.pop(k, None)
        for c in s["deck"]:
            for k in ("slot_flags", "star_level", "evolution_count", "hero_count"):
                c.pop(k, None)
    for e in raw["entities"]:
        for k in (
            "is_clone",
            "tags1",
            "tags2",
            "state",
            "delay_ms",
            "deploy_ms",
            "is_underground",
            "is_invisible",
            "ability_ticks",
            "attack_ms",
            "load_ms",
            "speed_percent",
            "hit_speed_percent",
            "aim_x",
            "aim_y",
            "time_left_ms",
            "radius",
        ):
            e.pop(k, None)

    restored = NullsBattleSnapshot.from_dict(raw)
    out = NullsToSchemaV4Adapter(TABLES_DIR).adapt(restored)
    assert out.obs.shape == (SchemaV4.SEAT_INTS,)
    assert out.obs[SchemaV4.H_ELIXIR_RESERVED] == SchemaV4.UNKNOWN
    assert out.obs[SchemaV4.H_HAND_REFILL_MS] == SchemaV4.UNKNOWN
    assert out.obs[SchemaV4.H_PENDING_COUNT] == SchemaV4.UNKNOWN
    assert out.obs[SchemaV4.H_ABILITY] == SchemaV4.UNKNOWN
    assert out.obs[SchemaV4.H_DECK_SLOTS] == SchemaV4.UNKNOWN


def test_corrupt_or_torn_pending_queue_returns_unknown_without_overflow():
    mem, base = _build_fake_memory()
    alloc = _alloc_cursor(0x42000000)
    st = int.from_bytes(mem.read(base + G_STATE, 8), "little")
    gm = int.from_bytes(mem.read(st + ST_GM_OFF, 8), "little")
    ctx = int.from_bytes(mem.read(gm + GM_CTX_OFF, 8), "little")
    lb = int.from_bytes(mem.read(ctx + CTX_LB_OFF, 8), "little")
    sum1 = int.from_bytes(mem.read(lb + LB_SUMMONERS_OFF + 8, 8), "little")

    pq = alloc(64)
    pq_ptrs = alloc(8)
    bad_it = alloc(PQ_ITEM_SIZE)
    wiz_dp = _write_logic_data(mem, alloc, "Wizard", 26000017, 5)
    mem.w_u64(pq_ptrs, bad_it)
    mem.w_u64(pq + PQ_ARR0_OFF, pq_ptrs)
    mem.w_i32(pq + PQ_CNT0_OFF, 1)
    mem.w_i32(bad_it + PQ_ITEM_SEQ_OFF, 1)
    mem.w_i32(bad_it + PQ_ITEM_DECK_IDX_OFF, 0)
    mem.w_u64(bad_it + PQ_ITEM_CARD_DP_OFF, wiz_dp)
    mem.w_i32(bad_it + PQ_ITEM_COST_OFF, 0x7FFFFFFF)  # freed/torn huge garbage cost
    mem.w_i32(bad_it + PQ_ITEM_IS_ABILITY_OFF, 0)
    mem.w_i32(bad_it + PQ_ITEM_TIMEOUT_TICK_OFF, 280)
    mem.w_u64(sum1 + SUMMONER_PENDING_PTR_OFF, pq)

    snap = NullsMemoryReader(mem, libg_base=base).read_snapshot()
    assert snap.summoners[1].pending_commands is None
    assert snap.summoners[1].elixir_reserved is None

    adapter = NullsToSchemaV4Adapter(TABLES_DIR)
    out = adapter.adapt(snap)
    assert out.obs[SchemaV4.H_ELIXIR_RESERVED] == SchemaV4.UNKNOWN
    assert out.obs[SchemaV4.H_PENDING_COUNT] == SchemaV4.UNKNOWN

    # Even if a corrupted serialized snapshot somehow carries huge elixir_reserved, adapt() never raises OverflowError
    bad_snap = replace(
        _sample_snapshot(),
        summoners=(
            _sample_snapshot().summoners[0],
            replace(_sample_snapshot().summoners[1], elixir_reserved=10_000_000_000),
        ),
    )
    out2 = adapter.adapt(bad_snap)
    assert out2.obs[SchemaV4.H_ELIXIR_RESERVED] == SchemaV4.UNKNOWN


