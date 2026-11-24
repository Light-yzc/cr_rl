"""Bounded, read-only /proc/<pid>/mem reader for Nulls Royale (ARM64 16.635.1).

Only verified pointer chains and struct offsets are decoded. Strictly read-only:
never writes process memory, never injects code, and hard-caps every snapshot
poll to <= 4 MiB.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import subprocess
import time
from typing import Any, Protocol


HARD_CAP_BYTES = 4 * 1024 * 1024  # 4 MiB absolute per-poll ceiling
DEFAULT_MAX_POLL_BYTES = 1024 * 1024  # 1 MiB default target budget per poll
DEFAULT_PACKAGE = "nullsroyale.rel.free"
DEFAULT_BUILD = "16.635.1"

# Verified Nulls 16.635.1 offsets relative to libg.so base
G_STATE = 0x1B7D210
G_LOCAL_ACC = 0x1B728B8

ST_GM_OFF = 0x28
ST_MODE_OFF = 0x38
GM_CTX_OFF = 0x90
CTX_V7_OFF = 0x40
CTX_LB_OFF = 0xA8

LB_SUBSTATE_OFF = 0x18
LB_PLAYER_OFFS = (0x30, 0x38)
LB_DECKS_OFF = 0x88
LB_SUMMONERS_OFF = 0xE0
LB_CLOCK_OFF = 0x190
LB_END_WRAP_OFF = 0x1B0

OBJ18_STATE_OFF = 0x74
OBJ18_CLK_FLAG_OFF = 0x110

CLK_TICK_OFF = 0x10
CLK_TIME_MS_OFF = 0x14
CLK_FLAGS_OFF = 0x18
CLK_SECTION_OFF = 0x20
CLK_SEC_EL_OFF = 0x24
CLK_SEC_LEN_OFF = 0x28
CLK_SEC_TYPE_OFF = 0x2C
CLK_RATE_IDX_OFF = 0x34
CLK_PHASE_OFF = 0x34  # backward-compatible alias for CLK_RATE_IDX_OFF
CLK_FULL_BAR_X10_OFF = 0x40
CLK_FULL_BAR_DIV_OFF = 0x44
CLK_REFILL_CFG_MS_OFF = 0x58

SECTION_NORMAL = 0
SECTION_OVERTIME = 1
SECTION_BONUS_TIME = 2
START_COOLDOWN_TICKS = 90

SUMMONER_HAND_REFILL_MS_OFF = 520
SUMMONER_HAND_PTR_OFF = 528
SUMMONER_QUEUE_PTR_OFF = 544
SUMMONER_PENDING_PTR_OFF = 584
SUMMONER_SEEN_OFF = 648
SUMMONER_SEEN_EVO_OFF = 680
SUMMONER_SEEN_EVO_NEED_OFF = 712
SUMMONER_ELIXIR_OFF = 760
SUMMONER_LEAKED_OFF = 836
SUMMONER_PLAYED_OFF = 840
SUMMONER_CHAMP_OFFS = (928, 936)
SUMMONER_READ_SIZE = 944

PQ_ARR0_OFF = 16
PQ_CNT0_OFF = 28
PQ_ARR1_OFF = 32
PQ_CNT1_OFF = 44
PQ_ITEM_SIZE = 56
PQ_ITEM_SEQ_OFF = 0
PQ_ITEM_DECK_IDX_OFF = 4
PQ_ITEM_CARD_DP_OFF = 8
PQ_ITEM_COST_OFF = 16
PQ_ITEM_IS_ABILITY_OFF = 20
PQ_ITEM_TIMEOUT_TICK_OFF = 24

CHAMP_CD_OFF = 120
CHAMP_CHARGES_OFF = 128
CHAMP_DECK_IDX_OFF = 140
CHAMP_DP_OFF = 144
CHAMP_STATE_OFF = 152
CHAMP_UNITS_VEC_OFF = 160
CHAMP_CMD_MS_OFF = 172
CHAMP_PENDING_SEQ_OFF = 180
CHAMP_READ_SIZE = 192

DECK_CARDS_ARR_OFF = 0x20
CARD_DP_OFF = 0x10
CARD_LVL_OFF = 0x18
CARD_SLOT_FLAGS_OFF = 0x1C
CARD_STAR_LVL_OFF = 0x24
CARD_EVO_CNT_OFF = 0x30
CARD_HERO_CNT_OFF = 0x34

V7_ARR_OFF = 0x08
V7_CNT_OFF = 0x14

ENT_VT_OFF = 0x00
ENT_ID_OFF = 0x08
ENT_SUB18_OFF = 0x18
ENT_K30_OFF = 0x30
ENT_DP_OFF = 0x48
ENT_SIDE_OFF = 0x78
ENT_X_OFF = 0x7C
ENT_Y_OFF = 0x80
ENT_Z_OFF = 0x8C
ENT_CARD_IDX_OFF = 148
ENT_IS_CLONE_OFF = 164
ENT_TAG1_OFF = 176
ENT_TAG2_OFF = 184

TAG_CHARGING = 1 << 3
TAG_NO_MOVE = 1 << 6
TAG_NO_ATTACK = 1 << 10
TAG_INACTIVE = 1 << 20
TAG_HIDDEN = 1 << 24
TAG_FORCE_GROUND = 1 << 39
TAG_FORCE_AIR = 1 << 44

CHAR_STATE_OFF = 284
CHAR_LVL_OFF = 288
ENT_LVL_OFF = 0x120  # backward-compatible alias for CHAR_LVL_OFF
CHAR_DELAY_MS_OFF = 296
CHAR_DEPLOY_MS_OFF = 348
CHAR_UNDERGROUND_OFF = 368
CHAR_INVISIBLE_OFF = 369
CHAR_ABILITY_TICKS_OFF = 488

PROJ_AIM_X_OFF = 288
PROJ_AIM_Y_OFF = 292
PROJ_LVL_OFF = 408

AOE_LVL_OFF = 252
AOE_TIME_LEFT_OFF = 256
AOE_LIFE_OVR_OFF = 276
AOE_DATA_MAX_RADIUS_OFF = 184
AOE_DATA_LIFE_OFF = 328
AOE_DATA_RADIUS_OFF = 332

DP_STR_LEN_OFF = 0x28
DP_STR_DATA_OFF = 0x30
DP_SCID_OFF = 0x40
DP_COST_OFF = 0x148

# Verified component & entity vtable RVAs (libg.so base)
VT_AREA_EFFECT = 0x1A4F9A0
VT_CHARACTER = 0x1A4FBB8
VT_PROJECTILE = 0x1A50348
VT_KING_TOWER = 0x1A50840
VT_CHAMPION_CTRL = 0x1A51C30

VT_BUFF_COMP = 0x1A55838
VT_COMBAT_COMP = 0x1A558B0
VT_HITPOINT_COMP = 0x1A559A0
VT_MOVE_COMP = 0x1A55A18

HP_VAL_OFF = 0x10
HP_MAX_OFF = 0x14
HP_SHIELD_OFF = 0x2C
HP_MAX_SHIELD_OFF = 0x30

BUFF_ARR_OFF = 0x18
BUFF_CNT_OFF = 0x24
BUFF_INST_TIME_OFF = 0x08
BUFF_INST_TOTAL_OFF = 0x0C
BUFF_INST_DP_OFF = 0x18
BUFF_INST_LVL_OFF = 0x28

BUFF_DP_HIT_SPEED_MULT_OFF = 224
BUFF_DP_SPEED_MULT_OFF = 228
BUFF_DP_INVISIBLE_OFF = 264

COMBAT_TARGET_PTR_OFF = 0x10
COMBAT_ATTACK_MS_OFF = 0x24
COMBAT_LOAD_MS_OFF = 0x28

_BATCH_PAGES = 32
_PTR_MASK = 0x00FFFFFFFFFFFFFF


class NotInBattleError(RuntimeError):
    """Raised when the client is in lobby, loading, or post-match overlay."""


class ReadBudgetExceededError(RuntimeError):
    """Raised if a single poll attempts to read beyond the configured byte cap."""


def is_ptr(val: int) -> bool:
    """Return True if a tagged or plain 64-bit ARM64 value looks like a heap/lib ptr."""
    low = val & _PTR_MASK
    return 0x10000000 < low < 0x7FFFFFFFFFFF


class MemoryBackend(Protocol):
    """Abstract read-only memory accessor for live ADB and offline tests."""

    @property
    def poll_bytes_read(self) -> int: ...

    def begin_poll(self) -> None: ...

    def prefetch_pages(self, addrs: list[int]) -> None: ...

    def read(self, addr: int, size: int) -> bytes: ...


class BoundedAdbMemReader:
    """Read-only page-cached `/proc/<pid>/mem` reader over ADB `dd`."""

    def __init__(
        self,
        serial: str,
        pid: str,
        *,
        adb: str = "adb",
        max_poll_bytes: int = DEFAULT_MAX_POLL_BYTES,
        transport=None,
    ) -> None:
        if not 4096 <= max_poll_bytes <= HARD_CAP_BYTES:
            raise ValueError(
                f"max_poll_bytes must be within [4096, {HARD_CAP_BYTES}], got {max_poll_bytes}"
            )
        self.serial = serial
        self.pid = str(pid)
        self.adb = adb
        self.max_poll_bytes = int(max_poll_bytes)
        self._transport = transport or self._default_transport
        self._page_cache: dict[int, bytes] = {}
        self._poll_bytes = 0
        self.total_bytes_read = 0

    @property
    def poll_bytes_read(self) -> int:
        return self._poll_bytes

    def _default_transport(self, args: list[str]) -> bytes:
        return subprocess.check_output([self.adb, "-s", self.serial, *args], timeout=5)

    def begin_poll(self) -> None:
        self._page_cache.clear()
        self._poll_bytes = 0

    def _charge(self, nbytes: int) -> None:
        if self._poll_bytes + nbytes > self.max_poll_bytes:
            raise ReadBudgetExceededError(
                f"Per-poll read budget exceeded: {self._poll_bytes + nbytes} > {self.max_poll_bytes}"
            )
        self._poll_bytes += nbytes
        self.total_bytes_read += nbytes

    def prefetch_pages(self, addrs: list[int]) -> None:
        pages = sorted(
            {
                (a & _PTR_MASK) & ~0xFFF
                for a in addrs
                if is_ptr(a) and ((a & _PTR_MASK) & ~0xFFF) not in self._page_cache
            }
        )
        for off in range(0, len(pages), _BATCH_PAGES):
            batch = pages[off : off + _BATCH_PAGES]
            self._charge(len(batch) * 4096)
            cmds = ";".join(
                f"dd if=/proc/{self.pid}/mem bs=4096 skip={p // 4096} count=1 2>/dev/null"
                for p in batch
            )
            raw = self._transport(["exec-out", f"sh -c '{cmds}'"])
            if len(raw) == len(batch) * 4096:
                for idx, page in enumerate(batch):
                    self._page_cache[page] = raw[idx * 4096 : (idx + 1) * 4096]

    def _read_pages(self, start_page: int, count: int) -> bytes:
        if count == 1 and start_page in self._page_cache:
            return self._page_cache[start_page]
        self._charge(count * 4096)
        raw = self._transport(
            [
                "exec-out",
                f"dd if=/proc/{self.pid}/mem bs=4096 skip={start_page // 4096} count={count} 2>/dev/null",
            ]
        )
        for idx in range(count):
            chunk = raw[idx * 4096 : (idx + 1) * 4096]
            if len(chunk) == 4096:
                self._page_cache[start_page + idx * 4096] = chunk
        return raw

    def read(self, addr: int, size: int) -> bytes:
        clean = addr & _PTR_MASK
        if clean < 0x10000 or size <= 0:
            return b""
        page = clean & ~0xFFF
        off = clean - page
        count = (off + size + 0xFFF) // 0x1000
        if all((page + idx * 4096) in self._page_cache for idx in range(count)):
            raw = b"".join(self._page_cache[page + idx * 4096] for idx in range(count))
        else:
            raw = self._read_pages(page, count)
        return raw[off : off + size]


def _u64(mem: MemoryBackend, addr: int) -> int:
    b = mem.read(addr, 8)
    return struct.unpack("<Q", b)[0] if len(b) == 8 else 0


def _i32(mem: MemoryBackend, addr: int) -> int:
    b = mem.read(addr, 4)
    return struct.unpack("<i", b)[0] if len(b) == 4 else 0


def _read_logic_name(mem: MemoryBackend, dp: int) -> str:
    """Decode Nulls 16.635.1 `LogicData` name at `dp+0x28` (inline if len <= 7)."""
    if not is_ptr(dp):
        return ""
    raw = mem.read(dp + DP_STR_LEN_OFF, 16)
    if len(raw) < 16:
        return ""
    length, _ = struct.unpack_from("<ii", raw, 0)
    if not (0 < length <= 128):
        return ""
    if length <= 7:
        return raw[8 : 8 + length].decode("utf-8", "replace")
    sptr = struct.unpack_from("<Q", raw, 8)[0]
    if not is_ptr(sptr):
        return ""
    data = mem.read(sptr, length)
    return data.decode("utf-8", "replace") if len(data) == length else ""


def _buff_multiplier_percent(vals: list[int]) -> int:
    """Compute C++ `sub_10B93E4`/`sub_10B9300` speed/hit-speed percent multiplier."""
    boost = 100
    slow = 0
    for v in vals:
        if v >= 1:
            if v > boost:
                boost = v
        elif v < 0:
            neg = -v
            if neg >= slow:
                slow = neg
    rem = max(0, min(100, 100 - slow))
    return (boost * rem) // 100


def _aoe_radius_units(base_r: int, max_r: int, life_ms: int, time_left_raw: int) -> int:
    """Compute effective AreaEffect radius matching `sub_1070D18`."""
    if max_r > 0 and life_ms > 0:
        num = (max_r - base_r) * time_left_raw
        q = abs(num) // life_ms
        if num < 0:
            q = -q
        val = max_r - q
        lo, hi = (base_r, max_r) if base_r <= max_r else (max_r, base_r)
        return max(lo, min(hi, val))
    return base_r


@dataclass(frozen=True)
class NullsBuff:
    name: str
    scid: int
    time_left: int
    total_ms: int | None = None
    speed_mult: int | None = None
    hit_speed_mult: int | None = None
    invisible: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scid": self.scid,
            "time_left": self.time_left,
            "total_ms": self.total_ms,
            "speed_mult": self.speed_mult,
            "hit_speed_mult": self.hit_speed_mult,
            "invisible": self.invisible,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NullsBuff:
        return cls(
            name=str(d["name"]),
            scid=int(d["scid"]),
            time_left=int(d["time_left"]),
            total_ms=None if d.get("total_ms") is None else int(d["total_ms"]),
            speed_mult=None if d.get("speed_mult") is None else int(d["speed_mult"]),
            hit_speed_mult=None if d.get("hit_speed_mult") is None else int(d["hit_speed_mult"]),
            invisible=None if d.get("invisible") is None else bool(d["invisible"]),
        )


@dataclass(frozen=True)
class NullsCard:
    name: str
    scid: int
    cost: int
    level_raw: int
    slot_flags: int | None = None
    evolution_count: int | None = None
    hero_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scid": self.scid,
            "cost": self.cost,
            "level_raw": self.level_raw,
            "slot_flags": self.slot_flags,
            "evolution_count": self.evolution_count,
            "hero_count": self.hero_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NullsCard:
        return cls(
            name=str(d["name"]),
            scid=int(d["scid"]),
            cost=int(d["cost"]),
            level_raw=int(d["level_raw"]),
            slot_flags=None if d.get("slot_flags") is None else int(d["slot_flags"]),
            evolution_count=None if d.get("evolution_count") is None else int(d["evolution_count"]),
            hero_count=None if d.get("hero_count") is None else int(d["hero_count"]),
        )


@dataclass(frozen=True)
class NullsPendingCommand:
    seq: int
    deck_idx: int
    card_name: str
    card_scid: int
    cost: int
    is_ability: bool
    timeout_tick: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "deck_idx": self.deck_idx,
            "card_name": self.card_name,
            "card_scid": self.card_scid,
            "cost": self.cost,
            "is_ability": self.is_ability,
            "timeout_tick": self.timeout_tick,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NullsPendingCommand:
        return cls(
            seq=int(d["seq"]),
            deck_idx=int(d["deck_idx"]),
            card_name=str(d["card_name"]),
            card_scid=int(d["card_scid"]),
            cost=int(d["cost"]),
            is_ability=bool(d["is_ability"]),
            timeout_tick=int(d["timeout_tick"]),
        )


@dataclass(frozen=True)
class NullsAbilityState:
    slot: int
    deck_idx: int
    champion_name: str
    champion_scid: int
    unit_eid: int | None
    state: int
    cooldown_ms: int
    charges: int
    command_ms: int
    pending_seq: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "deck_idx": self.deck_idx,
            "champion_name": self.champion_name,
            "champion_scid": self.champion_scid,
            "unit_eid": self.unit_eid,
            "state": self.state,
            "cooldown_ms": self.cooldown_ms,
            "charges": self.charges,
            "command_ms": self.command_ms,
            "pending_seq": self.pending_seq,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NullsAbilityState:
        return cls(
            slot=int(d["slot"]),
            deck_idx=int(d["deck_idx"]),
            champion_name=str(d["champion_name"]),
            champion_scid=int(d["champion_scid"]),
            unit_eid=None if d.get("unit_eid") is None else int(d["unit_eid"]),
            state=int(d["state"]),
            cooldown_ms=int(d["cooldown_ms"]),
            charges=int(d["charges"]),
            command_ms=int(d["command_ms"]),
            pending_seq=int(d["pending_seq"]),
        )


@dataclass(frozen=True)
class NullsSummoner:
    seat: int
    acc_id: tuple[int, int]
    elixir: int
    played_count: int
    hand: tuple[int, int, int, int]
    queue: tuple[int, ...]
    seen_scids: tuple[int, ...]
    deck: tuple[NullsCard, ...]
    hand_refill_ms: int | None = None
    elixir_reserved: int | None = None
    pending_commands: tuple[NullsPendingCommand, ...] | None = None
    abilities: tuple[NullsAbilityState, ...] | None = None
    seen_evo: tuple[int, ...] | None = None
    seen_evo_need: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "acc_id": list(self.acc_id),
            "elixir": self.elixir,
            "played_count": self.played_count,
            "hand": list(self.hand),
            "queue": list(self.queue),
            "seen_scids": list(self.seen_scids),
            "deck": [c.to_dict() for c in self.deck],
            "hand_refill_ms": self.hand_refill_ms,
            "elixir_reserved": self.elixir_reserved,
            "pending_commands": None if self.pending_commands is None else [p.to_dict() for p in self.pending_commands],
            "abilities": None if self.abilities is None else [a.to_dict() for a in self.abilities],
            "seen_evo": None if self.seen_evo is None else list(self.seen_evo),
            "seen_evo_need": None if self.seen_evo_need is None else list(self.seen_evo_need),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NullsSummoner:
        raw_pc = d.get("pending_commands")
        raw_ab = d.get("abilities")
        raw_se = d.get("seen_evo")
        raw_sen = d.get("seen_evo_need")
        return cls(
            seat=int(d["seat"]),
            acc_id=(int(d["acc_id"][0]), int(d["acc_id"][1])),
            elixir=int(d["elixir"]),
            played_count=int(d["played_count"]),
            hand=tuple(int(v) for v in d["hand"]),  # type: ignore[arg-type]
            queue=tuple(int(v) for v in d["queue"]),
            seen_scids=tuple(int(v) for v in d["seen_scids"]),
            deck=tuple(NullsCard.from_dict(c) for c in d["deck"]),
            hand_refill_ms=None if d.get("hand_refill_ms") is None else int(d["hand_refill_ms"]),
            elixir_reserved=None if d.get("elixir_reserved") is None else int(d["elixir_reserved"]),
            pending_commands=None if raw_pc is None else tuple(NullsPendingCommand.from_dict(p) for p in raw_pc),
            abilities=None if raw_ab is None else tuple(NullsAbilityState.from_dict(a) for a in raw_ab),
            seen_evo=None if raw_se is None else tuple(int(v) for v in raw_se),
            seen_evo_need=None if raw_sen is None else tuple(int(v) for v in raw_sen),
        )


@dataclass(frozen=True)
class NullsEntity:
    eid: int
    vt: int
    name: str
    scid: int
    side: int
    x: int
    y: int
    z: int
    level_raw: int | None
    hp: int | None
    max_hp: int | None
    shield: int | None
    max_shield: int | None
    target_eid: int | None
    buffs: tuple[NullsBuff, ...] | None
    state: int | None = None
    deploy_ms: int | None = None
    attack_ms: int | None = None
    delay_ms: int | None = None
    load_ms: int | None = None
    speed_percent: int | None = None
    hit_speed_percent: int | None = None
    tags1: int | None = None
    tags2: int | None = None
    is_clone: bool | None = None
    is_underground: bool | None = None
    is_invisible: bool | None = None
    aim_x: int | None = None
    aim_y: int | None = None
    time_left_ms: int | None = None
    radius: int | None = None
    ability_ticks: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eid": self.eid,
            "vt": self.vt,
            "name": self.name,
            "scid": self.scid,
            "side": self.side,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "level_raw": self.level_raw,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "shield": self.shield,
            "max_shield": self.max_shield,
            "target_eid": self.target_eid,
            "state": self.state,
            "deploy_ms": self.deploy_ms,
            "attack_ms": self.attack_ms,
            "delay_ms": self.delay_ms,
            "load_ms": self.load_ms,
            "speed_percent": self.speed_percent,
            "hit_speed_percent": self.hit_speed_percent,
            "tags1": self.tags1,
            "tags2": self.tags2,
            "is_clone": self.is_clone,
            "is_underground": self.is_underground,
            "is_invisible": self.is_invisible,
            "aim_x": self.aim_x,
            "aim_y": self.aim_y,
            "time_left_ms": self.time_left_ms,
            "radius": self.radius,
            "ability_ticks": self.ability_ticks,
            "buffs": None if self.buffs is None else [b.to_dict() for b in self.buffs],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NullsEntity:
        raw_buffs = d.get("buffs")
        def _opt_int(k: str) -> int | None:
            v = d.get(k)
            return None if v is None else int(v)
        def _opt_bool(k: str) -> bool | None:
            v = d.get(k)
            return None if v is None else bool(v)
        return cls(
            eid=int(d["eid"]),
            vt=int(d["vt"]),
            name=str(d["name"]),
            scid=int(d["scid"]),
            side=int(d["side"]),
            x=int(d["x"]),
            y=int(d["y"]),
            z=int(d["z"]),
            level_raw=_opt_int("level_raw"),
            hp=_opt_int("hp"),
            max_hp=_opt_int("max_hp"),
            shield=_opt_int("shield"),
            max_shield=_opt_int("max_shield"),
            target_eid=_opt_int("target_eid"),
            buffs=None if raw_buffs is None else tuple(NullsBuff.from_dict(b) for b in raw_buffs),
            state=_opt_int("state"),
            deploy_ms=_opt_int("deploy_ms"),
            attack_ms=_opt_int("attack_ms"),
            delay_ms=_opt_int("delay_ms"),
            load_ms=_opt_int("load_ms"),
            speed_percent=_opt_int("speed_percent"),
            hit_speed_percent=_opt_int("hit_speed_percent"),
            tags1=_opt_int("tags1"),
            tags2=_opt_int("tags2"),
            is_clone=_opt_bool("is_clone"),
            is_underground=_opt_bool("is_underground"),
            is_invisible=_opt_bool("is_invisible"),
            aim_x=_opt_int("aim_x"),
            aim_y=_opt_int("aim_y"),
            time_left_ms=_opt_int("time_left_ms"),
            radius=_opt_int("radius"),
            ability_ticks=_opt_int("ability_ticks"),
        )


@dataclass(frozen=True)
class NullsBattleSnapshot:
    mode: int
    battle_substate: int
    tick: int
    time_ms: int
    sec_elapsed_ms: int
    sec_len_ms: int
    phase: int
    clock_running: bool
    clock_ended: bool
    own_seat: int | None
    local_acc: tuple[int, int]
    summoners: tuple[NullsSummoner, NullsSummoner]
    entities: tuple[NullsEntity, ...]
    bytes_read: int
    monotonic_ts: float
    section: int | None = None
    section_type: int | None = None
    elixir_interval_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "battle_substate": self.battle_substate,
            "tick": self.tick,
            "time_ms": self.time_ms,
            "sec_elapsed_ms": self.sec_elapsed_ms,
            "sec_len_ms": self.sec_len_ms,
            "phase": self.phase,
            "clock_running": self.clock_running,
            "clock_ended": self.clock_ended,
            "own_seat": self.own_seat,
            "local_acc": list(self.local_acc),
            "summoners": [s.to_dict() for s in self.summoners],
            "entities": [e.to_dict() for e in self.entities],
            "bytes_read": self.bytes_read,
            "monotonic_ts": self.monotonic_ts,
            "section": self.section,
            "section_type": self.section_type,
            "elixir_interval_ms": self.elixir_interval_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NullsBattleSnapshot:
        return cls(
            mode=int(d["mode"]),
            battle_substate=int(d["battle_substate"]),
            tick=int(d["tick"]),
            time_ms=int(d["time_ms"]),
            sec_elapsed_ms=int(d["sec_elapsed_ms"]),
            sec_len_ms=int(d["sec_len_ms"]),
            phase=int(d["phase"]),
            clock_running=bool(d["clock_running"]),
            clock_ended=bool(d["clock_ended"]),
            own_seat=None if d.get("own_seat") is None else int(d["own_seat"]),
            local_acc=(int(d["local_acc"][0]), int(d["local_acc"][1])),
            summoners=(
                NullsSummoner.from_dict(d["summoners"][0]),
                NullsSummoner.from_dict(d["summoners"][1]),
            ),
            entities=tuple(NullsEntity.from_dict(e) for e in d["entities"]),
            bytes_read=int(d.get("bytes_read", 0)),
            monotonic_ts=float(d.get("monotonic_ts", 0.0)),
            section=None if d.get("section") is None else int(d["section"]),
            section_type=None if d.get("section_type") is None else int(d["section_type"]),
            elixir_interval_ms=None if d.get("elixir_interval_ms") is None else int(d["elixir_interval_ms"]),
        )


def resolve_pid_and_libg_base(
    serial: str,
    package: str = DEFAULT_PACKAGE,
    *,
    adb: str = "adb",
) -> tuple[str, int]:
    """Query the target process PID and `libg.so` base load address via ADB."""
    pid_raw = subprocess.check_output([adb, "-s", serial, "shell", f"pidof {package}"], timeout=5)
    pids = pid_raw.decode().strip().split()
    if not pids:
        raise NotInBattleError(f"Package {package!r} is not running on {serial}")
    pid = pids[0]
    maps_raw = subprocess.check_output(
        [adb, "-s", serial, "shell", f"grep libg.so /proc/{pid}/maps"], timeout=5
    ).decode()
    lines = [ln for ln in maps_raw.splitlines() if "libg.so" in ln]
    if not lines:
        raise NotInBattleError(f"libg.so not mapped in pid {pid}")
    base = int(lines[0].split("-")[0], 16)
    return pid, base


class NullsMemoryReader:
    """Page-batched read-only state reader for Nulls Royale `16.635.1`."""

    def __init__(
        self,
        backend: MemoryBackend,
        libg_base: int,
        *,
        clock=time.monotonic,
        require_running_clock: bool = True,
        cache_metadata: bool = True,
    ) -> None:
        self.mem = backend
        self.base = int(libg_base)
        self.clock = clock
        self.require_running_clock = bool(require_running_clock)
        self.cache_metadata = cache_metadata
        self._row_metadata: dict[int, tuple[str, int]] = {}
        self._card_costs: dict[int, int] = {}
        self._buff_meta: dict[int, tuple[int, int, bool]] = {}
        self._aoe_meta: dict[int, tuple[int, int, int]] = {}
        self._metadata_match: tuple[int, int] | None = None
        self._metadata_tick = -1

    def _clear_metadata_caches(self) -> None:
        self._row_metadata.clear()
        self._card_costs.clear()
        self._buff_meta.clear()
        self._aoe_meta.clear()

    def _prefetch_structs(self, addrs: list[int], size: int) -> None:
        """Prefetch start and (when page-spanning) end pages for fixed-size structs."""
        pages: list[int] = []
        for addr in addrs:
            if not is_ptr(addr):
                continue
            pages.append(addr)
            if (addr & 0xFFF) + size > 4096:
                pages.append(addr + size - 1)
        if pages:
            self.mem.prefetch_pages(pages)

    def _metadata(self, dp: int) -> tuple[str, int]:
        if not is_ptr(dp):
            return "", 0
        key = dp & _PTR_MASK
        if key in self._row_metadata:
            return self._row_metadata[key]
        name = _read_logic_name(self.mem, dp)
        scid = _i32(self.mem, dp + DP_SCID_OFF)
        if self.cache_metadata and name and scid > 0:
            self._row_metadata[key] = (name, scid)
        return name, scid

    def _cost(self, dp: int) -> int:
        if not is_ptr(dp):
            return 0
        key = dp & _PTR_MASK
        if key not in self._card_costs:
            value = _i32(self.mem, dp + DP_COST_OFF)
            if not self.cache_metadata:
                return value
            self._card_costs[key] = value
        return self._card_costs[key]

    def _buff_meta_for(self, bdp: int) -> tuple[int, int, bool, bool]:
        """Return `(hit_speed_mult, speed_mult, invisible, valid)` for a BuffData ptr."""
        if not is_ptr(bdp):
            return 0, 0, False, False
        key = bdp & _PTR_MASK
        if key in self._buff_meta:
            hm, sm, inv = self._buff_meta[key]
            return hm, sm, inv, True
        raw = self.mem.read(bdp, 0x110)
        if len(raw) < 0x110:
            return 0, 0, False, False
        hm, sm = struct.unpack_from("<ii", raw, BUFF_DP_HIT_SPEED_MULT_OFF)
        inv = bool(raw[BUFF_DP_INVISIBLE_OFF])
        if self.cache_metadata:
            self._buff_meta[key] = (hm, sm, inv)
        return hm, sm, inv, True

    def _aoe_meta_for(self, adp: int) -> tuple[int, int, int, bool]:
        """Return `(base_radius, max_radius, life_ms, valid)` for an AreaEffectData ptr."""
        if not is_ptr(adp):
            return 0, 0, 0, False
        key = adp & _PTR_MASK
        if key in self._aoe_meta:
            br, mr, lm = self._aoe_meta[key]
            return br, mr, lm, True
        raw = self.mem.read(adp, 0x154)
        if len(raw) < 0x154:
            return 0, 0, 0, False
        max_r = struct.unpack_from("<i", raw, AOE_DATA_MAX_RADIUS_OFF)[0]
        life_ms, base_r = struct.unpack_from("<ii", raw, AOE_DATA_LIFE_OFF)
        if self.cache_metadata:
            self._aoe_meta[key] = (base_r, max_r, life_ms)
        return base_r, max_r, life_ms, True

    @classmethod
    def connect(
        cls,
        serial: str,
        *,
        package: str = DEFAULT_PACKAGE,
        adb: str = "adb",
        max_poll_bytes: int = DEFAULT_MAX_POLL_BYTES,
        require_running_clock: bool = True,
    ) -> NullsMemoryReader:
        pid, base = resolve_pid_and_libg_base(serial, package, adb=adb)
        backend = BoundedAdbMemReader(serial, pid, adb=adb, max_poll_bytes=max_poll_bytes)
        return cls(backend, base, require_running_clock=require_running_clock)

    def read_snapshot(self) -> NullsBattleSnapshot:
        """Read one coherent battle snapshot or raise `NotInBattleError`."""
        self.mem.begin_poll()
        self.mem.prefetch_pages([self.base + G_STATE, self.base + G_LOCAL_ACC])

        st = _u64(self.mem, self.base + G_STATE)
        acc_obj = _u64(self.mem, self.base + G_LOCAL_ACC)
        if not is_ptr(st):
            raise NotInBattleError("Global game state pointer is null")

        self.mem.prefetch_pages([st, acc_obj])
        mode = _i32(self.mem, st + ST_MODE_OFF)
        gm = _u64(self.mem, st + ST_GM_OFF)
        local_acc = (0, 0)
        if is_ptr(acc_obj):
            acc_raw = self.mem.read(acc_obj + 0x28, 8)
            if len(acc_raw) == 8:
                local_acc = struct.unpack("<ii", acc_raw)

        if mode != 4 or not is_ptr(gm):
            raise NotInBattleError(f"Client is not in battle (mode={mode})")

        self.mem.prefetch_pages([gm])
        ctx = _u64(self.mem, gm + GM_CTX_OFF)
        if not is_ptr(ctx):
            raise NotInBattleError("Game context pointer is null")

        self.mem.prefetch_pages([ctx])
        v7 = _u64(self.mem, ctx + CTX_V7_OFF)
        lb = _u64(self.mem, ctx + CTX_LB_OFF)
        if not is_ptr(v7) or not is_ptr(lb):
            raise NotInBattleError("LogicBattle pointer is null")

        self.mem.prefetch_pages([v7, lb])
        substate = _i32(self.mem, lb + LB_SUBSTATE_OFF)
        p_ptrs = [_u64(self.mem, lb + off) for off in LB_PLAYER_OFFS]
        d_raw = self.mem.read(lb + LB_DECKS_OFF, 16)
        s_raw = self.mem.read(lb + LB_SUMMONERS_OFF, 16)
        if len(d_raw) != 16 or len(s_raw) != 16:
            raise NotInBattleError("Incomplete LogicBattle summoner/deck header")
        d0, d1 = struct.unpack("<QQ", d_raw)
        s0, s1 = struct.unpack("<QQ", s_raw)
        clk_ptr = _u64(self.mem, lb + LB_CLOCK_OFF)
        b1b0 = _u64(self.mem, lb + LB_END_WRAP_OFF)

        arr_v7 = _u64(self.mem, v7 + V7_ARR_OFF)
        cnt_raw = self.mem.read(v7 + V7_CNT_OFF - 4, 8)
        _, ent_cnt = struct.unpack("<ii", cnt_raw) if len(cnt_raw) == 8 else (0, 0)

        if not (is_ptr(s0) and is_ptr(s1) and is_ptr(d0) and is_ptr(d1) and is_ptr(clk_ptr)):
            raise NotInBattleError("Battle summoners, decks, or clock not initialized")
        if ent_cnt <= 0 or ent_cnt > 400 or not is_ptr(arr_v7):
            raise NotInBattleError(f"Entity list not active (cnt={ent_cnt})")

        # Batch fetch battle control structs + summoner + deck + entity pointer arrays
        batch2 = [clk_ptr, s0, s1, d0, d1, arr_v7, b1b0, *p_ptrs]
        self._prefetch_structs([s0, s1], SUMMONER_READ_SIZE)
        self.mem.prefetch_pages(batch2)

        clk_raw = self.mem.read(clk_ptr, 0x60)
        if len(clk_raw) < 0x40:
            raise NotInBattleError("Failed to read BattleClock")
        tick = struct.unpack_from("<i", clk_raw, CLK_TICK_OFF)[0]
        time_ms = struct.unpack_from("<i", clk_raw, CLK_TIME_MS_OFF)[0]
        clk_flags = struct.unpack_from("<I", clk_raw, CLK_FLAGS_OFF)[0]
        section = struct.unpack_from("<i", clk_raw, CLK_SECTION_OFF)[0]
        sec_el = struct.unpack_from("<i", clk_raw, CLK_SEC_EL_OFF)[0]
        sec_len = struct.unpack_from("<i", clk_raw, CLK_SEC_LEN_OFF)[0]
        section_type = struct.unpack_from("<i", clk_raw, CLK_SEC_TYPE_OFF)[0]
        phase = struct.unpack_from("<i", clk_raw, CLK_RATE_IDX_OFF)[0]
        elixir_interval_ms: int | None = None
        if len(clk_raw) >= CLK_FULL_BAR_DIV_OFF + 4:
            raw_40, raw_44 = struct.unpack_from("<ii", clk_raw, CLK_FULL_BAR_X10_OFF)
            if raw_40 > 0 and raw_44 > 0:
                elixir_interval_ms = raw_40 // raw_44

        if self._metadata_match != (gm, lb) or tick < self._metadata_tick:
            self._clear_metadata_caches()
        self._metadata_match = (gm, lb)
        self._metadata_tick = tick
        clock_ended = ((clk_flags >> 8) & 0xFF) != 0
        clock_running = ((clk_flags >> 16) & 0xFF) != 0

        obj18 = _u64(self.mem, b1b0 + 0x18) if is_ptr(b1b0) else 0
        if is_ptr(obj18):
            self.mem.prefetch_pages([obj18])
            end_74 = _i32(self.mem, obj18 + OBJ18_STATE_OFF)
            clk_110 = _i32(self.mem, obj18 + OBJ18_CLK_FLAG_OFF)
            if end_74 in (1, 2) or clk_110 != 0:
                clock_ended = True

        if self.require_running_clock and clock_ended:
            raise NotInBattleError("Battle has ended (post-match overlay)")

        own_seat: int | None = None
        player_accs: list[tuple[int, int]] = [(0, 0), (0, 0)]
        for seat_i, pp in enumerate(p_ptrs):
            if is_ptr(pp):
                acc_bytes = self.mem.read(pp, 8)
                if len(acc_bytes) == 8:
                    acc_val = struct.unpack("<ii", acc_bytes)
                    player_accs[seat_i] = acc_val
                    if local_acc != (0, 0) and acc_val == local_acc:
                        own_seat = seat_i

        # Prefetch summoner arrays, pending queues, champion slots, deck arrays, and entity pointers
        s_raws: list[bytes] = []
        d_arr_ptrs: list[int] = []
        hand_ptrs: list[int] = []
        queue_ptrs: list[int] = []
        pq_ptrs: list[int] = []
        champ_ptrs: list[int] = []
        for sp, dp in ((s0, d0), (s1, d1)):
            sr = self.mem.read(sp, SUMMONER_READ_SIZE)
            s_raws.append(sr)
            hand_ptrs.append(struct.unpack_from("<Q", sr, SUMMONER_HAND_PTR_OFF)[0])
            queue_ptrs.append(struct.unpack_from("<Q", sr, SUMMONER_QUEUE_PTR_OFF)[0])
            pq_p = struct.unpack_from("<Q", sr, SUMMONER_PENDING_PTR_OFF)[0]
            pq_ptrs.append(pq_p)
            if len(sr) >= SUMMONER_CHAMP_OFFS[1] + 8:
                champ_ptrs.extend(struct.unpack_from("<QQ", sr, SUMMONER_CHAMP_OFFS[0]))
            else:
                champ_ptrs.extend((0, 0))
            d_arr_ptrs.append(_u64(self.mem, dp + DECK_CARDS_ARR_OFF))

        self.mem.prefetch_pages([*hand_ptrs, *queue_ptrs, *pq_ptrs, *d_arr_ptrs])
        self._prefetch_structs(champ_ptrs, CHAMP_READ_SIZE)

        card_ptrs_per_seat: list[tuple[int, ...]] = []
        for dap in d_arr_ptrs:
            if is_ptr(dap):
                cp_raw = self.mem.read(dap, 64)
                card_ptrs_per_seat.append(struct.unpack("<8Q", cp_raw) if len(cp_raw) == 64 else ())
            else:
                card_ptrs_per_seat.append(())

        # Read PendingQueue headers & ChampionController structs in bulk
        pq_headers: dict[int, list[tuple[int, int]]] = {}
        pq_vec_ptrs: list[int] = []
        for pq_p in pq_ptrs:
            if is_ptr(pq_p):
                r584 = self.mem.read(pq_p, 48)
                if len(r584) == 48:
                    vecs: list[tuple[int, int]] = []
                    for a_off, c_off in ((PQ_ARR0_OFF, PQ_CNT0_OFF), (PQ_ARR1_OFF, PQ_CNT1_OFF)):
                        arr_p = struct.unpack_from("<Q", r584, a_off)[0]
                        cnt_v = struct.unpack_from("<i", r584, c_off)[0]
                        if 0 < cnt_v <= 16 and is_ptr(arr_p):
                            vecs.append((arr_p, cnt_v))
                            pq_vec_ptrs.append(arr_p)
                        elif cnt_v != 0:
                            # Corrupted / invalid count -> skip whole queue to avoid bogus 0
                            vecs = []
                            break
                    else:
                        pq_headers[pq_p] = vecs

        champ_raws: dict[int, bytes] = {}
        champ_vec_ptrs: list[int] = []
        for cp_p in champ_ptrs:
            if is_ptr(cp_p):
                craw = self.mem.read(cp_p, CHAMP_READ_SIZE)
                if len(craw) == CHAMP_READ_SIZE:
                    cvt = (struct.unpack_from("<Q", craw, 0)[0] & _PTR_MASK) - self.base
                    if cvt == VT_CHAMPION_CTRL:
                        champ_raws[cp_p] = craw
                        vec_p = struct.unpack_from("<Q", craw, CHAMP_UNITS_VEC_OFF)[0]
                        if is_ptr(vec_p):
                            champ_vec_ptrs.append(vec_p)

        eptrs_raw = self.mem.read(arr_v7, ent_cnt * 8)
        eptrs = list(struct.unpack(f"<{len(eptrs_raw) // 8}Q", eptrs_raw))
        all_card_ptrs = [cp for cps in card_ptrs_per_seat for cp in cps if is_ptr(cp)]
        self.mem.prefetch_pages([*eptrs, *pq_vec_ptrs, *champ_vec_ptrs])
        self._prefetch_structs(all_card_ptrs, 0x38)

        # Unpack PendingQueue item pointers & ChampionController unit arrays
        pq_item_ptrs: dict[int, list[int]] = {}
        all_pq_items: list[int] = []
        for pq_p, vecs in pq_headers.items():
            items: list[int] = []
            vec_ok = True
            for arr_p, cnt_v in vecs:
                raw_arr = self.mem.read(arr_p, cnt_v * 8)
                if len(raw_arr) != cnt_v * 8:
                    vec_ok = False
                    break
                ptrs = list(struct.unpack(f"<{cnt_v}Q", raw_arr))
                if not all(is_ptr(p) for p in ptrs):
                    vec_ok = False
                    break
                items.extend(ptrs)
            if vec_ok:
                pq_item_ptrs[pq_p] = items
                all_pq_items.extend(items)

        champ_unit_ptrs_arr: list[int] = []
        champ_unit_vec_info: dict[int, tuple[int, int]] = {}
        for vec_p in champ_vec_ptrs:
            vraw = self.mem.read(vec_p, 16)
            if len(vraw) == 16:
                uarr = struct.unpack_from("<Q", vraw, 0)[0]
                ucnt = struct.unpack_from("<i", vraw, 12)[0]
                if 0 < ucnt <= 16 and is_ptr(uarr):
                    champ_unit_vec_info[vec_p] = (uarr, ucnt)
                    champ_unit_ptrs_arr.append(uarr)

        if all_pq_items or champ_unit_ptrs_arr:
            self.mem.prefetch_pages([*all_pq_items, *champ_unit_ptrs_arr])

        # Read card objects (0x38 bytes: dp, lvl, slot_flags, star_lvl, evo_cnt, hero_cnt)
        card_records: list[tuple[int, int, int, int, int]] = []
        card_dps: list[int] = []
        for cp in all_card_ptrs:
            craw = self.mem.read(cp, 0x38)
            cdp = struct.unpack_from("<Q", craw, CARD_DP_OFF)[0] if len(craw) >= 24 else 0
            clv = struct.unpack_from("<i", craw, CARD_LVL_OFF)[0] if len(craw) >= 28 else -1
            csf = struct.unpack_from("<i", craw, CARD_SLOT_FLAGS_OFF)[0] if len(craw) >= 32 else -1
            cevo = struct.unpack_from("<i", craw, CARD_EVO_CNT_OFF)[0] if len(craw) >= 52 else -1
            chero = struct.unpack_from("<i", craw, CARD_HERO_CNT_OFF)[0] if len(craw) >= 56 else -1
            card_dps.append(cdp)
            card_records.append((cdp, clv, csf, cevo, chero))

        # Read entity headers (0x128 bytes safe for every kind), collect extra tails + component ptrs
        self._prefetch_structs(eptrs, 0x128)
        ent_headers: list[tuple[int, bytes, int]] = []
        char_tails: list[int] = []
        proj_tails: list[int] = []
        sub18_ptrs: list[int] = []
        ent_dp_ptrs: list[int] = []
        aoe_dp_ptrs: list[int] = []
        for ep in eptrs:
            if not is_ptr(ep):
                continue
            eraw = self.mem.read(ep, 0x128)
            if len(eraw) < 0x124:
                continue
            vt = (struct.unpack_from("<Q", eraw, ENT_VT_OFF)[0] & _PTR_MASK) - self.base
            ent_headers.append((ep, eraw, vt))
            sub18 = struct.unpack_from("<Q", eraw, ENT_SUB18_OFF)[0]
            dp = struct.unpack_from("<Q", eraw, ENT_DP_OFF)[0]
            # Projectile/area objects must not be followed as character component arrays.
            if vt in (VT_CHARACTER, VT_KING_TOWER):
                if (ep & 0xFFF) + 0x1F0 > 4096:
                    char_tails.append(ep + 0x1EF)
                if is_ptr(sub18):
                    sub18_ptrs.append(sub18)
            elif vt == VT_PROJECTILE:
                if (ep & 0xFFF) + 0x1A0 > 4096:
                    proj_tails.append(ep + 0x19F)
            elif vt == VT_AREA_EFFECT and is_ptr(dp) and (dp & _PTR_MASK) not in self._aoe_meta:
                aoe_dp_ptrs.append(dp)
            if is_ptr(dp):
                ent_dp_ptrs.append(dp)

        # Read pending items & collect their CardData ptrs and Champion dp ptrs
        pq_raw_items: dict[int, list[bytes]] = {}
        extra_dps: list[int] = []
        for pq_p, iptrs in pq_item_ptrs.items():
            rows: list[bytes] = []
            for ip in iptrs:
                ir = self.mem.read(ip, PQ_ITEM_SIZE)
                if len(ir) == PQ_ITEM_SIZE:
                    rows.append(ir)
                    cdp = struct.unpack_from("<Q", ir, PQ_ITEM_CARD_DP_OFF)[0]
                    if is_ptr(cdp):
                        extra_dps.append(cdp)
            pq_raw_items[pq_p] = rows

        for craw in champ_raws.values():
            cdp = struct.unpack_from("<Q", craw, CHAMP_DP_OFF)[0]
            if is_ptr(cdp):
                extra_dps.append(cdp)

        all_dp_ptrs = (*card_dps, *ent_dp_ptrs, *extra_dps)
        unknown_dps = [dp for dp in all_dp_ptrs if (dp & _PTR_MASK) not in self._row_metadata]
        unknown_costs = [dp for dp in card_dps if (dp & _PTR_MASK) not in self._card_costs]
        self.mem.prefetch_pages([*unknown_dps, *unknown_costs, *sub18_ptrs, *char_tails, *proj_tails])
        self._prefetch_structs(aoe_dp_ptrs, 0x154)

        ent_records: list[tuple[int, bytes, int]] = []
        for ep, eraw, vt in ent_headers:
            if vt in (VT_CHARACTER, VT_KING_TOWER):
                full = self.mem.read(ep, 0x1F0)
                if len(full) == 0x1F0:
                    eraw = full
            elif vt == VT_PROJECTILE:
                full = self.mem.read(ep, 0x1A0)
                if len(full) == 0x1A0:
                    eraw = full
            ent_records.append((ep, eraw, vt))

        # Collect heap string pointers for dp entries with length >= 8 & component pointers
        sptr_addrs: list[int] = []
        for dp in unknown_dps:
            if is_ptr(dp):
                raw_len = self.mem.read(dp + DP_STR_LEN_OFF, 16)
                if len(raw_len) == 16:
                    slen = struct.unpack_from("<i", raw_len, 0)[0]
                    if 8 <= slen <= 128:
                        sptr_addrs.append(struct.unpack_from("<Q", raw_len, 8)[0])

        comp_ptrs: list[int] = []
        sub18_map: dict[int, tuple[int, int, int]] = {}
        for sub18 in sub18_ptrs:
            s18raw = self.mem.read(sub18, 0x20)
            if len(s18raw) == 0x20:
                c_combat, _c_move, c_hp, c_buff = struct.unpack("<4Q", s18raw)
                sub18_map[sub18] = (c_combat, c_hp, c_buff)
                comp_ptrs.extend(p for p in (c_combat, c_hp, c_buff) if is_ptr(p))

        self.mem.prefetch_pages([*sptr_addrs, *comp_ptrs])

        # Build pointer -> entity ID lookup across all live entities
        ptr_to_eid = {
            ep & _PTR_MASK: struct.unpack_from("<i", eraw, ENT_ID_OFF)[0]
            for ep, eraw, _ in ent_records
        }

        # Decode decks & summoners
        card_idx = 0
        summoners: list[NullsSummoner] = []
        for seat_i in range(2):
            cps = card_ptrs_per_seat[seat_i]
            deck_cards: list[NullsCard] = []
            for cp in cps:
                if not is_ptr(cp):
                    continue
                cdp, clv, csf, cevo, chero = card_records[card_idx]
                card_idx += 1
                cname, cscid = self._metadata(cdp)
                ccost = self._cost(cdp)
                deck_cards.append(
                    NullsCard(
                        name=cname,
                        scid=cscid,
                        cost=ccost,
                        level_raw=clv,
                        slot_flags=None if csf < 0 else csf,
                        evolution_count=None if cevo < 0 else cevo,
                        hero_count=None if chero < 0 else chero,
                    )
                )

            sr = s_raws[seat_i]
            elix = struct.unpack_from("<i", sr, SUMMONER_ELIXIR_OFF)[0]
            played = struct.unpack_from("<i", sr, SUMMONER_PLAYED_OFF)[0]
            hand_refill_ms = struct.unpack_from("<i", sr, SUMMONER_HAND_REFILL_MS_OFF)[0]
            seen_scids = struct.unpack_from("<8i", sr, SUMMONER_SEEN_OFF)
            seen_evo = (
                struct.unpack_from("<8i", sr, SUMMONER_SEEN_EVO_OFF)
                if len(sr) >= SUMMONER_SEEN_EVO_OFF + 32
                else None
            )
            seen_evo_need = (
                struct.unpack_from("<8i", sr, SUMMONER_SEEN_EVO_NEED_OFF)
                if len(sr) >= SUMMONER_SEEN_EVO_NEED_OFF + 32
                else None
            )
            hp_p = hand_ptrs[seat_i]
            qp_p = queue_ptrs[seat_i]
            hand_raw = self.mem.read(hp_p, 16) if is_ptr(hp_p) else b""
            queue_raw = self.mem.read(qp_p, 32) if is_ptr(qp_p) else b""
            hand_vals = struct.unpack("<4i", hand_raw) if len(hand_raw) == 16 else (-1, -1, -1, -1)
            queue_vals = struct.unpack("<8i", queue_raw) if len(queue_raw) == 32 else ()

            pq_p = pq_ptrs[seat_i]
            pending_cmds: tuple[NullsPendingCommand, ...] | None = None
            elixir_reserved: int | None = None
            if is_ptr(pq_p) and pq_p in pq_item_ptrs:
                expected_cnt = sum(cnt_v for _, cnt_v in pq_headers[pq_p])
                raw_rows = pq_raw_items.get(pq_p, ())
                cmds: list[NullsPendingCommand] = []
                pq_valid = len(raw_rows) == expected_cnt
                if pq_valid:
                    for ir in raw_rows:
                        seq = struct.unpack_from("<i", ir, PQ_ITEM_SEQ_OFF)[0]
                        d_idx = struct.unpack_from("<i", ir, PQ_ITEM_DECK_IDX_OFF)[0]
                        cdp = struct.unpack_from("<Q", ir, PQ_ITEM_CARD_DP_OFF)[0]
                        cost_v = struct.unpack_from("<i", ir, PQ_ITEM_COST_OFF)[0]
                        ab_raw = ir[PQ_ITEM_IS_ABILITY_OFF]
                        t_tick = struct.unpack_from("<i", ir, PQ_ITEM_TIMEOUT_TICK_OFF)[0]
                        c_name, c_scid = self._metadata(cdp)
                        if not (
                            0 <= seq <= 100_000
                            and 0 <= d_idx < 8
                            and 0 <= cost_v <= 15
                            and ab_raw <= 1
                            and 0 <= t_tick <= 200_000
                            and bool(c_name)
                        ):
                            pq_valid = False
                            break
                        cmds.append(
                            NullsPendingCommand(
                                seq=seq,
                                deck_idx=d_idx,
                                card_name=c_name,
                                card_scid=c_scid,
                                cost=cost_v,
                                is_ability=bool(ab_raw),
                                timeout_tick=t_tick,
                            )
                        )
                if pq_valid:
                    pending_cmds = tuple(cmds)
                    elixir_reserved = sum(c.cost for c in cmds) * 10000

            abilities: tuple[NullsAbilityState, ...] | None = None
            seat_champs = champ_ptrs[seat_i * 2 : seat_i * 2 + 2]
            if any(cp in champ_raws for cp in seat_champs) or is_ptr(pq_p):
                ab_list: list[NullsAbilityState] = []
                for slot_i, cp_p in enumerate(seat_champs):
                    craw = champ_raws.get(cp_p)
                    if craw is None:
                        continue
                    cd_ms = struct.unpack_from("<i", craw, CHAMP_CD_OFF)[0]
                    chg = struct.unpack_from("<i", craw, CHAMP_CHARGES_OFF)[0]
                    d_idx = struct.unpack_from("<i", craw, CHAMP_DECK_IDX_OFF)[0]
                    cdp = struct.unpack_from("<Q", craw, CHAMP_DP_OFF)[0]
                    cst = struct.unpack_from("<i", craw, CHAMP_STATE_OFF)[0]
                    vec_p = struct.unpack_from("<Q", craw, CHAMP_UNITS_VEC_OFF)[0]
                    cmd_ms = struct.unpack_from("<i", craw, CHAMP_CMD_MS_OFF)[0]
                    p_seq = struct.unpack_from("<i", craw, CHAMP_PENDING_SEQ_OFF)[0]
                    ch_name, ch_scid = self._metadata(cdp)
                    unit_eid: int | None = None
                    if vec_p in champ_unit_vec_info:
                        uarr, ucnt = champ_unit_vec_info[vec_p]
                        u_raw = self.mem.read(uarr, ucnt * 8)
                        if len(u_raw) >= 8:
                            for up in struct.unpack(f"<{len(u_raw) // 8}Q", u_raw):
                                ukey = up & _PTR_MASK
                                if ukey in ptr_to_eid:
                                    unit_eid = ptr_to_eid[ukey]
                    ab_list.append(
                        NullsAbilityState(
                            slot=slot_i,
                            deck_idx=d_idx,
                            champion_name=ch_name,
                            champion_scid=ch_scid,
                            unit_eid=unit_eid,
                            state=cst,
                            cooldown_ms=cd_ms,
                            charges=chg,
                            command_ms=cmd_ms,
                            pending_seq=p_seq,
                        )
                    )
                abilities = tuple(ab_list)

            summoners.append(
                NullsSummoner(
                    seat=seat_i,
                    acc_id=player_accs[seat_i],
                    elixir=elix,
                    played_count=played,
                    hand=hand_vals,  # type: ignore[arg-type]
                    queue=queue_vals,
                    seen_scids=seen_scids,
                    deck=tuple(deck_cards),
                    hand_refill_ms=hand_refill_ms,
                    elixir_reserved=elixir_reserved,
                    pending_commands=pending_cmds,
                    abilities=abilities,
                    seen_evo=seen_evo,
                    seen_evo_need=seen_evo_need,
                )
            )

        # Decode active BuffComponent arrays & prefetch BuffData structs
        buff_comp_data: dict[int, tuple[int, int]] = {}
        buff_arr_ptrs: list[int] = []
        for _, _, c_buff in sub18_map.values():
            if is_ptr(c_buff):
                braw = self.mem.read(c_buff, 0x28)
                if len(braw) >= 0x28:
                    cvt = (struct.unpack_from("<Q", braw, 0)[0] & _PTR_MASK) - self.base
                    if cvt == VT_BUFF_COMP:
                        barr = struct.unpack_from("<Q", braw, BUFF_ARR_OFF)[0]
                        bcnt = struct.unpack_from("<i", braw, BUFF_CNT_OFF)[0]
                        buff_comp_data[c_buff] = (barr, bcnt)
                        if is_ptr(barr) and 0 < bcnt <= 32:
                            buff_arr_ptrs.append(barr)

        buff_inst_ptrs_by_comp: dict[int, tuple[int, ...]] = {}
        if buff_arr_ptrs:
            self.mem.prefetch_pages(buff_arr_ptrs)
            all_binst: list[int] = []
            for c_buff, (barr, bcnt) in buff_comp_data.items():
                if is_ptr(barr) and 0 < bcnt <= 32:
                    raw_arr = self.mem.read(barr, bcnt * 8)
                    ptrs = struct.unpack(f"<{len(raw_arr) // 8}Q", raw_arr)
                    buff_inst_ptrs_by_comp[c_buff] = ptrs
                    all_binst.extend(p for p in ptrs if is_ptr(p))
            if all_binst:
                self.mem.prefetch_pages(all_binst)
                bdp_ptrs = [_u64(self.mem, bp + BUFF_INST_DP_OFF) for bp in all_binst]
                unknown_bdps = [p for p in bdp_ptrs if is_ptr(p) and (p & _PTR_MASK) not in self._row_metadata]
                unknown_bmeta = [p for p in bdp_ptrs if is_ptr(p) and (p & _PTR_MASK) not in self._buff_meta]
                self.mem.prefetch_pages(unknown_bdps)
                self._prefetch_structs(unknown_bmeta, 0x110)
                bsptr_ptrs: list[int] = []
                for dp in unknown_bdps:
                    rl = self.mem.read(dp + DP_STR_LEN_OFF, 16)
                    if len(rl) == 16 and 8 <= struct.unpack_from("<i", rl, 0)[0] <= 128:
                        bsptr_ptrs.append(struct.unpack_from("<Q", rl, 8)[0])
                if bsptr_ptrs:
                    self.mem.prefetch_pages(bsptr_ptrs)

        entities: list[NullsEntity] = []
        for ep, eraw, vt in ent_records:
            eid = struct.unpack_from("<i", eraw, ENT_ID_OFF)[0]
            sub18 = struct.unpack_from("<Q", eraw, ENT_SUB18_OFF)[0]
            dp = struct.unpack_from("<Q", eraw, ENT_DP_OFF)[0]
            side = struct.unpack_from("<i", eraw, ENT_SIDE_OFF)[0]
            x, y = struct.unpack_from("<ii", eraw, ENT_X_OFF)
            z = struct.unpack_from("<i", eraw, ENT_Z_OFF)[0]
            is_clone = bool(eraw[ENT_IS_CLONE_OFF])
            tags1, tags2 = struct.unpack_from("<QQ", eraw, ENT_TAG1_OFF)
            name, scid = self._metadata(dp)

            hp_val: int | None = None
            max_hp_val: int | None = None
            shield_val: int | None = None
            max_shield_val: int | None = None
            target_eid: int | None = None
            buffs_val: tuple[NullsBuff, ...] | None = None
            state_val: int | None = None
            delay_val: int | None = None
            deploy_val: int | None = None
            attack_ms_val: int | None = None
            load_ms_val: int | None = None
            speed_pct_val: int | None = None
            hit_speed_pct_val: int | None = None
            is_underground: bool | None = None
            is_invisible: bool | None = None
            aim_x_val: int | None = None
            aim_y_val: int | None = None
            time_left_val: int | None = None
            radius_val: int | None = None
            ability_ticks_val: int | None = None
            lvl_raw: int | None = None

            if vt in (VT_CHARACTER, VT_KING_TOWER) and len(eraw) >= 0x1F0:
                lvl_raw = struct.unpack_from("<i", eraw, CHAR_LVL_OFF)[0]
                state_val = struct.unpack_from("<i", eraw, CHAR_STATE_OFF)[0]
                delay_val = struct.unpack_from("<i", eraw, CHAR_DELAY_MS_OFF)[0]
                deploy_val = struct.unpack_from("<i", eraw, CHAR_DEPLOY_MS_OFF)[0]
                is_underground = bool(eraw[CHAR_UNDERGROUND_OFF])
                char_invis = bool(eraw[CHAR_INVISIBLE_OFF])
                ability_ticks_val = struct.unpack_from("<i", eraw, CHAR_ABILITY_TICKS_OFF)[0]
                buff_invis = False

                if sub18 in sub18_map:
                    c_combat, c_hp, c_buff = sub18_map[sub18]
                    if is_ptr(c_hp):
                        hraw = self.mem.read(c_hp, 0x34)
                        if len(hraw) >= 0x34:
                            hvt = (struct.unpack_from("<Q", hraw, 0)[0] & _PTR_MASK) - self.base
                            if hvt == VT_HITPOINT_COMP:
                                hp_val = struct.unpack_from("<i", hraw, HP_VAL_OFF)[0]
                                max_hp_val = struct.unpack_from("<i", hraw, HP_MAX_OFF)[0]
                                shield_val = struct.unpack_from("<i", hraw, HP_SHIELD_OFF)[0]
                                max_shield_val = struct.unpack_from("<i", hraw, HP_MAX_SHIELD_OFF)[0]

                    if is_ptr(c_combat):
                        craw = self.mem.read(c_combat, 0x30)
                        if len(craw) >= 0x30:
                            cvt = (struct.unpack_from("<Q", craw, 0)[0] & _PTR_MASK) - self.base
                            if cvt == VT_COMBAT_COMP:
                                tptr = struct.unpack_from("<Q", craw, COMBAT_TARGET_PTR_OFF)[0] & _PTR_MASK
                                if tptr in ptr_to_eid:
                                    target_eid = ptr_to_eid[tptr]
                                attack_ms_val, load_ms_val = struct.unpack_from("<ii", craw, COMBAT_ATTACK_MS_OFF)
                    elif c_combat == 0:
                        attack_ms_val = 0
                        load_ms_val = 0

                    if c_buff in buff_comp_data:
                        _, bcnt = buff_comp_data[c_buff]
                        if bcnt == 0:
                            buffs_val = ()
                            speed_pct_val = 100
                            hit_speed_pct_val = 100
                        elif 0 < bcnt <= 32 and c_buff in buff_inst_ptrs_by_comp:
                            blist: list[NullsBuff] = []
                            sms: list[int] = []
                            hms: list[int] = []
                            meta_ok = True
                            for bp in buff_inst_ptrs_by_comp[c_buff]:
                                if not is_ptr(bp):
                                    meta_ok = False
                                    continue
                                iraw = self.mem.read(bp, 0x30)
                                if len(iraw) < 0x2C:
                                    meta_ok = False
                                    continue
                                bdp = struct.unpack_from("<Q", iraw, BUFF_INST_DP_OFF)[0]
                                btime = struct.unpack_from("<i", iraw, BUFF_INST_TIME_OFF)[0]
                                btotal = struct.unpack_from("<i", iraw, BUFF_INST_TOTAL_OFF)[0]
                                bname, bscid = self._metadata(bdp)
                                hm, sm, binv, bv = self._buff_meta_for(bdp)
                                if not bv:
                                    meta_ok = False
                                else:
                                    hms.append(hm)
                                    sms.append(sm)
                                    if binv:
                                        buff_invis = True
                                blist.append(
                                    NullsBuff(
                                        name=bname,
                                        scid=bscid,
                                        time_left=btime,
                                        total_ms=btotal,
                                        speed_mult=sm if bv else None,
                                        hit_speed_mult=hm if bv else None,
                                        invisible=binv if bv else None,
                                    )
                                )
                            buffs_val = tuple(blist)
                            if meta_ok and len(blist) == bcnt:
                                speed_pct_val = _buff_multiplier_percent(sms)
                                hit_speed_pct_val = _buff_multiplier_percent(hms)
                is_invisible = char_invis or buff_invis
            elif vt == VT_PROJECTILE and len(eraw) >= 0x1A0:
                aim_x_val, aim_y_val = struct.unpack_from("<ii", eraw, PROJ_AIM_X_OFF)
                lvl_raw = struct.unpack_from("<i", eraw, PROJ_LVL_OFF)[0]
            elif vt == VT_AREA_EFFECT and len(eraw) >= 0x118:
                lvl_raw = struct.unpack_from("<i", eraw, AOE_LVL_OFF)[0]
                t_raw = struct.unpack_from("<i", eraw, AOE_TIME_LEFT_OFF)[0]
                ovr = struct.unpack_from("<i", eraw, AOE_LIFE_OVR_OFF)[0]
                time_left_val = max(0, t_raw)
                base_r, max_r, d_life, a_ok = self._aoe_meta_for(dp)
                if a_ok:
                    life_eff = ovr if ovr >= 0 else d_life
                    radius_val = _aoe_radius_units(base_r, max_r, life_eff, t_raw)

            entities.append(
                NullsEntity(
                    eid=eid,
                    vt=vt,
                    name=name,
                    scid=scid,
                    side=side,
                    x=x,
                    y=y,
                    z=z,
                    level_raw=lvl_raw,
                    hp=hp_val,
                    max_hp=max_hp_val,
                    shield=shield_val,
                    max_shield=max_shield_val,
                    target_eid=target_eid,
                    buffs=buffs_val,
                    state=state_val,
                    deploy_ms=deploy_val,
                    attack_ms=attack_ms_val,
                    delay_ms=delay_val,
                    load_ms=load_ms_val,
                    speed_percent=speed_pct_val,
                    hit_speed_percent=hit_speed_pct_val,
                    tags1=tags1,
                    tags2=tags2,
                    is_clone=is_clone,
                    is_underground=is_underground,
                    is_invisible=is_invisible,
                    aim_x=aim_x_val,
                    aim_y=aim_y_val,
                    time_left_ms=time_left_val,
                    radius=radius_val,
                    ability_ticks=ability_ticks_val,
                )
            )

        return NullsBattleSnapshot(
            mode=mode,
            battle_substate=substate,
            tick=tick,
            time_ms=time_ms,
            sec_elapsed_ms=sec_el,
            sec_len_ms=sec_len,
            phase=phase,
            clock_running=clock_running,
            clock_ended=clock_ended,
            own_seat=own_seat,
            local_acc=local_acc,
            summoners=(summoners[0], summoners[1]),
            entities=tuple(entities),
            bytes_read=self.mem.poll_bytes_read,
            monotonic_ts=float(self.clock()),
            section=section,
            section_type=section_type,
            elixir_interval_ms=elixir_interval_ms,
        )
