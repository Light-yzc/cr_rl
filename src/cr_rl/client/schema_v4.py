"""Schema v4 observation, action, and vocabulary contract for crforge 16.635.3.

Constants and index definitions match ``org.crforge.env.Schema`` and
``org.crforge.env.Vocabulary`` in the vendored ``crforge`` simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


VOCAB_TABLES: tuple[str, ...] = (
    "spells_characters",
    "spells_other",
    "spells_buildings",
    "spells_evolved",
    "spells_hero_form",
    "characters",
    "buildings",
    "projectiles",
    "area_effect_objects",
)

SPELL_TABLES: tuple[str, ...] = (
    "spells_characters",
    "spells_other",
    "spells_buildings",
    "spells_evolved",
    "spells_hero_form",
)


class SchemaV4:
    """Flat integer layout for Schema v4 observations, actions, and masks."""

    VERSION = 4
    MASK_ALL_TILES = 1
    MASK_LANDING_TILES = 2
    MASK_VERSION = MASK_LANDING_TILES

    ID_BATTLE = 1
    ID_SEAT_FIRST_SIGHT = 2
    ID_VERSION = ID_SEAT_FIRST_SIGHT

    CELL_UNITS = 500
    TILE_UNITS = 1000
    COLS = 18
    ROWS = 32
    TILES = COLS * ROWS  # 576
    WIDTH_UNITS = COLS * TILE_UNITS  # 18000
    HEIGHT_UNITS = ROWS * TILE_UNITS  # 32000

    HAND_SLOTS = 4
    DECK_SIZE = 8
    WAIT = 0
    ABILITY_SLOTS = 2
    FIRST_ABILITY = 1 + HAND_SLOTS * TILES  # 2305
    ACTIONS = FIRST_ABILITY + ABILITY_SLOTS  # 2307

    PLAY_DELAY_TICKS = 20
    PENDING_ROWS = HAND_SLOTS
    PENDING_INTS = 4
    ENEMY_PLAY_ROWS = 8
    ENEMY_PLAY_INTS = 9
    QUEUE_SLOTS = 4
    ABILITY_INTS = 6
    BUFF_SLOTS = 3

    ELIXIR_SCALE = 10000
    MS_PER_TICK = 50
    PERCENT_SCALE = 100
    BUFF_FOREVER = -1

    # Header indices [0 .. 223]
    H_TICK = 0
    H_RESULT = 1
    H_ELIXIR = 2
    H_ELIXIR_RESERVED = 3
    H_CROWNS = 4
    H_CROWNS_ENEMY = 5
    H_HAND_REFILL_MS = 6
    H_HAND_CARD = 7
    H_HAND_COST = 11
    H_HAND_PENDING = 15
    H_NEXT_CARD = 19
    H_DECK = 20
    H_ENEMY_SEEN = 28
    H_TOWERS = 36
    H_ENTITY_COUNT = 48
    H_ENTITY_OVERFLOW = 49
    H_DECISION_TICKS = 50
    H_PLAY_DELAY_TICKS = 51
    H_SEAT = 52
    H_PENDING_COUNT = 53
    H_EPISODE_STEP = 54
    H_TIME_MS = 55
    H_SECTION = 56
    H_SECTION_TYPE = 57
    H_SECTION_LEFT_MS = 58
    H_FULL_BAR_MS = 59
    H_OVERTIME = 60
    H_PLAY_OPEN_IN_TICKS = 61
    H_LAST_REFUSAL = 62
    H_LAST_RUN_REFUSAL = 63
    H_ILLEGAL = 64
    H_PENDING = 65
    H_DECK_SLOTS = H_PENDING + PENDING_ROWS * PENDING_INTS  # 81
    H_DECK_EVOLUTION = H_DECK_SLOTS + DECK_SIZE  # 89
    H_HAND_FORM = H_DECK_EVOLUTION + DECK_SIZE  # 97
    H_ABILITY = H_HAND_FORM + HAND_SLOTS  # 101
    H_LAST_ABILITY_RUN_REFUSAL = H_ABILITY + ABILITY_SLOTS * ABILITY_INTS  # 113
    H_ENEMY_PLAY_COUNT = H_LAST_ABILITY_RUN_REFUSAL + 1  # 114
    H_ENEMY_ELIXIR = H_ENEMY_PLAY_COUNT + 1  # 115
    H_ENEMY_PLAYS = H_ENEMY_ELIXIR + 1  # 116
    H_QUEUE = H_ENEMY_PLAYS + ENEMY_PLAY_ROWS * ENEMY_PLAY_INTS  # 188
    H_DECK_LEVEL = H_QUEUE + QUEUE_SLOTS  # 192
    H_ENEMY_SEEN_EVOLUTION = H_DECK_LEVEL + DECK_SIZE  # 200
    H_ENEMY_SEEN_EVOLUTION_NEED = H_ENEMY_SEEN_EVOLUTION + DECK_SIZE  # 208
    HEADER_INTS = 224

    # Slot flags & play forms
    SLOT_EVOLUTION = 1
    SLOT_HERO = 1 << 1
    FORM_PLAIN = 0
    FORM_EVOLVED = 1
    FORM_HERO = 2

    # Enemy play kinds & flags
    PLAY_TROOP = 1
    PLAY_AREA = 2
    PLAY_LAUNCHED = 3
    PLAY_THROWN = 4
    PLAY_TUNNEL = 5

    PLAY_CARD_SHOWN = 1
    PLAY_POINT_SHOWN = 1 << 1
    PLAY_MIRROR = 1 << 2
    PLAY_EVOLUTION_DIAMOND = 1 << 3

    UNKNOWN = -1

    # Battle results
    RUNNING = 0
    WON = 1
    LOST = 2
    DRAW = 3

    # Timeline section types
    SECTION_NORMAL = 0
    SECTION_OVERTIME = 1
    SECTION_BONUS_TIME = 2

    # Refusals
    REFUSAL_NONE = 0
    REFUSAL_MASKED = 100
    REFUSAL_NOT_MODELLED = 101
    RUN_NOT_PLACED = 1000

    # Entity row indices [0 .. 33]
    E_ID = 0
    E_KIND = 1
    E_SIDE = 2
    E_ROW = 3
    E_X = 4
    E_Y = 5
    E_HP = 6
    E_MAX_HP = 7
    E_SHIELD = 8
    E_STATE = 9
    E_FLAGS = 10
    E_LEVEL = 11
    E_DX = 12
    E_DY = 13
    E_TARGET = 14
    E_TARGET_X = 15
    E_TARGET_Y = 16
    E_DEPLOY_MS = 17
    E_ATTACK_MS = 18
    E_LOAD_MS = 19
    E_SPEED_PERCENT = 20
    E_HIT_SPEED_PERCENT = 21
    E_BUFF_COUNT = 22
    E_BUFFS = 23
    E_AIM_X = E_BUFFS + 2 * BUFF_SLOTS  # 29
    E_AIM_Y = E_AIM_X + 1  # 30
    E_TIME_LEFT_MS = E_AIM_Y + 1  # 31
    E_RADIUS = E_TIME_LEFT_MS + 1  # 32
    E_ABILITY_TICKS = E_RADIUS + 1  # 33
    ENTITY_INTS = E_ABILITY_TICKS + 1  # 34

    NO_STATE = -1
    NO_TARGET = -1
    MAX_ENTITIES = 200
    SEAT_INTS = HEADER_INTS + MAX_ENTITIES * ENTITY_INTS  # 7024

    KIND_CHARACTER = 1
    KIND_TOWER = 2
    KIND_PROJECTILE = 3
    KIND_AREA = 4

    F_ALIVE = 1
    F_HIDDEN = 1 << 1
    F_INVISIBLE = 1 << 2
    F_UNDERGROUND = 1 << 3
    F_AIR = 1 << 4
    F_BUILDING = 1 << 5
    F_KING = 1 << 6
    F_CLONE = 1 << 7
    F_AWAKE = 1 << 8
    F_NO_MOVE = 1 << 9
    F_NO_ATTACK = 1 << 10
    F_CHARGED = 1 << 11

    STATE_NAMES: tuple[str, ...] = (
        "standing",
        "moving",
        "attacking",
        "dashing",
        "deploying",
        "jumping",
        "spawn_pathfind",
        "ingame_pathfind",
        "clone_setup",
        "morphing",
        "casting",
        "waiting_to_deploy",
        "following_removed",
        "following_removed_building",
        "components_disabled",
        "route_following_alternate",
        "ability_follow_up",
    )

    FLAG_NAMES: tuple[str, ...] = (
        "alive",
        "hidden",
        "invisible",
        "underground",
        "air",
        "building",
        "king",
        "clone",
        "awake",
        "no_move",
        "no_attack",
        "charged",
    )

    @classmethod
    def to_seat_x(cls, seat: int, x: int) -> int:
        """Transform battle X coordinate to seat frame (seat 0 mirrors width)."""
        return cls.WIDTH_UNITS - x if seat == 0 else x

    @classmethod
    def to_seat_y(cls, seat: int, y: int) -> int:
        """Transform battle Y coordinate to seat frame (seat 1 mirrors length)."""
        return cls.HEIGHT_UNITS - y if seat == 1 else y

    @classmethod
    def to_seat_dx(cls, seat: int, dx: int) -> int:
        return -dx if seat == 0 else dx

    @classmethod
    def to_seat_dy(cls, seat: int, dy: int) -> int:
        return -dy if seat == 1 else dy

    @classmethod
    def game_x(cls, seat: int, col: int) -> int:
        """Convert seat-frame tile column center to battle X in game units."""
        cx = col * cls.TILE_UNITS + cls.CELL_UNITS
        return cls.WIDTH_UNITS - cx if seat == 0 else cx

    @classmethod
    def game_y(cls, seat: int, row: int) -> int:
        """Convert seat-frame tile row center to battle Y in game units."""
        cy = row * cls.TILE_UNITS + cls.CELL_UNITS
        return cls.HEIGHT_UNITS - cy if seat == 1 else cy

    @classmethod
    def play(cls, slot: int, col: int, row: int) -> int:
        if not (0 <= slot < cls.HAND_SLOTS):
            raise ValueError(f"slot out of range: {slot}")
        if not (0 <= col < cls.COLS):
            raise ValueError(f"col out of range: {col}")
        if not (0 <= row < cls.ROWS):
            raise ValueError(f"row out of range: {row}")
        return 1 + slot * cls.TILES + row * cls.COLS + col

    @classmethod
    def play_tile(cls, slot: int, tile: int) -> int:
        if not (0 <= slot < cls.HAND_SLOTS):
            raise ValueError(f"slot out of range: {slot}")
        if not (0 <= tile < cls.TILES):
            raise ValueError(f"tile out of range: {tile}")
        return 1 + slot * cls.TILES + tile

    @classmethod
    def ability(cls, slot: int) -> int:
        if not (0 <= slot < cls.ABILITY_SLOTS):
            raise ValueError(f"ability slot out of range: {slot}")
        return cls.FIRST_ABILITY + slot

    @classmethod
    def decode_action(cls, action: int) -> dict[str, int | str]:
        """Decode an action integer into kind, slot, col, row, tile."""
        if action == cls.WAIT:
            return {"kind": "wait"}
        if cls.FIRST_ABILITY <= action < cls.ACTIONS:
            return {"kind": "ability", "slot": action - cls.FIRST_ABILITY}
        if 1 <= action < cls.FIRST_ABILITY:
            slot, tile = divmod(action - 1, cls.TILES)
            row, col = divmod(tile, cls.COLS)
            return {"kind": "play", "slot": slot, "col": col, "row": row, "tile": tile}
        raise ValueError(f"action out of range: {action}")


def _hash_names(names: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Vocabulary:
    """Sorted name-to-id tables matching ``org.crforge.env.Vocabulary``."""

    names: tuple[str, ...]
    buff_names: tuple[str, ...]
    name_to_id: dict[str, int]
    buff_to_id: dict[str, int]
    sha256_prefix: str
    buff_sha256_prefix: str
    relative_levels: dict[str, int]

    @classmethod
    def from_tables(cls, tables_dir: str | Path) -> Vocabulary:
        root = Path(tables_dir)
        row_names: set[str] = set()
        for tbl in VOCAB_TABLES:
            data = json.loads((root / f"{tbl}.json").read_text(encoding="utf-8"))
            row_names.update(data.get("rows", {}).keys())

        buff_names: set[str] = set()
        cb_data = json.loads((root / "character_buffs.json").read_text(encoding="utf-8"))
        buff_names.update(cb_data.get("rows", {}).keys())

        act_data = json.loads((root / "actions.json").read_text(encoding="utf-8"))
        for entry in act_data.get("actions", {}).values():
            if not isinstance(entry, dict):
                continue
            fields = entry.get("fields", entry)
            if not isinstance(fields, dict):
                continue
            if fields.get("SpawnType") == "BuffType":
                sd = fields.get("SpawnData")
                if isinstance(sd, dict) and isinstance(sd.get("Name"), str) and sd["Name"]:
                    buff_names.add(sd["Name"])

        sorted_names = tuple(sorted(row_names))
        sorted_buffs = tuple(sorted(buff_names))

        rel_levels: dict[str, int] = {}
        rarities_path = root / "rarities.json"
        if rarities_path.exists():
            rarities = json.loads(rarities_path.read_text(encoding="utf-8")).get("rows", {})
            for rname, rrow in rarities.items():
                cols = rrow.get("columns", {})
                rel = cols.get("RelativeLevel")
                rel_levels[rname] = int(rel) if isinstance(rel, (int, float)) else 0

        return cls(
            names=sorted_names,
            buff_names=sorted_buffs,
            name_to_id={name: i for i, name in enumerate(sorted_names)},
            buff_to_id={name: i for i, name in enumerate(sorted_buffs)},
            sha256_prefix=_hash_names(sorted_names),
            buff_sha256_prefix=_hash_names(sorted_buffs),
            relative_levels=rel_levels,
        )

    def row_id(self, name: str | None) -> int:
        if not name:
            return SchemaV4.UNKNOWN
        return self.name_to_id.get(name, SchemaV4.UNKNOWN)

    def buff_id(self, name: str | None) -> int:
        if not name:
            return SchemaV4.UNKNOWN
        return self.buff_to_id.get(name, SchemaV4.UNKNOWN)


def load_table_rows(tables_dir: str | Path, table_name: str) -> dict[str, dict[str, Any]]:
    """Load `{row_name: columns_dict}` for a single crforge JSON table."""
    path = Path(tables_dir) / f"{table_name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for name, row in data.get("rows", {}).items():
        cols = row.get("columns", {}) if isinstance(row, dict) else {}
        out[name] = cols
    return out
