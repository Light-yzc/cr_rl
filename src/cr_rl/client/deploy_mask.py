"""Exact port of crforge's DeployMask and PlacementSearch landing-tile logic.

Used both for building Schema v4 legal-action masks from live game state and
for offline unit verification against crforge 16.635.3 tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schema_v4 import SPELL_TABLES, SchemaV4, load_table_rows


def trunc_div(a: int, b: int) -> int:
    """C/Java integer division truncating toward zero."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


# Standard 1v1 arena routing cell bitmask (width=36, height=64; 500-unit cells).
# Bits: 0x20 = water, 0x10 = NOT_PLACEABLE (0x40 also non-placeable in MapCheck),
# 0x01/0x02 = pathing lane hints. Extracted directly from crforge TileMap.standard1v1().
STANDARD_1V1_BITS: tuple[int, ...] = (
    # fmt: off
    16,16,16,16,16,16,16,16,16,16,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,16,16,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,16,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,17,17,17,18,18,18,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,17,17,17,18,18,18,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,17,17,17,18,18,18,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,17,17,17,18,18,18,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,17,17,17,18,18,18,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,17,17,17,18,18,18,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    16,16,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,16,16,
    16,16,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,16,16,
    32,32,32,32,32,1,1,1,1,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,2,2,2,2,32,32,32,32,32,
    32,32,32,32,32,0,1,1,0,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,0,2,2,0,32,32,32,32,32,
    32,32,32,32,32,0,1,1,0,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,0,2,2,0,32,32,32,32,32,
    32,32,32,32,32,1,1,1,1,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,2,2,2,2,32,32,32,32,32,
    16,16,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,16,16,
    16,16,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,16,16,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,17,17,17,18,18,18,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,17,17,17,18,18,18,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,17,17,17,18,18,18,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,17,17,17,18,18,18,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,17,17,17,18,18,18,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,17,17,17,18,18,18,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    16,16,16,16,16,16,16,16,16,16,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,16,16,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,16,
    # fmt: on
)


@dataclass(frozen=True)
class UnitDeployInfo:
    name: str
    collision_radius: int
    tile_size_override: int
    building: bool
    king: bool
    summoner_tower: bool
    no_deploy_size_w: int
    no_deploy_size_h: int

    @property
    def footprint(self) -> int:
        if self.tile_size_override > 0:
            return self.tile_size_override
        return trunc_div(self.collision_radius + 499, 500) + 1


@dataclass(frozen=True)
class DeployCard:
    name: str
    mana_cost: int
    mirror: bool
    mana_from_summoner: bool
    spell: bool
    spell_as_deploy: bool
    can_deploy_on_enemy_side: bool
    can_place_on_buildings: bool
    can_place_on_water: bool
    full_lane_deploy: bool
    deploy_w_tile_margin: int
    deploy_start_y: int
    deploy_end_y: int
    unit: UnitDeployInfo | None
    search_unit: UnitDeployInfo | None

    @property
    def placement_unit(self) -> UnitDeployInfo | None:
        if self.spell or self.search_unit is not None:
            return self.search_unit
        return self.unit


@dataclass(frozen=True)
class MaskEntity:
    alive: bool
    side: int
    x: int
    y: int
    summoner_tower: bool
    summoner: bool
    building: bool
    no_deploy_size_w: int
    no_deploy_size_h: int
    footprint: int

    @property
    def team(self) -> int:
        return 2 if self.side == 100 else (self.side & 1)

    @classmethod
    def from_unit(cls, unit: UnitDeployInfo, side: int, x: int, y: int, alive: bool) -> MaskEntity:
        return cls(
            alive=alive,
            side=side,
            x=x,
            y=y,
            summoner_tower=unit.summoner_tower,
            summoner=unit.king,
            building=unit.building,
            no_deploy_size_w=unit.no_deploy_size_w,
            no_deploy_size_h=unit.no_deploy_size_h,
            footprint=unit.footprint,
        )


@dataclass(frozen=True)
class TileMap:
    width: int
    height: int
    bits: tuple[int, ...]

    WATER_BIT = 0x20
    NOT_PLACEABLE_BITS = 0x50

    @classmethod
    def standard_1v1(cls) -> TileMap:
        return cls(width=36, height=64, bits=STANDARD_1V1_BITS)

    def cell_ok(self, col: int, row: int, card: DeployCard) -> bool:
        b = self.bits[row * self.width + col] & 0x7F
        if b & self.WATER_BIT:
            return card.can_place_on_water
        return (b & self.NOT_PLACEABLE_BITS) == 0 or card.can_place_on_buildings


def _as_int(v: object, default: int = 0) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _as_str(v: object) -> str:
    return v if isinstance(v, str) else ""


class DeployCatalog:
    """Loaded card & unit deploy metadata from crforge 16.635.3 JSON tables."""

    def __init__(self, units: dict[str, UnitDeployInfo], cards: dict[str, DeployCard]) -> None:
        self.units = units
        self.cards = cards

    @classmethod
    def from_tables(cls, tables_dir: str | Path) -> DeployCatalog:
        root = Path(tables_dir)
        chars = load_table_rows(root, "characters")
        blds = load_table_rows(root, "buildings")
        projs = load_table_rows(root, "projectiles")
        aeos = load_table_rows(root, "area_effect_objects")

        units: dict[str, UnitDeployInfo] = {}
        for tbl_rows in (chars, blds):
            for name, c in tbl_rows.items():
                if name in units:
                    continue
                units[name] = UnitDeployInfo(
                    name=name,
                    collision_radius=_as_int(c.get("CollisionRadius")),
                    tile_size_override=_as_int(c.get("TileSizeOverride")),
                    building=bool(c.get("IsBuilding", False)),
                    king=bool(c.get("IsSummoner", False)),
                    summoner_tower=bool(c.get("IsSummonerTower", False)),
                    no_deploy_size_w=_as_int(c.get("NoDeploySizeW")),
                    no_deploy_size_h=_as_int(c.get("NoDeploySizeH")),
                )

        cards: dict[str, DeployCard] = {}
        for tbl_name in SPELL_TABLES:
            rows = load_table_rows(root, tbl_name)
            for name, c in rows.items():
                if name in cards:
                    continue
                summon_name = _as_str(c.get("SummonCharacter"))
                if not summon_name:
                    s_list = c.get("SummonCharactersList")
                    if isinstance(s_list, list) and s_list and isinstance(s_list[0], str):
                        summon_name = s_list[0]
                unit = units.get(summon_name) if summon_name else None
                spell = unit is None

                proj_name = _as_str(c.get("Projectile"))
                aeo_name = _as_str(c.get("AreaEffectObject"))
                search_unit: UnitDeployInfo | None = None
                if unit is not None:
                    morph_name = _as_str((chars.get(unit.name) or {}).get("SpawnPathfindMorph"))
                    if morph_name:
                        search_unit = units.get(morph_name)
                if spell and proj_name and proj_name in projs:
                    sc = _as_str(projs[proj_name].get("SpawnCharacter"))
                    if sc:
                        search_unit = units.get(sc)
                elif spell and aeo_name and aeo_name in aeos:
                    sc = _as_str(aeos[aeo_name].get("SpawnCharacter"))
                    if sc:
                        search_unit = units.get(sc)

                spell_as_deploy = bool(aeo_name) if not spell else bool(c.get("SpellAsDeploy", False))
                cards[name] = DeployCard(
                    name=name,
                    mana_cost=_as_int(c.get("ManaCost")),
                    mirror=bool(c.get("Mirror", False)),
                    mana_from_summoner=bool(c.get("ManaCostFromSummonerMana", False)),
                    spell=spell,
                    spell_as_deploy=spell_as_deploy,
                    can_deploy_on_enemy_side=bool(c.get("CanDeployOnEnemySide", False)),
                    can_place_on_buildings=bool(c.get("CanPlaceOnBuildings", False)),
                    can_place_on_water=bool(c.get("CanPlaceOnWater", False)),
                    full_lane_deploy=bool(c.get("FullLaneDeploy", False)),
                    deploy_w_tile_margin=_as_int(c.get("DeployWTileMargin")),
                    deploy_start_y=_as_int(c.get("DeployStartY")),
                    deploy_end_y=_as_int(c.get("DeployEndY")),
                    unit=unit,
                    search_unit=search_unit,
                )
        return cls(units=units, cards=cards)

    def resolve_play_card(
        self,
        card_name: str | None,
        *,
        last_card_name: str | None = None,
    ) -> tuple[DeployCard | None, int | None]:
        """Resolve a hand card (including Mirror -> last played card) to `(DeployCard, cost)`."""
        if not card_name or card_name not in self.cards:
            return None, None
        card = self.cards[card_name]
        if card.mirror:
            if not last_card_name or last_card_name not in self.cards:
                return None, None
            repeated = self.cards[last_card_name]
            if repeated.mirror:
                return None, None
            return repeated, repeated.mana_cost + 1
        return card, card.mana_cost


def _box_for_entity(
    e: MaskEntity,
    own_team0: bool,
    enemy_ok: bool,
    on_buildings: bool,
    touchdown: bool,
    same_rows: bool,
) -> tuple[int, int] | None:
    if not (e.alive and (e.building or e.summoner)):
        return None
    enemy = (e.team == 0) != own_team0
    no_h = (e.no_deploy_size_h - 12) if touchdown else e.no_deploy_size_h
    if on_buildings:
        to_box = (same_rows and enemy) if e.summoner else (same_rows and enemy and e.summoner_tower)
        if to_box and not enemy_ok:
            return (e.no_deploy_size_w, no_h)
        if e.building or e.no_deploy_size_w > 0:
            return None
    if same_rows and enemy and e.no_deploy_size_w >= 1 and not enemy_ok:
        return (e.no_deploy_size_w, no_h)
    return (e.footprint, e.footprint)


def build_deploy_mask(
    tile_map: TileMap,
    card: DeployCard,
    own_seat: int,
    entities: list[MaskEntity],
    *,
    touchdown: bool = False,
) -> list[bool]:
    """Build the 18x32 tile legality mask in battle-frame tile order `(row * 18 + col)`."""
    cols = tile_map.width >> 1
    rows = tile_map.height >> 1
    mask = [True] * (cols * rows)

    start_y = card.deploy_start_y
    end_y = card.deploy_end_y
    if end_y != start_y:
        if start_y >= 1:
            for r in range(min(start_y, rows)):
                mask[r * cols : (r + 1) * cols] = [False] * cols
        stop = rows if end_y == 0 else end_y
        if stop > 0:
            for r in range(stop, rows):
                mask[r * cols : (r + 1) * cols] = [False] * cols

    own_team0 = (own_seat & 1) == 0
    same_rows = start_y == end_y
    for e in entities:
        b = _box_for_entity(
            e,
            own_team0=own_team0,
            enemy_ok=card.can_deploy_on_enemy_side,
            on_buildings=card.can_place_on_buildings,
            touchdown=touchdown,
            same_rows=same_rows,
        )
        if b is None or b[0] == 0:
            continue
        w, h = b
        r0 = trunc_div(e.y - 500 * h, 1000)
        r1 = trunc_div(e.y + 500 * h, 1000)
        c0 = trunc_div(e.x - 500 * w, 1000)
        c1 = trunc_div(e.x + 500 * w, 1000)
        if r0 >= r1 or c0 >= c1:
            continue
        for r in range(max(0, r0), min(rows, r1)):
            base = r * cols
            for c in range(max(0, c0), min(cols, c1)):
                mask[base + c] = False

    if card.full_lane_deploy:
        for r in range(rows):
            base = r * cols
            if not all(mask[base : base + cols]):
                mask[base : base + cols] = [False] * cols

    return mask


def _character_snap(card: DeployCard) -> bool:
    return card.placement_unit is not None and not card.spell_as_deploy


def _search_footprint(card: DeployCard) -> int:
    pu = card.placement_unit
    if _character_snap(card) and pu is not None and pu.building:
        return pu.footprint
    return 1


def _snapped(card: DeployCard, coord: int) -> int:
    t = trunc_div(coord, 1000) * 1000
    pu = card.placement_unit
    if _character_snap(card) and pu is not None and pu.building and (pu.footprint & 1) == 0:
        return t
    return t + 500


def _tile_legal(
    tile_map: TileMap,
    card: DeployCard,
    mask: list[bool],
    cols: int,
    rows: int,
    px: int,
    py: int,
    fp: int,
    margin: int,
) -> bool:
    half = 500 * fp
    xlo = trunc_div(px - half, 1000)
    xhi = trunc_div(px + half, 1000)
    ylo = trunc_div(py - half, 1000)
    yhi = trunc_div(py + half, 1000)
    if xlo >= xhi:
        return True
    available = max(rows, ylo) - ylo
    for c in range(xlo, xhi):
        if ylo >= yhi:
            continue
        if c < margin or c >= cols - margin:
            return False
        if py - half < -999:
            return False
        left = available
        for r in range(ylo, yhi):
            if left == 0 or r < 0 or r >= rows or not mask[r * cols + c]:
                return False
            cc = 2 * c
            rr = 2 * r
            if not (
                tile_map.cell_ok(cc, rr, card)
                and tile_map.cell_ok(cc + 1, rr, card)
                and tile_map.cell_ok(cc, rr + 1, card)
                and tile_map.cell_ok(cc + 1, rr + 1, card)
            ):
                return False
            left -= 1
    return True


def landing_tiles(
    tile_map: TileMap,
    card: DeployCard,
    own_seat: int,
    entities: list[MaskEntity],
) -> list[bool]:
    """Compute battle-frame 18x32 `PlacementSearch.landingTiles` for `card`."""
    cols = tile_map.width >> 1
    rows = tile_map.height >> 1
    mask = build_deploy_mask(tile_map, card, own_seat, entities)
    fp = _search_footprint(card)
    margin = card.deploy_w_tile_margin
    out = [False] * (cols * rows)
    for row in range(rows):
        py = _snapped(card, row * 1000 + 500)
        for col in range(cols):
            px = _snapped(card, col * 1000 + 500)
            out[row * cols + col] = _tile_legal(
                tile_map, card, mask, cols, rows, px, py, fp, margin
            )
    return out


def build_legal_mask(
    catalog: DeployCatalog,
    tile_map: TileMap,
    own_seat: int,
    entities: list[MaskEntity],
    hand_cards: list[str | None],
    hand_costs: list[int],
    hand_pending: list[bool],
    available_elixir_whole: int,
    play_open: bool = True,
    match_running: bool = True,
    own_king_alive: bool = True,
    last_played_card: str | None = None,
    ability_ready: tuple[bool, bool] = (False, False),
) -> list[bool]:
    """Build Schema v4 `bool[2307]` legal action mask (`WAIT=0`, `play`, `ability`)."""
    legal = [False] * SchemaV4.ACTIONS
    legal[SchemaV4.WAIT] = True
    if not (match_running and own_king_alive and play_open):
        return legal

    for slot in range(SchemaV4.HAND_SLOTS):
        cname = hand_cards[slot] if slot < len(hand_cards) else None
        if not cname or hand_pending[slot]:
            continue
        resolved, default_cost = catalog.resolve_play_card(cname, last_card_name=last_played_card)
        if resolved is None:
            continue
        cost = hand_costs[slot] if hand_costs[slot] > 0 else (default_cost or 0)
        if available_elixir_whole < cost:
            continue
        lands = landing_tiles(tile_map, resolved, own_seat, entities)
        base = 1 + slot * SchemaV4.TILES
        for s_row in range(SchemaV4.ROWS):
            g_row = SchemaV4.game_y(own_seat, s_row) // 1000
            for s_col in range(SchemaV4.COLS):
                g_col = SchemaV4.game_x(own_seat, s_col) // 1000
                if lands[g_row * SchemaV4.COLS + g_col]:
                    legal[base + s_row * SchemaV4.COLS + s_col] = True

    for ab_slot in range(SchemaV4.ABILITY_SLOTS):
        if ab_slot < len(ability_ready) and ability_ready[ab_slot]:
            legal[SchemaV4.FIRST_ABILITY + ab_slot] = True
    return legal
