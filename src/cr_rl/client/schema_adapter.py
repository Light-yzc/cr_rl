"""Adapter mapping Nulls Royale 16.635.1 memory snapshots to Schema v4 + legal masks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .deploy_mask import DeployCatalog, MaskEntity, TileMap, build_legal_mask
from .nulls_reader import (
    SECTION_BONUS_TIME,
    SECTION_NORMAL,
    SECTION_OVERTIME,
    START_COOLDOWN_TICKS,
    TAG_CHARGING,
    TAG_FORCE_AIR,
    TAG_FORCE_GROUND,
    TAG_HIDDEN,
    TAG_INACTIVE,
    TAG_NO_ATTACK,
    TAG_NO_MOVE,
    VT_AREA_EFFECT,
    VT_CHARACTER,
    VT_KING_TOWER,
    VT_PROJECTILE,
    NullsBattleSnapshot,
    NullsBuff,
    NullsEntity,
)
from .schema_v4 import SPELL_TABLES, SchemaV4, Vocabulary, load_table_rows


# Verified 16.635.1 -> 16.635.3 row name overrides (populated only from observed diffs).
DEFAULT_NAME_ALIASES: dict[str, str] = {}
_CHAMPION_READY_STATE = 2
_PENDING_TIMEOUT_EXTRA_TICKS = 40
_ABILITY_CHARGE_PER_MS = 50


@dataclass(frozen=True)
class _EnemyPlayRecord:
    card_name: str | None
    cast_name: str | None
    form: int
    play_tick: int
    seat_x: int
    seat_y: int
    kind: int
    flags: int
    point_tick: int
    eids: tuple[int, ...] = ()


@dataclass(frozen=True)
class _MatchCheckpoint:
    tick: int
    time_ms: int | None
    episode_step: int
    full_bar_ms: int
    prev_pos: dict[int, tuple[int, str | None, int, int]]
    first_seen_ms: dict[int, int]
    tower_max_hp: dict[tuple[int, int], int]
    saw_side_towers: dict[int, bool]
    prev_played_count: int | None
    prev_hand_cards: tuple[str | None, ...]
    last_played_card: str | None
    seen_enemy_eids: set[int]
    prev_enemy_seen_scids: tuple[int, ...]
    recent_enemy_plays: deque[_EnemyPlayRecord]
    enemy_play_count: int | None
    ability_ran_tick: dict[int, int]
    champ_was_active: dict[int, bool]


@dataclass(frozen=True)
class SchemaObservation:
    """Single-tick Schema v4 observation tensor, legal mask, and UI metadata."""

    obs: np.ndarray  # int32[7024]
    legal_mask: np.ndarray  # bool[2307]
    own_seat: int
    tick: int
    time_ms: int
    hand_cards: tuple[str | None, str | None, str | None, str | None]
    hand_card_ids: tuple[int, int, int, int]
    unmatched_names: tuple[str, ...]
    rejected_entities: tuple[dict[str, int | str], ...] = ()


def _load_card_rarities(tables_dir: Path) -> dict[str, str]:
    rarities: dict[str, str] = {}
    for tbl in SPELL_TABLES:
        path = tables_dir / f"{tbl}.json"
        if not path.exists():
            continue
        for name, cols in load_table_rows(tables_dir, tbl).items():
            rarity = cols.get("Rarity")
            if isinstance(rarity, str) and rarity:
                rarities[name] = rarity
    return rarities


def _load_air_units(tables_dir: Path) -> set[str]:
    air: set[str] = set()
    chars_path = tables_dir / "characters.json"
    if chars_path.exists():
        for name, cols in load_table_rows(tables_dir, "characters").items():
            fh = cols.get("FlyingHeight")
            if isinstance(fh, (int, float)) and fh > 0:
                air.add(name)
    return air


def _buff_slot_rank(buff: NullsBuff) -> int:
    """Order active buffs per `BuffSlots.rank`: stops -> slows -> timed -> forever."""
    sm = buff.speed_mult or 0
    hsm = buff.hit_speed_mult or 0
    if sm <= -100 or hsm <= -100:
        return 0
    if sm < 0 or hsm < 0:
        return 1
    if buff.time_left >= 0:
        return 2
    return 3


def _entity_level(level_raw: int | None) -> int:
    """Decode grid entity `level_raw` (handles plain `abs_level - 1` or PackedLevel)."""
    if level_raw is None or level_raw < 0:
        return SchemaV4.UNKNOWN
    if level_raw >= 256:
        rel = (level_raw >> 8) & 0xFF
        steps = level_raw & 0xFF
        return max(1, steps + 1 + rel)
    return max(1, level_raw + 1)


class NullsToSchemaV4Adapter:
    """Stateful Schema v4 + legal-mask converter for live Nulls snapshots."""

    def __init__(
        self,
        tables_dir: str | Path,
        *,
        vocab: Vocabulary | None = None,
        catalog: DeployCatalog | None = None,
        tile_map: TileMap | None = None,
        aliases: dict[str, str] | None = None,
        decision_ticks: int = 8,
        default_own_seat: int | None = None,
        allow_abilities: bool = False,
    ) -> None:
        self.tables_dir = Path(tables_dir)
        self.vocab = vocab or Vocabulary.from_tables(self.tables_dir)
        self.catalog = catalog or DeployCatalog.from_tables(self.tables_dir)
        self.tile_map = tile_map or TileMap.standard_1v1()
        self.aliases = dict(DEFAULT_NAME_ALIASES)
        if aliases:
            self.aliases.update(aliases)
        self.decision_ticks = int(decision_ticks)
        self.default_own_seat = default_own_seat
        self.allow_abilities = bool(allow_abilities)

        self._unit_stats: dict[str, dict] = {}
        for table in ("characters", "buildings"):
            if (self.tables_dir / f"{table}.json").exists():
                self._unit_stats.update(load_table_rows(self.tables_dir, table))
        self._card_rarities = _load_card_rarities(self.tables_dir)
        self._air_units = _load_air_units(self.tables_dir)
        self._init_spell_metadata()
        self.unmatched_names: set[str] = set()
        self._scid_to_name: dict[int, str] = {}

        # Match-scoped tracking state
        self._last_tick: int = -1
        self._deck_sig: tuple[str, ...] = ()
        self._episode_step: int = 0
        self._full_bar_ms: int = 0
        self._seat_ids: dict[int, int] = {}
        self._next_seat_id: int = 1
        self._prev_pos: dict[int, tuple[int, str | None, int, int]] = {}
        self._tower_max_hp: dict[tuple[int, int], int] = {}
        self._saw_side_towers: dict[int, bool] = {0: False, 1: False}
        self._prev_played_count: int | None = None
        self._prev_hand_cards: tuple[str | None, ...] = ()
        self._last_played_card: str | None = None
        self.first_seen_ms: dict[int, int] = {}
        self.seen_enemy_eids: set[int] = set()
        self._prev_enemy_seen_scids: tuple[int, ...] = ()
        self.recent_enemy_plays: deque[_EnemyPlayRecord] = deque(maxlen=SchemaV4.ENEMY_PLAY_ROWS)
        self._enemy_play_count: int | None = None
        self._last_time_ms: int | None = None
        self._ability_ran_tick: dict[int, int] = {}
        self._champ_was_active: dict[int, bool] = {}
        self._pre_tick_ckpt: _MatchCheckpoint | None = None

    def _init_spell_metadata(self) -> None:
        """Build card form/evolution cost, play-kind, and reverse spawn lookup maps."""
        self._evo_need: dict[str, int] = {}
        self._evo_form_row: dict[str, str] = {}
        self._hero_form_row: dict[str, str] = {}
        self._card_play_kind: dict[str, int] = {}
        self._card_cost_by_name: dict[str, int] = {}
        self._spawn_to_card: dict[str, list[tuple[str, int, str]]] = {}

        tbl_rows: dict[str, dict[str, dict]] = {}
        for tbl in SPELL_TABLES:
            p = self.tables_dir / f"{tbl}.json"
            tbl_rows[tbl] = load_table_rows(self.tables_dir, tbl) if p.exists() else {}

        base_tables = ("spells_characters", "spells_other", "spells_buildings")
        base_cards: dict[str, dict] = {}
        for tbl in base_tables:
            base_cards.update(tbl_rows[tbl])

        def _classify_kind(cols: dict, *, is_spell_other: bool) -> int:
            summ = cols.get("SummonCharacter")
            if not summ and isinstance(cols.get("SummonCharactersList"), list) and cols["SummonCharactersList"]:
                summ = cols["SummonCharactersList"][0]
            if isinstance(summ, str) and summ and not is_spell_other:
                st = self._unit_stats.get(summ, {})
                spd = st.get("SpawnPathfindSpeed")
                if isinstance(spd, (int, float)) and spd != 0:
                    return SchemaV4.PLAY_TUNNEL
                return SchemaV4.PLAY_TROOP
            if cols.get("SpellAsDeploy"):
                return SchemaV4.PLAY_THROWN
            if cols.get("Projectile"):
                return SchemaV4.PLAY_LAUNCHED
            return SchemaV4.PLAY_AREA

        def _register_spawns(base_name: str, form: int, cast_name: str, cols: dict) -> None:
            keys: list[str] = []
            for k in ("SummonCharacter", "SummonCharacterSecond", "Projectile", "AreaEffectObject"):
                v = cols.get(k)
                if isinstance(v, str) and v:
                    keys.append(v)
            s_list = cols.get("SummonCharactersList")
            if isinstance(s_list, list):
                for item in s_list:
                    if isinstance(item, str) and item:
                        keys.append(item)
            for key in keys:
                lst = self._spawn_to_card.setdefault(key, [])
                entry = (base_name, form, cast_name)
                if entry not in lst:
                    lst.append(entry)

        for tbl in base_tables:
            is_other = tbl == "spells_other"
            for name, cols in tbl_rows[tbl].items():
                mc = cols.get("ManaCost")
                if isinstance(mc, (int, float)):
                    self._card_cost_by_name[name] = int(mc)
                self._card_play_kind[name] = _classify_kind(cols, is_spell_other=is_other)
                _register_spawns(name, SchemaV4.FORM_PLAIN, name, cols)

        for evo_name, cols in tbl_rows["spells_evolved"].items():
            base_name = evo_name[:-4] if evo_name.endswith("_EV1") else evo_name
            if base_name in base_cards:
                dc = cols.get("DarkElixirCost")
                if isinstance(dc, (int, float)) and dc > 0:
                    self._evo_need[base_name] = int(dc)
                self._evo_form_row[base_name] = evo_name
            mc = cols.get("ManaCost")
            if isinstance(mc, (int, float)):
                self._card_cost_by_name[evo_name] = int(mc)
            self._card_play_kind[evo_name] = _classify_kind(cols, is_spell_other=not bool(cols.get("SummonCharacter")))
            if base_name in base_cards:
                _register_spawns(base_name, SchemaV4.FORM_EVOLVED, evo_name, cols)

        for hero_name, cols in tbl_rows["spells_hero_form"].items():
            base_name = hero_name[:-5] if hero_name.endswith("_hero") else hero_name
            if base_name in base_cards:
                self._hero_form_row[base_name] = hero_name
            mc = cols.get("ManaCost")
            if isinstance(mc, (int, float)):
                self._card_cost_by_name[hero_name] = int(mc)
            self._card_play_kind[hero_name] = _classify_kind(cols, is_spell_other=False)
            if base_name in base_cards:
                _register_spawns(base_name, SchemaV4.FORM_HERO, hero_name, cols)

    def reset_match(self) -> None:
        """Clear per-match entity IDs, velocities, tower cache, and Mirror state."""
        self._last_tick = -1
        self._deck_sig = ()
        self._episode_step = 0
        self._full_bar_ms = 0
        self._seat_ids.clear()
        self._next_seat_id = 1
        self._prev_pos.clear()
        self._tower_max_hp.clear()
        self._saw_side_towers = {0: False, 1: False}
        self._prev_played_count = None
        self._prev_hand_cards = ()
        self._last_played_card = None
        self.first_seen_ms.clear()
        self.seen_enemy_eids.clear()
        self._prev_enemy_seen_scids = ()
        self.recent_enemy_plays.clear()
        self._enemy_play_count = None
        self._last_time_ms = None
        self._ability_ran_tick.clear()
        self._champ_was_active.clear()
        self._pre_tick_ckpt = None

    def _snapshot_checkpoint(self) -> _MatchCheckpoint:
        return _MatchCheckpoint(
            tick=self._last_tick,
            time_ms=self._last_time_ms,
            episode_step=self._episode_step,
            full_bar_ms=self._full_bar_ms,
            prev_pos=dict(self._prev_pos),
            first_seen_ms=dict(self.first_seen_ms),
            tower_max_hp=dict(self._tower_max_hp),
            saw_side_towers=dict(self._saw_side_towers),
            prev_played_count=self._prev_played_count,
            prev_hand_cards=self._prev_hand_cards,
            last_played_card=self._last_played_card,
            seen_enemy_eids=set(self.seen_enemy_eids),
            prev_enemy_seen_scids=self._prev_enemy_seen_scids,
            recent_enemy_plays=deque(self.recent_enemy_plays, maxlen=SchemaV4.ENEMY_PLAY_ROWS),
            enemy_play_count=self._enemy_play_count,
            ability_ran_tick=dict(self._ability_ran_tick),
            champ_was_active=dict(self._champ_was_active),
        )

    def _restore_checkpoint(self, c: _MatchCheckpoint) -> None:
        self._last_tick = c.tick
        self._last_time_ms = c.time_ms
        self._episode_step = c.episode_step
        self._full_bar_ms = c.full_bar_ms
        self._prev_pos = dict(c.prev_pos)
        self.first_seen_ms = dict(c.first_seen_ms)
        self._tower_max_hp = dict(c.tower_max_hp)
        self._saw_side_towers = dict(c.saw_side_towers)
        self._prev_played_count = c.prev_played_count
        self._prev_hand_cards = c.prev_hand_cards
        self._last_played_card = c.last_played_card
        self.seen_enemy_eids = set(c.seen_enemy_eids)
        self._prev_enemy_seen_scids = c.prev_enemy_seen_scids
        self.recent_enemy_plays = deque(c.recent_enemy_plays, maxlen=SchemaV4.ENEMY_PLAY_ROWS)
        self._enemy_play_count = c.enemy_play_count
        self._ability_ran_tick = dict(c.ability_ran_tick)
        self._champ_was_active = dict(c.champ_was_active)

    def normalize_name(self, raw_name: str | None) -> str | None:
        if not raw_name:
            return None
        mapped = self.aliases.get(raw_name, raw_name)
        if mapped in self.vocab.name_to_id or mapped in self.catalog.cards:
            return mapped
        self.unmatched_names.add(raw_name)
        return mapped

    def card_level(self, card_name: str | None, level_raw: int) -> int:
        if level_raw < 0 or not card_name:
            return SchemaV4.UNKNOWN
        rarity = self._card_rarities.get(card_name, "Common")
        rel = self.vocab.relative_levels.get(rarity, 0)
        return max(1, level_raw + 1 + rel)

    def _seat_id_of(self, eid: int) -> int:
        sid = self._seat_ids.get(eid)
        if sid is None:
            sid = self._next_seat_id
            self._next_seat_id += 1
            self._seat_ids[eid] = sid
        return sid

    def _entity_kind(self, ent: NullsEntity, norm_name: str | None) -> int | None:
        if ent.eid <= 0:
            return None
        if ent.vt == VT_PROJECTILE:
            return SchemaV4.KIND_PROJECTILE
        if ent.vt == VT_AREA_EFFECT:
            return SchemaV4.KIND_AREA
        if ent.vt == VT_KING_TOWER:
            return SchemaV4.KIND_TOWER
        # Names alone cannot establish a memory layout. In particular, tower
        # projectile subclasses were being silently treated as characters.
        if ent.vt != VT_CHARACTER:
            return None
        if norm_name in ("KingTower", "PrincessTower"):
            return SchemaV4.KIND_TOWER
        unit = self.catalog.units.get(norm_name or "")
        if unit is not None and (unit.king or unit.summoner_tower):
            return SchemaV4.KIND_TOWER
        return SchemaV4.KIND_CHARACTER

    def _resolve_enemy_play_card(
        self,
        ent: NullsEntity,
        kind: int,
        nname: str | None,
        enemy_deck_cards: list[str],
        newly_seen_cards: set[str],
    ) -> tuple[str | None, str | None, int]:
        """Resolve an enemy spawned entity to `(base_card, cast_card, form)`."""
        if not nname:
            return None, None, SchemaV4.UNKNOWN
        cands = self._spawn_to_card.get(nname, ())
        if cands:
            for base, form, cast in cands:
                if base in newly_seen_cards:
                    return base, cast, form
            for base, form, cast in cands:
                if base in enemy_deck_cards:
                    return base, cast, form
            for base, form, cast in cands:
                if base == nname or cast == nname:
                    return base, cast, form
            base, form, cast = cands[0]
            return base, cast, form
        if nname in self.catalog.cards or nname in self._card_play_kind:
            if nname.endswith("_EV1"):
                return nname[:-4], nname, SchemaV4.FORM_EVOLVED
            if nname.endswith("_hero"):
                return nname[:-5], nname, SchemaV4.FORM_HERO
            return nname, nname, SchemaV4.FORM_PLAIN
        return None, None, SchemaV4.UNKNOWN

    def _enemy_launched_hidden_eids(
        self,
        other_ents: list[tuple[NullsEntity, int, str | None]],
        live_eids: set[int],
        resolved_seat: int,
    ) -> set[int]:
        """Track enemy launched spell projectiles whose landing point is hidden until impact."""
        hidden: set[int] = set()
        for idx, rec in enumerate(list(self.recent_enemy_plays)):
            if (rec.flags & SchemaV4.PLAY_POINT_SHOWN) != 0 or not rec.eids:
                continue
            if rec.kind == SchemaV4.PLAY_LAUNCHED:
                still_flying = any(eid in live_eids for eid in rec.eids)
                if still_flying:
                    hidden.update(rec.eids)
                else:
                    self.recent_enemy_plays[idx] = _EnemyPlayRecord(
                        card_name=rec.card_name,
                        cast_name=rec.cast_name,
                        form=rec.form,
                        play_tick=rec.play_tick,
                        seat_x=rec.seat_x,
                        seat_y=rec.seat_y,
                        kind=rec.kind,
                        flags=rec.flags | SchemaV4.PLAY_POINT_SHOWN,
                        point_tick=self._last_tick,
                        eids=rec.eids,
                    )
        # Also mark newly visible projectile entities in current frame if their play is LAUNCHED
        for ent, kind, nname in other_ents:
            if kind == SchemaV4.KIND_PROJECTILE and ent.side == 1 - resolved_seat:
                card, _, _ = self._resolve_enemy_play_card(ent, kind, nname, [], set())
                if card and self._card_play_kind.get(card) == SchemaV4.PLAY_LAUNCHED:
                    hidden.add(ent.eid)
        return hidden

    @staticmethod
    def _cmd_ticks_left(timeout_tick: int, cur_tick: int, *, minimum: int = 0) -> int:
        exec_tick = timeout_tick - _PENDING_TIMEOUT_EXTRA_TICKS
        return max(minimum, min(SchemaV4.PLAY_DELAY_TICKS, exec_tick - cur_tick))

    def _track_hand_and_mirror(
        self, own_sum
    ) -> tuple[list[str | None], list[int], list[int]]:
        hand_names: list[str | None] = []
        hand_ids: list[int] = []
        hand_costs: list[int] = []
        for slot in range(SchemaV4.HAND_SLOTS):
            idx = own_sum.hand[slot] if slot < len(own_sum.hand) else -1
            if 0 <= idx < len(own_sum.deck):
                card = own_sum.deck[idx]
                cname = self.normalize_name(card.name)
                hand_names.append(cname)
                hand_ids.append(self.vocab.row_id(cname))
                hand_costs.append(card.cost)
            else:
                hand_names.append(None)
                hand_ids.append(-1)
                hand_costs.append(SchemaV4.UNKNOWN)

        if (
            self._prev_played_count is not None
            and own_sum.played_count > self._prev_played_count
            and len(self._prev_hand_cards) == SchemaV4.HAND_SLOTS
        ):
            for prev_c, cur_c in zip(self._prev_hand_cards, hand_names):
                if prev_c and prev_c != cur_c:
                    self._last_played_card = prev_c
                    break
        self._prev_played_count = own_sum.played_count
        self._prev_hand_cards = tuple(hand_names)
        return hand_names, hand_ids, hand_costs

    def _fill_deck_and_form_headers(
        self, o: np.ndarray, own_sum, enemy_sum, hand_names: list[str | None]
    ) -> None:
        next_name = (
            self.normalize_name(own_sum.deck[own_sum.queue[0]].name)
            if own_sum.queue and 0 <= own_sum.queue[0] < len(own_sum.deck)
            else None
        )
        o[SchemaV4.H_NEXT_CARD] = self.vocab.row_id(next_name) if next_name else -1
        for k in range(SchemaV4.QUEUE_SLOTS):
            qidx = own_sum.queue[k] if k < len(own_sum.queue) else -1
            if 0 <= qidx < len(own_sum.deck):
                qname = self.normalize_name(own_sum.deck[qidx].name)
                o[SchemaV4.H_QUEUE + k] = self.vocab.row_id(qname)
            else:
                o[SchemaV4.H_QUEUE + k] = -1

        for k in range(SchemaV4.DECK_SIZE):
            if k >= len(own_sum.deck):
                o[SchemaV4.H_DECK + k] = -1
                o[SchemaV4.H_DECK_LEVEL + k] = SchemaV4.UNKNOWN
                o[SchemaV4.H_DECK_SLOTS + k] = SchemaV4.UNKNOWN
                o[SchemaV4.H_DECK_EVOLUTION + k] = SchemaV4.UNKNOWN
                continue
            dc = own_sum.deck[k]
            dname = self.normalize_name(dc.name)
            o[SchemaV4.H_DECK + k] = self.vocab.row_id(dname)
            o[SchemaV4.H_DECK_LEVEL + k] = self.card_level(dname, dc.level_raw)
            if dc.slot_flags is None:
                o[SchemaV4.H_DECK_SLOTS + k] = SchemaV4.UNKNOWN
                o[SchemaV4.H_DECK_EVOLUTION + k] = SchemaV4.UNKNOWN
                continue
            sf = dc.slot_flags & (SchemaV4.SLOT_EVOLUTION | SchemaV4.SLOT_HERO)
            if (dc.evolution_count or 0) > 0:
                sf |= SchemaV4.SLOT_EVOLUTION
            o[SchemaV4.H_DECK_SLOTS + k] = sf
            need = self._evo_need.get(dname or "", 0)
            if dc.evolution_count is None:
                o[SchemaV4.H_DECK_EVOLUTION + k] = SchemaV4.UNKNOWN
            elif (sf & SchemaV4.SLOT_EVOLUTION) and need > 0 and (dc.slot_flags & SchemaV4.SLOT_EVOLUTION):
                o[SchemaV4.H_DECK_EVOLUTION + k] = max(dc.evolution_count, need)
            else:
                o[SchemaV4.H_DECK_EVOLUTION + k] = (
                    min(need, dc.evolution_count) if need > 0 else dc.evolution_count
                )

        for k in range(SchemaV4.HAND_SLOTS):
            idx = own_sum.hand[k] if k < len(own_sum.hand) else -1
            if not (0 <= idx < len(own_sum.deck)):
                o[SchemaV4.H_HAND_FORM + k] = SchemaV4.FORM_PLAIN
                continue
            dc = own_sum.deck[idx]
            cname = hand_names[k]
            if dc.slot_flags is None:
                o[SchemaV4.H_HAND_FORM + k] = SchemaV4.UNKNOWN
                continue
            sf = int(o[SchemaV4.H_DECK_SLOTS + idx])
            evo_c = int(o[SchemaV4.H_DECK_EVOLUTION + idx])
            need = self._evo_need.get(cname or "", 0)
            if (sf & SchemaV4.SLOT_EVOLUTION) and (need <= 0 or evo_c >= need):
                o[SchemaV4.H_HAND_FORM + k] = SchemaV4.FORM_EVOLVED
            elif (sf & SchemaV4.SLOT_HERO) and (cname in self._hero_form_row):
                o[SchemaV4.H_HAND_FORM + k] = SchemaV4.FORM_HERO
            else:
                o[SchemaV4.H_HAND_FORM + k] = SchemaV4.FORM_PLAIN

        for k in range(SchemaV4.DECK_SIZE):
            scid = enemy_sum.seen_scids[k] if k < len(enemy_sum.seen_scids) else -1
            sname = self._scid_to_name.get(scid) if scid > 0 else None
            o[SchemaV4.H_ENEMY_SEEN + k] = self.vocab.row_id(sname) if sname else -1
            if enemy_sum.seen_evo is not None and enemy_sum.seen_evo_need is not None:
                need = (
                    max(0, enemy_sum.seen_evo_need[k])
                    if (scid > 0 and k < len(enemy_sum.seen_evo_need))
                    else 0
                )
                o[SchemaV4.H_ENEMY_SEEN_EVOLUTION_NEED + k] = need
                o[SchemaV4.H_ENEMY_SEEN_EVOLUTION + k] = (
                    enemy_sum.seen_evo[k]
                    if (scid > 0 and need > 0 and k < len(enemy_sum.seen_evo))
                    else SchemaV4.UNKNOWN
                )
            else:
                o[SchemaV4.H_ENEMY_SEEN_EVOLUTION + k] = SchemaV4.UNKNOWN
                o[SchemaV4.H_ENEMY_SEEN_EVOLUTION_NEED + k] = SchemaV4.UNKNOWN

    def _collect_entities_and_towers(
        self,
        o: np.ndarray,
        snap: NullsBattleSnapshot,
        resolved_seat: int,
    ):
        world_ents: list[tuple[NullsEntity, int, str | None]] = []
        other_ents: list[tuple[NullsEntity, int, str | None]] = []
        live_towers: dict[tuple[int, int], NullsEntity] = {}
        mask_entities: list[MaskEntity] = []
        id_to_seat_pos: dict[int, tuple[int, int]] = {}
        live_eids: set[int] = set()
        ent_by_eid: dict[int, NullsEntity] = {}
        rejected_entities: list[dict[str, int | str]] = []

        for ent in snap.entities:
            nname = self.normalize_name(ent.name)
            kind = self._entity_kind(ent, nname)
            if kind is None:
                rejected_entities.append({'eid':ent.eid,'vtable':hex(ent.vt),'name':ent.name,
                    'reason':'invalid_entity_id' if ent.eid <= 0 else 'unverified_entity_layout'})
                continue
            if nname and ent.scid > 0:
                self._scid_to_name.setdefault(ent.scid, nname)
            sx = SchemaV4.to_seat_x(resolved_seat, ent.x)
            sy = SchemaV4.to_seat_y(resolved_seat, ent.y)
            id_to_seat_pos[ent.eid] = (sx, sy)
            live_eids.add(ent.eid)
            ent_by_eid[ent.eid] = ent
            self._seat_id_of(ent.eid)

            if kind in (SchemaV4.KIND_TOWER, SchemaV4.KIND_CHARACTER):
                world_ents.append((ent, kind, nname))
                alive = ent.hp is None or ent.hp > 0
                unit = self.catalog.units.get(nname or "")
                if kind == SchemaV4.KIND_TOWER and ent.side in (0, 1) and alive:
                    self._saw_side_towers[ent.side] = True
                    is_king = (
                        ent.vt == VT_KING_TOWER
                        or (unit is not None and unit.king)
                        or nname == "KingTower"
                    )
                    slot_kind = 0 if is_king else (1 if sx < (SchemaV4.WIDTH_UNITS // 2) else 2)
                    live_towers[(ent.side, slot_kind)] = ent
                    if ent.max_hp is not None and ent.max_hp > 0:
                        self._tower_max_hp[(ent.side, slot_kind)] = ent.max_hp
                if unit is not None and ent.side in (0, 1):
                    mask_entities.append(
                        MaskEntity.from_unit(unit, ent.side, ent.x, ent.y, alive=alive)
                    )
            else:
                other_ents.append((ent, kind, nname))

        tower_order = (
            (resolved_seat, 0),
            (resolved_seat, 1),
            (resolved_seat, 2),
            (1 - resolved_seat, 0),
            (1 - resolved_seat, 1),
            (1 - resolved_seat, 2),
        )
        for t_idx, key in enumerate(tower_order):
            tent = live_towers.get(key)
            if tent is not None and tent.hp is not None and tent.max_hp is not None:
                o[SchemaV4.H_TOWERS + 2 * t_idx] = tent.hp
                o[SchemaV4.H_TOWERS + 2 * t_idx + 1] = tent.max_hp
            else:
                o[SchemaV4.H_TOWERS + 2 * t_idx] = 0
                o[SchemaV4.H_TOWERS + 2 * t_idx + 1] = self._tower_max_hp.get(key, 0)

        def _crowns_for_capturer(defender_side: int) -> int:
            if not self._saw_side_towers.get(defender_side, False):
                return SchemaV4.UNKNOWN
            if (defender_side, 0) not in live_towers:
                return 3
            alive_princess = int((defender_side, 1) in live_towers) + int(
                (defender_side, 2) in live_towers
            )
            return 2 - alive_princess

        o[SchemaV4.H_CROWNS] = _crowns_for_capturer(1 - resolved_seat)
        o[SchemaV4.H_CROWNS_ENEMY] = _crowns_for_capturer(resolved_seat)
        return world_ents, other_ents, live_towers, mask_entities, id_to_seat_pos, live_eids, ent_by_eid, rejected_entities

    def _fill_pending_and_ability_headers(
        self,
        o: np.ndarray,
        own_sum,
        snap: NullsBattleSnapshot,
        resolved_seat: int,
        hand_ids: list[int],
        hand_costs: list[int],
        pending_slots: tuple[bool, bool, bool, bool],
        pending_targets: Sequence[tuple[int, int] | None] | None,
        live_eids: set[int],
        ent_by_eid: dict[int, NullsEntity],
    ) -> list[bool]:
        pending_card_cmds = (
            [c for c in own_sum.pending_commands if not c.is_ability]
            if own_sum.pending_commands is not None
            else None
        )
        pending_deck_idxs = (
            {c.deck_idx for c in own_sum.pending_commands if not c.is_ability}
            if own_sum.pending_commands is not None
            else set()
        )
        valid_reserved = (
            own_sum.elixir_reserved
            if own_sum.elixir_reserved is not None and 0 <= own_sum.elixir_reserved <= 200_000
            else None
        )
        o[SchemaV4.H_ELIXIR_RESERVED] = (
            valid_reserved if valid_reserved is not None else SchemaV4.UNKNOWN
        )
        o[SchemaV4.H_HAND_REFILL_MS] = (
            max(0, own_sum.hand_refill_ms)
            if own_sum.hand_refill_ms is not None
            else SchemaV4.UNKNOWN
        )
        for k in range(SchemaV4.HAND_SLOTS):
            o[SchemaV4.H_HAND_CARD + k] = hand_ids[k]
            o[SchemaV4.H_HAND_COST + k] = hand_costs[k]
            idx_k = own_sum.hand[k] if k < len(own_sum.hand) else -1
            o[SchemaV4.H_HAND_PENDING + k] = (
                1 if (pending_slots[k] or idx_k in pending_deck_idxs) else 0
            )

        if pending_card_cmds is not None:
            pcnt = min(len(pending_card_cmds), SchemaV4.PENDING_ROWS)
            o[SchemaV4.H_PENDING_COUNT] = pcnt
            for r in range(SchemaV4.PENDING_ROWS):
                at = SchemaV4.H_PENDING + r * SchemaV4.PENDING_INTS
                if r >= pcnt:
                    o[at : at + 4] = (-1, 0, 0, 0)
                    continue
                cmd = pending_card_cmds[r]
                cname = self.normalize_name(cmd.card_name)
                if not cname and 0 <= cmd.deck_idx < len(own_sum.deck):
                    cname = self.normalize_name(own_sum.deck[cmd.deck_idx].name)
                cid = self.vocab.row_id(cname) if cname else SchemaV4.UNKNOWN
                px, py = SchemaV4.UNKNOWN, SchemaV4.UNKNOWN
                if pending_targets is not None:
                    hand_slot = (
                        list(own_sum.hand).index(cmd.deck_idx)
                        if cmd.deck_idx in own_sum.hand
                        else -1
                    )
                    if 0 <= hand_slot < len(pending_targets) and pending_targets[hand_slot] is not None:
                        bx, by = pending_targets[hand_slot]  # type: ignore[misc]
                        px = SchemaV4.to_seat_x(resolved_seat, bx)
                        py = SchemaV4.to_seat_y(resolved_seat, by)
                ticks_left = self._cmd_ticks_left(cmd.timeout_tick, snap.tick)
                o[at : at + 4] = (cid, px, py, ticks_left)
        else:
            o[SchemaV4.H_PENDING_COUNT] = SchemaV4.UNKNOWN
            o[SchemaV4.H_PENDING : SchemaV4.H_DECK_SLOTS] = SchemaV4.UNKNOWN

        ability_ready_flags = [False, False]
        if own_sum.abilities is None:
            o[SchemaV4.H_ABILITY : SchemaV4.H_LAST_ABILITY_RUN_REFUSAL] = SchemaV4.UNKNOWN
            return ability_ready_flags

        by_slot = {a.slot: a for a in own_sum.abilities}
        ab_cmds = (
            {c.deck_idx: c for c in own_sum.pending_commands if c.is_ability}
            if own_sum.pending_commands is not None
            else {}
        )
        for k in range(SchemaV4.ABILITY_SLOTS):
            at = SchemaV4.H_ABILITY + k * SchemaV4.ABILITY_INTS
            ab = by_slot.get(k)
            if ab is None or ab.deck_idx < 0:
                o[at : at + 6] = (-1, -1, 0, 0, -1, 0)
                continue
            ch_name = self.normalize_name(ab.champion_name)
            if not ch_name and 0 <= ab.deck_idx < len(own_sum.deck):
                ch_name = self.normalize_name(own_sum.deck[ab.deck_idx].name)
            ch_id = self.vocab.row_id(ch_name) if ch_name else SchemaV4.UNKNOWN
            unit_live = ab.unit_eid is not None and ab.unit_eid in live_eids
            u_seat_id = self._seat_id_of(ab.unit_eid) if (ab.unit_eid is not None and unit_live) else -1
            cmd_ticks = 0
            if ab.deck_idx in ab_cmds:
                cmd_ticks = self._cmd_ticks_left(ab_cmds[ab.deck_idx].timeout_tick, snap.tick, minimum=1)
            elif ab.command_ms > 0:
                cmd_ticks = (ab.command_ms + _ABILITY_CHARGE_PER_MS - 1) // _ABILITY_CHARGE_PER_MS
            o[at : at + 6] = (
                ch_id,
                u_seat_id,
                ab.state,
                max(0, ab.cooldown_ms),
                ab.charges,
                cmd_ticks,
            )
            if ab.unit_eid is not None and unit_live:
                uent = ent_by_eid[ab.unit_eid]
                if cmd_ticks > 0 or ab.state in (7, 10) or uent.state == 10 or (uent.ability_ticks or 0) > 0:
                    self._ability_ran_tick.setdefault(ab.unit_eid, snap.tick)
            ability_ready_flags[k] = (
                self.allow_abilities
                and ab.state == _CHAMPION_READY_STATE
                and unit_live
                and cmd_ticks == 0
            )
        return ability_ready_flags

    def _update_and_write_enemy_plays(
        self,
        o: np.ndarray,
        snap: NullsBattleSnapshot,
        enemy_sum,
        resolved_seat: int,
        world_ents: list[tuple[NullsEntity, int, str | None]],
        other_ents: list[tuple[NullsEntity, int, str | None]],
        live_eids: set[int],
    ) -> set[int]:
        o[SchemaV4.H_ENEMY_PLAY_COUNT] = enemy_sum.played_count
        o[SchemaV4.H_ENEMY_ELIXIR] = enemy_sum.elixir

        newly_seen_scids = [
            s for s in enemy_sum.seen_scids
            if s > 0 and s not in self._prev_enemy_seen_scids
        ]
        self._prev_enemy_seen_scids = tuple(enemy_sum.seen_scids)
        newly_seen_cards = {
            self._scid_to_name[s] for s in newly_seen_scids if s in self._scid_to_name
        }
        enemy_deck_cards = [
            c for card in enemy_sum.deck if (c := self.normalize_name(card.name))
        ]

        delta = (
            0
            if self._enemy_play_count is None
            else max(0, enemy_sum.played_count - self._enemy_play_count)
        )
        clusters: dict[tuple[str, str, int], list[tuple[NullsEntity, int]]] = {}
        for ent, kind, nname in world_ents + other_ents:
            if ent.side != 1 - resolved_seat or ent.eid in self.seen_enemy_eids:
                continue
            self.seen_enemy_eids.add(ent.eid)
            if kind == SchemaV4.KIND_TOWER or ent.is_clone:
                continue
            base_c, cast_c, form = self._resolve_enemy_play_card(
                ent, kind, nname, enemy_deck_cards, newly_seen_cards
            )
            if base_c is not None and cast_c is not None:
                clusters.setdefault((base_c, cast_c, form), []).append((ent, kind))

        cluster_items = sorted(
            clusters.items(),
            key=lambda kv: (kv[0][0] in newly_seen_cards, len(kv[1])),
            reverse=True,
        )
        for i in range(min(delta, SchemaV4.ENEMY_PLAY_ROWS)):
            if i < len(cluster_items):
                (base_c, cast_c, form), body_pairs = cluster_items[i]
                bodies = [e for e, _ in body_pairs]
                p_kind = self._card_play_kind.get(
                    cast_c, self._card_play_kind.get(base_c, SchemaV4.PLAY_TROOP)
                )
                flags = SchemaV4.PLAY_CARD_SHOWN
                if p_kind == SchemaV4.PLAY_LAUNCHED and any(
                    k == SchemaV4.KIND_PROJECTILE for _, k in body_pairs
                ):
                    first_p = bodies[0]
                    bx = first_p.aim_x if first_p.aim_x is not None else first_p.x
                    by = first_p.aim_y if first_p.aim_y is not None else first_p.y
                    sx = SchemaV4.to_seat_x(resolved_seat, bx)
                    sy = SchemaV4.to_seat_y(resolved_seat, by)
                    p_tick = 0
                elif p_kind == SchemaV4.PLAY_TUNNEL and any(
                    e.is_underground or e.state == 6 for e in bodies
                ):
                    sx, sy, p_tick = 0, 0, 0
                else:
                    bx = int(sum(e.x for e in bodies) / len(bodies))
                    by = int(sum(e.y for e in bodies) / len(bodies))
                    sx = SchemaV4.to_seat_x(resolved_seat, bx)
                    sy = SchemaV4.to_seat_y(resolved_seat, by)
                    flags |= SchemaV4.PLAY_POINT_SHOWN
                    p_tick = snap.tick
                self.recent_enemy_plays.appendleft(
                    _EnemyPlayRecord(
                        card_name=base_c,
                        cast_name=cast_c,
                        form=form,
                        play_tick=snap.tick,
                        seat_x=sx,
                        seat_y=sy,
                        kind=p_kind,
                        flags=flags,
                        point_tick=p_tick,
                        eids=tuple(e.eid for e in bodies),
                    )
                )
            elif i < len(newly_seen_scids) and newly_seen_scids[i] in self._scid_to_name:
                sc_name = self._scid_to_name[newly_seen_scids[i]]
                self.recent_enemy_plays.appendleft(
                    _EnemyPlayRecord(
                        card_name=sc_name,
                        cast_name=sc_name,
                        form=SchemaV4.FORM_PLAIN,
                        play_tick=snap.tick,
                        seat_x=0,
                        seat_y=0,
                        kind=self._card_play_kind.get(sc_name, SchemaV4.PLAY_TROOP),
                        flags=SchemaV4.PLAY_CARD_SHOWN,
                        point_tick=0,
                    )
                )
            else:
                self.recent_enemy_plays.appendleft(
                    _EnemyPlayRecord(
                        card_name=None,
                        cast_name=None,
                        form=-1,
                        play_tick=snap.tick,
                        seat_x=-1,
                        seat_y=-1,
                        kind=-1,
                        flags=0,
                        point_tick=0,
                    )
                )
        self._enemy_play_count = enemy_sum.played_count

        hidden_proj_eids = self._enemy_launched_hidden_eids(other_ents, live_eids, resolved_seat)
        o[SchemaV4.H_ENEMY_PLAYS : SchemaV4.H_QUEUE] = 0
        for r in range(SchemaV4.ENEMY_PLAY_ROWS):
            at = SchemaV4.H_ENEMY_PLAYS + r * SchemaV4.ENEMY_PLAY_INTS
            if r >= len(self.recent_enemy_plays):
                o[at : at + 2] = -1
                continue
            rec = self.recent_enemy_plays[r]
            card_id = self.vocab.row_id(rec.card_name) if rec.card_name else -1
            cast_id = self.vocab.row_id(rec.cast_name) if rec.cast_name else -1
            pt_shown = (rec.flags & SchemaV4.PLAY_POINT_SHOWN) != 0
            o[at : at + 9] = (
                card_id,
                cast_id,
                rec.form if rec.card_name else -1,
                max(0, snap.tick - rec.play_tick),
                rec.seat_x if pt_shown else (0 if rec.card_name else -1),
                rec.seat_y if pt_shown else (0 if rec.card_name else -1),
                rec.kind,
                rec.flags,
                max(0, snap.tick - rec.point_tick) if pt_shown else 0,
            )
        return hidden_proj_eids

    @staticmethod
    def _clear_buff_slots(o: np.ndarray, at: int, *, unknown_ms: bool = False) -> None:
        ms_fill = SchemaV4.UNKNOWN if unknown_ms else 0
        for b_idx in range(SchemaV4.BUFF_SLOTS):
            o[at + SchemaV4.E_BUFFS + 2 * b_idx] = -1
            o[at + SchemaV4.E_BUFFS + 2 * b_idx + 1] = ms_fill

    def _write_entity_target(
        self,
        o: np.ndarray,
        at: int,
        target_eid: int | None,
        resolved_seat: int,
        id_to_seat_pos: dict[int, tuple[int, int]],
        ent_by_eid: dict[int, NullsEntity],
        *,
        hide_enemy_underground: bool = False,
    ) -> None:
        if target_eid is None or target_eid not in id_to_seat_pos:
            o[at + SchemaV4.E_TARGET] = SchemaV4.NO_TARGET
            o[at + SchemaV4.E_TARGET_X] = 0
            o[at + SchemaV4.E_TARGET_Y] = 0
            return
        if hide_enemy_underground:
            tgt = ent_by_eid.get(target_eid)
            if tgt is not None and tgt.side == 1 - resolved_seat and tgt.is_underground:
                o[at + SchemaV4.E_TARGET] = SchemaV4.NO_TARGET
                o[at + SchemaV4.E_TARGET_X] = 0
                o[at + SchemaV4.E_TARGET_Y] = 0
                return
        tx, ty = id_to_seat_pos[target_eid]
        o[at + SchemaV4.E_TARGET] = self._seat_id_of(target_eid)
        o[at + SchemaV4.E_TARGET_X] = tx
        o[at + SchemaV4.E_TARGET_Y] = ty

    def _compute_world_flags(
        self, ent: NullsEntity, kind: int, nname: str | None
    ) -> int:
        unit = self.catalog.units.get(nname or "")
        alive = ent.hp is None or ent.hp > 0
        flags = SchemaV4.F_ALIVE if alive else 0
        tags1 = ent.tags1 if ent.tags1 is not None else 0
        base_air = (ent.z > 0) or (nname in self._air_units)
        if ent.tags1 is not None:
            is_air = bool(tags1 & TAG_FORCE_AIR) or (base_air and not bool(tags1 & TAG_FORCE_GROUND))
        else:
            is_air = base_air
        if is_air:
            flags |= SchemaV4.F_AIR
        is_building = (unit is not None and unit.building) or kind == SchemaV4.KIND_TOWER
        is_king = (unit is not None and unit.king) or (
            kind == SchemaV4.KIND_TOWER and nname == "KingTower"
        )
        if is_building:
            flags |= SchemaV4.F_BUILDING
        if is_king:
            flags |= SchemaV4.F_KING
        if ent.is_clone:
            flags |= SchemaV4.F_CLONE
        if ent.is_underground:
            flags |= SchemaV4.F_UNDERGROUND
        if ent.is_invisible or (ent.buffs and any(b.invisible for b in ent.buffs)):
            flags |= SchemaV4.F_INVISIBLE
        if ent.tags1 is not None:
            if tags1 & TAG_HIDDEN:
                flags |= SchemaV4.F_HIDDEN
            if tags1 & TAG_NO_MOVE:
                flags |= SchemaV4.F_NO_MOVE
            if tags1 & TAG_NO_ATTACK:
                flags |= SchemaV4.F_NO_ATTACK
            if tags1 & TAG_CHARGING:
                flags |= SchemaV4.F_CHARGED
            if kind == SchemaV4.KIND_TOWER and not (tags1 & TAG_INACTIVE):
                flags |= SchemaV4.F_AWAKE
        elif kind == SchemaV4.KIND_TOWER:
            if not is_king or (ent.hp is not None and ent.max_hp is not None and ent.hp < ent.max_hp):
                flags |= SchemaV4.F_AWAKE
        return flags

    def _fill_entity_row(
        self,
        o: np.ndarray,
        at: int,
        ent: NullsEntity,
        kind: int,
        nname: str | None,
        snap: NullsBattleSnapshot,
        resolved_seat: int,
        initial_frame: bool,
        id_to_seat_pos: dict[int, tuple[int, int]],
        ent_by_eid: dict[int, NullsEntity],
        hidden_proj_eids: set[int],
        next_prev_pos: dict[int, tuple[int, str | None, int, int]],
    ) -> None:
        sx, sy = id_to_seat_pos[ent.eid]
        next_prev_pos[ent.eid] = (kind, nname, sx, sy)
        prev_info = self._prev_pos.get(ent.eid)
        if prev_info is not None and prev_info[0] == kind and prev_info[1] == nname:
            dx = sx - prev_info[2]
            dy = sy - prev_info[3]
        else:
            dx, dy = 0, 0

        rel_side = (
            0 if ent.side == resolved_seat else (1 if ent.side == 1 - resolved_seat else 2)
        )
        o[at + SchemaV4.E_ID] = self._seat_id_of(ent.eid)
        o[at + SchemaV4.E_KIND] = kind
        o[at + SchemaV4.E_SIDE] = rel_side
        o[at + SchemaV4.E_ROW] = self.vocab.row_id(nname) if nname else SchemaV4.UNKNOWN
        o[at + SchemaV4.E_X] = sx
        o[at + SchemaV4.E_Y] = sy
        o[at + SchemaV4.E_DX] = dx
        o[at + SchemaV4.E_DY] = dy

        if kind in (SchemaV4.KIND_PROJECTILE, SchemaV4.KIND_AREA):
            o[at + SchemaV4.E_HP] = 0
            o[at + SchemaV4.E_MAX_HP] = 0
            o[at + SchemaV4.E_SHIELD] = 0
            o[at + SchemaV4.E_STATE] = SchemaV4.NO_STATE
            o[at + SchemaV4.E_FLAGS] = SchemaV4.F_ALIVE
            o[at + SchemaV4.E_LEVEL] = _entity_level(ent.level_raw)
            self._write_entity_target(o, at, ent.target_eid, resolved_seat, id_to_seat_pos, ent_by_eid)
            o[at + SchemaV4.E_DEPLOY_MS] = 0
            o[at + SchemaV4.E_ATTACK_MS] = 0
            o[at + SchemaV4.E_LOAD_MS] = 0
            o[at + SchemaV4.E_SPEED_PERCENT] = 0
            o[at + SchemaV4.E_HIT_SPEED_PERCENT] = 0
            o[at + SchemaV4.E_BUFF_COUNT] = 0
            self._clear_buff_slots(o, at, unknown_ms=False)
            if kind == SchemaV4.KIND_PROJECTILE:
                if ent.eid in hidden_proj_eids or ent.aim_x is None or ent.aim_y is None:
                    o[at + SchemaV4.E_AIM_X] = SchemaV4.UNKNOWN
                    o[at + SchemaV4.E_AIM_Y] = SchemaV4.UNKNOWN
                else:
                    o[at + SchemaV4.E_AIM_X] = SchemaV4.to_seat_x(resolved_seat, ent.aim_x)
                    o[at + SchemaV4.E_AIM_Y] = SchemaV4.to_seat_y(resolved_seat, ent.aim_y)
                o[at + SchemaV4.E_TIME_LEFT_MS] = 0
                o[at + SchemaV4.E_RADIUS] = 0
            else:
                o[at + SchemaV4.E_AIM_X] = 0
                o[at + SchemaV4.E_AIM_Y] = 0
                o[at + SchemaV4.E_TIME_LEFT_MS] = (
                    max(0, ent.time_left_ms) if ent.time_left_ms is not None else SchemaV4.UNKNOWN
                )
                o[at + SchemaV4.E_RADIUS] = (
                    ent.radius if ent.radius is not None else SchemaV4.UNKNOWN
                )
            o[at + SchemaV4.E_ABILITY_TICKS] = SchemaV4.NO_STATE
            return

        flags = self._compute_world_flags(ent, kind, nname)
        if ent.side == 1 - resolved_seat and ent.is_underground:
            o[at + SchemaV4.E_HP] = SchemaV4.UNKNOWN
            o[at + SchemaV4.E_MAX_HP] = SchemaV4.UNKNOWN
            o[at + SchemaV4.E_SHIELD] = SchemaV4.UNKNOWN
            o[at + SchemaV4.E_STATE] = ent.state if ent.state is not None else 6
            o[at + SchemaV4.E_FLAGS] = flags
            o[at + SchemaV4.E_LEVEL] = SchemaV4.UNKNOWN
            o[at + SchemaV4.E_TARGET] = SchemaV4.NO_TARGET
            o[at + SchemaV4.E_TARGET_X] = 0
            o[at + SchemaV4.E_TARGET_Y] = 0
            o[at + SchemaV4.E_DEPLOY_MS] = 0
            o[at + SchemaV4.E_ATTACK_MS] = 0
            o[at + SchemaV4.E_LOAD_MS] = 0
            o[at + SchemaV4.E_SPEED_PERCENT] = 100
            o[at + SchemaV4.E_HIT_SPEED_PERCENT] = 100
            o[at + SchemaV4.E_BUFF_COUNT] = 0
            self._clear_buff_slots(o, at, unknown_ms=False)
            o[at + SchemaV4.E_AIM_X] = 0
            o[at + SchemaV4.E_AIM_Y] = 0
            o[at + SchemaV4.E_TIME_LEFT_MS] = 0
            o[at + SchemaV4.E_RADIUS] = 0
            o[at + SchemaV4.E_ABILITY_TICKS] = SchemaV4.NO_STATE
            return

        o[at + SchemaV4.E_HP] = ent.hp if ent.hp is not None else SchemaV4.UNKNOWN
        o[at + SchemaV4.E_MAX_HP] = ent.max_hp if ent.max_hp is not None else SchemaV4.UNKNOWN
        o[at + SchemaV4.E_SHIELD] = ent.shield if ent.shield is not None else SchemaV4.UNKNOWN

        stats = self._unit_stats.get(nname or "", {})
        deploy_time = max(0, int(stats.get("DeployTime", 1000) or 0))
        if ent.eid not in self.first_seen_ms:
            self.first_seen_ms[ent.eid] = (
                snap.time_ms - deploy_time
                if initial_frame or kind == SchemaV4.KIND_TOWER
                else snap.time_ms
            )
        rem_deploy = max(0, deploy_time - (snap.time_ms - self.first_seen_ms[ent.eid]))
        if dx or dy:
            self.first_seen_ms[ent.eid] = min(
                self.first_seen_ms[ent.eid], snap.time_ms - deploy_time
            )
            rem_deploy = 0
        if ent.deploy_ms is not None and ent.deploy_ms >= 0:
            rem_deploy = (
                (ent.delay_ms if ent.state == 11 and ent.delay_ms and ent.delay_ms > 0 else 0)
                + ent.deploy_ms
            )
        elif ent.state is not None:
            rem_deploy = 0 if kind == SchemaV4.KIND_TOWER else SchemaV4.UNKNOWN

        inferred_state = (
            4
            if rem_deploy > 0
            else (1 if dx or dy else (2 if ent.target_eid is not None and ent.target_eid > 0 else 0))
        )
        o[at + SchemaV4.E_STATE] = (
            ent.state if ent.state is not None and 0 <= ent.state <= 16 else inferred_state
        )
        o[at + SchemaV4.E_FLAGS] = flags
        o[at + SchemaV4.E_LEVEL] = _entity_level(ent.level_raw)
        self._write_entity_target(
            o, at, ent.target_eid, resolved_seat, id_to_seat_pos, ent_by_eid, hide_enemy_underground=True
        )
        o[at + SchemaV4.E_DEPLOY_MS] = (
            0 if (kind == SchemaV4.KIND_TOWER and ent.deploy_ms is None) else rem_deploy
        )
        o[at + SchemaV4.E_ATTACK_MS] = (
            ent.attack_ms if ent.attack_ms is not None and ent.attack_ms >= 0 else SchemaV4.UNKNOWN
        )
        o[at + SchemaV4.E_LOAD_MS] = (
            ent.load_ms if ent.load_ms is not None and ent.load_ms >= 0 else SchemaV4.UNKNOWN
        )
        o[at + SchemaV4.E_SPEED_PERCENT] = (
            ent.speed_percent if ent.speed_percent is not None else SchemaV4.UNKNOWN
        )
        o[at + SchemaV4.E_HIT_SPEED_PERCENT] = (
            ent.hit_speed_percent if ent.hit_speed_percent is not None else SchemaV4.UNKNOWN
        )

        if ent.buffs is None:
            o[at + SchemaV4.E_BUFF_COUNT] = SchemaV4.UNKNOWN
            self._clear_buff_slots(o, at, unknown_ms=True)
        else:
            sorted_buffs = sorted(ent.buffs, key=_buff_slot_rank)
            listed = min(len(sorted_buffs), SchemaV4.BUFF_SLOTS)
            o[at + SchemaV4.E_BUFF_COUNT] = len(sorted_buffs)
            for b_idx in range(SchemaV4.BUFF_SLOTS):
                slot_at = at + SchemaV4.E_BUFFS + 2 * b_idx
                if b_idx >= listed:
                    o[slot_at] = -1
                    o[slot_at + 1] = 0
                    continue
                buff = sorted_buffs[b_idx]
                bname = self.aliases.get(buff.name, buff.name)
                bid = self.vocab.buff_id(bname)
                if bid == SchemaV4.UNKNOWN and buff.name:
                    self.unmatched_names.add(buff.name)
                o[slot_at] = bid
                o[slot_at + 1] = buff.time_left

        o[at + SchemaV4.E_AIM_X] = 0
        o[at + SchemaV4.E_AIM_Y] = 0
        o[at + SchemaV4.E_TIME_LEFT_MS] = 0
        o[at + SchemaV4.E_RADIUS] = 0
        if ent.eid in self._ability_ran_tick:
            o[at + SchemaV4.E_ABILITY_TICKS] = max(0, snap.tick - self._ability_ran_tick[ent.eid])
        elif ent.ability_ticks is not None and ent.ability_ticks > 0:
            o[at + SchemaV4.E_ABILITY_TICKS] = ent.ability_ticks
        else:
            o[at + SchemaV4.E_ABILITY_TICKS] = SchemaV4.NO_STATE

    def adapt(
        self,
        snap: NullsBattleSnapshot,
        *,
        own_seat: int | None = None,
        pending_slots: tuple[bool, bool, bool, bool] = (False, False, False, False),
        pending_targets: Sequence[tuple[int, int] | None] | None = None,
        last_refusal: int | None = None,
        last_run_refusal: int | None = None,
        illegal: int | None = None,
        last_ability_run_refusal: int | None = None,
    ) -> SchemaObservation:
        resolved_seat = own_seat if own_seat is not None else snap.own_seat
        if resolved_seat is None:
            resolved_seat = self.default_own_seat
        if resolved_seat not in (0, 1):
            raise ValueError(
                f"Cannot determine own_seat from snapshot (got {resolved_seat}); pass own_seat explicitly"
            )

        own_sum = snap.summoners[resolved_seat]
        enemy_sum = snap.summoners[1 - resolved_seat]
        deck_sig = tuple(
            f"{self.normalize_name(c.name)}:{c.level_raw}" for c in own_sum.deck
        ) + (str(resolved_seat),)

        if self._last_tick >= 0 and (snap.tick < self._last_tick or deck_sig != self._deck_sig):
            self.reset_match()
        elif self._last_tick >= 0 and snap.tick == self._last_tick and self._pre_tick_ckpt is not None:
            self._restore_checkpoint(self._pre_tick_ckpt)

        self._pre_tick_ckpt = self._snapshot_checkpoint()
        initial_frame = self._last_time_ms is None
        if not initial_frame and self._last_time_ms is not None and snap.tick > self._last_tick:
            if own_sum.elixir >= 10 * SchemaV4.ELIXIR_SCALE:
                self._full_bar_ms += max(0, snap.time_ms - self._last_time_ms)
        self._last_tick = snap.tick
        self._deck_sig = deck_sig

        for summ in snap.summoners:
            for card in summ.deck:
                cname = self.normalize_name(card.name)
                if cname and card.scid > 0:
                    self._scid_to_name[card.scid] = cname

        hand_names, hand_ids, hand_costs = self._track_hand_and_mirror(own_sum)

        o = np.zeros(SchemaV4.SEAT_INTS, dtype=np.int32)
        o[SchemaV4.H_TICK] = snap.tick
        o[SchemaV4.H_RESULT] = SchemaV4.RUNNING if not snap.clock_ended else SchemaV4.UNKNOWN
        o[SchemaV4.H_ELIXIR] = own_sum.elixir

        self._fill_deck_and_form_headers(o, own_sum, enemy_sum, hand_names)
        (
            world_ents,
            other_ents,
            live_towers,
            mask_entities,
            id_to_seat_pos,
            live_eids,
            ent_by_eid,
            rejected_entities,
        ) = self._collect_entities_and_towers(o, snap, resolved_seat)

        ordered = [*world_ents, *reversed(other_ents)]
        row_count = min(len(ordered), SchemaV4.MAX_ENTITIES)
        o[SchemaV4.H_ENTITY_COUNT] = row_count
        o[SchemaV4.H_ENTITY_OVERFLOW] = max(0, len(ordered) - SchemaV4.MAX_ENTITIES)
        o[SchemaV4.H_DECISION_TICKS] = self.decision_ticks
        o[SchemaV4.H_PLAY_DELAY_TICKS] = SchemaV4.PLAY_DELAY_TICKS
        o[SchemaV4.H_SEAT] = resolved_seat

        ability_ready_flags = self._fill_pending_and_ability_headers(
            o,
            own_sum,
            snap,
            resolved_seat,
            hand_ids,
            hand_costs,
            pending_slots,
            pending_targets,
            live_eids,
            ent_by_eid,
        )

        o[SchemaV4.H_EPISODE_STEP] = self._episode_step
        o[SchemaV4.H_TIME_MS] = snap.time_ms
        sec_type = (
            snap.section_type
            if snap.section_type is not None
            else (SchemaV4.SECTION_OVERTIME if snap.phase > 0 else SchemaV4.SECTION_NORMAL)
        )
        o[SchemaV4.H_SECTION] = snap.section if snap.section is not None else snap.phase
        o[SchemaV4.H_SECTION_TYPE] = sec_type
        o[SchemaV4.H_SECTION_LEFT_MS] = (
            max(0, snap.sec_len_ms - snap.sec_elapsed_ms)
            if snap.sec_len_ms > 0
            else SchemaV4.UNKNOWN
        )
        o[SchemaV4.H_FULL_BAR_MS] = self._full_bar_ms
        o[SchemaV4.H_OVERTIME] = (
            1 if sec_type in (SchemaV4.SECTION_OVERTIME, SchemaV4.SECTION_BONUS_TIME) else 0
        )
        o[SchemaV4.H_PLAY_OPEN_IN_TICKS] = max(0, START_COOLDOWN_TICKS - max(0, snap.tick))
        o[SchemaV4.H_LAST_REFUSAL] = last_refusal if last_refusal is not None else SchemaV4.UNKNOWN
        o[SchemaV4.H_LAST_RUN_REFUSAL] = (
            last_run_refusal if last_run_refusal is not None else SchemaV4.UNKNOWN
        )
        o[SchemaV4.H_ILLEGAL] = illegal if illegal is not None else SchemaV4.UNKNOWN
        o[SchemaV4.H_LAST_ABILITY_RUN_REFUSAL] = (
            last_ability_run_refusal
            if last_ability_run_refusal is not None
            else SchemaV4.UNKNOWN
        )

        hidden_proj_eids = self._update_and_write_enemy_plays(
            o, snap, enemy_sum, resolved_seat, world_ents, other_ents, live_eids
        )

        next_prev_pos: dict[int, tuple[int, str | None, int, int]] = {}
        for row_idx in range(row_count):
            ent, kind, nname = ordered[row_idx]
            at = SchemaV4.HEADER_INTS + row_idx * SchemaV4.ENTITY_INTS
            self._fill_entity_row(
                o,
                at,
                ent,
                kind,
                nname,
                snap,
                resolved_seat,
                initial_frame,
                id_to_seat_pos,
                ent_by_eid,
                hidden_proj_eids,
                next_prev_pos,
            )

        self._last_time_ms = snap.time_ms
        self._prev_pos = next_prev_pos
        self._episode_step += 1

        own_king_alive = (resolved_seat, 0) in live_towers or not self._saw_side_towers.get(
            resolved_seat, False
        )
        reserved_for_mask = (
            own_sum.elixir_reserved
            if own_sum.elixir_reserved is not None and 0 <= own_sum.elixir_reserved <= 200_000
            else 0
        )
        available_whole = max(
            0, (own_sum.elixir - reserved_for_mask) // SchemaV4.ELIXIR_SCALE
        )
        effective_pending = [
            bool(o[SchemaV4.H_HAND_PENDING + k]) for k in range(SchemaV4.HAND_SLOTS)
        ]
        legal_list = build_legal_mask(
            self.catalog,
            self.tile_map,
            resolved_seat,
            mask_entities,
            hand_names,
            [max(0, c) for c in hand_costs],
            effective_pending,
            available_elixir_whole=available_whole,
            play_open=int(o[SchemaV4.H_PLAY_OPEN_IN_TICKS]) <= 0,
            match_running=not snap.clock_ended,
            own_king_alive=own_king_alive,
            last_played_card=self._last_played_card,
            ability_ready=(ability_ready_flags[0], ability_ready_flags[1]),
        )

        return SchemaObservation(
            obs=o,
            legal_mask=np.asarray(legal_list, dtype=bool),
            own_seat=resolved_seat,
            tick=snap.tick,
            time_ms=snap.time_ms,
            hand_cards=(
                hand_names[0],
                hand_names[1],
                hand_names[2],
                hand_names[3],
            ),
            hand_card_ids=(
                hand_ids[0],
                hand_ids[1],
                hand_ids[2],
                hand_ids[3],
            ),
            unmatched_names=tuple(sorted(self.unmatched_names)),
            rejected_entities=tuple(rejected_entities),
        )
