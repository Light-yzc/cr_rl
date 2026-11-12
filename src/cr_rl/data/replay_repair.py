"""Explicit, auditable repairs for event-to-simulator replay alignment.

This module plans repairs without mutating a simulator. It aligns ticks, snaps
placement requests and constructs same-card hand/queue swap plans. The pinned
JVM bridge in ``opening_cycle`` applies those plans. Every repair is represented
as a ledger record so repaired replays
cannot be confused with exact human-state replays.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np

GRID_WIDTH = 18
GRID_HEIGHT = 32
PLAY_ACTIONS = 4 * GRID_WIDTH * GRID_HEIGHT
PLAY_FIRST = 1
PLAY_LAST = PLAY_FIRST + PLAY_ACTIONS - 1
ACTION_COUNT = PLAY_LAST + 3


@dataclass(frozen=True)
class RepairConfig:
    """Conservative repair policy.

    ``time_tolerance_ticks`` is measured on the 50 ms simulator clock.  A
    placement can be snapped only within ``max_tile_shift`` Manhattan cells;
    hand/cycle and elixir mismatches are never silently repaired.
    """

    decision_ticks: int = 5
    time_tolerance_ticks: int = 2
    max_tile_shift: int = 2
    allow_tile_snap: bool = False
    allow_hand_replacement: bool = False
    max_hand_replacements: int | None = None

    def __post_init__(self):
        if self.decision_ticks < 1:
            raise ValueError("decision_ticks must be positive")
        if self.time_tolerance_ticks < 0:
            raise ValueError("time_tolerance_ticks cannot be negative")
        if self.max_tile_shift < 0:
            raise ValueError("max_tile_shift cannot be negative")
        if self.max_hand_replacements is not None and self.max_hand_replacements < 0:
            raise ValueError("max_hand_replacements cannot be negative")
        if self.allow_hand_replacement and self.max_hand_replacements == 0:
            raise ValueError("hand replacement requires a positive replacement budget")


@dataclass(frozen=True)
class TickAlignment:
    source_tick: int
    simulator_tick: int | None
    delta_ticks: int | None
    accepted: bool
    reason: str


@dataclass(frozen=True)
class RepairDecision:
    requested_action: int
    executed_action: int | None
    category: str
    exact: bool
    repaired: bool
    accepted: bool
    distance: int | None = None
    detail: str = ""


@dataclass
class RepairLedger:
    """Machine-readable audit trail for one replay."""

    records: list[dict] = field(default_factory=list)

    def add(self, record: RepairDecision, *, seat: int, source_tick: int):
        if int(seat) not in (0, 1) or int(source_tick) < 0:
            raise ValueError("Ledger seat and source tick must be valid")
        item = asdict(record)
        item.update(seat=int(seat), source_tick=int(source_tick))
        self.records.append(item)

    def summary(self):
        categories = Counter(row["category"] for row in self.records)
        return {
            "schema_version": 1,
            "attempted": len(self.records),
            "exact": sum(bool(row["exact"]) for row in self.records),
            "repaired": sum(bool(row["repaired"]) for row in self.records),
            "accepted": sum(bool(row["accepted"]) for row in self.records),
            "rejected": sum(not bool(row["accepted"]) for row in self.records),
            "categories": dict(sorted(categories.items())),
            "records": [dict(row) for row in self.records],
        }


def align_tick(source_tick: int, config: RepairConfig, *, mode="nearest") -> TickAlignment:
    """Align a source 20 Hz tick to the simulator's decision grid.

    Events farther than the configured tolerance from a decision boundary are
    rejected.  The returned tick is never silently rounded outside the stated
    tolerance.
    """

    if mode not in ("nearest", "ceil"):
        raise ValueError("Time alignment mode must be nearest or ceil")
    source_tick = int(source_tick)
    if source_tick < 0:
        return TickAlignment(source_tick, None, None, False, "negative_tick")
    quotient, remainder = divmod(source_tick, config.decision_ticks)
    if remainder == 0 or (mode == "nearest" and remainder <= config.decision_ticks - remainder):
        aligned = quotient * config.decision_ticks
    else:
        aligned = (quotient + 1) * config.decision_ticks
    delta = aligned - source_tick
    if abs(delta) > config.time_tolerance_ticks:
        return TickAlignment(source_tick, None, delta, False, "time_misaligned")
    return TickAlignment(source_tick, aligned, delta, True, "exact" if delta == 0 else "time_snap")


def decode_play_action(action: int):
    if not PLAY_FIRST <= int(action) <= PLAY_LAST:
        raise ValueError("Action is not a four-slot placement action")
    offset = int(action) - PLAY_FIRST
    slot, tile = divmod(offset, GRID_WIDTH * GRID_HEIGHT)
    return slot, tile % GRID_WIDTH, tile // GRID_WIDTH


def encode_play_action(slot: int, x: int, y: int):
    if not (0 <= slot < 4 and 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
        raise ValueError("Placement needs slot 0..3, x 0..17, y 0..31")
    return PLAY_FIRST + int(slot) * GRID_WIDTH * GRID_HEIGHT + int(y) * GRID_WIDTH + int(x)


def nearest_legal_action(action: int, legal_mask: Iterable[bool], max_distance: int | None):
    """Return a nearby legal tile for the same card slot, or ``None``.

    WAIT, abilities, and non-placement actions are never changed.  Legal tiles
    are chosen deterministically by Manhattan distance and then action index.
    """

    action = int(action)
    mask = np.asarray(legal_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != ACTION_COUNT:
        raise ValueError("legal_mask must be a one-dimensional 2307-action mask")
    if not 0 <= action < ACTION_COUNT:
        raise ValueError("Action is outside the 2307-action codec")
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance cannot be negative")
    if mask[action]:
        return action, 0
    if not PLAY_FIRST <= action <= PLAY_LAST:
        return None
    slot, x, y = decode_play_action(action)
    left = PLAY_FIRST + slot * GRID_WIDTH * GRID_HEIGHT
    right = left + GRID_WIDTH * GRID_HEIGHT
    candidates = np.flatnonzero(mask[left:right]) + left
    if not len(candidates):
        return None
    scored = []
    for candidate in candidates.tolist():
        _, cx, cy = decode_play_action(candidate)
        distance = abs(cx - x) + abs(cy - y)
        scored.append((distance, candidate))
    distance, candidate = min(scored)
    if max_distance is not None and distance > int(max_distance):
        return None
    return int(candidate), int(distance)


def repair_action(
    action: int,
    legal_mask: Iterable[bool],
    *,
    config: RepairConfig,
    hand_contains_card: bool | None = None,
    elixir_legal: bool | None = None,
):
    """Classify/repair one requested action without mutating simulator state.

    ``hand_contains_card`` and ``elixir_legal`` are optional caller-provided
    facts.  A false hand or elixir check is always rejected unless the caller
    explicitly implements a separately recorded external hand repair; this
    function never invents cards or adds elixir.
    """

    action = int(action)
    mask = np.asarray(legal_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != ACTION_COUNT:
        raise ValueError("legal_mask must be a one-dimensional 2307-action mask")
    if action < 0 or action >= len(mask):
        return RepairDecision(action, None, "action_out_of_range", False, False, False)
    if hand_contains_card is False:
        return RepairDecision(action, None, "not_in_hand", False, False, False)
    if elixir_legal is False:
        return RepairDecision(action, None, "elixir_insufficient", False, False, False)
    if mask[action]:
        return RepairDecision(action, action, "exact", True, False, True)
    if not config.allow_tile_snap:
        return RepairDecision(action, None, "mask_closed", False, False, False)
    candidate = nearest_legal_action(action, mask, config.max_tile_shift)
    if candidate is None:
        return RepairDecision(action, None, "mask_closed", False, False, False)
    repaired, distance = candidate
    return RepairDecision(
        action,
        repaired,
        "tile_snap",
        False,
        True,
        True,
        distance,
        "same-slot nearest legal tile",
    )


def plan_hand_swap(slots, queue, deck_index, pending=None, preferred_slot=None):
    """Plan a queue↔hand swap while preserving the native cycle's card multiset.

    Native ``Hand.rearrange`` addresses slots as 0..3 and queue entries as 4+j.
    Only a nonempty, nonpending slot can be displaced. A requested card already
    in a pending slot is rejected so the caller can wait for its command/refill.
    The displaced card occupies the requested card's exact queue position;
    remaining queue order, forms and runtime deck indices are unchanged.
    """
    slots, queue = [int(c) for c in slots], [int(c) for c in queue]
    deck_index = int(deck_index)
    pending = [False] * 4 if pending is None else [bool(p) for p in pending]
    if len(slots) != 4 or len(pending) != 4:
        raise ValueError("Hand and pending mask must have four slots")
    cards = [c for c in slots if c != -1] + queue
    if any(c < 0 or c >= 8 for c in cards) or len(cards) != len(set(cards)):
        raise ValueError("Live hand/queue must contain distinct runtime deck indices")
    if not 0 <= deck_index < 8:
        raise ValueError("Requested runtime deck index must be in 0..7")
    if preferred_slot is not None and not 0 <= int(preferred_slot) < 4:
        raise ValueError("preferred_slot is outside the four hand slots")
    if deck_index in slots:
        slot = slots.index(deck_index)
        if pending[slot]:
            raise ValueError("Requested card is pending; wait for the queued play")
        return {
            "slot": slot, "changed": False, "requested_deck_index": deck_index,
            "displaced_deck_index": None, "queue_index": None, "places": [], "cards": [],
            "before_slots": slots, "before_queue": queue,
            "after_slots": list(slots), "after_queue": list(queue),
        }
    if deck_index not in queue:
        raise ValueError("Requested card is not present in the live hand or queue")
    safe = [i for i, c in enumerate(slots) if c != -1 and not pending[i]]
    if preferred_slot is None:
        if not safe:
            raise ValueError("No nonempty, nonpending slot is available for hand repair")
        slot = safe[0]
    else:
        slot = int(preferred_slot)
        if slot not in safe:
            raise ValueError("Cannot replace an empty or pending hand slot")
    queue_index = queue.index(deck_index)
    displaced = slots[slot]
    after_slots, after_queue = list(slots), list(queue)
    after_slots[slot], after_queue[queue_index] = deck_index, displaced
    return {
        "slot": slot, "changed": True, "requested_deck_index": deck_index,
        "displaced_deck_index": displaced, "queue_index": queue_index,
        "places": [slot, 4 + queue_index], "cards": [deck_index, displaced],
        "before_slots": slots, "before_queue": queue,
        "after_slots": after_slots, "after_queue": after_queue,
    }

