"""Keep usable replay labels without treating omitted human commands as WAIT."""

from __future__ import annotations

import numpy as np

# Separate from the simulator's native row flags; training already rejects nonzero flags.
SOURCE_COMMAND_OMITTED = 1 << 16
GENERATED_WAIT_TAIL = 1 << 17


def prefix_cutoff(omitted, decision_ticks=5, max_omissions=0):
    """Cut both seats at the first command exceeding the declared omission budget."""
    if max_omissions < 0 or decision_ticks < 1:
        raise ValueError("Invalid omission budget or decision interval")
    events = sorted((int(t), int(s)) for t, s in omitted)
    return (
        events[max_omissions][0] // decision_ticks * decision_ticks
        if len(events) > max_omissions
        else None
    )


def mask_omitted_commands(
    flags, ticks, seats, omitted, decision_ticks=5, margin_ticks=0, both_seats=False
):
    result = np.array(flags, dtype=np.int32, copy=True)
    if omitted:
        windows = {int(t) // decision_ticks * decision_ticks * 2 + int(s) for t, s in omitted}
        keys = np.asarray(ticks, dtype=np.int64) * 2 + np.asarray(seats)
        result[np.isin(keys, list(windows))] |= SOURCE_COMMAND_OMITTED
        if margin_ticks:
            for tick, seat in omitted:
                close = np.abs(np.asarray(ticks) - int(tick)) <= margin_ticks
                selected = close if both_seats else close & (np.asarray(seats) == int(seat))
                result[selected] |= SOURCE_COMMAND_OMITTED
    return result


def isolated_wait_rows(waits, actions, replay_ids, seats, ticks, before_ticks=20, after_ticks=10):
    """Exclude near-action WAIT targets within the same match/seat; retain history frames."""
    if min(before_ticks, after_ticks) < 0:
        raise ValueError("WAIT exclusion margins cannot be negative")
    waits = np.asarray(waits)
    keep = np.ones(len(waits), dtype=bool)
    groups = {}
    for row in np.flatnonzero(np.asarray(actions) != 0):
        groups.setdefault((int(replay_ids[row]), int(seats[row])), []).append(ticks[row])
    wait_groups = {}
    for index, row in enumerate(waits):
        wait_groups.setdefault((int(replay_ids[row]), int(seats[row])), []).append(index)
    for key, events in groups.items():
        selected = np.asarray(wait_groups.get(key, []), dtype=np.int64)
        times = np.sort(events)
        query = ticks[waits[selected]]
        # Action time in [WAIT-after, WAIT+before] means this WAIT is ambiguous.
        left = np.searchsorted(times, query - after_ticks, side="left")
        right = np.searchsorted(times, query + before_ticks, side="right")
        keep[selected] = left == right
    return waits[keep]
