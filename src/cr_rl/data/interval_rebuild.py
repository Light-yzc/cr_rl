"""Causal command extraction/planning for a new simulator decision interval.

This module never downsamples already labelled observations. Plans are lower
bounds on submission times; actual masks, pending commands and native results
must be checked by a fresh simulator execution.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re


EVENT_KINDS = {"play_card", "ability", "activate_ability", "ability_activation"}


def recorded_source_events(replay, *, allow_incomplete=False):
    """Recover source commands from execution ledger, rejecting missing commands.

    A native command JSON is insufficient: hand repairs and original source
    times are not reproduced by that file. Failed/early-ended source events may
    have no embedded source_event, so completeness is checked against event IDs.
    """
    by_id = {}
    for row in replay.get("repair_records", []):
        event = row.get("source_event")
        if row.get("category") != "execution" or event is None:
            continue
        event_id = int(row["event_id"])
        source_tick = int(row["source_tick"])
        if int(event.get("replay_tick_20hz", source_tick)) != source_tick:
            raise ValueError("Ledger source time disagrees with original command")
        value = (source_tick, event)
        if event_id in by_id and by_id[event_id] != value:
            raise ValueError("Conflicting copies of the same source event")
        by_id[event_id] = value
    total = int(replay.get("total_play_events", len(by_id)))
    missing = sorted(set(range(total)) - by_id.keys())
    if any(key < 0 or key >= total for key in by_id):
        raise ValueError("Ledger event identity is outside its declared source stream")
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)}/{total} original commands; supply source parquet")
    events = []
    for event_id, (tick, event) in sorted(by_id.items(), key=lambda pair: (pair[1][0], pair[0])):
        item = deepcopy(event)
        item["replay_tick_20hz"] = tick
        item["original_event_id"] = event_id
        events.append(item)
    return events, {
        "mode": "execution_ledger",
        "declared_source_events": total,
        "available_source_events": len(events),
        "source_complete": not missing,
        "missing_event_ids": missing,
        "source_command_coverage": len(events) / max(total, 1),
    }


def causal_grid_plan(events, decision_ticks):
    """Stable FIFO, one action per seat/grid cell, never advance a command.

    Separate seats can submit simultaneously. Same-seat collisions are queued
    to later cells, never overwritten or silently dropped. This offline plan
    ignores resource/pending constraints, so it is not a simulator trajectory.
    """
    if decision_ticks < 1:
        raise ValueError("decision_ticks must be positive")
    previous = {"team": -decision_ticks, "opponent": -decision_ticks}
    occupied = Counter()
    result = []
    for index, event in sorted(enumerate(events), key=lambda pair: (int(pair[1]["replay_tick_20hz"]), pair[0])):
        if event.get("kind") not in EVENT_KINDS or event.get("side") not in previous:
            raise ValueError("Unsupported source event kind or seat")
        source_tick = int(event["replay_tick_20hz"])
        if source_tick < 0:
            raise ValueError("Negative source time")
        seat = event["side"]
        ceil_tick = (source_tick + decision_ticks - 1) // decision_ticks * decision_ticks
        occupied[(seat, ceil_tick)] += 1
        assigned_tick = max(ceil_tick, previous[seat] + decision_ticks)
        previous[seat] = assigned_tick
        result.append({"source_index": event.get("source_index", index), "seat": seat,
            "source_tick": source_tick, "ceil_tick": ceil_tick,
            "planned_submission_tick": assigned_tick,
            "grid_delay_ticks": ceil_tick-source_tick,
            "fifo_extra_ticks": assigned_tick-ceil_tick,
            "kind": event["kind"], "card_key": event.get("card_key")})
    return result, {
        "source_events": len(events),
        "same_seat_collision_events": sum(max(0, count-1) for count in occupied.values()),
        "collision_policy": "FIFO postponement; one command per seat per decision; no overwrite",
        "grid_delay_ticks": quantiles([r["grid_delay_ticks"] for r in result]),
        "fifo_extra_ticks": quantiles([r["fifo_extra_ticks"] for r in result]),
        "early_commands": sum(r["planned_submission_tick"] < r["source_tick"] for r in result),
    }


def canonicalise_events(events, decks, aliases):
    """Map source form keys to pinned native deck identities without changing time.

    Original keys are retained. Form flags still come from the original deck
    slots; this maps command identity, not evolution mechanics.
    """
    normalise = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
    result = []
    for event in events:
        item = deepcopy(event)
        seat = {"team": 0, "opponent": 1}.get(item.get("side"))
        if seat is None:
            raise ValueError("Unknown event seat")
        keys = (item.get("deck_card_key_candidates") or item.get("ability_source_candidates") or [])
        keys = list(keys) + ([item["card_key"]] if item.get("card_key") else [])
        mapped = []
        for key in keys:
            # Upstream form identity is a deck slot flag, not a different card.
            base = re.sub(r"-(ev\d+|hero)$", "", key)
            native = normalise(aliases.get(normalise(base), base))
            mapped.extend(card for card in decks[seat] if normalise(card) == native)
        mapped = list(dict.fromkeys(mapped))
        if not mapped:
            raise ValueError(f"Cannot map source command {keys!r} to pinned deck")
        if item["kind"] == "play_card" and len(mapped) != 1:
            raise ValueError("Ordinary card identity is ambiguous")
        item["original_deck_card_key_candidates"] = item.get("deck_card_key_candidates", [])
        item["deck_card_key_candidates"] = mapped
        result.append(item)
    return result


def quantiles(values):
    values = sorted(values)
    return {"count": len(values), **{name: values[round((len(values)-1)*fraction)] if values else None
        for name, fraction in [("min",0), ("p50",.5), ("p90",.9), ("p95",.95), ("p99",.99), ("max",1)]}}


def execution_interval_audit(stats, source_count):
    """Keep quantisation, extra waits, and fixed native command delay distinct."""
    confirmed = [r for r in stats.get("repair_records", [])
        if r.get("category") == "execution" and r.get("native_success") is True]
    delays = [r["simulator_tick"]-r["source_tick"] for r in confirmed]
    native = [r["actual_run_tick"]-r["simulator_tick"] for r in confirmed if "actual_run_tick" in r]
    if any(delay < 0 for delay in delays):
        raise ValueError("A source command was executed before its causal source time")
    return {"source_events": source_count, "confirmed_commands": len(confirmed),
        "command_retention": len({r["event_id"] for r in confirmed})/max(1,source_count),
        "submission_minus_source_ticks": quantiles(delays),
        "native_run_minus_submission_ticks": quantiles(native),
        "early_commands": 0,
        "native_run_refusals": len(stats.get("native_run_refusals", [])),
        "completion_kind": stats.get("trajectory_completion_kind"),
        "simulator_fault": stats.get("simulator_fault"),
        "scheduler": ("legacy FIFO; await same-seat native confirmation" if
            stats.get("serial_native_confirmation", True) else
            "causal FIFO one command per seat/grid; different legal pending cards allowed; native failure fatal")}


def strict_command_admission(stats, arrays, source_events, decision_ticks):
    """Require all original commands to have one distinct confirmed legal row.

    Partial games retain their external audit ledger but are never silently
    called all-preserved data. Terminal source tails cannot be fabricated.
    """
    import numpy as np
    reasons=[]
    ordered=[event for _,event in sorted(enumerate(source_events),
        key=lambda pair:(int(pair[1]["replay_tick_20hz"]),pair[0]))]
    count=len(ordered)
    by_id={}
    for record in stats.get("repair_records",[]):
        if record.get("category")=="execution":
            by_id.setdefault(int(record["event_id"]),[]).append(record)
    if stats.get("simulator_fault"):
        reasons.append("simulator_fault")
    if stats.get("native_run_refusals"):
        reasons.append("native_run_refusal")
    if stats.get("omitted_command_ticks"):
        reasons.append("omitted_source_commands")
    if not stats.get("all_source_events_executed"):
        reasons.append("source_stream_incomplete")
    if not stats.get("pending_commands_settled"):
        reasons.append("pending_commands_unsettled")
    if any(record.get("category")=="ability_skipped" for record in stats.get("repair_records",[])):
        reasons.append("ability_skipped")
    if arrays is None:
        reasons.append("no_live_rows")
        return {"accepted":False,"source_commands":count,"verified_supervised_commands":0,
                "retention":0.,"reasons":reasons}
    ticks=np.asarray(arrays["ticks"]);seats=np.asarray(arrays["seats"])
    actions=np.asarray(arrays["actions"]);flags=np.asarray(arrays["flags"])
    legal=np.asarray(arrays["masks"])
    if len(actions)==0 or np.any(ticks%decision_ticks):
        reasons.append("rows_outside_decision_grid_or_empty")
    row_keys={}
    for index,(tick,seat) in enumerate(zip(ticks,seats)):
        row_keys.setdefault((int(tick),int(seat)),[]).append(index)
    mapped_rows=set();verified=0;mapping=[];previous_submission={0:-decision_ticks,1:-decision_ticks}
    if set(by_id)-set(range(count)):
        reasons.append("unexpected_source_event_id")
    for event_id,event in enumerate(ordered):
        records=by_id.get(event_id,[])
        if len(records)!=1:
            reasons.append(f"event_{event_id}_missing_or_duplicate_submission")
            continue
        record=records[0]
        expected_seat={"team":0,"opponent":1}.get(event.get("side"))
        source_tick=int(event["replay_tick_20hz"])
        submitted=int(record["simulator_tick"])
        if (record.get("native_success") is not True or record.get("execution_status")!="confirmed"
                or record.get("accepted") is not True):
            reasons.append(f"event_{event_id}_not_native_confirmed")
            continue
        if record.get("seat")!=expected_seat or int(record["source_tick"])!=source_tick:
            reasons.append(f"event_{event_id}_source_identity_mismatch")
            continue
        if submitted<source_tick or submitted%decision_ticks:
            reasons.append(f"event_{event_id}_early_or_off_grid")
            continue
        if submitted<=previous_submission[expected_seat]:
            reasons.append(f"event_{event_id}_fifo_order_or_seat_grid_collision")
            continue
        previous_submission[expected_seat]=submitted
        if int(record.get("actual_run_tick",-1))<submitted:
            reasons.append(f"event_{event_id}_native_time_before_submission")
            continue
        candidates=row_keys.get((submitted,expected_seat),[])
        if len(candidates)!=1:
            reasons.append(f"event_{event_id}_missing_or_duplicate_live_row")
            continue
        row=candidates[0]
        action=int(record["executed_action"])
        if (row in mapped_rows or action<=0 or action>=legal.shape[-1] or
                int(actions[row])!=action or flags[row]!=0 or not bool(legal[row,action])):
            reasons.append(f"event_{event_id}_not_unique_legal_supervision")
            continue
        mapped_rows.add(row);verified+=1
        mapping.append({"event_id":event_id,"row":row,"source_tick":source_tick,
            "submission_tick":submitted,"seat":expected_seat,"action":action})
    supervised_actions=set(np.flatnonzero((actions!=0)&(flags==0)).tolist())
    if supervised_actions!=mapped_rows:
        reasons.append("unmatched_or_missing_nonwait_supervision")
    if set(np.flatnonzero(actions!=0).tolist())!=mapped_rows:
        reasons.append("flagged_or_extra_nonwait_command_row")
    accepted=not reasons and verified==count and count>0
    return {"accepted":accepted,"source_commands":count,"verified_supervised_commands":verified,
        "retention":verified/max(1,count),"reasons":reasons,"source_to_live_row":mapping}
