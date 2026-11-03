"""Adapt the published Cochon123 cleaned replay tables to the BC event interface.

Schema reference: https://github.com/cochon123/clash-royale-ai at
5609f61a10621ea5afb8669dfb787d6c6fde3cfe. No additional deduplication is performed.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REVISION = "10cca44e9d863d06cf078f142d3e744f78698b0a"
LEGACY_HEROES = {"mini-pekka", "musketeer", "giant", "knight"}


def convert_match(match, events):
    if any(len(match.get("participant_tags", {}).get(s, [])) != 1 for s in ("team", "opponent")):
        raise ValueError("requires_one_player_per_side")
    battle = {}
    for side in ("team", "opponent"):
        hero_cards = {
            e["ability_card"]
            for e in events
            if e.get("event_type") == "ability_activation"
            and e.get("side") == side
            and e.get("ability_card") in LEGACY_HEROES
        }
        cards = []
        for key in match["decks"][side]:
            base = key.removesuffix("-ev1").removesuffix("-hero")
            form_key = key + ("-hero" if base in hero_cards and "-hero" not in key else "")
            cards.append({"name": base, "card_key": form_key, "level": -1})
        battle[side] = {"players": [{"deck": cards}]}
    converted = []
    for event in events:
        ability = event["event_type"] == "ability_activation"
        key = event.get("ability_card") if ability else event.get("card")
        cards = battle[event["side"]]["players"][0]["deck"]
        found = next((c for c in cards if c["name"] == key), None)
        if found is None:
            raise ValueError("event_card_missing_from_deck")
        row = {
            "kind": "ability" if ability else "play_card",
            "side": event["side"],
            "replay_tick_20hz": int(event["ticks"]),
            "card_key": found["card_key"],
        }
        if not ability:
            row["coordinates"] = {
                "native_world_units": {"x": int(event["x"]), "y": int(event["y"])}
            }
        converted.append(row)
    return {
        "schema_version": "royaleapi-battle-actions.v1",
        "battle": battle,
        "events": converted,
        "source": {
            "dataset": "Cochon123/clash-royale-replays",
            "revision": REVISION,
            "replay_tag": "cochon_" + match["battle_id"],
            "source_subset": "cleaned-legacy",
            "card_levels": "unknown",
            "deck_forms": "explicit slugs and labelled legacy Hero abilities only",
            "reconstruction_assumptions": "Unspecified evolution forms use base form; "
            "not a verified reproduction of the original arena state",
            "ability_attributions": sorted(
                {
                    e.get("attribution") or "unavailable"
                    for e in events
                    if e["event_type"] == "ability_activation"
                }
            ),
        },
    }


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    events = collections.defaultdict(list)
    for line in (args.cleaned / "events.jsonl").open(encoding="utf-8"):
        event = json.loads(line)
        events[event["battle_id"]].append(event)
    rows, rejected = [], collections.Counter()
    for line in (args.cleaned / "matches.jsonl").open(encoding="utf-8"):
        match = json.loads(line)
        try:
            payload = convert_match(match, events[match["battle_id"]])
        except (ValueError, KeyError, TypeError) as error:
            rejected[str(error)] += 1
            continue
        rows.append(
            {
                "replay_tag": payload["source"]["replay_tag"],
                "battle_type": "unknown_1v1",
                "game_mode": "unknown",
                "payload_json": json.dumps(payload, separators=(",", ":")),
            }
        )
    if not rows:
        raise RuntimeError(f"No compatible published matches: {dict(rejected)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    report = {
        "source": "Cochon123/clash-royale-replays",
        "revision": REVISION,
        "subset": "cleaned-legacy",
        "accepted_candidates": len(rows),
        "additional_deduplication": False,
        "rejections": dict(rejected),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
