"""Download immutable human replay shards and compile source-labelled league candidates."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from import_il_replay import normalise, row_name, slot_flag

REVISION = "059d43a02138a34b1b3009cc2acc7630fb99a638"
BASE = f"https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/{REVISION}/"


def download(relative, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        urllib.request.urlretrieve(BASE + relative, temporary)
        temporary.replace(destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected and digest != expected:
        raise ValueError(f"Source SHA-256 mismatch: {relative}")
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/human_sources"))
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    download("manifest.json", args.out / "manifest.json")
    manifest = json.loads((args.out / "manifest.json").read_text())
    files = [f for f in manifest["files"] if f["path"].startswith("replays/")][: args.shards]
    tables, records = [], []
    for record in files:
        path = args.out / record["path"]
        digest = download(record["path"], path, record["sha256"])
        tables.append(pq.read_table(path))
        records.append({**record, "verified_sha256": digest})
        print(json.dumps({"downloaded": record["path"], "sha256": digest}), flush=True)
    combined = pa.concat_tables(tables)
    pq.write_table(combined, args.out / "replays.parquet")
    game_tables = Path.home() / ".crforge/tables/16.635.3"
    card_rows = set()
    for name in ("spells_characters", "spells_buildings", "spells_other"):
        card_rows.update(json.loads((game_tables / f"{name}.json").read_text())["rows"])
    vocabulary = {normalise(name): name for name in card_rows}
    frequencies, examples = collections.Counter(), {}
    for row in combined.select(["battle_type", "game_mode", "payload_json"]).to_pylist():
        if row["battle_type"] != "pathOfLegend" or row["game_mode"] != "Ranked":
            continue
        payload = json.loads(row["payload_json"])
        for side in ("team", "opponent"):
            players = payload.get("battle", {}).get(side, {}).get("players", [])
            if len(players) != 1:
                continue
            cards = players[0].get("deck", [])
            names = [row_name(c, vocabulary, card_rows) for c in cards]
            if len(cards) != 8 or None in names or len(set(names)) != 8:
                continue
            forms = [slot_flag(c) for c in cards]
            key = tuple(sorted(zip(names, forms, strict=True)))
            frequencies[key] += 1
            examples[key] = (names, forms)
    decks, slots, provenance = {}, {}, {}
    for i, (key, count) in enumerate(frequencies.most_common(32)):
        name = f"human_popular_{i + 1:02d}"
        decks[name], slots[name] = examples[key]
        provenance[name] = {
            "source": "IL_Replay",
            "revision": REVISION,
            "occurrences_in_scanned_shards": count,
        }
    snapshot = json.loads(Path("configs/meta_sources_20261103.json").read_text())
    rejected = []
    for item in snapshot["decks"]:
        cards = [
            {"name": key.removesuffix("-ev1").removesuffix("-hero"), "card_key": key}
            for key in item["card_keys"]
        ]
        names = [row_name(c, vocabulary, card_rows) for c in cards]
        if None in names:
            rejected.append({**item, "reason": "card absent from pinned game tables"})
            continue
        decks[item["id"]] = names
        slots[item["id"]] = [slot_flag(c) for c in cards]
        provenance[item["id"]] = {"source": item["url"], "retrieved_date": "2026-11-03"}
    canonical = {}
    aliases = {}
    for name in list(decks):
        key = tuple(sorted(zip(decks[name], slots[name], strict=True)))
        if key in canonical:
            primary = canonical[key]
            provenance[primary].setdefault("additional_sources", []).append(provenance[name])
            aliases[name] = primary
            for mapping in (decks, slots, provenance):
                del mapping[name]
        else:
            canonical[key] = name
    league = {
        "version": 2,
        "decks": decks,
        "slots": slots,
        "provenance": provenance,
        "compatibility": "candidates; must reset and step in pinned simulator before use",
        "source_aliases": aliases,
    }
    (args.out / "league_candidates.yaml").write_text(yaml.safe_dump(league, sort_keys=False))
    (args.out / "sources.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "files": records,
                "scanned_replays": len(combined),
                "unique_compatible_decks": len(frequencies),
                "candidate_decks": len(decks),
                "rejected_meta": rejected,
            },
            indent=2,
        )
    )
    print(json.dumps({"scanned_replays": len(combined), "candidates": len(decks)}), flush=True)


if __name__ == "__main__":
    main()
