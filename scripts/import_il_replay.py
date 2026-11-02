from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ALIASES = {
    "sparky": "ZapMachine",
    "minipekka": "MiniPekka",
    "healspirit": "Heal",
    "thelog": "Log",
    "icespirit": "IceSpirits",
    "firespirit": "FireSpirits",
    "barbarianbarrel": "BarbLog",
}


def normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def row_name(card: dict, vocabulary: dict[str, str], card_rows: set[str]) -> str | None:
    key = normalise(card["name"])
    result = ALIASES.get(key, vocabulary.get(key))
    return result if result in card_rows else None


def slot_flag(card: dict) -> int:
    key = card.get("card_key", "")
    if "-ev1" in key:
        return 1
    if "-hero" in key:
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a small, version-labelled BC subset from RoyaleAPI replay events."
    )
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("artifacts/human_bc_reconstructed"))
    parser.add_argument("--level", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    brew_java = Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
    if brew_java.joinpath("lib/server/libjvm.dylib").exists():
        os.environ["JAVA_HOME"] = str(brew_java)

    from crforge_env import LadderConfig, VectorBattle, label_replays

    vendor = Path(__file__).resolve().parents[1] / ".cache/vendor/crforge"
    install = vendor / "env/build/install/env"
    tables = Path.home() / ".crforge/tables/16.635.3"
    config = LadderConfig(
        decision_ticks=1,
        level=args.level,
        tower_level=args.level,
        mask_version=1,
        arena="Arena_L",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(
        args.parquet, columns=["replay_tag", "battle_type", "game_mode", "payload_json"]
    )
    accepted = []
    generated = []
    all_stats = []

    with VectorBattle(1, 1, install=install, tables=tables, config=config) as vector:
        card_rows = set()
        for table_name in ("spells_characters", "spells_buildings", "spells_other"):
            card_table = json.loads((tables / f"{table_name}.json").read_text(encoding="utf-8"))
            card_rows.update(card_table["rows"])
        vocabulary = {normalise(name): name for name in card_rows}
        for row in table.to_pylist():
            if len(accepted) >= args.limit:
                break
            payload = json.loads(row["payload_json"])
            battle = payload.get("battle", {})
            if row.get("battle_type") != "pathOfLegend" or row.get("game_mode") != "Ranked":
                continue
            if payload.get("schema_version") != "royaleapi-battle-actions.v1":
                continue

            decks: list[list[str]] = []
            slots: list[list[int]] = []
            valid = True
            for side in ("team", "opponent"):
                players = battle.get(side, {}).get("players", [])
                if len(players) != 1:
                    valid = False
                    break
                cards = players[0].get("deck", [])
                names = [row_name(card, vocabulary, card_rows) for card in cards]
                levels = {int(card.get("level", -1)) for card in cards}
                if len(cards) != 8 or any(name is None for name in names):
                    valid = False
                    break
                if len(set(names)) != 8 or levels != {args.level}:
                    valid = False
                    break
                decks.append(names)
                slots.append([slot_flag(card) for card in cards])
            if not valid or not any(
                card.get("name", "").lower() == "sparky"
                for side in ("team", "opponent")
                for card in battle[side]["players"][0]["deck"]
            ):
                continue

            events: dict[int, list[dict]] = {}
            for event in payload.get("events", []):
                if event.get("kind") == "play_card":
                    events.setdefault(int(event["replay_tick_20hz"]), []).append(event)
            if not events:
                continue

            replay_tag = payload.get("source", {}).get("replay_tag", row["replay_tag"])
            seed = (
                int.from_bytes(hashlib.sha256(replay_tag.encode()).digest()[:4], "big") & 0x7FFFFFFF
            )
            vector.reset([seed ^ args.seed], decks[0], decks[1], slots[0], slots[1])
            skipped: dict[str, int] = {}
            used = 0
            collisions = 0
            for tick in range(max(events) + 1):
                actions = np.zeros((1, 2), dtype=np.int32)
                occupied: set[int] = set()
                for event in events.get(tick, []):
                    side = 0 if event.get("side") == "team" else 1
                    if side in occupied:
                        collisions += 1
                        continue
                    occupied.add(side)
                    candidates = event.get("deck_card_key_candidates") or [event.get("card_key")]
                    deck_cards = battle["team" if side == 0 else "opponent"]["players"][0]["deck"]
                    deck_index = next(
                        (
                            i
                            for i, card in enumerate(deck_cards)
                            if card.get("card_key") in candidates
                        ),
                        None,
                    )
                    if deck_index is None:
                        skipped["card_not_in_deck"] = skipped.get("card_not_in_deck", 0) + 1
                        continue
                    card_id = vector.card_id(decks[side][deck_index])
                    hand_fields = vector.schema.header["hand_card"]
                    hand = [int(vector.observations[0, side, field]) for field in hand_fields]
                    if card_id not in hand:
                        skipped["not_in_hand"] = skipped.get("not_in_hand", 0) + 1
                        continue
                    xy = event.get("coordinates", {}).get("native_world_units")
                    if not xy:
                        skipped["missing_coordinates"] = skipped.get("missing_coordinates", 0) + 1
                        continue
                    x, y = int(xy["x"]), int(xy["y"])
                    seat_x = 18000 - x if side == 0 else x
                    seat_y = y if side == 0 else 32000 - y
                    col = max(0, min(17, seat_x // 1000))
                    grid_row = max(0, min(31, seat_y // 1000))
                    action = vector.schema.play(hand.index(card_id), col, grid_row)
                    if not vector.masks[0, side, action]:
                        skipped["mask_closed"] = skipped.get("mask_closed", 0) + 1
                        continue
                    actions[0, side] = action
                    used += 1
                vector.actions[:] = actions
                vector.step()
                if vector.dones[0]:
                    break

            total_plays = sum(len(v) for v in events.values())
            coverage = used / max(1, total_plays)
            if used < 20 or coverage < 0.75:
                all_stats.append(
                    {
                        "replay_tag": replay_tag,
                        "accepted": False,
                        "used": used,
                        "total_play_events": total_plays,
                        "coverage": coverage,
                        "skipped": skipped,
                        "collisions": collisions,
                    }
                )
                continue
            replay_path = args.out / "reconstructed_replays" / f"{replay_tag}.json"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_path.write_text(vector.replay_json(0), encoding="utf-8")
            generated.append(replay_path)
            stats = {
                "replay_tag": replay_tag,
                "accepted": True,
                "used": used,
                "total_play_events": total_plays,
                "coverage": coverage,
                "skipped": skipped,
                "collisions": collisions,
                "battle_type": row.get("battle_type"),
                "game_mode": row.get("game_mode"),
                "source_schema": payload.get("schema_version"),
                "source_game_version": "unknown",
            }
            accepted.append(stats)
            all_stats.append(stats)

    if not generated:
        raise RuntimeError("No compatible Sparky replays could be reconstructed from this shard")

    with tempfile.TemporaryDirectory(prefix="cr-bc-labels-") as temp:
        labelled = label_replays(
            generated,
            install,
            tables,
            decision_ticks=5,
            threads=1,
            check=False,
            out_dir=temp,
        )
        observations, masks, actions, ticks, seats, flags, replay_ids = [], [], [], [], [], [], []
        counts_path = Path(temp) / "counts.jsonl"
        counts = [json.loads(line) for line in counts_path.read_text().splitlines() if line.strip()]
        for replay_id, item in enumerate(counts):
            path = Path(item["file"])
            with np.load(path) as data:
                keep = data["flags"] == 0
                observations.append(data["observations"][keep])
                masks.append(data["masks"][keep])
                actions.append(data["actions"][keep])
                ticks.append(data["ticks"][keep])
                seats.append(data["seats"][keep])
                flags.append(data["flags"][keep])
                replay_ids.append(np.full(int(keep.sum()), replay_id, dtype=np.int32))
        dataset_path = args.out / "human_bc_reconstructed.npz"
        np.savez_compressed(
            dataset_path,
            observations=np.concatenate(observations),
            masks=np.concatenate(masks),
            actions=np.concatenate(actions),
            ticks=np.concatenate(ticks),
            seats=np.concatenate(seats),
            flags=np.concatenate(flags),
            replay_ids=np.concatenate(replay_ids),
        )
        metadata = {
            "schema": labelled.metadata["schema"],
            "vocabulary": labelled.metadata["vocabulary"],
            "source": "VanguardX101/IL_Replay",
            "source_schema": "royaleapi-battle-actions.v1",
            "source_game_version": "unknown",
            "reconstruction_backend": "crforge",
            "backend_commit": "7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6",
            "data_version": "16.635.3",
            "content_sha": vector.content_sha,
            "decision_ticks": 5,
            "mask_version": 2,
            "rows": int(sum(len(x) for x in actions)),
            "replays": accepted,
            "label_totals": labelled.totals(),
            "warning": (
                "The source contains actions and coordinates but no full game-state trace or "
                "verified client/data version. Observations are reconstructed by replaying those "
                "actions in crforge 16.635.3. Use as approximate BC warm-start only, not as a "
                "claim of original-state fidelity."
            ),
        }
        (args.out / "human_bc_reconstructed.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (args.out / "import_stats.json").write_text(
            json.dumps(all_stats, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "rows": metadata["rows"],
                    "replays": len(accepted),
                    "label_totals": labelled.totals(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
