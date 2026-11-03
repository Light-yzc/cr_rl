from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ALIASES = {
    "archers": "Archer",
    "bandit": "Assassin",
    "cannoncart": "MovingCannon",
    "dartgoblin": "BlowdartGoblin",
    "elitebarbarians": "AngryBarbarians",
    "furnace": "FirespiritHut",
    "giantsnowball": "Snowball",
    "guards": "SkeletonWarriors",
    "icegolem": "IceGolemite",
    "lumberjack": "RageBarbarian",
    "motherwitch": "WitchMother",
    "nightwitch": "DarkWitch",
    "royalghost": "Ghost",
    "skeletonbarrel": "SkeletonBalloon",
    "zappies": "MiniSparkys",
    "sparky": "ZapMachine",
    "minipekka": "MiniPekka",
    "healspirit": "Heal",
    "thelog": "Log",
    "icespirit": "IceSpirits",
    "firespirit": "FireSpirits",
    "barbarianbarrel": "BarbLog",
    "executioner": "AxeMan",
    "magicarcher": "EliteArcher",
    "flyingmachine": "DartBarrell",
    "runegiant": "GiantBuffer",
    "void": "DarkMagic",
    "spiritempress": "MergeMaiden",
}


def normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def row_name(card: dict, vocabulary: dict[str, str], card_rows: set[str]) -> str | None:
    key = normalise(card["name"])
    result = ALIASES.get(key, vocabulary.get(key))
    return result if result in card_rows else None


def slot_flag(card: dict) -> int:
    key = card.get("card_key", "")
    return int("-ev1" in key) | (2 * int("-hero" in key))


def game_signature(decks, slots, payload):
    """Deduplicate equivalent sources/seat views using the actual model action grid."""
    sides = ("team", "opponent")
    events = []
    for event in payload.get("events", []):
        if event.get("kind") not in (
            "play_card",
            "ability",
            "activate_ability",
            "ability_activation",
        ):
            continue
        if event.get("side") not in sides:
            return None
        side = sides.index(event["side"])
        cards = payload["battle"][event["side"]]["players"][0]["deck"]
        keys = event.get("deck_card_key_candidates") or [event.get("card_key")]
        index = next((i for i, card in enumerate(cards) if card["card_key"] in keys), None)
        if index is None:
            return None
        point = event.get("coordinates", {}).get("native_world_units")
        xy = None
        if point:
            x, y = int(point["x"]), int(point["y"])
            xy = ((18000 - x) // 1000, y // 1000) if side == 0 else (x // 1000, (32000 - y) // 1000)
        kind = "play_card" if event["kind"] == "play_card" else "ability"
        events.append((int(event["replay_tick_20hz"]), side, kind, decks[side][index], xy))
    deck_sets = [sorted(zip(d, s, strict=True)) for d, s in zip(decks, slots, strict=True)]
    normal = json.dumps([deck_sets, sorted(events)], sort_keys=True)
    flipped = json.dumps(
        [deck_sets[::-1], sorted((t, 1 - s, k, c, xy) for t, s, k, c, xy in events)], sort_keys=True
    )
    return hashlib.sha256(min(normal, flipped).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt human replay events to approximate state-conditioned BC labels."
    )
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--source-revision", default="local-unversioned")
    parser.add_argument("--max-per-deck", type=int, default=1000000)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--min-plays", type=int, default=1)
    parser.add_argument("--max-omissions", type=int, default=0)
    parser.add_argument("--omission-margin-ticks", type=int, default=5)
    parser.add_argument("--ranked-only", action="store_true")
    parser.add_argument("--reject-collisions", action="store_true")
    parser.add_argument(
        "--allow-omitted-suffix",
        action="store_true",
        help="Legacy diagnostic only: retain states after missed source commands",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/human_bc_reconstructed"))
    parser.add_argument("--level", type=int, default=11)
    parser.add_argument(
        "--source-level",
        type=int,
        default=0,
        help="Optional source-level filter; 0 accepts all source levels",
    )
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument(
        "--infer-opening-cycle",
        action="store_true",
        help="Infer a plausible latent opening cycle from source card plays; experimental",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument(
        "--registry",
        type=Path,
        help="Shared accepted-game signatures and deck counts across shards/sources",
    )
    args = parser.parse_args()
    if min(args.max_omissions, args.omission_margin_ticks) < 0:
        parser.error("omission budget and margin cannot be negative")
    prefix_policy = "bounded_omission_prefix" if args.max_omissions else "first_omission_prefix"
    if not 0 <= args.min_coverage <= 1 or args.min_plays < 1:
        parser.error("coverage must be in [0,1] and min-plays must be positive")

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
        mask_version=2,
        arena="Arena_L",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(
        args.parquet, columns=["replay_tag", "battle_type", "game_mode", "payload_json"]
    )
    accepted = []
    generated = []
    all_stats = []
    deck_counts = {}
    registry = {"signatures": [], "deck_counts": {}}
    if args.registry and args.registry.exists():
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
    seen = set(registry["signatures"])
    next_row = args.start_row
    rejection_counts = Counter()
    omitted_by_path = {}
    cutoff_by_path = {}

    with VectorBattle(1, 1, install=install, tables=tables, config=config) as vector:
        card_rows = set()
        for table_name in ("spells_characters", "spells_buildings", "spells_other"):
            card_table = json.loads((tables / f"{table_name}.json").read_text(encoding="utf-8"))
            card_rows.update(card_table["rows"])
        vocabulary = {normalise(name): name for name in card_rows}
        for row_index, row in enumerate(table.to_pylist()):
            if row_index < args.start_row:
                continue
            if len(accepted) >= args.limit:
                break
            next_row = row_index + 1
            payload = json.loads(row["payload_json"])
            battle = payload.get("battle", {})
            is_cleaned_cochon = payload.get("source", {}).get("dataset") == (
                "Cochon123/clash-royale-replays"
            )
            if (
                args.ranked_only
                and not is_cleaned_cochon
                and (row.get("battle_type") != "pathOfLegend" or row.get("game_mode") != "Ranked")
            ):
                rejection_counts["ranked_only"] += 1
                continue
            if payload.get("schema_version") != "royaleapi-battle-actions.v1":
                rejection_counts["unsupported_schema"] += 1
                continue

            decks: list[list[str]] = []
            slots: list[list[int]] = []
            source_levels = []
            valid = True
            for side in ("team", "opponent"):
                players = battle.get(side, {}).get("players", [])
                if len(players) != 1:
                    valid = False
                    break
                cards = players[0].get("deck", [])
                names = [row_name(card, vocabulary, card_rows) for card in cards]
                levels = {int(card.get("level", -1)) for card in cards}
                source_levels.append([int(card.get("level", -1)) for card in cards])
                if len(cards) != 8 or any(name is None for name in names):
                    valid = False
                    break
                if len(set(names)) != 8 or (
                    args.source_level > 0 and levels != {args.source_level}
                ):
                    valid = False
                    break
                decks.append(names)
                slots.append([slot_flag(card) for card in cards])
            if not valid:
                rejection_counts["deck_or_player_structure_or_requested_level"] += 1
                continue
            deck_keys = [
                json.dumps(sorted(zip(deck, form, strict=True)))
                for deck, form in zip(decks, slots, strict=True)
            ]
            if any(registry["deck_counts"].get(key, 0) >= args.max_per_deck for key in deck_keys):
                continue
            signature = (
                game_signature(decks, slots, payload)
                if args.registry
                else (hashlib.sha256(str(row["replay_tag"]).encode()).hexdigest())
            )
            if signature is None or (args.registry and signature in seen):
                continue

            events: dict[int, list[dict]] = {}
            for event in payload.get("events", []):
                if event.get("kind") in (
                    "play_card",
                    "ability",
                    "activate_ability",
                    "ability_activation",
                ):
                    events.setdefault(int(event["replay_tick_20hz"]), []).append(event)
            if not events:
                rejection_counts["no_events"] += 1
                continue

            replay_tag = payload.get("source", {}).get("replay_tag", row["replay_tag"])
            seed = (
                int.from_bytes(hashlib.sha256(replay_tag.encode()).digest()[:4], "big") & 0x7FFFFFFF
            )
            vector.reset([seed ^ args.seed], decks[0], decks[1], slots[0], slots[1])
            if vector.dones[0] or vector.info(0)["fault"]:
                rejection_counts["simulator_reset_refused"] += 1
                continue
            opening_inference = []
            if args.infer_opening_cycle:
                from cr_rl.data.opening_cycle import (
                    infer_opening_cycle,
                    native_opening_deals,
                    reorder_for_deal,
                )

                native = native_opening_deals(vector)
                runtime_decks = [list(deck) for deck in decks]
                runtime_slots = [list(form) for form in slots]
                for side, label in enumerate(("team", "opponent")):
                    card_keys = [card["card_key"] for card in battle[label]["players"][0]["deck"]]
                    sequence = []
                    for tick in sorted(events):
                        for event in events[tick]:
                            if event.get("side") != label or event["kind"] != "play_card":
                                continue
                            keys = event.get("deck_card_key_candidates") or [event.get("card_key")]
                            index = next(
                                (i for i, key in enumerate(card_keys) if key in keys), None
                            )
                            if index is None:
                                sequence.append(-1)
                            else:
                                sequence.append(index)
                    inferred = infer_opening_cycle(sequence) if native[side]["ordinary"] else None
                    opening_inference.append(inferred)
                    if inferred:
                        runtime_decks[side], runtime_slots[side] = reorder_for_deal(
                            decks[side], slots[side], native[side]["order"], inferred["order"]
                        )
                vector.reset(
                    [seed ^ args.seed],
                    runtime_decks[0],
                    runtime_decks[1],
                    runtime_slots[0],
                    runtime_slots[1],
                )
                if vector.dones[0] or vector.info(0)["fault"]:
                    rejection_counts["inferred_deal_reset_refused"] += 1
                    continue
                actual = native_opening_deals(vector)
                for side, inferred in enumerate(opening_inference):
                    if inferred and (
                        [runtime_decks[side][i] for i in actual[side]["order"]]
                        != [decks[side][i] for i in inferred["order"]]
                    ):
                        raise ValueError(
                            "Opening-cycle inversion did not reproduce the inferred deal"
                        )
            skipped: dict[str, int] = {}
            used = 0
            used_abilities = 0
            collisions = 0
            omitted = []
            executed = []
            actions = np.zeros((1, 2), dtype=np.int32)
            for tick in range(max(events) + 1):
                actions.fill(0)
                occupied: set[int] = set()
                for event in events.get(tick, []):
                    side = 0 if event.get("side") == "team" else 1
                    if side in occupied:
                        collisions += 1
                        omitted.append((tick, side))
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
                        omitted.append((tick, side))
                        continue
                    card_id = vector.card_id(decks[side][deck_index])
                    if event["kind"] != "play_card":
                        possible = []
                        wanted = normalise(decks[side][deck_index])
                        for slot, field in enumerate(vector.schema.header["ability_champion"]):
                            champion = int(vector.observations[0, side, field])
                            name = vector.vocabulary[champion] if champion >= 0 else ""
                            matches = normalise(name).replace("hero", "") == wanted
                            action = 2305 + slot
                            if matches and vector.masks[0, side, action]:
                                possible.append(action)
                        if len(possible) != 1:
                            omitted.append((tick, side))
                            skipped["ability_unavailable_or_ambiguous"] = (
                                skipped.get("ability_unavailable_or_ambiguous", 0) + 1
                            )
                            continue
                        actions[0, side] = possible[0]
                        used += 1
                        used_abilities += 1
                        executed.append((tick, True))
                        continue
                    hand_fields = vector.schema.header["hand_card"]
                    hand = [int(vector.observations[0, side, field]) for field in hand_fields]
                    if card_id not in hand:
                        omitted.append((tick, side))
                        skipped["not_in_hand"] = skipped.get("not_in_hand", 0) + 1
                        continue
                    xy = event.get("coordinates", {}).get("native_world_units")
                    if not xy:
                        omitted.append((tick, side))
                        skipped["missing_coordinates"] = skipped.get("missing_coordinates", 0) + 1
                        continue
                    x, y = int(xy["x"]), int(xy["y"])
                    seat_x = 18000 - x if side == 0 else x
                    seat_y = y if side == 0 else 32000 - y
                    col = max(0, min(17, seat_x // 1000))
                    grid_row = max(0, min(31, seat_y // 1000))
                    action = vector.schema.play(hand.index(card_id), col, grid_row)
                    if not vector.masks[0, side, action]:
                        omitted.append((tick, side))
                        skipped["mask_closed"] = skipped.get("mask_closed", 0) + 1
                        continue
                    actions[0, side] = action
                    used += 1
                    executed.append((tick, False))
                if len(omitted) > args.max_omissions and not args.allow_omitted_suffix:
                    break
                vector.actions[:] = actions
                vector.step()
                if vector.dones[0]:
                    break

            total_plays = sum(len(v) for v in events.values())
            from cr_rl.data.replay_supervision import prefix_cutoff

            cutoff = (
                prefix_cutoff(omitted, max_omissions=args.max_omissions)
                if not args.allow_omitted_suffix
                else None
            )
            if cutoff is not None:
                used = sum(t < cutoff for t, _ in executed)
                used_abilities = sum(t < cutoff and ability for t, ability in executed)
            coverage = used / max(1, total_plays)
            if (
                used < args.min_plays
                or coverage < args.min_coverage
                or (args.reject_collisions and collisions)
            ):
                rejection_counts["requested_reconstruction_threshold"] += 1
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
            replay_path = args.out / "reconstructed_replays" / f"{signature}_{row_index:07d}.json"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_path.write_text(vector.replay_json(0), encoding="utf-8")
            generated.append(replay_path)
            omitted_by_path[replay_path.resolve()] = omitted
            cutoff_by_path[replay_path.resolve()] = cutoff
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
                "source": payload.get("source", {}).get("dataset", "VanguardX101/IL_Replay"),
                "source_revision": payload.get("source", {}).get("revision", args.source_revision),
                "source_metadata": payload.get("source", {}),
                "game_signature": signature,
                "additional_deduplication": bool(args.registry),
                "used_abilities": used_abilities,
                "source_card_levels": source_levels,
                "reconstruction_level": args.level,
                "omitted_commands": len(omitted),
                "omitted_command_ticks": omitted,
                "opening_cycle_inference": opening_inference,
                "initial_deal_policy": (
                    "source_consistent_latent_cycle"
                    if any(opening_inference)
                    else "seeded_simulator_deal"
                ),
                "max_omissions": args.max_omissions,
                "omission_budget_enforced_in_loop": True,
                "omission_margin_ticks": args.omission_margin_ticks if args.max_omissions else 0,
                "supervision_cutoff_tick": cutoff,
                "retained_source_event_fraction": coverage,
                "prefix_event_execution_coverage": used
                / max(1, sum(len(v) for t, v in events.items() if cutoff is None or t < cutoff)),
                "reconstruction_policy": "legacy_suffix"
                if args.allow_omitted_suffix
                else prefix_policy,
                "canonical_game_signature": game_signature(decks, slots, payload),
                "decks": decks,
                "slots": slots,
            }
            accepted.append(stats)
            all_stats.append(stats)
            for key in deck_keys:
                deck_counts[key] = deck_counts.get(key, 0) + 1
                registry["deck_counts"][key] = registry["deck_counts"].get(key, 0) + 1
            seen.add(signature)
            print(
                json.dumps(
                    {
                        "accepted": len(accepted),
                        "coverage": coverage,
                        "unique_decks": len(deck_counts),
                    }
                ),
                flush=True,
            )

    (args.out / "cursor.json").write_text(
        json.dumps(
            {
                "next_row": next_row,
                "total_rows": len(table),
                "accepted": len(accepted),
                "rejections": dict(rejection_counts),
            }
        ),
        encoding="utf-8",
    )
    if not generated:
        raise RuntimeError(
            "No compatible multi-deck replays could be reconstructed from this shard"
        )

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
        from cr_rl.data.replay_supervision import mask_omitted_commands

        if len(counts) != len(generated):
            raise ValueError("Labeller output count differs from reconstructed replay count")
        for replay_id, item in enumerate(counts):
            path = Path(item["file"])
            with np.load(path) as data:
                # Retain all observations for a correctly spaced history. Flags only
                # determine which rows can be supervised, not which frames exist.
                cutoff = cutoff_by_path[generated[replay_id].resolve()]
                keep = (
                    np.ones(len(data["actions"]), dtype=bool)
                    if cutoff is None
                    else data["ticks"] < cutoff
                )
                observations.append(data["observations"][keep])
                masks.append(data["masks"][keep])
                actions.append(data["actions"][keep])
                ticks.append(data["ticks"][keep])
                seats.append(data["seats"][keep])
                flags.append(
                    mask_omitted_commands(
                        data["flags"][keep],
                        data["ticks"][keep],
                        data["seats"][keep],
                        omitted_by_path[generated[replay_id].resolve()],
                        decision_ticks=5,
                        margin_ticks=args.omission_margin_ticks if args.max_omissions else 0,
                        both_seats=args.max_omissions > 0,
                    )
                )
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
            "source": sorted({r["source"] for r in accepted}),
            "source_revision": args.source_revision,
            "source_file_sha256": hashlib.file_digest(
                args.parquet.open("rb"), "sha256"
            ).hexdigest(),
            "source_level": args.source_level,
            "import_policy": {
                "ranked_only": args.ranked_only,
                "source_level_filter": args.source_level,
                "min_coverage": args.min_coverage,
                "min_plays": args.min_plays,
                "reject_collisions": args.reject_collisions,
                "levels_normalized_to": args.level,
                "omitted_command_windows_excluded": True,
                "reconstruction_policy": "legacy_suffix"
                if args.allow_omitted_suffix
                else prefix_policy,
                "max_omissions": args.max_omissions,
                "omission_budget_enforced_in_loop": True,
                "omission_margin_ticks": args.omission_margin_ticks if args.max_omissions else 0,
            },
            "next_row": next_row,
            "level": args.level,
            "tower_level": args.level,
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
        if args.registry:
            from cr_rl.training.persistence import atomic_json

            registry["signatures"] = sorted(seen)
            atomic_json(args.registry, registry)
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
