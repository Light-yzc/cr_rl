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
        from cr_rl.data.replay_execution import source_event_keys

        keys = source_event_keys(event)
        indices = [i for i, card in enumerate(cards) if card["card_key"] in keys]
        if not indices:
            return None
        card_identity = (
            decks[side][indices[0]]
            if len(indices) == 1
            else "/".join(sorted(decks[side][i] for i in indices))
        )
        point = (event.get("coordinates") or {}).get("native_world_units")
        xy = None
        if point:
            x, y = int(point["x"]), int(point["y"])
            xy = ((18000 - x) // 1000, y // 1000) if side == 0 else (x // 1000, (32000 - y) // 1000)
        kind = "play_card" if event["kind"] == "play_card" else "ability"
        events.append((int(event["replay_tick_20hz"]), side, kind, card_identity, xy))
    deck_sets = [sorted(zip(d, s, strict=True)) for d, s in zip(decks, slots, strict=True)]
    normal = json.dumps([deck_sets, sorted(events)], sort_keys=True)
    flipped = json.dumps(
        [deck_sets[::-1], sorted((t, 1 - s, k, c, xy) for t, s, k, c, xy in events)], sort_keys=True
    )
    return hashlib.sha256(min(normal, flipped).encode()).hexdigest()


def publish_live_dataset(args, accepted, paths, vector_metadata, next_row, registry, seen):
    """Publish states captured during repaired execution, never native re-labellings."""
    if not paths:
        raise RuntimeError("No repaired replay met the requested thresholds; see import_stats.json")
    from cr_rl.data.replay_supervision import mask_omitted_commands

    fields = ("observations", "masks", "actions", "ticks", "seats", "flags")
    pieces = {name: [] for name in (*fields, "replay_ids")}
    for replay_id, (path, stats) in enumerate(zip(paths, accepted, strict=True)):
        with np.load(path) as data:
            cutoff = stats["supervision_cutoff_tick"]
            keep = np.ones(len(data["actions"]), bool) if cutoff is None else data["ticks"] < cutoff
            for name in fields:
                values = data[name][keep]
                if name == "flags":
                    values = mask_omitted_commands(
                        values,
                        data["ticks"][keep],
                        data["seats"][keep],
                        stats["omitted_command_ticks"],
                        decision_ticks=5,
                        margin_ticks=args.omission_margin_ticks if args.max_omissions else 0,
                        both_seats=args.max_omissions > 0,
                    )
                pieces[name].append(values)
            pieces["replay_ids"].append(np.full(int(keep.sum()), replay_id, np.int32))
    arrays = {name: np.concatenate(values) for name, values in pieces.items()}
    if not len(arrays["actions"]):
        raise RuntimeError("All live rows were excluded by prefix cutoffs; see import_stats.json")
    illegal = ~arrays["masks"][np.arange(len(arrays["actions"])), arrays["actions"]].astype(bool)
    if np.any(illegal & (arrays["flags"] == 0)):
        raise RuntimeError("Live BC contains an unflagged illegal label")
    dataset_path = args.out / "human_bc_reconstructed.npz"
    np.savez_compressed(dataset_path, **arrays)
    with args.parquet.open("rb") as source:
        source_sha = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "schema": vector_metadata["schema"],
        "vocabulary": vector_metadata["vocabulary"],
        "dataset_kind": "repaired",
        "label_source": "live_pre_action_capture",
        "source_seed_xor": args.seed,
        "trajectory_admission": (
            "source_or_native_terminal"
            if args.require_source_or_terminal
            else "requested_reconstruction_thresholds"
        ),
        "source": sorted({r["source"] for r in accepted}),
        "source_revision": args.source_revision,
        "source_file_sha256": source_sha,
        "next_row": next_row,
        "level": args.level,
        "tower_level": args.level,
        "source_game_version": "unknown",
        "reconstruction_backend": "crforge",
        "backend_commit": "7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6",
        "backend_provenance": vector_metadata["backend_provenance"],
        "backend_patch_sha256": vector_metadata["backend_provenance"].get("patch_manifest_sha256"),
        "data_version": "16.635.3",
        "content_sha": vector_metadata.get("content_sha"),
        "decision_ticks": 5,
        "mask_version": 2,
        "rows": len(arrays["actions"]),
        "replays": accepted,
        "label_totals": {
            "rows": len(arrays["actions"]),
            "labelled": int(np.count_nonzero(arrays["actions"])),
            "label_masked": int(illegal.sum()),
            "flagged": int(np.count_nonzero(arrays["flags"])),
        },
        "import_policy": {
            "dataset_kind": "repaired",
            "require_source_or_terminal": args.require_source_or_terminal,
            "reconstruction_policy": "repaired_live_capture",
            "repair_is_exact_bc": False,
            "repair_hand": args.repair_hand,
            "max_hand_repairs": args.max_hand_repairs,
            "repair_tile": args.repair_tile,
            "max_tile_shift": args.max_tile_shift,
            "repair_time": args.repair_time,
            "time_tolerance_ticks": args.time_tolerance_ticks,
            "max_delay_ticks": args.max_delay_ticks,
            "max_omissions": args.max_omissions,
            "omission_budget_enforced_in_loop": True,
            "omitted_command_windows_excluded": True,
        },
        "warning": "These are actual simulator states after explicit repairs, not original human "
        "states. Native command replay JSON alone cannot reproduce live hand swaps. "
        "Keep this dataset in a separate repaired-BC experiment.",
    }
    (args.out / "human_bc_reconstructed.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    from cr_rl.training.persistence import atomic_json, file_sha256

    metadata_path = args.out / "human_bc_reconstructed.json"
    atomic_json(
        args.out / "dataset.manifest.json",
        {
            "format": "crrl-bc-shards-v1",
            "dataset_kind": "repaired",
            "rows": metadata["rows"],
            "replays": len(accepted),
            "metadata": metadata_path.name,
            "metadata_sha256": file_sha256(metadata_path),
            "shards": [
                {
                    "path": dataset_path.name,
                    "sha256": file_sha256(dataset_path),
                    "rows": metadata["rows"],
                    "replays": len(accepted),
                    "replay_offset": 0,
                }
            ],
        },
    )
    if args.registry:
        # Repaired imports must use their own registry: source identity stays the same.
        from cr_rl.training.persistence import atomic_json

        registry["dataset_kind"] = "repaired"
        registry["backend_patch_sha256"] = metadata["backend_patch_sha256"]
        registry["signatures"] = sorted(seen)
        atomic_json(args.registry, registry)
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "rows": metadata["rows"],
                "replays": len(accepted),
                "dataset_kind": "repaired",
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt human replay events to approximate state-conditioned BC labels."
    )
    parser.add_argument("parquet", type=Path)
    parser.add_argument(
        "--simulator-install",
        type=Path,
        help="Explicit installed backend for a controlled comparison",
    )
    parser.add_argument(
        "--allow-unpatched-backend",
        action="store_true",
        help="Comparison only: explicitly allow an unpatched backend without a receipt",
    )
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--source-revision", default="local-unversioned")
    parser.add_argument("--max-per-deck", type=int, default=1000000)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--min-plays", type=int, default=1)
    parser.add_argument("--max-omissions", type=int, default=0)
    parser.add_argument(
        "--skip-unavailable-abilities",
        action="store_true",
        help="Skip unresolved/refused ability events and continue; skipped rows are never BC labels",
    )
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
    parser.add_argument(
        "--repair-hand",
        action="store_true",
        help=(
            "Diagnostic only: swap a requested queued card into a live hand slot. "
            "Every repair is recorded and the resulting replay is not exact BC data."
        ),
    )
    parser.add_argument(
        "--max-hand-repairs",
        type=int,
        default=-1,
        help="Live hand swap budget per game; -1 means unlimited",
    )
    parser.add_argument(
        "--repair-tile",
        action="store_true",
        help="Diagnostic only: snap a closed placement to the nearest legal tile",
    )
    parser.add_argument("--max-tile-shift", type=int, default=2)
    parser.add_argument("--repair-time", action="store_true")
    parser.add_argument("--time-tolerance-ticks", type=int, default=4)
    parser.add_argument(
        "--max-delay-ticks",
        type=int,
        default=200,
        help="Maximum per-event wait at 20Hz; elixir is never injected",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=0,
        help="Scan at most this many source rows; 0 scans all",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Execute and report completion without materialising BC state tensors",
    )
    parser.add_argument(
        "--require-source-or-terminal",
        action="store_true",
        help="Repair collection: require full source execution or a confirmed normal native result",
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
    if (
        args.max_hand_repairs < -1
        or min(
            args.max_tile_shift,
            args.time_tolerance_ticks,
            args.max_delay_ticks,
            args.max_source_rows,
        )
        < 0
    ):
        parser.error("repair budgets cannot be negative")
    if args.max_hand_repairs != -1 and not args.repair_hand:
        parser.error("--max-hand-repairs requires --repair-hand")
    repair_enabled = args.repair_hand or args.repair_tile or args.repair_time
    if args.require_source_or_terminal and not repair_enabled:
        parser.error("--require-source-or-terminal requires the live repair path")
    prefix_policy = "bounded_omission_prefix" if args.max_omissions else "first_omission_prefix"
    if not 0 <= args.min_coverage <= 1 or args.min_plays < 1:
        parser.error("coverage must be in [0,1] and min-plays must be positive")

    brew_java = Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
    if brew_java.joinpath("lib/server/libjvm.dylib").exists():
        os.environ["JAVA_HOME"] = str(brew_java)

    from crforge_env import LadderConfig, VectorBattle, label_replays

    vendor = Path(__file__).resolve().parents[1] / ".cache/vendor/crforge"
    install = args.simulator_install or vendor / "env/build/install/env"
    tables = Path.home() / ".crforge/tables/16.635.3"
    config = LadderConfig(
        decision_ticks=5 if repair_enabled else 1,
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
    if args.registry and args.registry.exists():
        expected_kind = "repaired" if repair_enabled else "exact"
        if registry.get("dataset_kind", "exact") != expected_kind:
            raise ValueError("Use a separate registry for exact and repaired imports")
    seen = set(registry["signatures"])
    next_row = args.start_row
    rejection_counts = Counter()
    omitted_by_path = {}
    cutoff_by_path = {}
    live_paths = []

    with VectorBattle(1, 1, install=install, tables=tables, config=config) as vector:
        vector.set_autoreset(False)
        from cr_rl.envs.backend_patch import installed_backend, loaded_backend

        backend_provenance = installed_backend(
            install, require_patch=not args.allow_unpatched_backend
        )
        if backend_provenance.get("patches"):
            backend_provenance = loaded_backend(install)
        else:
            import jpype

            core = jpype.JClass("org.crforge.core.battle.unit.Standard1v1Battle")
            if hasattr(core, "commandLookupPatch"):
                raise ValueError(
                    "Patched JVM is missing its build receipt; publish it before import"
                )
        if args.registry and args.registry.exists():
            if registry.get("backend_patch_sha256") != backend_provenance.get(
                "patch_manifest_sha256"
            ):
                raise ValueError("Use a new replay registry for a different simulator patch")
        registry["backend_patch_sha256"] = backend_provenance.get("patch_manifest_sha256")
        live_metadata = {
            **dict(vector.metadata),
            "content_sha": vector.content_sha,
            "backend_provenance": backend_provenance,
        }
        card_rows = set()
        for table_name in ("spells_characters", "spells_buildings", "spells_other"):
            card_table = json.loads((tables / f"{table_name}.json").read_text(encoding="utf-8"))
            card_rows.update(card_table["rows"])
        vocabulary = {normalise(name): name for name in card_rows}
        for row_index, row in enumerate(table.to_pylist()):
            if row_index < args.start_row:
                continue
            if args.max_source_rows and row_index >= args.start_row + args.max_source_rows:
                break
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
            runtime_decks = [list(deck) for deck in decks]
            runtime_slots = [list(form) for form in slots]
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
            live_arrays = None
            live_stats = None
            if repair_enabled or args.audit_only:
                from cr_rl.data.native_replay import NativeReplayObserver
                from cr_rl.data.replay_execution import ExecutionConfig, execute_repaired_events

                keys = [
                    [card["card_key"] for card in battle[label]["players"][0]["deck"]]
                    for label in ("team", "opponent")
                ]
                source_seconds = (
                    payload.get("replay", {}).get("duration", {}).get("timeline_seconds")
                )
                source_end_tick = (
                    int(round(float(source_seconds) * 20)) if source_seconds is not None else None
                )
                live_stats, live_arrays, executed, omitted = execute_repaired_events(
                    vector,
                    events,
                    decks,
                    keys,
                    runtime_decks,
                    ExecutionConfig(
                        decision_ticks=5 if repair_enabled else 1,
                        repair_hand=args.repair_hand,
                        max_hand_repairs=args.max_hand_repairs,
                        repair_tile=args.repair_tile,
                        max_tile_shift=args.max_tile_shift,
                        repair_time=args.repair_time,
                        time_tolerance_ticks=args.time_tolerance_ticks,
                        max_delay_ticks=args.max_delay_ticks,
                        max_omissions=args.max_omissions,
                        allow_omitted_suffix=args.allow_omitted_suffix,
                        skip_unavailable_abilities=args.skip_unavailable_abilities,
                        capture_rows=not args.audit_only,
                        source_end_tick=source_end_tick,
                    ),
                    native_observer=NativeReplayObserver(vector),
                )
                live_stats["backend_patch_sha256"] = backend_provenance.get("patch_manifest_sha256")
                skipped = live_stats["skipped"]
                used, used_abilities = live_stats["used"], live_stats["used_abilities"]
                collisions = live_stats["collisions"]
                repair_records = live_stats["repair_records"]
                hand_repairs = live_stats["repair_summary"]["hand_replacements"]
            else:
                skipped: dict[str, int] = {}
                used = 0
                used_abilities = 0
                collisions = 0
                repair_records = []
                hand_repairs = 0
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
                        candidates = event.get("deck_card_key_candidates") or [
                            event.get("card_key")
                        ]
                        deck_cards = battle["team" if side == 0 else "opponent"]["players"][0][
                            "deck"
                        ]
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
                        xy = (event.get("coordinates") or {}).get("native_world_units")
                        if not xy:
                            omitted.append((tick, side))
                            skipped["missing_coordinates"] = (
                                skipped.get("missing_coordinates", 0) + 1
                            )
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
                or (
                    args.require_source_or_terminal
                    and not (live_stats or {}).get("source_or_native_terminal_completed", False)
                )
            ):
                rejection_counts["requested_reconstruction_threshold"] += 1
                all_stats.append(
                    {
                        **(live_stats or {}),
                        "source_row_index": row_index,
                        "replay_tag": replay_tag,
                        "accepted": False,
                        "used": used,
                        "total_play_events": total_plays,
                        "coverage": coverage,
                        "skipped": skipped,
                        "collisions": collisions,
                        "repair_records": repair_records,
                    }
                )
                continue
            replay_path = args.out / "reconstructed_replays" / f"{signature}_{row_index:07d}.json"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.audit_only:
                replay_path.write_text(vector.replay_json(0), encoding="utf-8")
                generated.append(replay_path)
                if live_arrays is not None:
                    live_path = replay_path.with_suffix(".live.npz")
                    np.savez_compressed(live_path, **live_arrays)
                    live_paths.append(live_path)
            omitted_by_path[replay_path.resolve()] = omitted
            cutoff_by_path[replay_path.resolve()] = cutoff
            stats = {
                **(live_stats or {}),
                "source_row_index": row_index,
                "initial_simulator_seed": seed ^ args.seed,
                "source_seed_xor": args.seed,
                "replay_tag": replay_tag,
                "accepted": True,
                "used": used,
                "total_play_events": total_plays,
                "coverage": coverage,
                "skipped": skipped,
                "collisions": collisions,
                "repair_records": repair_records,
                "repair_summary": (live_stats or {}).get(
                    "repair_summary",
                    {
                        "hand_replacements": hand_repairs,
                        "tile_snaps": sum(r.get("category") == "tile_snap" for r in repair_records),
                    },
                ),
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
                "repair_hand": args.repair_hand,
                "max_hand_repairs": args.max_hand_repairs,
                "repair_tile": args.repair_tile,
                "max_tile_shift": args.max_tile_shift,
                "repair_time": args.repair_time,
                "max_delay_ticks": args.max_delay_ticks,
                "omission_budget_enforced_in_loop": True,
                "omission_margin_ticks": args.omission_margin_ticks if args.max_omissions else 0,
                "supervision_cutoff_tick": cutoff,
                "retained_source_event_fraction": coverage,
                "prefix_event_execution_coverage": used
                / max(1, sum(len(v) for t, v in events.items() if cutoff is None or t < cutoff)),
                "reconstruction_policy": "repaired_live_capture"
                if repair_enabled
                else ("legacy_suffix" if args.allow_omitted_suffix else prefix_policy),
                "canonical_game_signature": game_signature(decks, slots, payload),
                "decks": decks,
                "slots": slots,
            }
            accepted.append(stats)
            all_stats.append(stats)
            if live_stats is not None and not args.audit_only:
                replay_path.with_suffix(".repair.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "seed": seed ^ args.seed,
                            "runtime_decks": runtime_decks,
                            "runtime_slots": runtime_slots,
                            "source_events": payload.get("events", []),
                            "warning": (
                                "Native command replay does not reproduce live hand mutations; "
                                "use live.npz for BC and this ledger for audit."
                            ),
                            **stats,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
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
    (args.out / "import_stats.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    if args.audit_only:
        print(
            json.dumps(
                {
                    "audit_only": True,
                    "attempted": len(all_stats),
                    "complete": sum(r.get("all_source_events_executed", False) for r in all_stats),
                }
            ),
            flush=True,
        )
        return
    if repair_enabled:
        publish_live_dataset(args, accepted, live_paths, live_metadata, next_row, registry, seen)
        return
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
                "reconstruction_policy": "repaired_live_capture"
                if repair_enabled
                else ("legacy_suffix" if args.allow_omitted_suffix else prefix_policy),
                "max_omissions": args.max_omissions,
                "omission_budget_enforced_in_loop": True,
                "omission_margin_ticks": args.omission_margin_ticks if args.max_omissions else 0,
                "repair_hand": args.repair_hand,
                "max_hand_repairs": args.max_hand_repairs,
                "repair_tile": args.repair_tile,
                "max_tile_shift": args.max_tile_shift,
                "repair_time": args.repair_time,
                "max_delay_ticks": args.max_delay_ticks,
                "repair_is_exact_bc": not repair_enabled,
            },
            "next_row": next_row,
            "level": args.level,
            "tower_level": args.level,
            "source_schema": "royaleapi-battle-actions.v1",
            "source_game_version": "unknown",
            "reconstruction_backend": "crforge",
            "backend_commit": "7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6",
            "backend_provenance": backend_provenance,
            "backend_patch_sha256": backend_provenance.get("patch_manifest_sha256"),
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
