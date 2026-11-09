"""Compare replay reconstruction arms on one fixed, small source cohort.

Uses CPU/JVM only. Completion of the source event stream and a simulator terminal
are separate outcomes. No model training, GPU allocation or source download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from cr_rl.training.persistence import atomic_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "exact": [],
    "cycle": ["--infer-opening-cycle"],
    "repair": [
        "--infer-opening-cycle", "--repair-time", "--repair-tile",
    ],
}
CODE_FILES = (
    "scripts/import_il_replay.py",
    "src/cr_rl/data/replay_execution.py",
    "src/cr_rl/data/opening_cycle.py",
    "src/cr_rl/data/replay_repair.py",
    "src/cr_rl/data/native_replay.py",
    "src/cr_rl/data/replay_supervision.py",
    "src/cr_rl/envs/backend_patch.py",
    "scripts/manage_crforge_patch.py",
    "configs/backends.yaml",
)


def code_fingerprints():
    names = set(CODE_FILES)
    names.update(
        str(path.relative_to(ROOT))
        for path in (ROOT / "patches").rglob("*") if path.is_file()
    )
    return {
        name: file_sha256(ROOT / name) if (ROOT / name).is_file() else None
        for name in sorted(names)
    }


def verified_installed_backend(install):
    from cr_rl.envs.backend_patch import installed_backend

    return installed_backend(install, verify=True, require_patch=True)


def backend_fingerprints():
    """Record source, installed binaries and a verified receipt without starting a JVM."""
    vendor = ROOT / ".cache/vendor/crforge"
    install = vendor / "env/build/install/env"

    def git_output(*arguments):
        result = subprocess.run(
            ["git", "-C", str(vendor), *arguments], capture_output=True, check=False
        )
        return result.stdout if result.returncode == 0 else None

    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    diff = git_output("diff", "--binary", "HEAD")
    untracked = git_output("ls-files", "--others", "--exclude-standard", "-z")
    receipt = install / "cr_rl_backend.json"
    verified, validation_error = None, None
    if receipt.is_file():
        try:
            verified = verified_installed_backend(install)
        except Exception as error:
            # Preserve after-arm evidence even if a receipt or JAR changed during the run.
            validation_error = f"{type(error).__name__}: {error}"
    return {
        "install": str(install.resolve()),
        "vendor_head": commit.decode().strip() if commit is not None else None,
        "vendor_status_porcelain": status.decode().splitlines() if status is not None else None,
        "vendor_tracked_diff_sha256": hashlib.sha256(diff).hexdigest()
        if diff is not None else None,
        "vendor_untracked_sha256": {
            name: file_sha256(vendor / name)
            for name in (untracked.decode().split("\0") if untracked is not None else [])
            if name and (vendor / name).is_file()
        },
        "installed_jars_sha256": {
            path.name: file_sha256(path) for path in sorted((install / "lib").glob("*.jar"))
        },
        "build_receipt_sha256": file_sha256(receipt) if receipt.is_file() else None,
        "verified_installed_backend": verified,
        "backend_validation_error": validation_error,
    }


def fixed_cohort(source, destination, *, start, count):
    """Read at most ``start+count`` source rows, without scanning images/blobs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    selected, seen = [], 0
    columns = ["replay_tag", "battle_type", "game_mode", "payload_json"]
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            if seen >= start and len(selected) < count:
                selected.append(row)
            seen += 1
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("No source rows selected")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), destination)
    traits = []
    for index, row in enumerate(selected):
        payload = json.loads(row["payload_json"])
        events = payload.get("events", [])
        battle = payload.get("battle", {})
        keys = [
            card.get("card_key") or ""
            for side in ("team", "opponent")
            for player in battle.get(side, {}).get("players", [])
            for card in player.get("deck", [])
        ]
        traits.append({
            "source_row_index": index,
            "replay_tag": row["replay_tag"],
            "source_events": len(events),
            "last_source_tick": max((int(e.get("replay_tick_20hz", 0)) for e in events), default=0),
            "has_hero_marker": any("-hero" in key for key in keys),
            "has_evolution_marker": any("-ev1" in key for key in keys),
        })
    return {
        "source": str(Path(source).resolve()),
        "source_file_sha256": file_sha256(source),
        "cohort_file_sha256": file_sha256(destination),
        "source_start_row": start,
        "source_rows": len(selected),
        "replay_tags": [row["replay_tag"] for row in selected],
        "source_traits": traits,
    }


def summarize(stats, cohort_size, traits=None):
    """Only report metrics actually supplied by the importer, never infer done."""
    # New importers report all attempts; old ones may skip rejected structures.
    processed = len(stats)
    complete_known = [row for row in stats if "source_event_stream_completed" in row]
    terminal_known = [row for row in stats if "simulator_terminated" in row]
    executed_known = [row for row in stats if "all_source_events_executed" in row]
    native_known = [row for row in stats if "native_successful_commands" in row]
    normal_known = [row for row in stats if "normal_simulator_terminal" in row]
    combined_known = [row for row in stats if "source_or_native_terminal_completed" in row]
    used, total = sum(row.get("used", 0) for row in stats), sum(
        row.get("total_play_events", 0) for row in stats
    )
    omissions = Counter()
    repairs = Counter()
    for row in stats:
        omissions.update(row.get("skipped", {}))
        repairs.update(row.get("repair_summary", {}).get("categories", {}))
    early_normal = [row for row in stats if normal_early_terminal(row)]
    return {
        "cohort_rows": cohort_size,
        "attempt_records": processed,
        "rows_without_attempt_record": cohort_size - processed,
        "accepted_replays": sum(bool(row.get("accepted")) for row in stats),
        "source_rows_with_native_run_refusals": sum(
            bool(row.get("native_run_refusals")) for row in stats
        ),
        "early_simulator_decisions_before_source_complete": sum(
            bool(row.get("early_terminal_before_last_source_event")) for row in stats
        ),
        "normal_early_terminal_episodes": len(early_normal),
        "normal_early_terminal_episodes_with_pending_to_mask": sum(
            not bool(row.get("pending_commands_settled")) for row in early_normal
        ),
        "full_source_or_normal_early_terminal_candidates": sum(
            bool(row.get("all_source_events_executed")) or normal_early_terminal(row)
            for row in stats
        ),
        "normal_simulator_terminal_known_rows": len(normal_known),
        "normal_simulator_terminal": sum(
            bool(row["normal_simulator_terminal"]) for row in normal_known
        ) if normal_known else None,
        "source_or_native_terminal_completed_known_rows": len(combined_known),
        "source_or_native_terminal_completed": sum(
            bool(row["source_or_native_terminal_completed"]) for row in combined_known
        ) if combined_known else None,
        "source_or_native_terminal_completion_rate_raw": (
            sum(bool(row["source_or_native_terminal_completed"]) for row in combined_known)
            / cohort_size if combined_known else None
        ),
        "source_or_native_terminal_completion_rate_attempted": (
            sum(bool(row["source_or_native_terminal_completed"]) for row in combined_known)
            / processed if combined_known and processed else None
        ),
        "native_winner_counts": dict(Counter(
            str(row["native_winner"]) for row in stats
            if row.get("normal_simulator_terminal") and row.get("native_winner") in (0, 1)
        )),
        "raw_native_winner_values": dict(Counter(
            str(row["native_winner"]) for row in stats if "native_winner" in row
        )),
        "source_declared_horizon_reached_rows": sum(
            bool(row.get("reached_declared_end")) for row in stats
        ),
        "all_events_complete_and_simulator_terminated": sum(
            bool(row.get("all_source_events_executed") and row.get("simulator_terminated"))
            for row in stats
        ),
        "all_events_complete_and_source_horizon_reached": sum(
            bool(row.get("all_source_events_executed") and row.get("reached_declared_end"))
            for row in stats
        ),
        "omission_reasons": dict(omissions),
        "repair_categories": dict(repairs),
        "all_source_events_executed": (
            sum(bool(row["all_source_events_executed"]) for row in executed_known)
            if executed_known else None
        ),
        "all_source_events_execution_rate": (
            sum(bool(row["all_source_events_executed"]) for row in executed_known) / cohort_size
            if executed_known else None
        ),
        "all_source_events_execution_rate_attempted": (
            sum(bool(row["all_source_events_executed"]) for row in executed_known) / processed
            if executed_known and processed else None
        ),
        "source_event_stream_completion_known_rows": len(complete_known),
        "source_event_stream_completed": (
            sum(bool(row["source_event_stream_completed"]) for row in complete_known)
            if complete_known else None
        ),
        "source_event_stream_completion_rate": (
            sum(bool(row["source_event_stream_completed"]) for row in complete_known) / cohort_size
            if complete_known else None
        ),
        "simulator_terminal_known_rows": len(terminal_known),
        "simulator_terminals": (
            sum(bool(row["simulator_terminated"]) for row in terminal_known)
            if terminal_known else None
        ),
        "simulator_terminal_rate": (
            sum(bool(row["simulator_terminated"]) for row in terminal_known) / cohort_size
            if terminal_known else None
        ),
        "requested_or_submitted_commands": used,
        "source_commands": total,
        "submitted_source_command_ratio": used / total if total else None,
        "native_successful_commands": (
            sum(row["native_successful_commands"] for row in native_known)
            if len(native_known) == cohort_size else None
        ),
        "scope": (
            "Source completion, native execution and simulator terminal are independent metrics"
        ),
        "strata": stratified_summary(stats, traits or []),
    }


def normal_early_terminal(row):
    if "native_winner" in row and row["native_winner"] not in (0, 1):
        return False
    if "normal_simulator_terminal" in row and not row["normal_simulator_terminal"]:
        return False
    return bool(
        row.get("early_terminal_before_last_source_event")
        and row.get("simulator_match_decided")
        and not row.get("simulator_fault", True)
        and not row.get("simulator_truncated", True)
        and row.get("stop_reason") == "native_match_decided"
    )


def stratified_summary(stats, traits):
    def group(name):
        if name == "long_source":
            return [t for t in traits if t["source_events"] >= 20 and t["last_source_tick"] >= 1200]
        if name == "plain_markers":
            return [t for t in traits if not t["has_hero_marker"] and not t["has_evolution_marker"]]
        return [t for t in traits if t[name]]

    result = {}
    for name in ("long_source", "plain_markers", "has_hero_marker", "has_evolution_marker"):
        selected = group(name)
        indices = {row["source_row_index"] for row in selected}
        rows = [row for row in stats if row.get("source_row_index") in indices]
        complete = sum(bool(row.get("all_source_events_executed")) for row in rows)
        combined = sum(bool(row.get("source_or_native_terminal_completed")) for row in rows)
        result[name] = {
            "source_rows": len(selected), "attempt_records": len(rows),
            "complete": complete,
            "complete_per_source_row": complete / len(selected) if selected else None,
            "complete_per_attempt": complete / len(rows) if rows else None,
            "source_or_native_terminal_completed": combined,
            "source_or_native_terminal_per_source_row": combined / len(selected)
            if selected else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--allow-hand-repair", action="store_true")
    parser.add_argument("--max-hand-repairs", type=int, default=1)
    parser.add_argument("--max-tile-shift", type=int, default=2)
    parser.add_argument("--max-delay-ticks", type=int, default=200)
    parser.add_argument("--require-patched-backend", action="store_true")
    parser.add_argument("--skip-unavailable-abilities", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.start_row < 0 or args.timeout_seconds < 1:
        parser.error("Invalid cohort size, source start or timeout")
    if args.max_hand_repairs < -1 or min(args.max_tile_shift, args.max_delay_ticks) < 0:
        parser.error("Invalid repair budget")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new output directory; benchmark cohorts are immutable")
    args.out.mkdir(parents=True, exist_ok=True)
    cohort_path = args.out / "cohort.parquet"
    cohort = fixed_cohort(args.parquet, cohort_path, start=args.start_row, count=args.count)
    backend = backend_fingerprints()
    if args.require_patched_backend and backend["verified_installed_backend"] is None:
        parser.error("A verified patched backend receipt is required")
    protocol = {
        **cohort, "source_revision": args.source_revision, "seed": args.seed,
        "arms": args.arms, "allow_hand_repair": args.allow_hand_repair,
        "max_hand_repairs": args.max_hand_repairs, "max_tile_shift": args.max_tile_shift,
        "max_delay_ticks": args.max_delay_ticks,
        "skip_unavailable_abilities": args.skip_unavailable_abilities,
        "importer_sha256": file_sha256(ROOT / "scripts/import_il_replay.py"),
        "code_sha256": code_fingerprints(),
        "backend_fingerprints": backend,
        "require_patched_backend": args.require_patched_backend,
        "limitation": "A completed source stream is not original-state fidelity or expert quality",
    }
    atomic_json(args.out / "protocol.json", protocol)
    results = {}
    for name in args.arms:
        arm = args.out / name
        arm.mkdir()
        extra = list(ARMS[name])
        if name == "repair":
            extra += ["--max-tile-shift", str(args.max_tile_shift)]
            extra += ["--max-delay-ticks", str(args.max_delay_ticks)]
            if args.allow_hand_repair:
                extra += ["--repair-hand", "--max-hand-repairs", str(args.max_hand_repairs)]
            if args.skip_unavailable_abilities:
                extra += ["--skip-unavailable-abilities"]
        command = [
            sys.executable, str(ROOT / "scripts/import_il_replay.py"), str(cohort_path),
            "--out", str(arm), "--limit", str(cohort["source_rows"]),
            "--max-source-rows", str(cohort["source_rows"]), "--audit-only",
            "--source-revision", args.source_revision, "--seed", str(args.seed),
            "--min-plays", "1", "--min-coverage", "0", "--max-omissions", "0",
            *extra,
        ]
        atomic_json(arm / "command.json", {"command": command})
        code_before = code_fingerprints()
        backend_before = backend_fingerprints()
        started = time.monotonic()
        timed_out = False
        with (arm / "import.log").open("w") as stream:
            try:
                process = subprocess.run(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=args.timeout_seconds,
                )
                code = process.returncode
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        stats_path = arm / "import_stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else []
        code_after = code_fingerprints()
        backend_after = backend_fingerprints()
        results[name] = {
            "returncode": code, "timed_out": timed_out,
            "code_sha256_before": code_before,
            "code_sha256_after": code_after,
            "code_matches_registered_protocol": (
                code_before == code_after == protocol["code_sha256"]
            ),
            "backend_fingerprints_before": backend_before,
            "backend_fingerprints_after": backend_after,
            "backend_matches_registered_protocol": (
                backend_before == backend_after == protocol["backend_fingerprints"]
            ),
            "seconds": time.monotonic() - started,
            **summarize(stats, cohort["source_rows"], cohort["source_traits"]),
        }
        atomic_json(args.out / "summary.json", {"protocol": protocol, "arms": results})
    print(json.dumps({"out": str(args.out), "arms": results}, indent=2))
    if any(
        row["returncode"] != 0 or row["timed_out"]
        or not row["code_matches_registered_protocol"]
        or not row["backend_matches_registered_protocol"]
        for row in results.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
