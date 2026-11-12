"""Incrementally publish CPU-only PPO trial logs to the existing TensorBoard tree.

The publisher reads JSON logs and evaluation reports only. It never imports
Torch, opens checkpoints or starts model/simulator processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

from cr_rl.monitoring.tensorboard_bridge import (
    numeric,
    read_lines,
    summary_scalars,
    timing_scalars,
)

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("baseline", "adaptive_lr", "longer_gae", "diverse_pool")


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluation_scalars(row):
    """Keep comparable aggregate outcomes; exclude individual replay decisions."""
    values = {
        key: float(row[key])
        for key in (
            "games", "completed_games", "planned_games", "invalid_games", "wins",
            "losses", "draws", "mean_score", "win_rate", "seconds", "complete",
        )
        if numeric(row.get(key))
    }
    games = row.get("games", row.get("completed_games"))
    if games is None and all(numeric(row.get(key)) for key in ("wins", "losses", "draws")):
        games = sum(row[key] for key in ("wins", "losses", "draws"))
    if numeric(games) and games > 0:
        values["games"] = games
        if numeric(row.get("wins")):
            values["win_rate"] = row["wins"] / games
        if numeric(row.get("wins")) and numeric(row.get("draws")):
            values["score"] = (row["wins"] + 0.5 * row["draws"]) / games
    for key in ("deck_pair_cluster_bootstrap_95_interval", "paired_bootstrap_95_interval"):
        interval = row.get(key)
        if isinstance(interval, list) and len(interval) == 2 and all(map(numeric, interval)):
            values["score_ci_low"], values["score_ci_high"] = interval
            break
    return values


def evaluation_step(row, path, latest_update):
    for key in ("candidate_update", "update", "updates", "checkpoint_update"):
        if numeric(row.get(key)) and row[key] >= 0:
            return int(row[key])
    match = re.search(r"(?:update|checkpoint)[_-]?(\d+)", path.parent.name)
    return int(match[1]) if match else latest_update


def changed_json(path, previous):
    """Read a report only when its stat changed, then compare actual content."""
    stat = path.stat()
    stamp = [stat.st_mtime_ns, stat.st_size, stat.st_ino]
    if previous.get("stamp") == stamp:
        return None, previous
    row = json.loads(path.read_text())
    signature = digest(row)
    next_state = {"stamp": stamp, "digest": signature}
    return (row if signature != previous.get("digest") else None), next_state


def publish_once(trials_root, branches, state, emit):
    """Use durable per-source cursors/hashes so restarts do not replay history."""
    counts = {"metrics": 0, "timings": 0, "evaluations": 0}
    for branch in branches:
        directory = trials_root / branch
        current = state.setdefault("branches", {}).setdefault(branch, {})
        known = current.setdefault("metrics_by_update", {})
        latest_update = int(current.get("latest_update", 0))
        rows, cursor = read_lines(directory / "metrics.jsonl", current.get("metrics_cursor", {}))
        for row in rows:
            update = int(row["update"])
            latest_update = max(latest_update, update)
            signature = digest(row)
            if known.get(str(update)) == signature:
                continue
            timing = directory / "timings" / f"update_{update:06d}.json"
            stamp = timing.stat().st_mtime if timing.exists() else time.time()
            emit(branch, summary_scalars(row), update, stamp)
            known[str(update)] = signature
            counts["metrics"] += 1
        current["metrics_cursor"] = cursor
        current["latest_update"] = latest_update
        for category, paths in (
            ("timings", sorted((directory / "timings").glob("*.json"))),
            ("evaluations", sorted((directory / "evaluation").glob("*/evaluation.json"))),
        ):
            files = current.setdefault(category, {})
            for path in paths:
                key = str(path.relative_to(directory))
                try:
                    row, file_state = changed_json(path, files.get(key, {}))
                except (OSError, json.JSONDecodeError):
                    # A writer may still be replacing a report; retry on the next pass.
                    continue
                if row is not None:
                    if category == "timings":
                        values, step = timing_scalars(row), row["update"]
                    else:
                        prefix = f"evaluation/{path.parent.name}/"
                        values = {prefix + k: v for k, v in evaluation_scalars(row).items()}
                        step = evaluation_step(row, path, latest_update)
                    emit(branch, values, step, path.stat().st_mtime)
                    counts[category] += 1
                files[key] = file_state
    return counts


def run(args):
    # TensorBoard's protobuf writer does not need torch.utils.tensorboard.
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    args.logdir.mkdir(parents=True, exist_ok=True)
    state_path = args.logdir / "convergence_trials_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    source = str(args.trials_root.resolve())
    if state.get("source", source) != source:
        raise ValueError("Publisher state belongs to another trials root; use a separate logdir")
    state["source"] = source
    writers = {}

    def emit(branch, values, step, wall_time):
        values = {key: float(value) for key, value in values.items() if numeric(value)}
        if not values:
            return
        if branch not in writers:
            writers[branch] = EventFileWriter(str(args.logdir / f"convergence_{branch}"))
        writers[branch].add_event(Event(
            wall_time=wall_time,
            step=int(step),
            summary=Summary(
                value=[Summary.Value(tag=k, simple_value=v) for k, v in values.items()]
            ),
        ))

    try:
        while True:
            counts = publish_once(args.trials_root, args.branches, state, emit)
            for writer in writers.values():
                writer.flush()
            state["heartbeat_unix"] = time.time()
            temporary = state_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(state_path)
            if any(counts.values()) or not args.watch:
                print(json.dumps({"published": counts, "logdir": str(args.logdir)}), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    finally:
        for writer in writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials-root", type=Path, default=ROOT / "artifacts/ppo_convergence_trials_20261108"
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=ROOT / "artifacts/ppo_convergence_trials_20261108/tensorboard",
    )
    parser.add_argument("--branches", nargs="+", choices=BRANCHES, default=list(BRANCHES))
    parser.add_argument("--once", action="store_true", help="Publish one pass (the default)")
    parser.add_argument("--watch", action="store_true", help="Poll continuously instead of once")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("Polling interval must be finite and positive")
    if args.once and args.watch:
        parser.error("Choose either --once or --watch")
    run(args)


if __name__ == "__main__":
    main()
