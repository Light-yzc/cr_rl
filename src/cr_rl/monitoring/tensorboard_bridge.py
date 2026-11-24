"""Tail immutable/append-only trainer logs without importing Torch or touching training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def flatten(value, prefix=""):
    result = {}
    for key, child in value.items():
        tag = f"{prefix}/{key}" if prefix else key
        if isinstance(child, dict):
            result.update(flatten(child, tag))
        elif numeric(child):
            result[tag] = float(child)
    return result


def read_lines(path, cursor):
    """Never consume a partial JSON line; support atomic rotation/truncation."""
    if not path.exists():
        return [], cursor
    stat = path.stat()
    if cursor.get("inode") != stat.st_ino or cursor.get("offset", 0) > stat.st_size:
        cursor = {"inode": stat.st_ino, "offset": 0}
    rows = []
    with path.open("rb") as stream:
        stream.seek(cursor.get("offset", 0))
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                stream.seek(start)
                break
            if line.strip():
                rows.append(json.loads(line))
        cursor = {"inode": stat.st_ino, "offset": stream.tell()}
    return rows, cursor


def summary_scalars(row):
    groups = {
        "ppo_update": row.get("metrics", {}),
        "critic_parallel": row.get("critic_parallel", {}),
        "rollout_parallel": row.get("rollout_parallel", {}),
        "sample_utilization": row.get("sample_utilization", {}),
        "decision_coverage": row.get("decision_coverage", {}),
        "actor_lr_schedule": row.get("actor_lr_schedule", {}),
        "opponent_retention": row.get("opponent_retention", {}),
        "reward_components": row.get("reward_component_sums", {}),
        "peak_gpu_memory_gib": row.get("peak_cuda_per_device_gib", {}),
        "learning_rates": {k: v[0] for k, v in row.get("learning_rates", {}).items() if v},
        "train_traj": row.get("training_win_rates", {}),
    }
    result = flatten(groups)
    retention = row.get("opponent_retention", {})
    if retention:
        result["opponent_retention/confirmed_expert_count"] = len(
            retention.get("confirmed_expert_ids", [])
        )
        result["opponent_retention/specialty_count"] = len(retention.get("specialties", []))
    for key in (
        "learner_samples_total",
        "valid_learner_samples",
        "actor_training_samples",
        "critic_training_samples",
        "completed_training_games_total",
        "terminal_games",
        "faulted_samples",
        "initial_logp_max_error",
        "learner_reward_sum",
        "learner_raw_reward_sum",
        "actor_kl_stopped",
    ):
        if numeric(row.get(key)):
            result[f"training/{key}"] = row[key]
    if (
        "train_traj/overall_win_rate" not in result
        and (row.get("terminal_games", 0) or 0) > 0
        and numeric(row.get("terminal_seat_wins"))
    ):
        result["train_traj/overall_win_rate"] = float(row["terminal_seat_wins"]) / float(
            row["terminal_games"]
        )
    result["performance/update_seconds"] = row["seconds"]
    result["performance/effective_new_samples_per_second"] = (
        row["valid_learner_samples"] / row["seconds"]
    )
    result["performance/completed_games_per_hour"] = (
        row["completed_training_games"] * 3600 / row["seconds"]
    )
    return result


def validation_scalars(results):
    """Extract numeric validation scalars, skipping errored zero-game entries."""
    scalars = {}
    for label, vals in (results or {}).items():
        if not isinstance(vals, dict):
            continue
        if int(vals.get("games", 1) or 0) == 0 and vals.get("error"):
            continue
        for key in (
            "win_rate",
            "wins",
            "games",
            "invalid_games",
            "p50_ms",
            "p99_ms",
            "max_ms",
            "latency_violations",
        ):
            if numeric(vals.get(key)):
                scalars[f"validation/{label}_{key}"] = vals[key]
    return scalars


def event_scalars(row):
    phase = row["phase"]
    if phase == "optimization":
        step = row.get("optimizer_steps_total", row["optimizer_steps"])
        return step, flatten({k: v for k, v in row.items() if numeric(v)}, "actor_step")
    if phase == "critic_optimization":
        return row["critic_optimizer_steps"], flatten(
            {k: v for k, v in row.items() if numeric(v)}, "critic_step"
        )
    step = row.get("learner_samples_total", row["update"])
    if phase == "policy":
        return step, {
            f"rollout_policy/{k}": row[k]
            for k in ("play_fraction", "entropy", "max_action_probability")
            if numeric(row.get(k))
        }
    if phase == "rollout" and row.get("seconds", 0) > 0:
        return step, {
            "rollout/valid_samples_per_second": row["rollout_samples"] / row["seconds"],
            "rollout/fraction_complete": row["rollout_step"] / row["rollout_total"],
        }
    if phase == "kl_stop":
        return step, flatten({k: v for k, v in row.items() if numeric(v)}, "kl_guard")
    if phase == "amp_overflow":
        return step, flatten(
            {k: v for k, v in row.items() if numeric(v)},
            "amp_overflow/" + row["component"],
        )
    if phase == "storage_retention":
        return row["update"], {
            f"storage/{k}": row[k]
            for k in ("removed_bytes", "removed_files", "removed_opponents")
            if numeric(row.get(k))
        }
    if phase == "validation":
        return row["update"], validation_scalars(row.get("results", {}))
    return step, {}


def timing_scalars(row):
    result = {}
    for key in (
        "rollout_seconds",
        "optimization_seconds",
        "serialization_seconds",
        "admission_seconds",
        "backup_seconds",
        "update_wall_seconds",
    ):
        if numeric(row.get(key)):
            result["timing/" + key] = row[key]
            if key != "update_wall_seconds" and row.get("update_wall_seconds", 0) > 0:
                result["timing_fraction/" + key] = row[key] / row["update_wall_seconds"]
    return result


def gpu_stats(indices):
    query = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=8,
    )
    result = {}
    for cells in csv.reader(process.stdout.splitlines()):
        index = cells[0].strip()
        if index not in indices:
            continue
        for key, value in zip(
            (
                "utilization_percent",
                "memory_used_mib",
                "memory_total_mib",
                "power_watts",
                "temperature_celsius",
            ),
            cells[1:],
            strict=True,
        ):
            try:
                result[f"gpu_{index}/{key}"] = float(value)
            except ValueError:
                pass
    return result


def host_memory_stats(run_dir, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    """Read RAM pressure without importing Torch or creating another model process."""
    result = {}
    try:
        for line in (proc_root / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                result["ram/host_available_gib"] = int(line.split()[1]) / 2**20
        pid = int((run_dir / "trainer.pid").read_text())
        for line in (proc_root / str(pid) / "status").read_text().splitlines():
            for field, tag in (("VmRSS:", "trainer_rss_gib"), ("VmSwap:", "trainer_swap_gib")):
                if line.startswith(field):
                    result["ram/" + tag] = int(line.split()[1]) / 2**20
        groups = (proc_root / str(pid) / "cgroup").read_text().splitlines()
        unified = next(line[3:] for line in groups if line.startswith("0::"))
        group = cgroup_root / unified.lstrip("/")
        for field, tag in (
            ("memory.current", "service_current_gib"),
            ("memory.peak", "service_peak_gib"),
            ("memory.max", "service_limit_gib"),
            ("memory.swap.current", "service_swap_gib"),
            ("memory.swap.max", "service_swap_limit_gib"),
        ):
            value = (group / field).read_text().strip()
            if value != "max":
                result["ram/" + tag] = int(value) / 2**30
        events = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
        result["ram/service_oom_kills"] = int(events.get("oom_kill", 0))
    except (OSError, ValueError, StopIteration):
        pass
    return result


def run(args):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    args.logdir.mkdir(parents=True, exist_ok=True)
    state_path = args.logdir / "bridge_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    writer = EventFileWriter(str(args.logdir / "local_dual_v100"), flush_secs=5)
    state.setdefault("timings", {})
    state.setdefault("validations", {})
    state.setdefault("episodes", {"count": 0, "wins": 0, "score_sum": 0})
    last_gpu = 0

    def emit(values, step, wall_time=None):
        values = {key: float(value) for key, value in values.items() if numeric(value)}
        if values:
            writer.add_event(
                Event(
                    wall_time=wall_time or time.time(),
                    step=int(step),
                    summary=Summary(
                        value=[Summary.Value(tag=k, simple_value=v) for k, v in values.items()]
                    ),
                )
            )

    try:
        while True:
            for name in ("metrics.jsonl", "progress.jsonl"):
                rows, cursor = read_lines(args.run_dir / name, state.get(name, {}))
                for row in rows:
                    if name == "metrics.jsonl":
                        timing = args.run_dir / "timings" / f"update_{row['update']:06d}.json"
                        wall_time = timing.stat().st_mtime if timing.exists() else None
                        emit(summary_scalars(row), row["update"], wall_time)
                    else:
                        step, values = event_scalars(row)
                        if (
                            row["phase"] == "episode"
                            and not row.get("fault")
                            and not row.get("truncated")
                        ):
                            episodes = state["episodes"]
                            episodes["count"] += 1
                            episodes["wins"] += row["reward"] > 0
                            episodes["score_sum"] += row["reward"]
                            values.update(
                                {
                                    "training_episodes/return": row["reward"],
                                    "training_episodes/length": row["episode_length"],
                                    "training_episodes/observed_win_rate": episodes["wins"]
                                    / episodes["count"],
                                    "training_episodes/observed_mean_return": episodes["score_sum"]
                                    / episodes["count"],
                                }
                            )
                        emit(values, step, row.get("time_unix"))
                        if row["phase"] == "validation" and values:
                            val_summary = (
                                args.run_dir
                                / "validation"
                                / f"update_{int(row['update']):08d}"
                                / "summary.json"
                            )
                            state["validations"][str(int(row["update"]))] = (
                                val_summary.stat().st_mtime_ns
                                if val_summary.exists()
                                else "event"
                            )
                state[name] = cursor
            for path in sorted((args.run_dir / "timings").glob("*.json")):
                stamp = path.stat().st_mtime_ns
                if state["timings"].get(path.name) != stamp:
                    row = json.loads(path.read_text())
                    emit(timing_scalars(row), row["update"])
                    state["timings"][path.name] = stamp
            for path in sorted((args.run_dir / "validation").glob("update_*/summary.json")):
                suffix = path.parent.name.removeprefix("update_")
                if not suffix.isdigit():
                    continue
                update_key = str(int(suffix))
                stamp = path.stat().st_mtime_ns
                if state["validations"].get(update_key) in (stamp, "event"):
                    continue
                results = json.loads(path.read_text())
                emit(validation_scalars(results), int(update_key), path.stat().st_mtime)
                state["validations"][update_key] = stamp
            now = time.time()
            if now - last_gpu >= 10:
                try:
                    emit(
                        gpu_stats(set(args.gpu_ids.split(","))),
                        int(now - state.setdefault("started", now)),
                        now,
                    )
                except (subprocess.SubprocessError, OSError) as error:
                    print("GPU telemetry unavailable:", error, flush=True)
                last_gpu = now
                emit(
                    {
                        **host_memory_stats(args.run_dir),
                        "storage/disk_free_gib": os.statvfs(args.run_dir).f_bavail
                        * os.statvfs(args.run_dir).f_frsize
                        / 2**30,
                    },
                    int(now - state["started"]),
                    now,
                )
            writer.flush()
            state["heartbeat_unix"] = now
            tmp = state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(state_path)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="1,2")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
