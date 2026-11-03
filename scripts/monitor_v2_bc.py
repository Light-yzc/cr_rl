"""Persist real BC progress and diagnose loss/prior health without changing training."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified


def json_lines(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def epoch_health(report):
    baseline = report.get("natural_baseline", {})
    placement_baseline = report.get("baseline", {}).get("position_loss", float("nan"))
    history = report.get("history", [])
    output = []
    prior = float(baseline.get("label_nonwait_fraction", float("nan")))
    for row in history:
        natural = row.get("natural_validation", {})
        balanced = row.get("validation", {})
        probability = natural.get("policy_nonwait_probability", float("nan"))
        action_rows = natural.get("nonwait_probability_on_action_rows", float("nan"))
        wait_rows = natural.get("wait_false_action_probability", float("nan"))
        placement_nll = balanced.get("position_loss", float("nan"))
        ratio = probability / prior if prior > 0 and math.isfinite(probability) else float("nan")
        notes = []
        reasons = []
        if not math.isfinite(natural.get("global_loss", float("nan"))):
            reasons.append("nonfinite natural validation loss")
        elif natural["global_loss"] >= baseline.get("global_loss", float("inf")):
            reasons.append("natural validation loss has not beaten initialization")
        if not math.isfinite(ratio) or ratio < 0.5:
            reasons.append("action probability is below half the held-out label rate")
        elif ratio > 2.0:
            reasons.append("action probability is over twice the held-out label rate")
        if math.isfinite(action_rows) and math.isfinite(wait_rows) and action_rows <= wait_rows:
            reasons.append("the policy does not distinguish action rows from WAIT rows")
        if (
            math.isfinite(placement_nll)
            and math.isfinite(placement_baseline)
            and placement_nll >= placement_baseline
        ):
            reasons.append("conditional placement loss has not beaten initialization")
        # The training loss corrects the sampled WAIT/PLAY prior to its natural
        # replay frequency. Balanced-set global loss is therefore diagnostic,
        # not a directly comparable selection objective.
        if natural.get("play_recall") == 0:
            notes.append(
                "greedy argmax predicts WAIT; closed-loop inference samples the categorical policy"
            )
        output.append(
            {
                "epoch": row.get("epoch"),
                "natural_loss": natural.get("global_loss"),
                "balanced_loss": balanced.get("global_loss"),
                "placement_nll": placement_nll,
                "label_action_rate": prior,
                "predicted_action_rate": probability,
                "action_rate_ratio": ratio,
                "action_row_probability": action_rows,
                "wait_row_probability": wait_rows,
                "greedy_play_recall": natural.get("play_recall"),
                "play_slot_accuracy": balanced.get("play_slot_accuracy"),
                "tile_exact_accuracy": balanced.get("play_tile_accuracy"),
                "tile_mean_l1_distance": balanced.get("play_tile_mean_l1_distance"),
                "tile_within_2": balanced.get("play_tile_within_2"),
                "status": "warning" if reasons else "healthy",
                "reasons": reasons,
                "notes": notes,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    root = Path("/content/cr-rl-lab/artifacts") / args.run_name
    drive = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    monitor_root = root / "monitor"
    monitor_drive = drive / "monitor"
    monitor_root.mkdir(parents=True, exist_ok=True)
    monitor_drive.mkdir(parents=True, exist_ok=True)
    status_path = monitor_root / "health_monitor_status.json"
    report_path = monitor_root / "loss_health.json"
    log_path = monitor_root / "loss_health.jsonl"
    deadline = time.monotonic() + args.hours * 3600
    previous_epoch = -1
    while time.monotonic() < deadline:
        now = time.time()
        job = json.loads((root / "status.json").read_text())
        report_file = root / "bc.json"
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        health = epoch_health(report)
        train_rows = [r for r in json_lines(root / "train.log") if r.get("phase") == "bc_train"]
        steps = train_rows[-1].get("optimizer_steps") if train_rows else None
        recent = [
            r["optimizer_batch_loss"]
            for r in train_rows[-32:]
            if math.isfinite(r.get("optimizer_batch_loss", float("nan")))
        ]
        trailing_loss = sum(recent) / len(recent) if recent else None
        reasons = []
        if health and health[-1]["status"] == "warning":
            reasons.extend(health[-1]["reasons"])
        if trailing_loss is not None and not math.isfinite(trailing_loss):
            reasons.append("nonfinite recent training loss")
        checkpoint_path = drive / "bc" / "latest.json"
        checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
        result = {
            "observed_unix": now,
            "phase": job.get("phase"),
            "worker_pid": job.get("worker_pid"),
            "training_pid": job.get("training_pid"),
            "worker_alive": Path(f"/proc/{job.get('worker_pid')}/stat").exists(),
            "optimizer_steps_from_log": steps,
            "rolling_32_optimizer_batch_loss": trailing_loss,
            "completed_epochs": report.get("completed_epochs", 0),
            "epochs": health,
            "latest_drive_checkpoint": checkpoint,
            "status": "warning" if reasons else "running",
            "warning_reasons": reasons,
            "wandb_run": json.loads((root / "bc.wandb.json").read_text()).get("url")
            if (root / "bc.wandb.json").exists()
            else None,
        }
        atomic_json(report_path, result)
        with log_path.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        snapshot = root / ".loss_health.jsonl"
        snapshot.write_bytes(log_path.read_bytes())
        copy_verified(report_path, monitor_drive / "loss_health.json")
        copy_verified(snapshot, monitor_drive / "loss_health.jsonl")
        atomic_json(status_path, {"phase": "running", "pid": os.getpid(), "heartbeat_unix": now})
        copy_verified(status_path, monitor_drive / "health_monitor_status.json")
        new_epoch = report.get("completed_epochs", 0)
        if new_epoch > previous_epoch:
            print(json.dumps(result, ensure_ascii=False), flush=True)
            previous_epoch = new_epoch
        if job.get("phase") in ("failed", "completed"):
            status = (
                "completed" if job.get("phase") == "completed" and job.get("success") else "failed"
            )
            atomic_json(
                status_path, {"phase": status, "pid": os.getpid(), "heartbeat_unix": time.time()}
            )
            copy_verified(status_path, monitor_drive / "health_monitor_status.json")
            return
        time.sleep(args.interval)
    atomic_json(
        status_path, {"phase": "timeout", "pid": os.getpid(), "heartbeat_unix": time.time()}
    )
    copy_verified(status_path, monitor_drive / "health_monitor_status.json")


if __name__ == "__main__":
    main()
