"""Reattach monitoring and the BC-to-PPO handoff to a surviving BC worker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256

ROOT = Path("/content/cr-rl-lab")


def process_state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (OSError, IndexError):
        return None


def last_training_event(path: Path) -> dict:
    latest = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("phase") == "bc_train":
                latest = row
    except OSError:
        pass
    return latest


def update_job_handles(records: list[dict]) -> None:
    path = Path("/content/v2_job_handles.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="121m_v2_bc_20261103")
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--ppo-updates", type=int, default=6)
    parser.add_argument("--handoff-suffix", default="_handoff_recovered")
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()

    run = ROOT / "artifacts" / args.run_name
    drive = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    monitor = run / "monitor"
    monitor_drive = drive / "monitor"
    monitor.mkdir(parents=True, exist_ok=True)
    monitor_drive.mkdir(parents=True, exist_ok=True)
    recovery_path = monitor / "recovery_status.json"
    recovery_drive = monitor_drive / "recovery_status.json"
    status_path = run / "status.json"
    drive_status = drive / "status.json"
    handoff_root = ROOT / "artifacts" / (args.run_name + args.handoff_suffix)
    handoff_status_path = handoff_root / "status.json"

    state = process_state(args.training_pid)
    if state in (None, "Z"):
        raise RuntimeError(
            f"BC training PID {args.training_pid} is not alive; refusing to fake a recovery"
        )

    status = json.loads(status_path.read_text())
    status.pop("error", None)
    status.update(
        phase="bc_training_recovered",
        success=False,
        worker_pid=args.training_pid,
        training_pid=args.training_pid,
        recovered_supervision=True,
    )
    atomic_json(status_path, status)
    copy_verified(status_path, drive_status)

    recovery = {
        "phase": "running",
        "pid": os.getpid(),
        "training_pid": args.training_pid,
        "started_unix": time.time(),
        "handoff_suffix": args.handoff_suffix,
    }
    atomic_json(recovery_path, recovery)
    copy_verified(recovery_path, recovery_drive)

    health_path = monitor / "health_monitor_status.json"
    health_log = monitor / "health_monitor.log"
    with health_log.open("w") as log:
        health_monitor = subprocess.Popen(
            [
                sys.executable,
                "scripts/monitor_v2_bc.py",
                "--run-name",
                args.run_name,
                "--interval",
                "60",
                "--hours",
                str(args.hours),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    handles = [
        {"name": "v2_bc_health", "pid": health_monitor.pid, "status": str(health_path)},
        {"name": "v2_bc_recovery", "pid": os.getpid(), "status": str(recovery_path)},
    ]
    update_job_handles(handles)

    if handoff_status_path.exists():
        raise RuntimeError(f"Handoff path already exists: {handoff_status_path}")
    handoff_log = monitor / "handoff_recovered.log"
    with handoff_log.open("w") as log:
        handoff = subprocess.Popen(
            [
                sys.executable,
                "scripts/run_v2_handoff.py",
                "--bc-run",
                args.run_name,
                "--ppo-updates",
                str(args.ppo_updates),
                "--suffix",
                args.handoff_suffix,
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    handles.append(
        {"name": "v2_handoff_recovered", "pid": handoff.pid, "status": str(handoff_status_path)}
    )
    update_job_handles(handles)

    deadline = time.monotonic() + args.hours * 3600
    previous_step = None
    step_changed_at = time.time()
    while time.monotonic() < deadline:
        now = time.time()
        process = process_state(args.training_pid)
        event = last_training_event(run / "train.log")
        step = event.get("optimizer_steps")
        if step != previous_step:
            previous_step = step
            step_changed_at = now

        if process in (None, "Z"):
            report_path = run / "bc.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            manifest_path = drive / "bc" / "latest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            checkpoint = run / "bc.pt"
            checkpoint_sha = file_sha256(checkpoint) if checkpoint.exists() else None
            passed = (
                report.get("completed_epochs") == status.get("epochs")
                and report.get("optimizer_steps", 0) > 0
                and manifest.get("optimizer_steps") == report.get("optimizer_steps")
                and manifest.get("sha256") == checkpoint_sha
            )
            if passed:
                status.update(
                    phase="completed",
                    success=True,
                    optimizer_steps=report["optimizer_steps"],
                    completed_epochs=report["completed_epochs"],
                    recovered_completion_unix=now,
                )
            else:
                status.update(
                    phase="failed",
                    success=False,
                    error=(
                        "Recovered BC worker exited without matching complete report "
                        "and Drive checkpoint"
                    ),
                    recovered_exit_unix=now,
                )
            atomic_json(status_path, status)
            copy_verified(status_path, drive_status)

        handoff_status = None
        if handoff_status_path.exists():
            handoff_status = json.loads(handoff_status_path.read_text())
            phase = handoff_status.get("phase")
            if phase in ("completed", "failed"):
                recovery.update(
                    phase=phase,
                    handoff_phase=phase,
                    last_optimizer_steps=step,
                    finished_unix=now,
                )
                atomic_json(recovery_path, recovery)
                copy_verified(recovery_path, recovery_drive)
                return

        recovery.update(
            phase="running",
            heartbeat_unix=now,
            training_process_state=process,
            optimizer_steps=step,
            seconds_without_new_step=now - step_changed_at,
            handoff_phase=handoff_status.get("phase") if handoff_status else "starting",
        )
        atomic_json(recovery_path, recovery)
        copy_verified(recovery_path, recovery_drive)
        time.sleep(args.interval)

    recovery.update(phase="timeout", heartbeat_unix=time.time(), optimizer_steps=previous_step)
    atomic_json(recovery_path, recovery)
    copy_verified(recovery_path, recovery_drive)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
