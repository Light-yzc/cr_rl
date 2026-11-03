"""Print a compact, read-only snapshot of the live v2 Colab training jobs."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/content/cr-rl-lab/artifacts/121m_v2_bc_20261103")
DRIVE = Path("/content/drive/MyDrive/cr_rl/runs/121m_v2_bc_20261103")
HANDOFF = Path("/content/cr-rl-lab/artifacts/121m_v2_bc_20261103_handoff_health")
RECOVERED_HANDOFF = Path("/content/cr-rl-lab/artifacts/121m_v2_bc_20261103_handoff_recovered")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def main():
    status = read_json(ROOT / "status.json", {})
    health = read_json(ROOT / "monitor" / "loss_health.json", {}) or read_json(
        ROOT / "loss_health.json", {}
    )
    report = read_json(ROOT / "bc.json", {})
    history = report.get("history", [])
    monitor = read_json(ROOT / "monitor" / "health_monitor_status.json", {}) or read_json(
        ROOT / "health_monitor_status.json", {}
    )
    recovery = read_json(ROOT / "monitor" / "recovery_status.json", {})
    checkpoint = read_json(DRIVE / "bc" / "latest.json", {})
    recovered_handoff = read_json(RECOVERED_HANDOFF / "status.json", {})
    handoff = recovered_handoff or read_json(HANDOFF / "status.json", {})
    handles = read_json(Path("/content/v2_job_handles.json"), [])

    train_rows = []
    try:
        for line in (ROOT / "train.log").read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("phase") == "bc_train":
                train_rows.append(row)
    except OSError:
        pass

    recent = [
        {
            key: row.get(key)
            for key in ("optimizer_steps", "epoch", "optimizer_batch_loss", "grad_norm")
            if key in row
        }
        for row in train_rows[-8:]
    ]
    pids = [
        status.get("worker_pid"),
        status.get("training_pid"),
        monitor.get("pid"),
        handoff.get("worker_pid"),
        recovery.get("pid"),
        recovery.get("handoff_pid"),
        *(job.get("pid") for job in handles if isinstance(job, dict)),
    ]
    pid_arg = ",".join(str(pid) for pid in pids if pid)
    processes = (
        subprocess.run(
            ["bash", "-lc", f"ps -o pid=,stat=,pcpu=,etime=,cmd= -p {pid_arg}"],
            capture_output=True,
            text=True,
            check=False,
        )
        .stdout.strip()
        .splitlines()
        if pid_arg
        else []
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    recovery_log = Path(
        "/content/cr-rl-lab/artifacts/121m_v2_bc_20261103/monitor/recovery_launcher.log"
    )
    recovery_log_tail = (
        recovery_log.read_text(errors="replace").splitlines()[-8:] if recovery_log.exists() else []
    )
    metric_keys = (
        "global_loss",
        "card_loss",
        "position_loss",
        "wait_accuracy",
        "play_recall",
        "play_slot_accuracy",
        "play_tile_accuracy",
        "play_tile_mean_l1_distance",
        "play_tile_within_2",
        "policy_nonwait_probability",
        "label_nonwait_fraction",
        "nonwait_probability_on_action_rows",
        "wait_false_action_probability",
    )
    last_epoch = history[-1] if history else {}
    last_epoch_summary = {
        "epoch": last_epoch.get("epoch"),
        "balanced_validation": {
            key: last_epoch.get("validation", {}).get(key)
            for key in metric_keys
            if key in last_epoch.get("validation", {})
        },
        "natural_validation": {
            key: last_epoch.get("natural_validation", {}).get(key)
            for key in metric_keys
            if key in last_epoch.get("natural_validation", {})
        },
    }
    drive_monitor = DRIVE / "monitor"

    snapshot = {
        "observed_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "job_status": {
            key: status.get(key)
            for key in (
                "phase",
                "worker_pid",
                "training_pid",
                "optimizer_steps",
                "completed_epochs",
                "success",
            )
        },
        "last_train_rows": recent,
        "report_completed_epochs": report.get("completed_epochs"),
        "last_epoch_metrics": last_epoch_summary,
        "epoch_health": health.get("epochs"),
        "rolling_32_loss": health.get("rolling_32_optimizer_batch_loss"),
        "health_monitor": {key: monitor.get(key) for key in ("phase", "pid", "heartbeat_unix")},
        "recovery_supervisor": {
            key: recovery.get(key)
            for key in (
                "phase",
                "pid",
                "heartbeat_unix",
                "training_pid",
                "training_process_state",
                "optimizer_steps",
                "seconds_without_new_step",
                "handoff_phase",
                "error",
            )
        },
        "job_handles": handles,
        "wandb_key_present": bool(__import__("os").environ.get("WANDB_API_KEY")),
        "drive_checkpoint": {
            key: checkpoint.get(key)
            for key in ("path", "optimizer_steps", "epoch", "sha256", "bytes", "mid_epoch")
        },
        "drive_monitor_files": {
            name: (drive_monitor / name).exists()
            for name in (
                "loss_health.json",
                "loss_health.jsonl",
                "health_monitor_status.json",
                "recovery_status.json",
            )
        },
        "handoff_status": {
            key: handoff.get(key)
            for key in ("phase", "worker_pid", "child_pid", "bc_run", "passed")
        },
        "recovered_handoff_status_exists": recovered_handoff is not None,
        "handoff_error": str(handoff.get("error", ""))[:500],
        "recovery_log_tail": recovery_log_tail,
        "processes": processes,
        "gpu": gpu,
    }
    print(json.dumps(snapshot, ensure_ascii=False))


if __name__ == "__main__":
    main()
