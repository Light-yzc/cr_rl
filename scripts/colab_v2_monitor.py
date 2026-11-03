"""Kernel-side supervision of real v2 worker processes; never simulates activity."""

import json
import os
import time
from pathlib import Path


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def main():
    handles = Path(os.environ.get("CRRL_V2_HANDLES", "/content/v2_job_handles.json"))
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        jobs = json.loads(handles.read_text())
        active = False
        for job in jobs:
            status_path = Path(job["status"])
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            running = alive(job["pid"])
            report = {
                "job": job["name"],
                "process_alive": running,
                "phase": status.get("phase"),
                "observed_unix": time.time(),
            }
            if job["name"] == "v2_bc_health":
                health_path = status_path.parent / "loss_health.json"
                health = json.loads(health_path.read_text()) if health_path.exists() else {}
                report.update(
                    optimizer_steps=health.get("optimizer_steps_from_log"),
                    rolling_32_loss=health.get("rolling_32_optimizer_batch_loss"),
                    completed_epochs=health.get("completed_epochs"),
                    loss_status=health.get("status"),
                    warning_reasons=health.get("warning_reasons", []),
                    last_epoch=health.get("epochs", [])[-1] if health.get("epochs") else None,
                    drive_checkpoint=health.get("latest_drive_checkpoint"),
                )
            print(json.dumps(report), flush=True)
            if status.get("phase") == "failed":
                raise RuntimeError(f"{job['name']} failed; runtime retained for diagnosis")
            if not running and status.get("phase") != "completed":
                raise RuntimeError(f"{job['name']} exited before reporting completion")
            active |= running
        if not active:
            print(
                "V2 supervised jobs completed; runtime retained for the next training phase",
                flush=True,
            )
            return
        time.sleep(20)
    raise TimeoutError("v2 supervision window ended; inspect verified Drive checkpoints")


if __name__ == "__main__":
    main()
