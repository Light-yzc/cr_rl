"""Replace the live BC health monitor without touching the training worker."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/cr-rl-lab")
RUN_NAME = "121m_v2_bc_20261103"
MONITOR_ROOT = ROOT / "artifacts" / RUN_NAME / "monitor"


def save_handles(path: Path, handles: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(handles, indent=2))
    temporary.replace(path)


def main():
    handles_path = Path("/content/v2_job_handles.json")
    handles = json.loads(handles_path.read_text())
    old = [job for job in handles if job.get("name") == "v2_bc_health"]
    handles = [job for job in handles if job.get("name") != "v2_bc_health"]
    save_handles(handles_path, handles)
    for job in old:
        try:
            os.kill(int(job["pid"]), signal.SIGTERM)
        except (KeyError, ProcessLookupError, PermissionError, ValueError):
            pass

    log_path = MONITOR_ROOT / "health_monitor_revised.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "scripts/monitor_v2_bc.py",
                "--run-name",
                RUN_NAME,
                "--interval",
                "60",
                "--hours",
                "12",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    handles.append(
        {
            "name": "v2_bc_health",
            "pid": process.pid,
            "status": str(MONITOR_ROOT / "health_monitor_status.json"),
        }
    )
    save_handles(handles_path, handles)
    print(
        json.dumps(
            {"old_monitor_pids": [job.get("pid") for job in old], "new_monitor_pid": process.pid}
        )
    )


if __name__ == "__main__":
    main()
