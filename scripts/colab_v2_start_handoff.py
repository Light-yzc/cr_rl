"""Restart the recovered handoff using W&B credentials inherited by the BC worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/content/cr-rl-lab")
RUN_NAME = "121m_v2_bc_20261103"
TRAINING_PID = 30046
SUFFIX = "_handoff_recovered"


def main():
    raw_environment = Path(f"/proc/{TRAINING_PID}/environ").read_bytes().split(b"\0")
    token = next(
        (item.split(b"=", 1)[1] for item in raw_environment if item.startswith(b"WANDB_API_KEY=")),
        None,
    )
    if not token:
        raise RuntimeError("The surviving BC worker has no inherited W&B key; no handoff launched")
    os.environ["WANDB_API_KEY"] = token.decode()

    out = ROOT / "artifacts" / (RUN_NAME + SUFFIX)
    status_path = out / "status.json"
    if status_path.exists():
        raise RuntimeError(f"Recovered handoff already has a status file: {status_path}")
    log_path = ROOT / "artifacts" / RUN_NAME / "monitor" / "handoff_recovered.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "scripts/run_v2_handoff.py",
                "--bc-run",
                RUN_NAME,
                "--ppo-updates",
                "6",
                "--suffix",
                SUFFIX,
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    handles_path = Path("/content/v2_job_handles.json")
    try:
        handles = json.loads(handles_path.read_text())
    except (OSError, json.JSONDecodeError):
        handles = []
    handles = [job for job in handles if job.get("name") != "v2_handoff_recovered"]
    handles.append(
        {
            "name": "v2_handoff_recovered",
            "pid": process.pid,
            "status": str(status_path),
        }
    )
    temporary = handles_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(handles, indent=2))
    temporary.replace(handles_path)
    time.sleep(3)
    try:
        phase = json.loads(status_path.read_text()).get("phase")
    except (OSError, json.JSONDecodeError):
        phase = "starting"
    print(json.dumps({"handoff_pid": process.pid, "phase": phase, "wandb_key_present": True}))


if __name__ == "__main__":
    main()
