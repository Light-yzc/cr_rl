"""Execute this file in a Colab kernel cell to resume and supervise the BC trial.

    colab exec --session SESSION --file scripts/colab_resume_bc_repair.py --timeout 15000

Requires the project/simulator installed and Drive already mounted. The trainer
is detached for transport resilience, while this cell supervises its actual work
until completion/failure. Closing a client connection is still not a lifetime
guarantee; platform limits apply. Does not create or provision another runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified
from cr_rl.training.runtime_supervision import process_alive, supervise_trial


def main():
    root = Path("/content/cr-rl-lab")
    run = root / "artifacts/121m_bc_repair_20261104"
    backup = Path("/content/drive/MyDrive/cr_rl/runs/121m_bc_repair_20261104")
    if not Path("/var/colab/hostname").exists() or not backup.is_dir():
        raise RuntimeError("Run in the existing prepared Colab kernel after mounting Drive")
    if not os.environ.get("TBE_RUNTIME_ADDR"):
        raise RuntimeError("Run in the Colab kernel; runtime management endpoint is missing")
    setup = json.loads((root / "verification_setup.json").read_text())
    tests = json.loads((root / "verification_tests.json").read_text())
    if setup["phase"] != "completed" or tests["exit"] != 0:
        raise RuntimeError("Complete simulator setup and tests before launching training")
    run.mkdir(parents=True, exist_ok=True)
    marker = run / "verification_worker.json"
    status_path = run / "status.json"
    worker = json.loads(marker.read_text()) if marker.exists() else {}
    if not process_alive(worker.get("pid")):
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["phase"] == "completed":
                print(json.dumps(status), flush=True)
                return
            if process_alive(status.get("pid")) or process_alive(status.get("child_pid")):
                raise RuntimeError("Untracked trial process is active; refusing duplicate launch")
        copy_verified(backup / "protocol.json", run / "protocol.json")
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
        command = [
            sys.executable,
            "scripts/run_bc_repair_trial.py",
            "--resume-trial",
            "--evaluate-results",
            "--evaluation-pairs",
            "8",
        ]
        with (run / "controller_resume.log").open("a") as log:
            child = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        worker = {"pid": child.pid, "created_unix": time.time(), "command": command}
        atomic_json(marker, worker)
        copy_verified(marker, backup / marker.name)
        # Do not arm the release guard before this controller publishes status.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status.get("pid") == child.pid:
                    break
            if child.poll() is not None:
                raise RuntimeError(
                    "Controller exited during startup; inspect controller_resume.log"
                )
            time.sleep(1)
        else:
            raise TimeoutError("Controller did not publish status; inspect existing process")
    guard_path = run / "release_guard.json"
    guard = json.loads(guard_path.read_text()) if guard_path.exists() else {}
    if not process_alive(guard.get("pid")):
        with (run / "release_guard.log").open("a") as log:
            guard_child = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/finish_and_release.py",
                    "--run",
                    str(run),
                    "--backup",
                    str(backup),
                ],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        worker["guard_pid"] = guard_child.pid
        atomic_json(marker, worker)
        copy_verified(marker, backup / marker.name)
    atomic_json(
        run / "kernel_supervision.json",
        {
            "kernel_pid": os.getpid(),
            "controller_pid": worker["pid"],
            "started_unix": time.time(),
            "budget_seconds": 4 * 3600,
            "purpose": "Actual trial supervision; no simulated input or synthetic GPU work",
        },
    )
    copy_verified(run / "kernel_supervision.json", backup / "kernel_supervision.json")
    supervise_trial(status_path, emit=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
