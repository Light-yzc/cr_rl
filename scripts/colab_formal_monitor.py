"""Colab cell supervision of the real formal job, with backup flush on success."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


def main():
    from google.colab import runtime

    run_name = os.environ.get("CRRL_RUN_NAME", "119m_formal_bc_expanded_20261103")
    pid = int(os.environ.get("CRRL_WORKER_PID", "15295"))
    root = Path("/content/cr-rl-lab/artifacts") / run_name
    status_path = root / "status.json"
    deadline = time.monotonic() + 3 * 3600
    while time.monotonic() < deadline:
        status = json.loads(status_path.read_text())
        proc = Path(f"/proc/{pid}/stat")
        alive = proc.exists() and proc.read_text().split()[2] != "Z"
        print(
            json.dumps(
                {
                    "phase": status["phase"],
                    "worker_alive": alive,
                    "heartbeat_age_seconds": time.time() - status["heartbeat_unix"],
                }
            ),
            flush=True,
        )
        if status["phase"] == "failed":
            print(json.dumps(status), flush=True)
            return  # Preserve the runtime for diagnosis; no fake activity.
        if status["phase"] == "completed" and not alive:
            drive = Path("/content/drive/MyDrive/cr_rl/runs") / run_name
            report = {
                "training_completed": True,
                "flush_requested": True,
                "runtime_release_planned": True,
            }
            destination = drive / "release_status.json"
            with destination.open("w") as stream:
                json.dump(report, stream)
                stream.flush()
                os.fsync(stream.fileno())
            result = subprocess.run(
                [
                    "/opt/google/drive/drive",
                    "--push_changes_and_quit",
                    "--single_process",
                    "--timeout_sec=180",
                ],
                capture_output=True,
                timeout=200,
            )
            report["drive_flush_returncode"] = result.returncode
            (root / "release_status.json").write_text(json.dumps(report))
            print(json.dumps(report), flush=True)
            if result.returncode != 0:
                raise RuntimeError("Drive flush failed; runtime retained for recovery")
            runtime.unassign()
            return
        if not alive:
            raise RuntimeError("Formal worker exited without a completed/failed status")
        if time.time() - status["heartbeat_unix"] > 1800:
            raise RuntimeError("Formal job heartbeat stale for 30 minutes")
        time.sleep(20)
    raise TimeoutError("Formal monitoring exceeded its three-hour supervision budget")


if __name__ == "__main__":
    main()
