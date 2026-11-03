"""Supervise current BC and cloud-backed data expansion in a real Colab cell."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def main():
    from google.colab import runtime

    jobs = json.loads(Path("/content/broad_bc_worker.json").read_text())
    bc = Path(jobs["bc_status"])
    data = Path(jobs["data_status"])
    destination = Path(jobs["drive"]) / "monitor_status.json"
    deadline = time.monotonic() + float(os.environ.get("CRRL_SUPERVISION_HOURS", "12")) * 3600
    while time.monotonic() < deadline:
        training = json.loads(bc.read_text())
        switch_path = Path("/content/all_bc_switch.json")
        switch = json.loads(switch_path.read_text()) if switch_path.exists() else {}
        retired = switch.get("old_training_retired", False)
        if retired:
            training["phase"] = "retired_for_full_bc"
        building = json.loads(data.read_text()) if data.exists() else {"phase": "starting"}
        queue_path = Path(jobs["queue_status"]) if jobs.get("queue_status") else None
        queue = json.loads(queue_path.read_text()) if queue_path and queue_path.exists() else None
        report = {
            "bc_phase": training["phase"],
            "data_phase": building["phase"],
            "replays": building.get("replays", 0),
            "data_worker_alive": alive(jobs["data_pid"]),
            "heartbeat_unix": time.time(),
            "all_bc_phase": queue["phase"] if queue else None,
        }
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report))
        temporary.replace(destination)
        print(json.dumps(report), flush=True)
        if (
            training["phase"] == "failed"
            or building["phase"] == "failed"
            or (queue and queue["phase"] == "failed")
        ):
            raise RuntimeError("A supervised job failed; runtime retained for diagnosis")
        bc_done = (training["phase"] == "completed" or retired) and not alive(jobs["bc_pid"])
        data_done = building["phase"] == "completed" and not alive(jobs["data_pid"])
        queue_done = not jobs.get("queue_pid") or (
            queue and queue["phase"] == "completed" and not alive(jobs["queue_pid"])
        )
        if bc_done and data_done and queue_done:
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
            if result.returncode:
                raise RuntimeError("Drive flush failed; runtime retained for recovery")
            runtime.unassign()
            return
        if not report["data_worker_alive"] and not data_done:
            raise RuntimeError("Data worker exited before completion; verified shards retained")
        if jobs.get("queue_pid") and not alive(jobs["queue_pid"]) and not queue_done:
            raise RuntimeError("All-data BC worker exited before completion; runtime retained")
        time.sleep(20)
    # The data worker was launched as its own process group, including its importer children.
    if alive(jobs["data_pid"]):
        os.killpg(jobs["data_pid"], signal.SIGTERM)
    if jobs.get("queue_pid") and alive(jobs["queue_pid"]):
        os.killpg(jobs["queue_pid"], signal.SIGTERM)
    report.update(data_phase="time_budget_reached", verified_shards_retained=True)
    destination.write_text(json.dumps(report))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
