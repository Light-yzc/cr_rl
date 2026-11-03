"""Run in a Colab kernel cell to supervise actual work, not simulated user activity."""

from __future__ import annotations

import json
import time
from pathlib import Path

run = Path("/content/crrl-live/artifacts/119m_20261103_pilot01")
deadline = time.monotonic() + 3 * 3600
last_print = 0
while time.monotonic() < deadline:
    status = json.loads((run / "job_status.json").read_text())
    phase = status["phase"]
    if time.monotonic() - last_print >= 60:
        progress = status.get("last_event", {})
        print(
            json.dumps(
                {
                    "phase": phase,
                    "active_update": status.get("active_update"),
                    "completed_updates": status["completed_updates"],
                    "drive_backed_up_update": status["backed_up_update"],
                    "rollout_step": progress.get("rollout_step"),
                    "heartbeat_age_seconds": time.time()
                    - status.get("heartbeat_unix", time.time()),
                }
            ),
            flush=True,
        )
        last_print = time.monotonic()
    if phase in ("completed", "failed"):
        print(json.dumps(status), flush=True)
        break
    # Backup hashing can legitimately take a few minutes; flag a real stall, not silence.
    heartbeat = status.get("heartbeat_unix", time.time())
    if time.time() - heartbeat > 1800:
        raise RuntimeError("Training supervisor heartbeat stale for 30 minutes")
    time.sleep(10)
else:
    raise TimeoutError("Training monitor exceeded its three-hour supervision budget")
