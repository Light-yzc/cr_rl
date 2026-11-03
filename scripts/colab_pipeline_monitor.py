"""Keep the kernel occupied by monitoring the actual bounded pipeline task."""

from __future__ import annotations

import json
import time
from pathlib import Path

status_path = Path("/content/crrl-live/artifacts/119m_multideck_bc_20261103/pipeline_status.json")
started = time.monotonic()
while time.monotonic() - started < 3 * 3600:
    if status_path.exists():
        status = json.loads(status_path.read_text())
        print(
            json.dumps(
                {key: status.get(key) for key in ("phase", "heartbeat_unix", "bc_gate", "error")}
            ),
            flush=True,
        )
        if status["phase"] in ("failed", "completed"):
            break
        if time.time() - status["heartbeat_unix"] > 600:
            raise RuntimeError("Pipeline heartbeat stale; inspect actual worker before restarting")
    time.sleep(30)
