"""Start v2 recovery supervision in the current Colab runtime."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/cr-rl-lab")
RUN = ROOT / "artifacts" / "121m_v2_bc_20261103"
PID = 30046


def main():
    stat = Path(f"/proc/{PID}/stat")
    if not stat.exists() or stat.read_text().split()[2] == "Z":
        raise RuntimeError(f"BC training worker {PID} is no longer alive")
    monitor = RUN / "monitor"
    monitor.mkdir(parents=True, exist_ok=True)
    with (monitor / "recovery_launcher.log").open("a") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "scripts/recover_v2_bc_supervision.py",
                "--run-name",
                "121m_v2_bc_20261103",
                "--training-pid",
                str(PID),
                "--ppo-updates",
                "6",
                "--handoff-suffix",
                "_handoff_recovered",
                "--interval",
                "20",
                "--hours",
                "12",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps({"recovery_pid": proc.pid, "training_pid": PID, "status": "started"}))


if __name__ == "__main__":
    main()
