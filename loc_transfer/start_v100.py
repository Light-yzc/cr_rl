"""Start V100 training in a bounded user service, refusing duplicate trainers."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = Path(os.environ.get("RUN_DIR", root / "artifacts/local_dual_v100")).resolve()
out.mkdir(parents=True, exist_ok=True)
pid_path = out / "trainer.pid"
unit = "crrl-v100.service"
if subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0:
    raise SystemExit(f"Trainer service already running: {unit}")
if pid_path.exists():
    pid = int(pid_path.read_text())
    command = Path(f"/proc/{pid}/cmdline")
    if command.exists() and b"local_dual_v100" in command.read_bytes():
        raise SystemExit(f"Trainer already running: PID {pid}")
environment = dict(os.environ, RUN_DIR=str(out))
allowed = (
    "RUN_DIR",
    "PYTHON",
    "JAVA_HOME",
    "GPU_IDS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "ENVS",
    "ENV_THREADS",
    "STEPS",
    "UPDATES",
    "BATCH",
    "ACTOR_BATCH",
    "CRITIC_BATCH",
    "ACCUM",
    "AMP_DTYPE",
    "OPPONENT_CACHE_SIZE",
    "TRAINING_OPPONENT_CACHE_SIZE",
    "CRRL_RESUME_REASON",
)
command = [
    "systemd-run",
    "--user",
    "--unit=" + unit,
    "--collect",
    "--service-type=exec",
    "--working-directory=" + str(root),
    "--property=MemoryHigh=5G",
    "--property=MemoryMax=6G",
    "--property=MemorySwapMax=1G",
    "--property=OOMPolicy=stop",
    "--property=KillMode=control-group",
    "--property=StandardOutput=append:" + str(out / "train.log"),
    "--property=StandardError=append:" + str(out / "train.log"),
]
for key in allowed:
    if key in environment:
        command.append("--setenv=" + key + "=" + environment[key])
command += ["bash", str(root / "loc_transfer/run_v100.sh"), *sys.argv[1:]]
subprocess.run(command, check=True)
for _ in range(20):
    pid = int(
        subprocess.check_output(
            ["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"], text=True
        ).strip()
        or "0"
    )
    if pid:
        break
    time.sleep(0.1)
else:
    raise SystemExit("Trainer service failed to start; inspect journalctl --user -u " + unit)
pid_path.write_text(str(pid) + "\n")
print(
    json.dumps(
        {
            "pid": pid,
            "unit": unit,
            "run_dir": str(out),
            "memory_high_gib": 5,
            "memory_max_gib": 6,
            "swap_max_gib": 1,
            "log": str(out / "train.log"),
        }
    )
)
