"""Launch a detached read-only log bridge and local TensorBoard server."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, default=root / "artifacts/local_dual_v100")
parser.add_argument("--port", type=int, default=6006)
args = parser.parse_args()
run = args.run_dir.resolve()
logdir = run / "tensorboard"
logdir.mkdir(exist_ok=True)
environment = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)
commands = {
    "bridge": [
        sys.executable,
        "-u",
        "-m",
        "cr_rl.monitoring.tensorboard_bridge",
        "--run-dir",
        str(run),
        "--logdir",
        str(logdir),
    ],
    "server": [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--reload_interval",
        "5",
    ],
}
result = {}
for name, command in commands.items():
    pidfile = logdir / f"{name}.pid"
    if pidfile.exists():
        pid = int(pidfile.read_text())
        proc = Path(f"/proc/{pid}/cmdline")
        if proc.exists() and str(logdir).encode() in proc.read_bytes():
            result[name] = {"pid": pid, "already_running": True}
            continue
    with (logdir / f"{name}.log").open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pidfile.write_text(str(process.pid) + "\n")
    result[name] = {"pid": process.pid}
print(json.dumps({"url": f"http://127.0.0.1:{args.port}", "logdir": str(logdir), **result}))
