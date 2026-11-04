"""Prepare one already allocated L4 for the registered frozen seed replication."""

import os
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path

root = Path("/content/cr-rl-lab")
root.mkdir(exist_ok=True)
with tarfile.open("/content/crrl_replication_source.tar.gz") as archive:
    archive.extractall(root, filter="data")
os.chdir(root)
backup = Path("/content/drive/MyDrive/cr_rl/runs/121m_ppo_seed_replication_20261104")
if not backup.parent.is_dir():
    from google.colab import drive

    previous = os.environ.get("TBE_CREDS_ADDR")
    os.environ["TBE_CREDS_ADDR"] = os.environ["TBE_EPHEM_CREDS_ADDR"]
    try:
        drive._mount("/content/drive", timeout_ms=300000, ephemeral=False)
    finally:
        if previous is None:
            os.environ.pop("TBE_CREDS_ADDR", None)
        else:
            os.environ["TBE_CREDS_ADDR"] = previous
backup.mkdir(exist_ok=True)
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
java = Path("/usr/lib/jvm/java-21-openjdk-amd64")
commands = [[sys.executable, "-m", "pip", "install", "-q", "-e", str(root), "pytest"]]
if not java.exists():
    commands += [
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "-qq", "openjdk-21-jdk-headless"],
    ]
os.environ["JAVA_HOME"] = str(java)
commands += [[sys.executable, "scripts/setup_simulator.py"], [sys.executable, "-m", "pytest", "-q"]]
for index, command in enumerate(commands):
    log_path = backup / f"bootstrap_{index:02d}.log"
    print(f"Bootstrap step {index + 1}/{len(commands)}", flush=True)
    with log_path.open("a") as log:
        result = subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=1800
        )
    if result.returncode:
        raise RuntimeError(f"Bootstrap step failed; inspect {log_path.name}")
sys.path.insert(0, str(root / "src"))
runpy.run_path(str(root / "scripts/colab_seed_replication.py"), run_name="__main__")
