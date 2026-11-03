"""Prepare a mounted Colab Linux VM; launch through `colab exec -f` after upload."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

ROOT = Path("/content/crrl-live")


def main():
    status = {"ready": False}
    try:
        mounts = Path("/proc/mounts").read_text()
        if " /content/drive " not in mounts or not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Authorize and mount Google Drive before preparing training")
        probe = Path("/content/drive/MyDrive/cr_rl/.write_probe")
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("drive-write-read-verified")
        assert probe.read_text() == "drive-write-read-verified"
        probe.unlink()
        subprocess.run(
            [
                "bash",
                "-lc",
                "apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
                "apt-get install -y -qq openjdk-21-jdk-headless git",
            ],
            check=True,
        )
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
        os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "-e",
                str(ROOT),
                "pytest",
                "wandb",
                "wandb-workspaces",
            ],
            check=True,
        )
        subprocess.run([sys.executable, "scripts/setup_simulator.py"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "cr_rl.cli", "doctor"], cwd=ROOT, check=True)
        status.update(ready=True, drive_write_read_verified=True)
    except Exception:
        status["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (ROOT / "setup_status.json").write_text(json.dumps(status))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        main()
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open("/content/crrl-live-src.tar.gz") as archive:
            archive.extractall(ROOT, filter="data")
        with (ROOT / "setup.log").open("w") as log:
            child = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/colab_prepare_training.py"), "--worker"],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        print("SETUP_WORKER", child.pid, flush=True)
