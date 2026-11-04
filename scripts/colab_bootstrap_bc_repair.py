import importlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

root = Path("/content/cr-rl-lab")
root.mkdir(exist_ok=True)
with tarfile.open("/content/crrl_l4_live_source.tar.gz") as f:
    f.extractall(root, filter="data")
status = root / "live_bootstrap_status.json"


def publish(phase, **details):
    row = {"phase": phase, "kernel_pid": os.getpid(), "observed_unix": time.time(), **details}
    tmp = status.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.replace(status)
    print(json.dumps(row), flush=True)


def run(command, phase, timeout=1200):
    with (root / "live_setup.log").open("a") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        began = time.monotonic()
        while child.poll() is None:
            publish(phase, child_pid=child.pid)
            if time.monotonic() - began > timeout:
                child.terminate()
                child.wait(timeout=30)
                raise TimeoutError(phase)
            time.sleep(15)
        if child.returncode:
            raise RuntimeError(f"{phase} failed; see live_setup.log")


try:
    publish("mounting_drive")
    if not Path("/content/drive/MyDrive/cr_rl/runs").is_dir():
        from google.colab import drive

        old = os.environ.get("TBE_CREDS_ADDR")
        os.environ["TBE_CREDS_ADDR"] = os.environ["TBE_EPHEM_CREDS_ADDR"]
        try:
            drive._mount("/content/drive", timeout_ms=300000, ephemeral=False)
        finally:
            if old is None:
                os.environ.pop("TBE_CREDS_ADDR", None)
            else:
                os.environ["TBE_CREDS_ADDR"] = old
    assert Path("/content/drive/MyDrive/cr_rl/runs").is_dir(), "Drive not mounted"
    publish("drive_ready")
    run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(root), "pytest", "ruff"],
        "installing_dependencies",
    )
    java = Path("/usr/lib/jvm/java-21-openjdk-amd64")
    if not java.exists():
        run(["apt-get", "update", "-qq"], "updating_package_index")
        run(["apt-get", "install", "-y", "-qq", "openjdk-21-jdk-headless"], "installing_java")
    if java.exists():
        os.environ["JAVA_HOME"] = str(java)
    run([sys.executable, "scripts/setup_simulator.py"], "building_simulator")
    (root / "verification_setup.json").write_text(
        json.dumps({"phase": "completed", "observed_unix": time.time()})
    )
    publish("testing_source")
    tested = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (root / "verification_tests.json").write_text(
        json.dumps({"exit": tested.returncode, "stdout": tested.stdout, "stderr": tested.stderr})
    )
    print(tested.stdout[-1500:], flush=True)
    if tested.returncode:
        raise RuntimeError("Source tests failed")
    importlib.invalidate_caches()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from cr_rl.training.persistence import copy_verified

    backup = Path("/content/drive/MyDrive/cr_rl/runs/121m_bc_repair_20261104")
    copy_verified(status, backup / "live_bootstrap_status.json")
    copy_verified(root / "verification_tests.json", backup / "live_source_tests.json")
    publish("starting_supervised_trial")
    from colab_resume_bc_repair import main

    main()
    publish("supervised_trial_completed")
except Exception as error:
    publish("bootstrap_or_supervision_failed", error_type=type(error).__name__)
    raise
