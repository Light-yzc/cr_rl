from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

ROOT = Path("/content/cr-rl-lab")
ARCHIVE = Path("/content/cr-rl-lab-src.tar.gz")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Upload the source archive first: {ARCHIVE}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")

    print("=== Colab runtime ===", flush=True)
    run(
        [
            sys.executable,
            "-c",
            "import torch\n"
            "print(torch.__version__, torch.cuda.is_available())\n"
            "print(torch.cuda.get_device_name(0))\n"
            "print(round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), 'GiB')",
        ]
    )
    run(
        [
            "bash",
            "-lc",
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
            "apt-get install -y -qq openjdk-21-jdk-headless git",
        ]
    )
    java_home = "/usr/lib/jvm/java-21-openjdk-amd64"
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:" + os.environ["PATH"]
    run(["java", "-version"])

    run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(ROOT), "pytest", "ruff"])
    run([sys.executable, "scripts/setup_simulator.py"], cwd=ROOT)
    if os.environ.get("CRRL_SETUP_ONLY") == "1":
        return

    print("=== Linux self-play acceptance ===", flush=True)
    try:
        run([sys.executable, "scripts/validate_training.py", "--device", "cuda"], cwd=ROOT)
    finally:
        result_archive = Path("/content/colab_smoke_results.tar.gz")
        results = ROOT / "artifacts/linux_validation"
        if results.exists():
            with tarfile.open(result_archive, "w:gz") as archive:
                archive.add(results, arcname="linux_validation")
            run([sys.executable, "scripts/colab_split_results.py"], cwd=ROOT)
    print("=== Linux training validation complete ===", flush=True)


if __name__ == "__main__":
    if os.environ.get("CRRL_BACKGROUND_WORKER") == "1":
        status = {"success": False}
        try:
            main()
            status["success"] = True
        except Exception:
            status["error"] = traceback.format_exc()
            traceback.print_exc()
        finally:
            Path("/content/crrl_job_status.json").write_text(json.dumps(status))
    else:
        # A detached worker lets the CLI poll completion even when the notebook's
        # execute reply is lost. In particular, CLI exit status alone is insufficient.
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ARCHIVE, "r:gz") as source:
            source.extractall(ROOT, filter="data")
        Path("/content/crrl_job_status.json").unlink(missing_ok=True)
        with Path("/content/crrl_job.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/colab_remote_smoke.py")],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CRRL_BACKGROUND_WORKER": "1"},
                start_new_session=True,
            )
        print(f"Linux validation worker started: {process.pid}", flush=True)
