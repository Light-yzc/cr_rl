"""Launch Linux/CUDA acceptance from Windows, macOS or Linux using Colab CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def download_results(session: list[str], output: Path) -> None:
    def fetch(remote: str, local: Path) -> None:
        for attempt in range(3):
            try:
                subprocess.run(
                    ["colab", "download", *session, remote, str(local)],
                    cwd=ROOT,
                    check=True,
                    timeout=120,
                )
                return
            except (subprocess.SubprocessError, OSError):
                if attempt == 2:
                    raise
                time.sleep(3)

    manifest_path = output / "results_manifest.json"
    fetch("/content/colab_results_manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    combined_path = output / "results.tar.gz"
    # Preserve already verified prefix chunks when recovering an interrupted transfer.
    if combined_path.exists():
        with combined_path.open("rb") as stream:
            for index, part in enumerate(manifest["parts"]):
                data = stream.read(part["size"])
                if hashlib.sha256(data).hexdigest() != part["sha256"]:
                    break
                (output / f"download_part_{index:04d}").write_bytes(data)
    ssh = shutil.which("ssh")
    windows_ssh = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/OpenSSH/ssh.exe"
    if not ssh and windows_ssh.exists():
        ssh = str(windows_ssh)
    proxy_args = ["colab", "ssh", "--proxy-mode", *session]
    proxy = subprocess.list2cmdline(proxy_args) if os.name == "nt" else shlex.join(proxy_args)

    def fetch_part(item: tuple[int, dict]) -> Path:
        index, part = item
        local = output / f"download_part_{index:04d}"
        if not local.exists() or local.stat().st_size != part["size"]:
            if ssh:
                for attempt in range(3):
                    try:
                        received = local.stat().st_size if local.exists() else 0
                        with local.open("ab") as stream:
                            subprocess.run(
                                [
                                    ssh,
                                    "-o",
                                    f"ProxyCommand={proxy}",
                                    "-o",
                                    "BatchMode=yes",
                                    "-o",
                                    "StrictHostKeyChecking=no",
                                    "-o",
                                    f"UserKnownHostsFile={os.devnull}",
                                    "root@colab-runtime",
                                    "dd if=/content/colab_smoke_results.tar.gz "
                                    "bs=1M iflag=skip_bytes,count_bytes "
                                    f"skip={index * 16 * 1024**2 + received} "
                                    f"count={part['size'] - received} status=none",
                                ],
                                stdout=stream,
                                stderr=subprocess.PIPE,
                                check=True,
                                timeout=600,
                            )
                        print(
                            f"Downloaded result chunk {index + 1}/{len(manifest['parts'])}",
                            flush=True,
                        )
                        break
                    except subprocess.SubprocessError:
                        if attempt == 2:
                            raise
                        time.sleep(5)
            else:
                fetch("/content/" + part["name"], local)
        if (
            local.stat().st_size != part["size"]
            or hashlib.sha256(local.read_bytes()).hexdigest() != part["sha256"]
        ):
            raise ValueError(f"Result chunk failed integrity check: {part['name']}")
        return local

    with (
        ThreadPoolExecutor(max_workers=1 if ssh else 4) as executor,
        (output / "results.tar.gz").open("wb") as combined,
    ):
        for local in executor.map(fetch_part, enumerate(manifest["parts"])):
            with local.open("rb") as stream:
                shutil.copyfileobj(stream, combined)
            local.unlink()
    if (output / "results.tar.gz").stat().st_size != manifest["size"]:
        raise ValueError("Result archive size mismatch")
    with tarfile.open(output / "results.tar.gz") as archive:
        archive.extractall(output, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=["T4", "L4"], default="L4")
    parser.add_argument("--session", default="crrl-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = (args.out or ROOT / "artifacts" / args.session).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists() for name in ("results.tar.gz", "source.tar.gz", "job_status.json")
    ):
        parser.error("Use a fresh output directory")
    source = output / "source.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for name in (
            "src",
            "configs",
            "scripts",
            "tests",
            "docs",
            "README.md",
            "pyproject.toml",
            "uv.lock",
        ):
            archive.add(
                ROOT / name,
                arcname=name,
                filter=lambda entry: (
                    None
                    if "__pycache__" in Path(entry.name).parts or ".egg-info" in entry.name
                    else entry
                ),
            )
    session = ["-s", args.session]
    run(["colab", "new", *session, "--gpu", args.gpu])
    try:
        run(["colab", "upload", *session, str(source), "/content/cr-rl-lab-src.tar.gz"])
        try:
            run(
                [
                    "colab",
                    "exec",
                    *session,
                    "--timeout",
                    "60",
                    "-f",
                    "scripts/colab_remote_smoke.py",
                ]
            )
            deadline = time.monotonic() + 3600
            status_path = output / "job_status.json"
            while not status_path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Colab validation exceeded one hour")
                time.sleep(10)
                try:
                    subprocess.run(
                        [
                            "colab",
                            "download",
                            *session,
                            "/content/crrl_job_status.json",
                            str(status_path),
                        ],
                        cwd=ROOT,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=45,
                    )
                except subprocess.TimeoutExpired:
                    continue
            status = json.loads(status_path.read_text())
            if not status.get("success"):
                raise RuntimeError(status.get("error", "Remote validation failed"))
        finally:
            # The remote script archives logs/checkpoints even when validation fails.
            run(
                [
                    "colab",
                    "download",
                    *session,
                    "/content/crrl_job.log",
                    str(output / "colab_job.log"),
                ]
            )
            download_results(session, output)
    finally:
        run(["colab", "stop", *session])
    report = output / "linux_validation/validation.json"
    if not report.exists() or not json.loads(report.read_text()).get("passed"):
        sys.exit(f"Validation did not finish; inspect logs in {output}")
    print(f"Linux/CUDA acceptance passed: {report}")


if __name__ == "__main__":
    main()
