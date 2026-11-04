"""Release this Colab VM only after its bounded trial ends and backups verify."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return bool(pid) and path.exists() and path.read_text().split()[2] != "Z"


def verified_results(run, backup, status):
    """Success requires both registered endpoints; failure preserves local checkpoints."""
    if status["phase"] not in ("completed", "failed"):
        raise ValueError("Trial is not finished")
    if alive(status.get("pid")) or alive(status.get("child_pid")):
        raise ValueError("Trial processes are still active")
    protocol = json.loads((run / "protocol.json").read_text())
    target = protocol["parent"]["optimizer_steps"] + protocol["new_updates_per_arm"]
    verified = {}
    for name in protocol["arms"]:
        arm = run / name
        for path in arm.glob("*"):
            if path.suffix in (".json", ".jsonl", ".log"):
                copy_verified(path, backup / name / path.name)
        if status["phase"] == "failed":
            checkpoint = arm / "bc.pt"
            if checkpoint.exists():
                digest = file_sha256(checkpoint)
                verified[name] = copy_verified(
                    checkpoint, backup / name / "recovery" / f"{digest}.pt"
                )
            continue
        record = json.loads((backup / name / "bc/latest.json").read_text())
        if record["optimizer_steps"] != target or Path(record["path"]).name != record["path"]:
            raise ValueError("Backup does not match the registered final step")
        checkpoint = backup / name / "bc" / record["path"]
        if (
            checkpoint.stat().st_size != record["bytes"]
            or file_sha256(checkpoint) != record["sha256"]
        ):
            raise ValueError("Final Drive checkpoint checksum mismatch")
        verified[name] = record
    if status["phase"] == "completed":
        if file_sha256(run / "comparison.json") != file_sha256(backup / "comparison.json"):
            raise ValueError("Comparison report not backed up")
        if (run / "evaluation_request.json").exists():
            marker = Path("evaluation/completed.json")
            if file_sha256(run / marker) != file_sha256(backup / marker):
                raise ValueError("Requested evaluation is incomplete or not backed up")
    for path in (run / "evaluation").rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
            copy_verified(path, backup / path.relative_to(run))
    copy_verified(run / "status.json", backup / "status.json")
    return verified


def release_runtime():
    # Same backend operation as the installed google.colab.runtime.unassign,
    # without its subsequent browser-JavaScript call (this worker has no UI).
    import httplib2

    address = os.environ.get("TBE_RUNTIME_ADDR")
    if not address:
        raise RuntimeError("Colab runtime management address is unavailable")
    response, _ = httplib2.Http(timeout=30).request(f"http://{address}/unassign", "POST")
    if response.status != 200:
        raise RuntimeError(f"Runtime release returned HTTP {response.status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TBE_RUNTIME_ADDR"):
        raise RuntimeError("Refusing to arm without a runtime release endpoint")
    marker = args.run / "release_guard.json"
    if marker.exists() and alive(json.loads(marker.read_text()).get("pid")):
        raise RuntimeError("Release guard already active")

    def publish(phase, **details):
        record = dict(phase=phase, pid=os.getpid(), observed_unix=time.time(), **details)
        atomic_json(marker, record)
        copy_verified(marker, args.backup / marker.name)

    publish("armed", policy="Wait for bounded trial completion/failure and verified backups")
    while True:
        try:
            status = json.loads((args.run / "status.json").read_text())
            if (
                status["phase"] not in ("completed", "failed")
                or alive(status.get("pid"))
                or alive(status.get("child_pid"))
            ):
                time.sleep(45)
                continue
            verified = verified_results(args.run, args.backup, status)
            publish("backups_verified", trial_phase=status["phase"], checkpoints=verified)
            time.sleep(15)
            publish("requesting_release", trial_phase=status["phase"], checkpoints=verified)
            release_runtime()
            return
        except Exception:
            publish("waiting_after_error", error=traceback.format_exc())
            time.sleep(60)


if __name__ == "__main__":
    main()
