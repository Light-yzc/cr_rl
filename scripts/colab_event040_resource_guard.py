"""Release only the idle runtime owned by /content/state_event040.

The worker never stops a trainer or starts a pipeline. Arm it separately after
the task files are installed. Completion releases immediately after task exit;
a persisted failed status permits release after fifteen minutes. Active
training, an owned child, or a held pipeline lock always defers release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("/content/state_event040")
DRIVE = Path("/content/drive/MyDrive/cr_rl/runs/state_global_event040_20261110")
FAILED_GRACE_SECONDS = 15 * 60
TRAINER_NAMES = {"train_state_bc.py", "train_bc.py", "train_selfplay.py"}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def active_processes(root=ROOT, *, proc_root=Path("/proc"), own_pid=None):
    """Exclude this guard and zombies; block on any live trainer or task child."""
    own_pid = own_pid or os.getpid()
    if not proc_root.is_dir():
        raise RuntimeError("Resource guard requires Linux /proc process accounting")
    result = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            # The parenthesized Linux comm field can itself contain spaces.
            state = (entry / "stat").read_text().rsplit(")", 1)[1].strip().split()[0]
            if state in ("Z", "X"):
                continue
            command = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            command = [argument for argument in command if argument]
            if not command:
                continue
            names = {Path(argument).name for argument in command}
            # A second guard has its own flock, and is not useful workload.
            if "colab_event040_resource_guard.py" in names:
                continue
            try:
                cwd = (entry / "cwd").resolve(strict=True)
            except (OSError, RuntimeError):
                cwd = None
            rooted = any(str(root) in argument for argument in command)
            rooted |= cwd is not None and (cwd == root or cwd.is_relative_to(root))
            if names & TRAINER_NAMES or rooted:
                result.append({"pid": int(entry.name), "program": Path(command[0]).name,
                               "trainer": bool(names & TRAINER_NAMES)})
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process exited during the inspection.
        except PermissionError as error:
            raise RuntimeError("Cannot safely inspect all active task processes") from error
    return result


def release_decision(status, *, now, active, pipeline_idle):
    if not status or status.get("phase") not in ("complete", "failed"):
        return False, "task_not_terminal"
    if active:
        return False, "active_processes"
    if not pipeline_idle:
        return False, "pipeline_lock_held"
    timestamp = status.get("time_unix")
    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp > now:
        return False, "invalid_terminal_timestamp"
    if status["phase"] == "failed" and now - timestamp < FAILED_GRACE_SECONDS:
        return False, "failed_grace_period"
    return True, "complete" if status["phase"] == "complete" else "failed_for_15_minutes"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_latest_backup(root=ROOT, drive=DRIVE):
    """Preserve the atomic latest checkpoint without tensor/model allocations."""
    source = root / "training/latest.pt"
    if not source.exists():
        return {"status": "no_local_checkpoint"}
    # torch.save's completed ZIP has a central directory and data.pkl. Stream
    # CRC verification also catches a truncated/partial artifact before release.
    with zipfile.ZipFile(source) as archive:
        if not any(name.endswith("/data.pkl") for name in archive.namelist()):
            raise ValueError("Local latest is not a completed Torch checkpoint archive")
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"Local latest checkpoint CRC failed: {damaged}")
    source_digest = file_sha256(source)
    destination = drive / "training/latest.pt"
    if destination.exists() and file_sha256(destination) == source_digest:
        return {"status": "already_verified", "sha256": source_digest, "bytes": source.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".guard-uploading")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024**2)
        dst.flush()
        os.fsync(dst.fileno())
    if file_sha256(temporary) != source_digest or file_sha256(source) != source_digest:
        raise OSError("Latest checkpoint changed or failed backup verification")
    temporary.replace(destination)
    return {"status": "copied_verified", "sha256": source_digest, "bytes": source.stat().st_size}


def snapshot_small_evidence(root=ROOT, drive=DRIVE):
    """The trainer publishes checkpoints; retain task logs without duplicating weights."""
    report = {"copied": [], "errors": []}
    candidates = [root / "status.json", root / "pipeline_progress.json"]
    candidates.extend(root.glob("*.log"))
    candidates.extend(root.glob("training/*.json"))
    candidates.extend(root.glob("training/*.jsonl"))
    candidates.extend(root.glob("evaluation/*/*.json"))
    for source in candidates:
        if not source.is_file():
            continue
        try:
            if source.stat().st_size > 64 * 1024**2:
                report["errors"].append({"path": str(source.relative_to(root)), "error": "log_exceeds_64MiB"})
                continue
            destination = drive / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".guard-uploading")
            shutil.copy2(source, temporary)
            if file_sha256(source) != file_sha256(temporary):
                raise OSError("Guard log backup checksum differs")
            temporary.replace(destination)
            report["copied"].append(str(source.relative_to(root)))
        except OSError as error:
            report["errors"].append({"path": str(source.relative_to(root)), "error": str(error)})
    return report


def unassign_runtime():
    """Colab backend release without requiring a notebook/browser callback."""
    address = os.environ.get("TBE_RUNTIME_ADDR")
    if not address:
        raise RuntimeError("Missing Colab runtime release endpoint")
    request = urllib.request.Request(f"http://{address}/unassign", data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Runtime release returned HTTP {response.status}")


def main(argv=None):
    global ROOT, DRIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--drive-dir", type=Path, default=DRIVE)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Report a decision without releasing")
    parser.add_argument("--once", action="store_true", help="Inspect one time then exit")
    args = parser.parse_args(argv)
    ROOT, DRIVE = args.root.resolve(), args.drive_dir.resolve()
    if ROOT.parent != Path('/content') or not DRIVE.is_relative_to(Path('/content/drive/MyDrive/cr_rl/runs')):
        parser.error('Guard paths must be a task directory under /content and its private Drive run')
    if not math.isfinite(args.poll_seconds) or not 1 <= args.poll_seconds <= 60:
        parser.error("Poll interval must be between one and sixty seconds")
    if not ROOT.is_dir():
        raise RuntimeError(f"Task root {ROOT} is not installed")
    if not args.dry_run and not os.environ.get("TBE_RUNTIME_ADDR"):
        raise RuntimeError("Refusing to arm outside a Colab runtime")
    import fcntl

    marker = ROOT / "resource_guard.json"
    guard_lock = (ROOT / "resource_guard.lock").open("a")
    fcntl.flock(guard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pipeline_lock = (ROOT / "pipeline.lock").open("a")

    def publish(phase, **details):
        value = {"phase": phase, "guard_pid": os.getpid(), "time_unix": time.time(),
                 "task_root": str(ROOT), **details}
        atomic_json(marker, value)
        try:
            atomic_json(DRIVE / marker.name, value)
        except OSError:
            pass  # A failed Drive mount must not burn idle GPU indefinitely.
        print(json.dumps(value), flush=True)

    publish("armed", failed_grace_seconds=FAILED_GRACE_SECONDS, dry_run=args.dry_run)
    while True:
        held = False
        try:
            status = read_json(ROOT / "status.json")
            active = active_processes(root=ROOT)
            try:
                fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
            except BlockingIOError:
                pass
            eligible, reason = release_decision(status, now=time.time(), active=active, pipeline_idle=held)
            if eligible:
                if args.dry_run:
                    publish("would_release", reason=reason, terminal_status=status)
                    return
                if not os.path.ismount("/content/drive") or not Path("/content/drive/MyDrive").is_dir():
                    raise RuntimeError("Drive FUSE mount is unavailable; latest checkpoint must remain local")
                checkpoint = ensure_latest_backup(root=ROOT, drive=DRIVE)
                evidence = snapshot_small_evidence(root=ROOT, drive=DRIVE)
                # The lock prevents this task from restarting between inspection
                # and release. Recheck status and independent trainer processes.
                current = read_json(ROOT / "status.json")
                active = active_processes(root=ROOT)
                ready, reason = release_decision(current, now=time.time(), active=active, pipeline_idle=held)
                if ready and current == status:
                    publish("requesting_release", reason=reason, terminal_status=current,
                            checkpoint=checkpoint, evidence=evidence)
                    unassign_runtime()
                    publish("release_requested", reason=reason)
                    return
            publish("waiting", reason=reason, active=active,
                    task_phase=status.get("phase") if status else None)
        except Exception as error:
            publish("waiting_after_error", error_type=type(error).__name__, error=str(error))
        finally:
            if held:
                fcntl.flock(pipeline_lock, fcntl.LOCK_UN)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
