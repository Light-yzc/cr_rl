"""Verified, generation-based backups of a complete self-play run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def copy_verified(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(source)
    if not destination.exists() or file_sha256(destination) != digest:
        temporary = destination.with_name(destination.name + ".uploading")
        with source.open("rb") as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024**2)
            dst.flush()
            os.fsync(dst.fileno())
        if file_sha256(temporary) != digest:
            raise OSError(f"Backup checksum mismatch: {destination.name}")
        temporary.replace(destination)
    return {"path": destination.name, "sha256": digest, "bytes": source.stat().st_size}


def sync_logs(local: Path, destination: Path) -> None:
    for name in (
        "train.log",
        "metrics.jsonl",
        "progress.jsonl",
        "wandb_run.json",
        "wandb_workspace.json",
        "job_status.json",
        "run_config.json",
        "summary.json",
        "supervisor.log",
        "league.yaml",
    ):
        source = local / name
        if source.exists():
            # Copy a stable prefix of an actively appended log into a local snapshot.
            snapshot = local / ".log_snapshots" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
            copy_verified(snapshot, destination / name)


def publish_update(local: Path, destination: Path, update: int) -> dict:
    """Advance latest.json only after checkpoint, pool and all raw data are verified."""
    pool = json.loads((local / "opponent_pool.json").read_text())
    files = []
    for entry in pool["entries"]:
        source = Path(entry["checkpoint"])
        source = source if source.is_absolute() else local / source
        record = copy_verified(source, destination / "opponents" / source.name)
        record["path"] = "opponents/" + source.name
        files.append(record)
        entry["checkpoint"] = "../opponents/" + source.name
    for source in sorted((local / "rollouts").glob("*")):
        if source.is_file():
            record = copy_verified(source, destination / "rollouts" / source.name)
            record["path"] = "rollouts/" + source.name
            files.append(record)
    checkpoint = f"checkpoints/update_{update:06d}.pt"
    record = copy_verified(local / "ppo_latest.pt", destination / checkpoint)
    record["path"] = checkpoint
    files.append(record)
    pool_path = f"checkpoints/update_{update:06d}.pool.json"
    if (local / "actor_export.pt").exists():
        export = f"exports/update_{update:06d}.actor.pt"
        record = copy_verified(local / "actor_export.pt", destination / export)
        record["path"] = export
        files.append(record)
    atomic_json(destination / pool_path, pool)
    sync_logs(local, destination)
    summary_path = local / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    manifest = {
        "update": update,
        "checkpoint": checkpoint,
        "pool": pool_path,
        "pool_sha256": file_sha256(destination / pool_path),
        "files": files,
        "resume_semantics": summary.get(
            "resume_semantics", "model_optimizer_rng_restored_simulator_episodes_reset"
        ),
    }
    atomic_json(destination / f"checkpoints/update_{update:06d}.manifest.json", manifest)
    atomic_json(destination / "latest.json", manifest)
    return manifest


def restore_run(source: Path, destination: Path) -> dict:
    manifest = json.loads((source / "latest.json").read_text())
    if file_sha256(source / manifest["pool"]) != manifest["pool_sha256"]:
        raise OSError("Backup opponent pool checksum mismatch")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    for record in manifest["files"]:
        path = source / record["path"]
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise OSError(f"Backup integrity check failed: {record['path']}")
        if record["path"] == manifest["checkpoint"]:
            target = destination / "ppo_latest.pt"
        elif record["path"].startswith("opponents/"):
            target = destination / path.name
        else:
            target = destination / record["path"]
        copy_verified(path, target)
    pool = json.loads((source / manifest["pool"]).read_text())
    for entry in pool["entries"]:
        entry["checkpoint"] = Path(entry["checkpoint"]).name
    atomic_json(destination / "opponent_pool.json", pool)
    for name in (
        "metrics.jsonl",
        "progress.jsonl",
        "wandb_run.json",
        "run_config.json",
        "league.yaml",
    ):
        if (source / name).exists():
            copy_verified(source / name, destination / name)
    return manifest
