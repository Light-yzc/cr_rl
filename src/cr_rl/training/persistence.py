"""Verified, generation-based backups of a complete self-play run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
        return digest.hexdigest()


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
        "history_bootstrap.json",
    ):
        source = local / name
        if source.exists():
            # Copy a stable prefix of an actively appended log into a local snapshot.
            snapshot = local / ".log_snapshots" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
            copy_verified(snapshot, destination / name)


def prune_backup(
    destination: Path,
    update: int,
    recent: int = 2,
    keep_every: int = 64,
    rollout_recent: int = 2,
    max_periodic: int | None = None,
) -> dict:
    """Bound immutable publication storage without removing restore points.

    Full checkpoints and exports are retained for the latest ``recent`` updates
    and periodic ``keep_every`` checkpoints. Rollouts keep only the latest
    ``rollout_recent`` updates. Opponent snapshots remain because retained pool
    manifests may reference older snapshots.
    """
    # Checkpoint cadence can be sparse (e.g. every 800 updates). Retain the
    # latest actual restore files, rather than looking for nonexistent u-1.
    existing = sorted({int(match.group(1)) for path in (destination / "checkpoints").glob("update_*.pt")
                       if (match := re.fullmatch(r"update_(\d+)\.pt", path.name))
                       and int(match.group(1)) <= update})
    keep_updates = {update-i for i in range(recent)}
    if recent:
        keep_updates.update(existing[-recent:])
    periodic = [u for u in existing if u % keep_every == 0] if keep_every else []
    if max_periodic is not None:
        periodic = periodic[-max_periodic:] if max_periodic else []
    keep_updates.update(periodic)
    removed = 0
    removed_bytes = 0
    patterns = (
        (destination / "checkpoints", r"update_(\d+)(?:\.pt|\.pool\.json|\.manifest\.json)$"),
        (destination / "exports", r"update_(\d+)\.actor\.pt$"),
    )
    for root, pattern in patterns:
        if not root.exists():
            continue
        for path in root.iterdir():
            match = re.fullmatch(pattern, path.name)
            if match and int(match.group(1)) not in keep_updates and path.is_file():
                size = path.stat().st_size
                path.unlink()
                removed += 1
                removed_bytes += size
    rollout_floor = max(0, update - rollout_recent + 1)
    rollout_root = destination / "rollouts"
    if rollout_root.exists():
        for path in rollout_root.iterdir():
            match = re.match(r"rollout_(\d+)", path.name)
            if match and int(match.group(1)) < rollout_floor and path.is_file():
                size = path.stat().st_size
                path.unlink()
                removed += 1
                removed_bytes += size
    return {
        "removed_files": removed,
        "removed_bytes": removed_bytes,
        "keep_updates": sorted(keep_updates),
    }


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
    prune_backup(destination, update)
    files = [record for record in files if (destination / record["path"]).exists()]
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


def prune_local_opponents(local: Path, active=()) -> dict:
    """Collect only owned opponents unreferenced by current/retained restore pools.

    The handoff/source archive and any paths outside local/opponents are never touched.
    Malformed reference files abort collection rather than risking a restore dependency.
    """
    local = Path(local)
    references = set()
    pools = [local / "opponent_pool.json", *(local / "checkpoints").glob("*.pool.json")]
    for pool_path in pools:
        pool = json.loads(pool_path.read_text())
        for entry in pool["entries"]:
            source = Path(entry["checkpoint"])
            references.add(
                (source if source.is_absolute() else pool_path.parent / source).resolve()
            )
    for source in active:
        references.add(Path(source).resolve())
    # Frozen eval/publication jobs must also retain their snapshot dependencies.
    for root in (local / "evaluation", local / ".publication_snapshots"):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix == ".pt":
                references.add(path.resolve())
            elif path.is_file() and path.suffix == ".json":

                def collect(value, path=path):
                    if isinstance(value, dict):
                        for child in value.values():
                            collect(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect(child)
                    elif isinstance(value, str) and value.endswith(".pt"):
                        item = Path(value)
                        references.add(
                            (item if item.is_absolute() else path.parent / item).resolve()
                        )

                collect(json.loads(path.read_text()))
    removed = removed_bytes = 0
    reference_names = {path.name for path in references}
    for path in [*(local / "opponents").glob("opponent_*.pt"), *local.glob("opponent_*.pt")]:
        if (
            not path.is_symlink()
            and path.resolve() not in references
            and path.name not in reference_names
        ):
            removed_bytes += path.stat().st_size
            path.unlink()
            removed += 1
    return {"removed_opponents": removed, "removed_opponent_bytes": removed_bytes}
