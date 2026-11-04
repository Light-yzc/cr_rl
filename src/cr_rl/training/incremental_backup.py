"""Opt-in immutable rollout publication with local verification receipts.

Only newly published files are rehashed every generation. Historical data is
trusted from receipts until a periodic full audit or a full restore validation.
No background serialization or mutation of live checkpoint tensors is involved.
"""

import json
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256, sync_logs


def signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def publish_incremental(local: Path, destination: Path, update: int, audit_every=16):
    if update < 1 or audit_every < 1:
        raise ValueError("Positive generation and audit interval required")
    receipt_path = local / ".publication_receipts.json"
    prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    identity = str(destination.resolve())
    if prior and prior["destination"] != identity:
        raise ValueError("Publication receipt belongs to a different backup directory")
    if prior and update != prior["update"] + 1:
        raise ValueError(
            "Incremental publication must advance one generation; audit recovery first"
        )
    receipts = dict(prior["receipts"]) if prior else {}

    def immutable(source, relative):
        previous = receipts.get(relative)
        current = signature(source)
        if previous:
            if previous["source_signature"] != current:
                # Metadata changes alone are allowed only after exact content verification.
                if file_sha256(source) != previous["record"]["sha256"]:
                    raise ValueError(f"Immutable published source changed: {relative}")
                previous = dict(previous, source_signature=current)
                receipts[relative] = previous
            return previous["record"]
        record = dict(copy_verified(source, destination / relative), path=relative)
        if signature(source) != current:
            raise ValueError(f"Source changed during immutable publication: {relative}")
        receipts[relative] = dict(source_signature=current, record=record)
        return record

    pool = json.loads((local / "opponent_pool.json").read_text())
    files = []
    for entry in pool["entries"]:
        source = Path(entry["checkpoint"])
        source = source if source.is_absolute() else local / source
        relative = "opponents/" + source.name
        files.append(immutable(source, relative))
        entry["checkpoint"] = "../" + relative
    if prior is None:
        # One initial discovery, including history restored before this mode was enabled.
        rollouts = sorted(p for p in (local / "rollouts").iterdir() if p.is_file())
    else:
        rollouts = [
            local / "rollouts" / f"rollout_{update:06d}.npz",
            local / "rollouts" / f"rollout_{update:06d}.npz.json",
        ]
        if not all(p.is_file() for p in rollouts):
            raise ValueError("Both new rollout and provenance metadata must exist")
    for source in rollouts:
        immutable(source, "rollouts/" + source.name)
    files.extend(
        receipts[name]["record"] for name in sorted(receipts) if name.startswith("rollouts/")
    )
    audited = update % audit_every == 0
    if audited:
        for row in files:
            path = destination / row["path"]
            if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
                raise OSError(f"Historical backup audit failed: {row['path']}")
    checkpoint = f"checkpoints/update_{update:06d}.pt"
    files.append(
        dict(copy_verified(local / "ppo_latest.pt", destination / checkpoint), path=checkpoint)
    )
    if (local / "actor_export.pt").exists():
        export = f"exports/update_{update:06d}.actor.pt"
        files.append(
            dict(copy_verified(local / "actor_export.pt", destination / export), path=export)
        )
    pool_path = f"checkpoints/update_{update:06d}.pool.json"
    atomic_json(destination / pool_path, pool)
    sync_logs(local, destination)
    manifest = dict(
        update=update,
        checkpoint=checkpoint,
        pool=pool_path,
        pool_sha256=file_sha256(destination / pool_path),
        files=files,
        resume_semantics="continuous_simulator_between_updates_reset_on_process_restart",
        publication=dict(
            mode="incremental_immutable_receipts",
            audit_every=audit_every,
            historical_full_audit=audited,
        ),
    )
    atomic_json(destination / f"checkpoints/update_{update:06d}.manifest.json", manifest)
    atomic_json(destination / "latest.json", manifest)
    # A stale receipt after a crash fails closed on the next generation.
    atomic_json(receipt_path, dict(destination=identity, update=update, receipts=receipts))
    return manifest
