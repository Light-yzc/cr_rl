"""Compare historical-rollout I/O using real payloads and tiny common weights.

This isolates backup-history growth; it excludes real model serialization and
does not pretend tiny placeholder weights represent a full PPO publication.
"""

import argparse
import json
import time
from pathlib import Path

import cr_rl.training.incremental_backup as incremental
import cr_rl.training.persistence as persistence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.local.exists() or args.backup.exists():
        raise ValueError("Fresh benchmark directories required")
    results = []
    for mode in ("full", "incremental"):
        local, backup = args.local / mode, args.backup / mode
        (local / "rollouts").mkdir(parents=True)
        (local / "ppo_latest.pt").write_bytes(b"benchmark-placeholder-full-state")
        (local / "opponent.pt").write_bytes(b"benchmark-placeholder-anchor")
        (local / "opponent_pool.json").write_text(
            json.dumps(dict(entries=[dict(policy_id=0, checkpoint="opponent.pt", generation=0)]))
        )
        for generation in range(1, 11):
            for suffix in ("npz", "npz.json"):
                name = f"rollout_{generation:06d}.{suffix}"
                (local / "rollouts" / name).symlink_to(args.source_run / "rollouts" / name)
        publish = persistence.publish_update if mode == "full" else incremental.publish_incremental
        publish(local, backup, 10)
        for generation in (11, 12):
            for suffix in ("npz", "npz.json"):
                (local / "rollouts" / f"rollout_{generation:06d}.{suffix}").symlink_to(
                    args.source_run / "rollouts" / f"rollout_000010.{suffix}"
                )
            counts = {"local_hashed_bytes": 0, "backup_hashed_bytes": 0, "hash_calls": 0}
            original = persistence.file_sha256

            def counted(path, counts=counts, backup=backup, original=original):
                counts["hash_calls"] += 1
                key = "backup_hashed_bytes" if path.is_relative_to(backup) else "local_hashed_bytes"
                counts[key] += path.stat().st_size
                return original(path)

            persistence.file_sha256 = incremental.file_sha256 = counted
            began = time.perf_counter()
            try:
                manifest = publish(local, backup, generation)
            finally:
                persistence.file_sha256 = incremental.file_sha256 = original
            row = dict(
                mode=mode,
                generation=generation,
                seconds=time.perf_counter() - began,
                rollout_files=sum(r["path"].startswith("rollouts/") for r in manifest["files"]),
                **counts,
            )
            results.append(row)
            print(json.dumps(row), flush=True)
    persistence.atomic_json(
        args.out,
        dict(
            results=results,
            limitation=(
                "Real rollout payloads, benchmark-only generations and tiny weights; "
                "not full update speed"
            ),
            source_run=str(args.source_run),
            checkpoint_placeholder_bytes=len(b"benchmark-placeholder-full-state"),
        ),
    )


if __name__ == "__main__":
    main()
