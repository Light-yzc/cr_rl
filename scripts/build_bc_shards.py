"""Reconstruct bounded BC shards using existing event tables, without deduplication."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--il-parquet", type=Path, required=True)
    parser.add_argument("--cochon-parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drive", type=Path, required=True)
    parser.add_argument("--target", type=int, default=0, help="0 scans all supplied candidates")
    parser.add_argument(
        "--strict-subset",
        action="store_true",
        help="Reproduce the earlier ranked/level-16/high-coverage subset",
    )
    parser.add_argument("--shard-replays", type=int, default=16)
    parser.add_argument("--max-omissions", type=int, default=0)
    parser.add_argument("--infer-opening-cycle", action="store_true")
    args = parser.parse_args()
    if args.target < 0:
        parser.error("target cannot be negative")
    if args.max_omissions < 0:
        parser.error("omission budget cannot be negative")
    target = args.target or sys.maxsize
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [
        (
            "Cochon123/clash-royale-replays",
            args.cochon_parquet,
            "10cca44e9d863d06cf078f142d3e744f78698b0a",
            0,
            min(128, target // 4) if args.strict_subset else target,
        ),
        (
            "VanguardX101/IL_Replay",
            args.il_parquet,
            "059d43a02138a34b1b3009cc2acc7630fb99a638",
            16 if args.strict_subset else 0,
            target,
        ),
    ]
    state_path = args.out / "build_status.json"
    status = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "phase": "building",
            "replays": 0,
            "rows": 0,
            "valid_plays": 0,
            "source_counts": {},
            "cursors": {},
            "shards": [],
            "started_unix": time.time(),
            "selection_policy": "strict" if args.strict_subset else "broad",
            "max_omissions": args.max_omissions,
            "omission_loop_version": 2,
            "infer_opening_cycle": args.infer_opening_cycle,
        }
    )
    policy = "strict" if args.strict_subset else "broad"
    if status.get("selection_policy", "strict") != policy:
        raise ValueError("Use a new output directory when changing the selection policy")
    if status.get("max_omissions", 0) != args.max_omissions:
        raise ValueError("Use a new output directory when changing the omission budget")
    if args.max_omissions and status.get("omission_loop_version") != 2:
        raise ValueError("Rebuild omission-budget data into a new versioned directory")
    if status.get("infer_opening_cycle", False) != args.infer_opening_cycle:
        raise ValueError("Use a new output directory when changing opening-cycle inference")
    metadata_path = args.out / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None

    def publish():
        atomic_json(state_path, status)
        copy_verified(state_path, args.drive / state_path.name)
        if metadata is None:
            return
        atomic_json(metadata_path, metadata)
        copy_verified(metadata_path, args.drive / metadata_path.name)
        manifest = {
            "format": "crrl-bc-shards-v1",
            "metadata": metadata_path.name,
            "metadata_sha256": file_sha256(metadata_path),
            "shards": status["shards"],
            "rows": status["rows"],
            "replays": status["replays"],
            "additional_deduplication": False,
        }
        manifest_path = args.out / "dataset.manifest.json"
        atomic_json(manifest_path, manifest)
        copy_verified(manifest_path, args.drive / manifest_path.name)

    try:
        status["phase"] = "building"
        publish()
        for source, parquet, revision, level, quota in sources:
            copy_verified(parquet, args.drive / "sources" / (parquet.stem + ".parquet"))
            start = status["cursors"].get(source, 0)
            count = status["source_counts"].get(source, 0)
            while status["replays"] < target and count < quota:
                if shutil.disk_usage(args.out).free < 6 * 1024**3:
                    raise RuntimeError("Less than 6 GiB free; cloud-backed shards remain resumable")
                index = len(status["shards"])
                output = args.out / "shards" / f"shard_{index:04d}"
                output.mkdir(parents=True, exist_ok=True)
                limit = min(args.shard_replays, quota - count, target - status["replays"])
                command = [
                    sys.executable,
                    "scripts/import_il_replay.py",
                    str(parquet),
                    "--out",
                    str(output),
                    "--limit",
                    str(limit),
                    "--start-row",
                    str(start),
                    "--source-revision",
                    revision,
                    "--source-level",
                    str(level),
                    "--max-per-deck",
                    "1000000",
                    "--max-omissions",
                    str(args.max_omissions),
                ]
                if args.strict_subset:
                    command += [
                        "--ranked-only",
                        "--min-coverage",
                        "0.9",
                        "--min-plays",
                        "20",
                        "--reject-collisions",
                    ]
                if args.infer_opening_cycle:
                    command.append("--infer-opening-cycle")
                with (output / "import.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                cursor_path = output / "cursor.json"
                if not cursor_path.exists():
                    raise RuntimeError((output / "import.log").read_text()[-4000:])
                cursor = json.loads(cursor_path.read_text())
                start = cursor["next_row"]
                status["cursors"][source] = start
                if not cursor["accepted"]:
                    break
                if result.returncode:
                    raise RuntimeError((output / "import.log").read_text()[-4000:])
                shard_meta = json.loads((output / "human_bc_reconstructed.json").read_text())
                data_path = output / "human_bc_reconstructed.npz"
                with np.load(data_path) as data:
                    a, m, f = data["actions"], data["masks"], data["flags"]
                    valid = (f == 0) & m[np.arange(len(a)), a].astype(bool)
                    status["valid_plays"] += int((valid & (a > 0) & (a < 2305)).sum())
                if metadata is None:
                    metadata = {
                        **shard_meta,
                        "source": [s[0] for s in sources],
                        "replays": [],
                        "label_totals": {},
                        "additional_deduplication": False,
                        "source_files": [
                            {"source": s[0], "revision": s[2], "sha256": file_sha256(s[1])}
                            for s in sources
                        ],
                    }
                for key in ("schema", "vocabulary", "backend_commit", "content_sha", "level"):
                    if metadata[key] != shard_meta[key]:
                        raise ValueError(f"Incompatible shard {key}")
                relative = data_path.relative_to(args.out)
                record = copy_verified(data_path, args.drive / relative)
                record.update(
                    path=relative.as_posix(),
                    replay_offset=status["replays"],
                    replays=len(shard_meta["replays"]),
                    rows=shard_meta["rows"],
                    source=source,
                )
                copy_verified(
                    output / "human_bc_reconstructed.json",
                    args.drive / relative.with_suffix(".json"),
                )
                for name in ("import_stats.json", "cursor.json", "import.log"):
                    copy_verified(output / name, args.drive / relative.parent / name)
                native = output / "reconstructed_replays.tar.gz"
                with tarfile.open(native, "w:gz") as archive:
                    archive.add(output / "reconstructed_replays", arcname="replays")
                copy_verified(native, args.drive / relative.parent / native.name)
                status["shards"].append(record)
                metadata["replays"].extend(shard_meta["replays"])
                for key, value in shard_meta["label_totals"].items():
                    metadata["label_totals"][key] = metadata["label_totals"].get(key, 0) + value
                status["rows"] += shard_meta["rows"]
                status["replays"] += len(shard_meta["replays"])
                metadata["rows"] = status["rows"]
                count += len(shard_meta["replays"])
                status["source_counts"][source] = count
                elapsed = time.time() - status["started_unix"]
                status["eta_seconds"] = (
                    elapsed / status["replays"] * (target - status["replays"])
                    if args.target
                    else None
                )
                publish()
                print(
                    json.dumps(
                        {
                            k: status[k]
                            for k in ("replays", "valid_plays", "source_counts", "eta_seconds")
                        }
                    ),
                    flush=True,
                )
                if start >= cursor["total_rows"]:
                    break
        status["phase"] = "completed"
        publish()
    except Exception:
        status["phase"] = "failed"
        publish()
        raise


if __name__ == "__main__":
    main()
