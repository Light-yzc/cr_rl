"""Prepare an A-tier-only BC snapshot and a bounded single-GPU train command.

This command never starts training.  It audits an immutable sharded dataset,
excludes B/C replay identities through the existing metadata mechanism, copies
the shard files into a new snapshot, and writes a reviewable ``train_command``.
The source dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from cr_rl.data.bc_provenance import require_dataset_kind
from cr_rl.training.persistence import atomic_json, file_sha256

try:
    from scripts.audit_bc_coverage import audit_dataset
except ModuleNotFoundError:  # direct ``python scripts/prepare_clean_bc_run.py``
    from audit_bc_coverage import audit_dataset


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != file_sha256(source):
            raise ValueError(f"Snapshot target differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_snapshot(
    dataset: Path,
    out: Path,
    *,
    min_a_games: int,
    thresholds: dict,
    allow_repaired: bool = False,
):
    source_manifest_path = dataset / "dataset.manifest.json" if dataset.is_dir() else dataset
    source_root = source_manifest_path.parent
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_metadata_path = source_root / source_manifest["metadata"]
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if file_sha256(source_metadata_path) != source_manifest["metadata_sha256"]:
        raise ValueError("BC source metadata checksum mismatch")
    kind = require_dataset_kind(
        metadata, allow_repaired=allow_repaired, manifest=source_manifest
    )
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    summary, records = audit_dataset(
        source_manifest_path,
        verify_shards=True,
        **thresholds,
    )
    active = [row for row in records if not row["excluded"]]
    a_ids = [row["replay_id"] for row in active if row["quality_tier"] == "A"]
    if kind == "repaired":
        # The user can run a bounded repair-BC experiment without pretending the
        # repaired trajectories pass the exact-prefix A-tier gate.  Keep the
        # native legality/history checks, and retain quality tiers as diagnostics.
        selected_ids = [
            row["replay_id"]
            for row in active
            if row["complete_paired_history_grid"]
            and row["valid_card_plays"] + row["valid_abilities"] > 0
            and row["unflagged_illegal_rows"] == 0
        ]
    else:
        selected_ids = a_ids
    if len(selected_ids) < min_a_games:
        raise ValueError(
            f"Only {len(selected_ids)} eligible {kind} replays; need at least {min_a_games}. "
            "Inspect the quality report or scan more source games."
        )

    old_excluded = set(int(value) for value in metadata.get("excluded_replay_ids", []))
    all_ids = set(range(len(metadata["replays"])))
    excluded = sorted((all_ids - set(selected_ids)) | old_excluded)
    metadata.update(
        {
            "excluded_replay_ids": excluded,
            "quality_tier_counts": {
                tier: sum(row["quality_tier"] == tier for row in active)
                for tier in ("A", "B", "C")
            },
            "supervised_replays": len(selected_ids),
            "dataset_kind": kind,
            "clean_snapshot": kind == "exact",
            "experimental_repair_snapshot": kind == "repaired",
            "clean_snapshot_protocol": (
                "A-tier-only; source manifest and shard hashes preserved"
                if kind == "exact"
                else "Explicit repair experiment; paired native histories and legal labels"
            ),
        }
    )

    for shard in source_manifest["shards"]:
        _copy_or_link(source_root / shard["path"], out / shard["path"])
    metadata_path = out / "dataset_metadata.json"
    atomic_json(metadata_path, metadata)
    manifest = {
        **source_manifest,
        "metadata": metadata_path.name,
        "metadata_sha256": file_sha256(metadata_path),
        "additional_deduplication": bool(source_manifest.get("additional_deduplication", False)),
        "quality_tier_counts": metadata["quality_tier_counts"],
        "supervised_replays": len(selected_ids),
        "dataset_kind": kind,
    }
    atomic_json(out / "dataset.manifest.json", manifest)
    atomic_json(
        out / "quality_report.json",
        {
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": file_sha256(source_manifest_path),
            "summary": summary,
            "a_tier_replay_ids": a_ids,
            "selected_replay_ids": selected_ids,
            "dataset_kind": kind,
            "excluded_replay_ids": excluded,
            "records": records,
        },
    )
    return summary, len(selected_ids), len(excluded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", type=Path, help="Registry-backed BC manifest or directory"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="New exact or experimental repair snapshot"
    )
    parser.add_argument("--min-a-games", type=int, default=None)
    parser.add_argument(
        "--allow-repaired",
        action="store_true",
        help="Prepare a separate experimental repair-BC snapshot; default minimum is two games",
    )
    parser.add_argument("--init-from-bc", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.min_a_games is None:
        args.min_a_games = 2 if args.allow_repaired else 128
    if args.min_a_games < 1 or args.epochs < 1:
        parser.error("--min-a-games and --epochs must be positive")
    thresholds = {
        "a_horizon_seconds": 60.0,
        "a_min_labels": 8,
        "b_horizon_seconds": 30.0,
        "b_min_labels": 4,
    }
    summary, selected_games, excluded = prepare_snapshot(
        args.dataset,
        args.out,
        min_a_games=args.min_a_games,
        thresholds=thresholds,
        allow_repaired=args.allow_repaired,
    )
    command = [
        "python",
        "scripts/train_bc.py",
        str(args.out),
        "--model",
        "production",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--all-plays",
        "--stable-split",
        "--ability-fraction",
        "0.1",
        "--tile-label-sigma",
        "1.0",
        "--position-coef",
        "1.0",
        "--bc-objective",
        "conditional",
        "--timing-objective",
        "natural",
        "--no-card-prior-correction",
        "--wait-margin-before",
        "4",
        "--wait-margin-after",
        "2",
        "--natural-validation-samples",
        "4096",
        "--stream-cache-gib",
        "8",
        "--shard-mix-window",
        "64",
        "--batch-size",
        "16",
        "--grad-accumulation",
        "6",
        "--learning-rate",
        "1e-5",
        "--tile-learning-rate",
        "3e-5",
        "--checkpoint-every-steps",
        "128",
        "--validation-every-steps",
        "32",
        "--holdout-decks",
        "configs/decks_eval.yaml",
        "--out",
        str(args.out / "bc.pt"),
    ]
    if args.init_from_bc:
        command += ["--init-from-bc", str(args.init_from_bc)]
    atomic_json(
        args.out / "clean_run.json",
        {
            "phase": "prepared_not_started",
            "a_tier_games": summary["quality_tiers"]["A"],
            "selected_games": selected_games,
            "allow_repaired": args.allow_repaired,
            "excluded_games": excluded,
            "quality_thresholds": thresholds,
            "command": command,
            "source_summary": summary,
        },
    )
    print(
        json.dumps(
            {"snapshot": str(args.out), "selected_games": selected_games, "command": command}
        )
    )


if __name__ == "__main__":
    main()
