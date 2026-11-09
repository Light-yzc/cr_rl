"""Freeze published replay shards while their producer continues appending data."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

from cr_rl.data.bc_provenance import require_dataset_kind
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def read_published(root):
    manifest_bytes = (root / "dataset.manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    metadata_bytes = (root / manifest["metadata"]).read_bytes()
    if hashlib.sha256(metadata_bytes).hexdigest() != manifest["metadata_sha256"]:
        raise ValueError("Producer is publishing a new manifest; retry the snapshot")
    return manifest, json.loads(metadata_bytes), hashlib.sha256(manifest_bytes).hexdigest()


def freeze_datasets(sources, destination, deduplicate=False, *, allow_repaired=False):
    """Freeze homogeneous exact or explicitly requested repaired BC datasets."""
    # Check every source before publishing/linking anything.  A repair experiment
    # can contain unmodified games, but exact and repaired *datasets* must not be
    # silently merged and advertised as one clean corpus.
    checked = []
    for local, cloud in sources:
        manifest, metadata, fingerprint = read_published(Path(local))
        kind = require_dataset_kind(metadata, allow_repaired=allow_repaired, manifest=manifest)
        checked.append((local, cloud, manifest, metadata, fingerprint, kind))
    kinds = {row[-1] for row in checked}
    if len(kinds) != 1:
        raise ValueError("Freeze a separate BC snapshot for exact and repaired datasets")
    kind = kinds.pop()
    destination.mkdir(parents=True, exist_ok=True)
    records, replay_offset, rows, locations = [], 0, 0, []
    combined = None
    fingerprints = []
    excluded = []
    seen = set()
    for source_index, (local, cloud, manifest, metadata, fingerprint, _) in enumerate(checked):
        prior_locations_path = Path(local) / "source_locations.json"
        prior_locations = (
            {
                item["path"]: item["source"]
                for item in json.loads(prior_locations_path.read_text())["files"]
            }
            if prior_locations_path.exists()
            else {}
        )
        fingerprints.append(fingerprint)
        source_base = replay_offset
        if combined is None:
            combined = {
                **metadata,
                "replays": [],
                "label_totals": {},
                "additional_deduplication": deduplicate,
                "dataset_kind": kind,
            }
        for key in (
            "schema",
            "vocabulary",
            "backend_commit",
            "content_sha",
            "level",
            "backend_patch_sha256",
        ):
            if combined.get(key) != metadata.get(key):
                raise ValueError(f"Cannot merge BC shards with different {key}")
        for record in manifest["shards"]:
            relative = Path("shards") / f"source_{source_index}" / record["path"]
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                source = Path(local) / record["path"]
                try:
                    os.link(source, target)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    copied = copy_verified(source, target)
                    if copied["sha256"] != record["sha256"]:
                        raise ValueError(f"Copied BC shard checksum mismatch: {relative}") from None
            records.append({**record, "path": relative.as_posix(), "replay_offset": replay_offset})
            locations.append(
                {
                    "path": relative.as_posix(),
                    "source": prior_locations.get(
                        record["path"], str(Path(cloud) / record["path"])
                    ),
                    "sha256": record["sha256"],
                }
            )
            replay_offset += record["replays"]
            rows += record["rows"]
        previous_exclusions = set(metadata.get("excluded_replay_ids", []))
        for index, replay in enumerate(metadata["replays"]):
            key = replay.get("canonical_game_signature") or json.dumps(
                [replay.get("source"), replay.get("source_revision"), replay["replay_tag"]]
            )
            if index in previous_exclusions or (deduplicate and key in seen):
                excluded.append(source_base + index)
            else:
                seen.add(key)
            combined["replays"].append(replay)
        for key, count in metadata.get("label_totals", {}).items():
            combined["label_totals"][key] = combined["label_totals"].get(key, 0) + count
    if replay_offset != len(combined["replays"]):
        raise ValueError("Replay metadata and shard offsets disagree")
    combined["rows"] = rows
    combined["excluded_replay_ids"] = excluded
    combined["supervised_replays"] = replay_offset - len(excluded)
    combined["source_manifest_sha256"] = fingerprints
    combined["source"] = sorted({r["source"] for r in combined["replays"]})
    atomic_json(destination / "dataset_metadata.json", combined)
    atomic_json(
        destination / "dataset.manifest.json",
        {
            "format": "crrl-bc-shards-v1",
            "shards": records,
            "metadata": "dataset_metadata.json",
            "metadata_sha256": file_sha256(destination / "dataset_metadata.json"),
            "rows": rows,
            "replays": replay_offset,
            "additional_deduplication": deduplicate,
            "supervised_replays": replay_offset - len(excluded),
            "dataset_kind": kind,
        },
    )
    atomic_json(destination / "source_locations.json", {"files": locations})
    return {
        "replays": replay_offset,
        "supervised_replays": replay_offset - len(excluded),
        "duplicates_excluded": len(excluded),
        "rows": rows,
        "source_manifest_sha256": fingerprints,
        "dataset_kind": kind,
    }
