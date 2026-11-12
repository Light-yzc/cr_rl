"""Immutable, SHA-cached catalogs of compatible 0.4-second state-BC datasets.

This is a catalog of already-built compressed shards. It does not reconstruct
raw replay parquet, change labels, or make an offline sequence a native rollout.
Only one-game shards are supported, so original compressed bytes can be reused
and global replay ids changed solely through ``replay_offset``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import struct
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

FORMAT = "crrl-bc-shards-v1"
LABEL_SOURCE = "offline_delayed_bc_supervision"
FIELDS = ("observations", "masks", "actions", "ticks", "seats", "flags", "replay_ids")
CONTRACT_FIELDS = (
    "schema",
    "vocabulary",
    "dataset_kind",
    "label_source",
    "decision_ticks",
    "mask_version",
    "source_decision_ticks",
    "reconstruction_backend",
    "backend_commit",
    "backend_patch_sha256",
    "data_version",
    "content_sha",
    "level",
    "tower_level",
    "native_trajectory",
    "ppo_rollouts_allowed",
    "native_parity_claim_allowed",
    "import_policy",
    "source_game_version",
    "backend_provenance_status",
)
RESAMPLE_FIELDS = (
    "kind",
    "source_decision_ticks",
    "logical_decision_ticks",
    "all_original_games_preserved",
    "all_original_nonwait_rows_preserved",
    "observation_clock",
    "mutable_observation_columns",
    "native_trajectory",
    "ppo_rollouts_allowed",
    "native_parity_claim_allowed",
    "mapping",
    "source_rows_definition",
)
REPLAY_FIELDS = (
    "canonical_game_signature",
    "game_signature",
    "replay_tag",
    "source",
    "source_revision",
    "dataset_kind",
    "repair_class",
    "repair_is_exact_bc",
    "source_row_index",
    "source_sha256",
    "decks",
    "deck_names",
    "deck_keys",
    "slot_flags",
    "level",
    "tower_level",
    "source_event_stream_completed",
    "all_source_events_executed",
    "normal_simulator_terminal",
    "source_or_native_terminal_completed",
    "trajectory_completion_kind",
    "coverage",
    "used",
    "used_abilities",
    "total_play_events",
    "native_successful_commands",
    "stop_reason",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inside(parent, name):
    path = (parent / name).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"Dataset path escapes its manifest directory: {name}")
    return path


def _contract(manifest, metadata):
    if manifest.get("format") != FORMAT:
        raise ValueError(f"State catalog requires {FORMAT}")
    for document in (manifest, metadata):
        if document.get("decision_ticks") != 8:
            raise ValueError(
                "State catalog accepts only decision_ticks=8; "
                "legacy 0.25s data must be rebuilt first"
            )
        if document.get("label_source") != LABEL_SOURCE:
            raise ValueError(f"State catalog accepts only label_source={LABEL_SOURCE}")
        if document.get("training_ready") is not True or document.get("smoke_only", False):
            raise ValueError("Dataset is not training-ready or is smoke-only")
    if manifest.get("dataset_kind") != metadata.get("dataset_kind"):
        raise ValueError("Manifest and metadata dataset_kind disagree")
    if not metadata.get("schema") or not metadata.get("vocabulary"):
        raise ValueError("State catalog requires schema and vocabulary")
    provenance_status = metadata.get("backend_provenance_status")
    if provenance_status != "legacy_unverified" and (
        not metadata.get("backend_commit") or not metadata.get("backend_patch_sha256")
    ):
        raise ValueError(
            "State catalog requires backend fingerprints or explicit "
            "backend_provenance_status=legacy_unverified"
        )
    if (
        "backend_provenance_status" in manifest
        and manifest["backend_provenance_status"] != provenance_status
    ):
        raise ValueError("Manifest and metadata backend_provenance_status disagree")
    for key in ("native_trajectory", "ppo_rollouts_allowed", "native_parity_claim_allowed"):
        if metadata.get(key) is not False:
            raise ValueError(f"Offline delayed BC requires explicit {key}=false")
        if key in manifest and manifest[key] != metadata[key]:
            raise ValueError(f"Manifest and metadata {key} disagree")
    result = {key: metadata.get(key) for key in CONTRACT_FIELDS}
    resampling = metadata.get("interval_resampling", {})
    result["interval_resampling"] = {key: resampling.get(key) for key in RESAMPLE_FIELDS}
    return result


def _identity(replay):
    value = replay.get("canonical_game_signature") or replay.get("game_signature")
    if value:
        return str(value)
    if not replay.get("replay_tag"):
        raise ValueError("Every replay requires a stable replay_tag or canonical_game_signature")
    # Exact compatibility with training.state_bc.replay_identity, including spaces.
    return json.dumps([replay.get("source"), replay.get("source_revision"), replay["replay_tag"]])


def _identity_keys(replay):
    keys = {_identity(replay)}
    for key in ("canonical_game_signature", "game_signature"):
        if replay.get(key):
            keys.add(str(replay[key]))
    if replay.get("replay_tag"):
        values = [replay.get("source"), replay.get("source_revision"), replay["replay_tag"]]
        keys.update((json.dumps(values), json.dumps(values, separators=(",", ":"))))
    return keys


def _npy_header(stream):
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Shard member is not an NPY array")
    version = stream.read(2)
    if version[0] == 1:
        size = struct.unpack("<H", stream.read(2))[0]
    elif version[0] in (2, 3):
        size = struct.unpack("<I", stream.read(4))[0]
    else:
        raise ValueError("Unsupported NPY format version")
    if size > 1024 * 1024:
        raise ValueError("Oversized NPY header")
    header = ast.literal_eval(
        stream.read(size).decode("utf-8" if version[0] == 3 else "latin1").strip()
    )
    if header.get("fortran_order"):
        raise ValueError("Fortran-order shard arrays are unsupported")
    return header


def _shard_layout(path):
    """Read only array headers and tiny replay ids; never decompress observations."""
    with zipfile.ZipFile(path) as archive:
        headers = {}
        for name in FIELDS:
            try:
                with archive.open(name + ".npy") as stream:
                    headers[name] = _npy_header(stream)
                    dtype = headers[name]["descr"]
                    allowed = "biu" if name == "masks" else "iu"
                    if not isinstance(dtype, str) or len(dtype) < 3 or dtype[1] not in allowed:
                        raise ValueError(
                            f"Shard array {name} must use a non-object integer/bool dtype"
                        )
                    expected_dimensions = 2 if name in ("observations", "masks") else 1
                    if len(headers[name]["shape"]) != expected_dimensions:
                        raise ValueError(f"Unexpected {name} array dimensions")
                    if name == "masks" and headers[name]["shape"][1] != 2307:
                        raise ValueError("State catalog requires the 2307-action mask codec")
                    if name == "replay_ids":
                        dtype = headers[name]["descr"]
                        if dtype not in ("<i4", "<i8", "<u4", "<u8", ">i4", ">i8", ">u4", ">u8"):
                            raise ValueError(
                                "replay_ids must be a non-object 32/64-bit integer array"
                            )
                        if len(headers[name]["shape"]) != 1:
                            raise ValueError("replay_ids must be one-dimensional")
                        count = int(headers[name]["shape"][0])
                        size = int(dtype[2:])
                        payload = stream.read()
                        if len(payload) != count * size:
                            raise ValueError("Truncated replay_ids array")
                        code = {("i", 4): "i", ("i", 8): "q", ("u", 4): "I", ("u", 8): "Q"}[
                            dtype[1], size
                        ]
                        ids = {value[0] for value in struct.iter_unpack(dtype[0] + code, payload)}
            except KeyError as error:
                raise ValueError(f"Missing required shard array: {name}") from error
        if count < 1 or any(not h["shape"] or h["shape"][0] != count for h in headers.values()):
            raise ValueError("Shard arrays must be nonempty and have matching row counts")
        if len(ids) != 1:
            raise ValueError(
                "State catalog supports exactly one replay per compressed shard; "
                "split multi-game shards first"
            )
        local_id = next(iter(ids))
        if local_id < 0:
            raise ValueError("Negative local replay id")
        return count, local_id


def _copy_verified(source, target, checksum, *, trust_data_files=False):
    if target.exists():
        if not trust_data_files and _sha(target) != checksum:
            raise ValueError(f"Corrupt existing immutable/cache file: {target.name}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".copy-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        if not trust_data_files and _sha(temporary) != checksum:
            raise ValueError("Compressed shard checksum mismatch after copying source")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _cached_shard(source, cache_root, checksum, *, trust_data_files=False):
    """Use locally verified SHA bytes without rereading an unchanged remote shard."""
    target = cache_root / (checksum + ".npz")
    receipt = cache_root / (checksum + ".verified.json")
    copied = _copy_verified(source, target, checksum, trust_data_files=trust_data_files)
    if trust_data_files:
        # A declared checksum remains a cache key, not a verified-content claim.
        return target, copied
    expected = {
        "format": "crrl-compressed-shard-verification-v1",
        "sha256": checksum,
        "bytes": target.stat().st_size,
        "verified": True,
        "verification": "local content SHA256 matches declared source manifest checksum",
    }
    if receipt.exists() and _load(receipt) != expected:
        raise ValueError("Compressed cache verification receipt disagrees with local shard")
    if not receipt.exists():
        with tempfile.NamedTemporaryFile(
            dir=cache_root, prefix=".receipt-", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            temporary.write_bytes(_json(expected) + b"\n")
            os.replace(temporary, receipt)
        finally:
            temporary.unlink(missing_ok=True)
    return target, copied


def _partition(replay, excluded, registry):
    values = {registry[key] for key in _identity_keys(replay) if key in registry}
    if len(values) > 1:
        raise ValueError("Canonical and legacy split registry keys disagree")
    chosen = (
        next(iter(values))
        if values
        else (
            "validation"
            if int.from_bytes(hashlib.sha256(_identity(replay).encode()).digest()[:8], "big")
            / 2**64
            < 0.1
            else "train"
        )
    )
    if excluded:
        chosen = "heldout"
    if chosen not in ("train", "validation", "heldout"):
        raise ValueError(f"Unknown split partition: {chosen}")
    return chosen


def build_catalog_snapshot(
    source_manifests,
    destination,
    cache_root=None,
    *,
    split_registry=None,
    min_train_games=0,
    trust_data_files=False,
):
    """Build/reuse ``destination/<catalog_sha>/dataset.manifest.json``.

    Inputs are explicitly named already-built manifests, not raw replay files.
    The result includes ``manifest_path``, ``fingerprint`` (manifest SHA), games,
    rows, partition counts, deduplication and new/reused compressed-cache counts.
    Split registry uses the same identity/hash semantics as state BC. Its copied
    output path must be passed to the trainer with ``--split-registry``.
    """
    paths = [
        Path(path) / "dataset.manifest.json" if Path(path).is_dir() else Path(path)
        for path in source_manifests
    ]
    if not paths:
        raise ValueError("At least one explicit source manifest is required")
    if min_train_games < 0:
        raise ValueError("min_train_games cannot be negative")
    registry = (
        _load(split_registry)
        if isinstance(split_registry, (str, Path))
        else dict(split_registry or {})
    )
    destination = Path(destination).resolve()
    cache_root = Path(cache_root).resolve() if cache_root else destination / ".compressed-cache"
    expected_contract = None
    candidates, aliases, sources = [], {}, []
    verified_cache = {}
    duplicates = 0
    conflicting_duplicates = 0
    for path in paths:
        path = path.resolve()
        manifest = _load(path)
        metadata_path = _inside(path.parent, manifest["metadata"])
        if _sha(metadata_path) != manifest["metadata_sha256"]:
            raise ValueError(f"Metadata checksum mismatch: {path}")
        metadata = _load(metadata_path)
        contract = _contract(manifest, metadata)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            different = [key for key in contract if contract[key] != expected_contract[key]]
            raise ValueError("Incompatible state dataset contracts: " + ", ".join(different))
        excluded = set(map(int, metadata.get("excluded_replay_ids", [])))
        source_rows = 0
        for record in manifest["shards"]:
            source = _inside(path.parent, record["path"])
            checksum = record["sha256"]
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError("Shard SHA must be lowercase SHA-256")
            if checksum not in verified_cache:
                verified_cache[checksum] = _cached_shard(
                    source, cache_root, checksum, trust_data_files=trust_data_files
                )
            cached, cache_was_copied = verified_cache[checksum]
            count, local_id = _shard_layout(cached)
            if record.get("replays", 1) != 1:
                raise ValueError("State catalog supports only one-replay shards")
            if record.get("rows", count) != count:
                raise ValueError("Shard row count differs from manifest")
            source_rows += count
            old_id = local_id + int(record.get("replay_offset", 0))
            if old_id < 0 or old_id >= len(metadata["replays"]):
                raise ValueError("Shard replay_offset points outside source metadata")
            replay = metadata["replays"][old_id]
            keys = _identity_keys(replay)
            known = {aliases[key] for key in keys if key in aliases}
            partition = _partition(replay, old_id in excluded, registry)
            if known:
                if len(known) != 1:
                    raise ValueError(
                        "Conflicting duplicate replay identities bridge multiple games"
                    )
                previous = candidates[next(iter(known))]
                if partition != previous["partition"]:
                    raise ValueError(
                        "Duplicate replay identity has conflicting split/heldout assignment"
                    )
                duplicates += 1
                conflicting_duplicates += checksum != previous["sha256"]
                for key in keys:
                    aliases[key] = next(iter(known))
                continue
            index = len(candidates)
            for key in keys:
                aliases[key] = index
            compact = {key: replay[key] for key in REPLAY_FIELDS if key in replay}
            candidates.append(
                {
                    "identity": _identity(replay),
                    "replay": compact,
                    "source": source,
                    "sha256": checksum,
                    "rows": count,
                    "local_id": local_id,
                    "partition": partition,
                    "cached": cached,
                }
            )
        if manifest.get("rows", source_rows) != source_rows:
            raise ValueError("Source manifest total row count disagrees with its shards")
        sources.append(
            {
                "manifest_sha256": _sha(path),
                "rows": source_rows,
                "referenced_games": len(manifest["shards"]),
            }
        )
    if not candidates:
        raise ValueError("Catalog has no games")
    candidates.sort(key=lambda row: row["identity"])
    partition_counts = Counter(row["partition"] for row in candidates)
    counts = {key: partition_counts[key] for key in ("train", "validation", "heldout")}
    if counts.get("train", 0) < min_train_games:
        raise ValueError(
            f"Only {counts.get('train', 0)} independent train games; require {min_train_games}"
        )
    signature = {
        "contract": expected_contract,
        "games": [
            {
                key: row[key]
                for key in ("identity", "replay", "sha256", "rows", "local_id", "partition")
            }
            for row in candidates
        ],
    }
    catalog_sha = hashlib.sha256(_json(signature)).hexdigest()
    target = destination / catalog_sha
    destination.mkdir(parents=True, exist_ok=True)
    new_cache = sum(copied for _, copied in verified_cache.values())
    reused_cache = len(verified_cache) - new_cache
    metadata = dict(expected_contract)
    metadata.update(
        training_ready=True,
        smoke_only=False,
        replays=[row["replay"] for row in candidates],
        excluded_replay_ids=[
            i for i, row in enumerate(candidates) if row["partition"] == "heldout"
        ],
        rows=sum(row["rows"] for row in candidates),
        catalog={
            "format": "crrl-state-catalog-v1",
            "catalog_sha256": catalog_sha,
            "label_scope": "already-built compatible shards; no raw replay reconstruction",
        },
    )
    metadata_bytes = _json(metadata) + b"\n"
    manifest = {
        key: metadata[key]
        for key in (
            "dataset_kind",
            "decision_ticks",
            "label_source",
            "native_trajectory",
            "ppo_rollouts_allowed",
            "native_parity_claim_allowed",
            "backend_provenance_status",
            "training_ready",
            "smoke_only",
            "rows",
        )
    }
    manifest.update(
        format=FORMAT,
        replays=len(candidates),
        metadata="dataset_metadata.json",
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        shards=[
            {
                "path": "shards/" + row["sha256"] + ".npz",
                "sha256": row["sha256"],
                "rows": row["rows"],
                "replays": 1,
                "replay_offset": i - row["local_id"],
            }
            for i, row in enumerate(candidates)
        ],
    )
    manifest_bytes = _json(manifest) + b"\n"
    output_registry = {_identity(row["replay"]): row["partition"] for row in candidates}
    registry_bytes = _json(output_registry) + b"\n"
    summary = {
        "manifest_path": str(target / "dataset.manifest.json"),
        "metadata_path": str(target / "dataset_metadata.json"),
        "split_registry_path": str(target / "split_registry.json"),
        "fingerprint": hashlib.sha256(manifest_bytes).hexdigest(),
        "catalog_sha256": catalog_sha,
        "games": len(candidates),
        "rows": metadata["rows"],
        "counts": counts,
        "duplicate_games_skipped": duplicates,
        "duplicate_games_with_different_shard_sha": conflicting_duplicates,
        "duplicate_policy": "first explicitly supplied source wins; identities never count twice",
        "new_cache_shards": new_cache,
        "reused_cache_shards": reused_cache,
        "source_manifests": sources,
        "snapshot_reused": target.exists(),
    }
    if target.exists():
        for name, expected in (
            ("dataset.manifest.json", manifest_bytes),
            ("dataset_metadata.json", metadata_bytes),
            ("split_registry.json", registry_bytes),
        ):
            if not (target / name).is_file() or (target / name).read_bytes() != expected:
                raise ValueError("Existing immutable catalog differs or is incomplete")
        for row in candidates:
            shard = target / "shards" / (row["sha256"] + ".npz")
            if not shard.is_file() or (not trust_data_files and _sha(shard) != row["sha256"]):
                raise ValueError("Existing immutable snapshot shard is corrupt")
        return summary
    staging = Path(tempfile.mkdtemp(prefix=".catalog-", dir=destination))
    try:
        (staging / "shards").mkdir()
        for row in candidates:
            output = staging / "shards" / (row["sha256"] + ".npz")
            if output.exists():
                continue
            try:
                os.link(row["cached"], output)
            except OSError:
                shutil.copyfile(row["cached"], output)
        (staging / "dataset_metadata.json").write_bytes(metadata_bytes)
        (staging / "split_registry.json").write_bytes(registry_bytes)
        (staging / "summary.json").write_bytes(_json(summary) + b"\n")
        # The trainer-visible manifest is committed last, followed by atomic directory publication.
        (staging / "dataset.manifest.json").write_bytes(manifest_bytes)
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary
