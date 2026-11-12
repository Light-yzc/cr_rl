"""Single-file state-BC transfer, trusting Drive infrastructure for integrity.

NPZ shards are already compressed: the outer archive uses ZIP_STORED + Zip64.
No application-level file hashing, data revalidation, or cache rescanning runs
in this path. Declared producer fingerprints identify versions only. Zipfile's
normal one-pass CRC handling remains; paths and special members stay restricted.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

FORMAT = "crrl-state-bc-bundle-v1"
BLOCK_BYTES = 4 * 1024**2
RESERVE_BYTES = 256 * 1024**2


def _emit(progress, phase, **fields):
    if progress is not None:
        progress({"phase": phase, **fields})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(path):
    """Legacy helper retained for callers; the transfer path never invokes it."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Unsafe bundle member path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in (".", "..") for p in name.split("/")) or str(path) != name:
        raise ValueError("Bundle member escapes its root")
    return name


def _source_path(root, name):
    """Lexical join; the trusted source root is resolved once, never per file."""
    return Path(root) / _name(name)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    try:
        temporary.write_bytes(_canonical(value) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _space(path, needed):
    available = shutil.disk_usage(path).free
    if available < needed + RESERVE_BYTES:
        raise OSError(
            f"Insufficient local SSD space: need {needed + RESERVE_BYTES}, available {available}"
        )


def _json_file(root, name, declared_sha=None, *, parse=True):
    path = _source_path(root, name)
    raw = path.read_bytes()
    # This is one read for layout, reused when archiving. The producer SHA is not checked.
    return (json.loads(raw) if parse else None), {
        "source": path,
        "sha256": declared_sha,
        "bytes": len(raw),
        "raw": raw,
    }


def _inventory(root, expected_games, expected_source_shards, progress):
    collection, record = _json_file(root, "collection.manifest.json")
    files = {"collection.manifest.json": record}
    if (
        collection.get("format") != "crrl-state-bc-collection-v1"
        or collection.get("migration_complete") is not True
    ):
        raise ValueError("Only a completed producer-committed collection can be packed")
    for field, expected in (
        ("source_shards_total", expected_source_shards),
        ("source_shards_migrated", expected_source_shards),
        ("source_replays_total", expected_games),
        ("accepted_replays", expected_games),
    ):
        if collection.get(field) != expected:
            raise ValueError("Collection count differs: " + field)
    if len(collection.get("batches", [])) != expected_source_shards:
        raise ValueError("Committed batch count differs")
    split_name = _name(collection.get("split_registry", "split_registry.json"))
    registry, record = _json_file(root, split_name, collection.get("split_registry_sha256"))
    files[split_name] = record
    split_counts = registry.get("source_identity_counts")
    if split_counts is None:
        split_counts = Counter(registry.get("assignments", registry).values())
    for position, batch in enumerate(collection["batches"]):
        name = _name(batch["manifest"])
        manifest, record = _json_file(root, name, batch.get("sha256"))
        if name in files:
            raise ValueError("Duplicate manifest path")
        files[name] = record
        parent = PurePosixPath(name).parent
        metadata_name = _name(str(parent / _name(manifest["metadata"])))
        _, record = _json_file(root, metadata_name, manifest.get("metadata_sha256"), parse=False)
        if metadata_name in files:
            raise ValueError("Duplicate metadata path")
        files[metadata_name] = record
        for item in manifest["shards"]:
            shard_name = _name(str(parent / _name(item["path"])))
            if not shard_name.endswith(".npz") or shard_name in files:
                raise ValueError("Duplicate or non-NPZ shard path")
            files[shard_name] = {
                "source": _source_path(root, shard_name),
                "sha256": item.get("sha256"),
                "bytes": None,
            }
        _emit(
            progress, "inventory", completed_files=position + 1, total_files=expected_source_shards
        )
    summary = {
        "games": collection["accepted_replays"],
        "rows": collection["rows"],
        "source_shards": expected_source_shards,
        "split_counts": {
            key: split_counts.get(key, 0) for key in ("train", "validation", "heldout")
        },
        "source_manifest_sha256": collection.get("source_manifest_sha256"),
        "split_registry_sha256": collection.get("split_registry_sha256"),
    }
    # Hash small declared version identifiers, never source file contents.
    bundle_id = hashlib.sha256(
        _canonical(
            {
                "declared_files": {name: value["sha256"] for name, value in files.items()},
                "source_version": summary,
            }
        )
    ).hexdigest()
    return files, summary, bundle_id


def _prefetch(name, record, scratch):
    target = _source_path(scratch, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = io.BytesIO(record["raw"]) if "raw" in record else record["source"].open("rb")
    size = 0
    with source, target.open("wb") as destination:
        for block in iter(lambda: source.read(BLOCK_BYTES), b""):
            _space(scratch, len(block))
            destination.write(block)
            size += len(block)
    return target, size


def _copy_stream(source, target, *, total_bytes=None, progress=None, phase="download"):
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with Path(source).open("rb") as src, target.open("wb") as dst:
        for block in iter(lambda: src.read(BLOCK_BYTES), b""):
            dst.write(block)
            size += len(block)
            _emit(progress, phase, completed_bytes=size, total_bytes=total_bytes)
    return size


def build_state_bundle(
    collection_root,
    out_dir,
    scratch_root,
    *,
    workers=8,
    expected_games=10654,
    expected_source_shards=333,
    progress=None,
):
    """Prefetch each compressed member once and publish one infrastructure-trusted ZIP."""
    if workers < 1 or workers > 64 or expected_games < 1 or expected_source_shards < 1:
        raise ValueError("Invalid worker/game/shard bounds")
    root, out, scratch = map(lambda p: Path(p).resolve(), (collection_root, out_dir, scratch_root))
    scratch.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if scratch == root or scratch.is_relative_to(root) or root.is_relative_to(scratch):
        raise ValueError("Scratch must be separate from the source collection")
    files, summary, bundle_id = _inventory(root, expected_games, expected_source_shards, progress)
    current = out / "current.json"
    if current.is_file():
        existing = json.loads(current.read_text())
        if existing.get("format") == FORMAT and existing.get("bundle_id") == bundle_id:
            _emit(progress, "cache_hit", **summary, archive=str(out / existing["archive"]["name"]))
            return {**existing, "index_path": str(current), "reused": True}
    _space(scratch, 0)
    with tempfile.TemporaryDirectory(prefix="state-bundle-", dir=scratch) as temporary:
        work = Path(temporary)
        archive_path, members = work / "data.zip", {}
        names, archived_bytes = sorted(files), 0
        with (
            ThreadPoolExecutor(max_workers=workers) as pool,
            zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive,
        ):
            pending, next_submit = {}, 0
            for position, name in enumerate(names):
                while next_submit < min(len(names), position + 2 * workers):
                    pending[next_submit] = pool.submit(
                        _prefetch, names[next_submit], files[names[next_submit]], work / "members"
                    )
                    next_submit += 1
                staged, size = pending.pop(position).result()
                _space(work, size)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type, info.create_system = zipfile.ZIP_STORED, 3
                info.external_attr = 0o100644 << 16
                with staged.open("rb") as source, archive.open(info, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(source, dst, BLOCK_BYTES)
                staged.unlink()
                members[name] = {"sha256": files[name]["sha256"], "bytes": size}
                archived_bytes += size
                _emit(
                    progress,
                    "archive",
                    completed_files=position + 1,
                    total_files=len(names),
                    completed_bytes=archived_bytes,
                    total_bytes=None,
                    member=name,
                )
        archive_bytes = archive_path.stat().st_size
        archive_name = "state_bc_full." + bundle_id + ".zip"
        published = out / archive_name
        uploading = published.with_name(published.name + ".uploading")
        try:
            _copy_stream(
                archive_path,
                uploading,
                total_bytes=archive_bytes,
                progress=progress,
                phase="publish",
            )
            os.replace(uploading, published)
        finally:
            uploading.unlink(missing_ok=True)
        receipt = {
            "format": FORMAT,
            "bundle_id": bundle_id,
            "source_inventory_sha256": bundle_id,
            "archive": {"name": archive_name, "sha256": None, "bytes": archive_bytes},
            "members": members,
            "member_count": len(members),
            "uncompressed_member_bytes": archived_bytes,
            "zip_compression": "ZIP_STORED",
            "zip64": True,
            "source_collection": str(root),
            **summary,
            "application_content_checks": False,
            "integrity_provider": "Google Drive infrastructure; normal ZIP CRC on extraction",
            "member_sha_semantics": (
                "Producer declarations copied as version metadata; not recalculated"
            ),
            "scope": (
                "Necessary committed data only; one source read, one ZIP upload, no revalidation"
            ),
        }
        _atomic_json(current, receipt)
        _emit(progress, "complete", **summary, archive=str(published), index_path=str(current))
        return {**receipt, "index_path": str(current), "reused": False}


def _index_members(index):
    if index.get("format") != FORMAT:
        raise ValueError("Unsupported state bundle format")
    _name(index["archive"]["name"])
    members = index["members"]
    for name in members:
        _name(name)
    if not {"collection.manifest.json", "split_registry.json"}.issubset(members):
        raise ValueError("Bundle is missing collection or split manifest")
    return members


def stage_state_bundle(bundle_index_path, local_root, *, progress=None):
    """Copy one ZIP and extract once; cache hits read only the small cache marker."""
    index_path, local = Path(bundle_index_path).resolve(), Path(local_root).resolve()
    index = json.loads(index_path.read_bytes())
    members = _index_members(index)
    # Declared bundle id/size is a version key, not a recomputed content digest.
    bundle_id = index.get("bundle_id") or index.get("source_inventory_sha256")
    _name(bundle_id)
    target = local / bundle_id
    local.mkdir(parents=True, exist_ok=True)
    marker = ".state_bundle_cache.json"
    expected_marker = {
        "format": FORMAT,
        "bundle_id": bundle_id,
        "archive_bytes": index["archive"]["bytes"],
        "application_content_checks": False,
    }
    if target.exists():
        marker_path = target / marker
        if (
            target.is_symlink()
            or not marker_path.is_file()
            or json.loads(marker_path.read_text()) != expected_marker
        ):
            raise ValueError("Existing local bundle cache has a different version marker")
        _emit(
            progress,
            "cache_hit",
            local_root=str(target),
            games=index.get("games"),
            rows=index.get("rows"),
        )
        return target
    _space(local, index["archive"]["bytes"] + index["uncompressed_member_bytes"])
    archive_source = _source_path(index_path.parent, index["archive"]["name"])
    with tempfile.TemporaryDirectory(prefix=".state-stage-", dir=local) as temporary:
        staging = Path(temporary)
        archive_path, collection = staging / "bundle.zip", staging / "collection"
        collection.mkdir()
        _copy_stream(
            archive_source, archive_path, total_bytes=index["archive"]["bytes"], progress=progress
        )
        with zipfile.ZipFile(archive_path) as archive:
            records = archive.infolist()
            names = [record.filename for record in records]
            if len(set(names)) != len(names):
                raise ValueError("Duplicate ZIP members")
            if set(names) != set(members):
                raise ValueError("ZIP has extra/missing members")
            completed = 0
            for number, info in enumerate(records):
                name = _name(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or stat.S_ISLNK(mode)
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or info.flag_bits & 1
                    or info.compress_type != zipfile.ZIP_STORED
                ):
                    raise ValueError(
                        "ZIP contains symlink, special, encrypted or non-stored members"
                    )
                output = _source_path(collection, name)
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, output.open("xb") as destination:
                    shutil.copyfileobj(source, destination, BLOCK_BYTES)
                completed += info.file_size
                _emit(
                    progress,
                    "extract",
                    completed_files=number + 1,
                    total_files=len(records),
                    completed_bytes=completed,
                    total_bytes=index["uncompressed_member_bytes"],
                )
        _atomic_json(collection / marker, expected_marker)
        os.rename(collection, target)
    _emit(
        progress,
        "complete",
        local_root=str(target),
        games=index.get("games"),
        rows=index.get("rows"),
    )
    return target
