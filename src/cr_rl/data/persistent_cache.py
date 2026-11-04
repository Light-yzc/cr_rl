"""Compressed, content-addressed validation caches; never mmap arrays on Drive."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import zstandard as zstd

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256

FORMAT = "crrl-validation-cache-zstd-v1"
FILES = ("observations.npy", "masks.npy", "complete.json")


def _signature(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid validation cache signature")
    return value


def _binding(record, signature, dataset_sha256, context):
    if (record.get("signature"), record.get("dataset_sha256"), record.get("context")) != (
        signature,
        dataset_sha256,
        context,
    ):
        raise ValueError("Persistent validation cache binding mismatch")


class _HashedReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def _publish_validation_cache(local, persistent_root, *, reserve_bytes=1024**3):
    """Stream three immutable files into one level-1 zstd archive; publish metadata last.

    Uses one compression thread and bounded buffers. Existing source arrays and
    training state are untouched. File hashes are checked during compression.
    """
    began = time.monotonic()
    local, persistent_root = Path(local), Path(persistent_root)
    record = json.loads((local / "complete.json").read_text())
    signature = _signature(record["signature"])
    if signature != local.name or set(record["sha256"]) != {"observations", "masks"}:
        raise ValueError("Unexpected compact cache layout")
    expected = {f"{key}.npy": value for key, value in record["sha256"].items()}
    expected["complete.json"] = file_sha256(local / "complete.json")
    remote = persistent_root / signature
    pointer = remote / "archive.json"
    if pointer.exists():
        old = json.loads(pointer.read_text())
        archive_name = f"{old.get('archive_sha256', '')}.tar.zst"
        archive = remote / archive_name
        if (
            old.get("format") == FORMAT
            and old.get("file_sha256") == expected
            and archive.is_file()
            and file_sha256(archive) == old.get("archive_sha256")
        ):
            return {
                "event": "already_published",
                "signature": signature,
                "archive_bytes": old["archive_bytes"],
                "seconds": time.monotonic() - began,
            }
    sizes = {name: (local / name).stat().st_size for name in FILES}
    raw_bytes = sum(sizes.values())
    # A conservative bound also handles an unexpectedly incompressible cache.
    if shutil.disk_usage(local.parent).free < raw_bytes + reserve_bytes:
        raise OSError("Insufficient scratch to publish compressed validation cache")
    with tempfile.TemporaryDirectory(prefix=".cache-publish-", dir=local.parent) as scratch:
        compressed = Path(scratch) / "cache.tar.zst"
        compress_began = time.monotonic()
        with compressed.open("wb") as output:
            with zstd.ZstdCompressor(level=1, threads=0, write_checksum=True).stream_writer(
                output,
                closefd=False,
            ) as encoded:
                with tarfile.open(
                    fileobj=encoded,
                    mode="w|",
                    bufsize=1024**2,
                    copybufsize=1024**2,
                ) as archive:
                    for name in FILES:
                        info = tarfile.TarInfo(name)
                        info.size, info.mode = sizes[name], 0o600
                        with (local / name).open("rb") as source:
                            reader = _HashedReader(source)
                            archive.addfile(info, reader)
                            if reader.digest.hexdigest() != expected[name]:
                                raise ValueError(f"Source cache checksum mismatch: {name}")
            output.flush()
            os.fsync(output.fileno())
        compression_seconds = time.monotonic() - compress_began
        digest = file_sha256(compressed)
        upload_began = time.monotonic()
        copied = copy_verified(compressed, remote / f"{digest}.tar.zst")
        upload_seconds = time.monotonic() - upload_began
        manifest = {
            "format": FORMAT,
            "signature": signature,
            "dataset_sha256": record["dataset_sha256"],
            "context": record["context"],
            "archive_sha256": digest,
            "archive_bytes": copied["bytes"],
            "uncompressed_bytes": raw_bytes,
            "file_sizes": sizes,
            "file_sha256": expected,
            "codec": "zstd",
            "level": 1,
            "compression_seconds": compression_seconds,
            "verified_upload_seconds": upload_seconds,
            "created_unix": time.time(),
        }
        metadata = Path(scratch) / "archive.json"
        atomic_json(metadata, manifest)
        copy_verified(metadata, pointer)
    return {"event": "published", **manifest, "seconds": time.monotonic() - began}


def _restore_validation_cache(
    persistent_root,
    destination,
    *,
    signature,
    dataset_sha256,
    context,
    maximum_bytes,
    reserve_bytes=1024**3,
):
    """Copy only compressed bytes to SSD, verify, then decompress and hash once.

    Cache misses return None. Invalid or partial caches raise and are never
    published locally. Caller can log a failure and rebuild from original shards.
    """
    began = time.monotonic()
    signature = _signature(signature)
    remote, destination = Path(persistent_root) / signature, Path(destination)
    pointer = remote / "archive.json"
    if not pointer.exists():
        return None
    metadata = json.loads(pointer.read_text())
    _binding(metadata, signature, dataset_sha256, context)
    digest = _signature(metadata["archive_sha256"])
    if metadata.get("format") != FORMAT or set(metadata["file_sizes"]) != set(FILES):
        raise ValueError("Unexpected persistent cache format or files")
    if set(metadata["file_sha256"]) != set(FILES):
        raise ValueError("Missing persistent cache checksums")
    sizes = metadata["file_sizes"]
    if (
        any(type(size) is not int or size < 0 for size in sizes.values())
        or sum(sizes.values()) != metadata["uncompressed_bytes"]
        or sum(sizes.values()) > maximum_bytes
        or type(metadata["archive_bytes"]) is not int
        or metadata["archive_bytes"] <= 0
    ):
        raise ValueError("Invalid persistent cache size")
    if (destination / "complete.json").exists():
        raise FileExistsError("Refusing to overwrite a completed local cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = metadata["uncompressed_bytes"] + metadata["archive_bytes"] + reserve_bytes
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError("Insufficient SSD space to restore validation cache")
    with tempfile.TemporaryDirectory(prefix=".cache-restore-", dir=destination.parent) as scratch:
        scratch = Path(scratch)
        compressed = scratch / "cache.tar.zst"
        copied_hash, copied_bytes = hashlib.sha256(), 0
        download_began = time.monotonic()
        with (remote / f"{digest}.tar.zst").open("rb") as source, compressed.open("wb") as output:
            while block := source.read(4 * 1024**2):
                copied_hash.update(block)
                copied_bytes += len(block)
                if copied_bytes > metadata["archive_bytes"]:
                    raise ValueError("Persistent archive larger than its manifest")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if copied_hash.hexdigest() != digest or copied_bytes != metadata["archive_bytes"]:
            raise ValueError("Persistent archive checksum mismatch")
        download_seconds = time.monotonic() - download_began
        unpacked, seen = scratch / "unpacked", set()
        unpacked.mkdir()
        extract_began = time.monotonic()
        with (
            compressed.open("rb") as source,
            zstd.ZstdDecompressor().stream_reader(source) as decoded,
        ):
            with tarfile.open(fileobj=decoded, mode="r|", bufsize=1024**2) as archive:
                for item in archive:
                    if (
                        item.name not in FILES
                        or item.name in seen
                        or not item.isfile()
                        or item.size != sizes[item.name]
                    ):
                        raise ValueError("Unexpected persistent archive member")
                    seen.add(item.name)
                    field_hash = hashlib.sha256()
                    with (
                        archive.extractfile(item) as content,
                        (unpacked / item.name).open("wb") as output,
                    ):
                        while block := content.read(4 * 1024**2):
                            output.write(block)
                            field_hash.update(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if field_hash.hexdigest() != metadata["file_sha256"][item.name]:
                        raise ValueError(f"Restored cache checksum mismatch: {item.name}")
        if seen != set(FILES):
            raise ValueError("Incomplete persistent archive")
        record = json.loads((unpacked / "complete.json").read_text())
        _binding(record, signature, dataset_sha256, context)
        if record["sha256"] != {
            name[:-4]: metadata["file_sha256"][name] for name in FILES if name.endswith(".npy")
        }:
            raise ValueError("Compact and persistent manifests disagree")
        destination.mkdir(parents=True, exist_ok=True)
        for name in FILES:  # complete.json is the final local publication marker.
            (unpacked / name).replace(destination / name)
        extraction_seconds = time.monotonic() - extract_began
    return {
        "event": "restored",
        "signature": signature,
        "archive_bytes": metadata["archive_bytes"],
        "uncompressed_bytes": metadata["uncompressed_bytes"],
        "verified_download_seconds": download_seconds,
        "extract_seconds": extraction_seconds,
        "seconds": time.monotonic() - began,
    }


def publish_validation_cache(*args, **kwargs):
    """Publish an immutable cache; normalize compression errors for optional fallback."""
    try:
        return _publish_validation_cache(*args, **kwargs)
    except (zstd.ZstdError, tarfile.TarError, EOFError) as error:
        raise ValueError(f"Invalid compressed validation cache: {error}") from error


def restore_validation_cache(*args, **kwargs):
    """Restore to SSD or return None on a miss; never use corrupt archive bytes."""
    try:
        return _restore_validation_cache(*args, **kwargs)
    except (zstd.ZstdError, tarfile.TarError, EOFError) as error:
        raise ValueError(f"Invalid compressed validation cache: {error}") from error
