"""Compressed BC shards on Drive, bounded-memory read-only arrays during training."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from cr_rl.training.persistence import file_sha256


class MappedShardArray:
    """Axis-zero indexing across read-only .npy shards, without concatenating states."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.offsets = np.cumsum([0, *(len(a) for a in arrays)])
        self.shape = (int(self.offsets[-1]), *arrays[0].shape[1:])
        self.dtype = arrays[0].dtype
        if any(a.shape[1:] != self.shape[1:] or a.dtype != self.dtype for a in arrays):
            raise ValueError("BC shard array schemas differ")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        first, *rest = key if isinstance(key, tuple) else (key,)
        if isinstance(first, slice) and first.indices(len(self))[2] == 1:
            left, right, _ = first.indices(len(self))
            pieces = []
            for index, array in enumerate(self.arrays):
                a = max(left, int(self.offsets[index])) - int(self.offsets[index])
                b = min(right, int(self.offsets[index + 1])) - int(self.offsets[index])
                if b > a:
                    pieces.append(array[(slice(a, b), *rest)])
            return np.concatenate(pieces) if pieces else self.arrays[0][(slice(0, 0), *rest)]
        if np.isscalar(first):
            row = int(first)
            row = row + len(self) if row < 0 else row
            if not 0 <= row < len(self):
                raise IndexError(row)
            shard = int(np.searchsorted(self.offsets[1:], row, side="right"))
            return self.arrays[shard][(row - int(self.offsets[shard]), *rest)]
        rows = np.arange(len(self))[first] if isinstance(first, slice) else np.asarray(first)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        if not rows.size:
            rows = rows.astype(np.int64)
        rows = np.where(rows < 0, rows + len(self), rows)
        if rows.ndim != 1 or ((rows < 0) | (rows >= len(self))).any():
            raise IndexError("Expected in-range one-dimensional BC row indices")
        shards = np.searchsorted(self.offsets[1:], rows, side="right")
        result = None
        for index in np.unique(shards):
            selected = np.flatnonzero(shards == index)
            local = rows[selected] - self.offsets[index]
            columns = [
                np.asarray(v)[selected]
                if not np.isscalar(v) and not isinstance(v, slice) and len(v) == len(rows)
                else v
                for v in rest
            ]
            piece = self.arrays[index][(local, *columns)]
            if result is None:
                result = np.empty((len(rows), *piece.shape[1:]), dtype=piece.dtype)
            result[selected] = piece
        if result is None:
            return self.arrays[0][(np.array([], dtype=int), *rest)]
        return result


class ShardArrayCache:
    """Bound decompressed state files; returned slices own their memory."""

    def __init__(self, root, max_bytes, minimum_free_bytes=0):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.entries = OrderedDict()
        self.bytes = 0
        self.evictions = 0
        self.extractions = 0
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.root.iterdir():
            if directory.is_dir() and len(directory.name) == 64:
                for field in ("observations", "masks"):
                    path = directory / f"{field}.npy"
                    if path.is_file():
                        self.entries[path] = path.stat().st_size
                        self.bytes += path.stat().st_size
        self._evict(0)

    def _evict(self, required):
        while self.entries and (
            self.bytes + required > self.max_bytes
            or shutil.disk_usage(self.root).free < required + self.minimum_free_bytes
        ):
            path, size = self.entries.popitem(last=False)
            path.unlink(missing_ok=True)
            self.bytes -= size
            self.evictions += 1
        if required and shutil.disk_usage(self.root).free < required + self.minimum_free_bytes:
            raise OSError("Insufficient scratch space for BC cache after evicting cached arrays")

    def read(self, source, digest, field, key):
        target = self.root / digest / f"{field}.npy"
        if target in self.entries:
            self.entries.move_to_end(target)
        else:
            with zipfile.ZipFile(source) as archive:
                record = archive.getinfo(target.name)
                if record.file_size > self.max_bytes:
                    raise ValueError("One BC state array exceeds the streaming cache budget")
                self._evict(record.file_size)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".partial")
                with archive.open(record) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=4 * 1024**2)
                temporary.replace(target)
                self.entries[target] = record.file_size
                self.bytes += record.file_size
                self.extractions += 1
        array = np.load(target, mmap_mode="r", allow_pickle=False)
        try:
            return np.array(array[key], copy=True)
        finally:
            array._mmap.close()


class CachedShardArray:
    def __init__(self, source, digest, field, cache):
        self.source, self.digest, self.field, self.cache = source, digest, field, cache
        with zipfile.ZipFile(source) as archive, archive.open(f"{field}.npy") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError(f"Unsupported BC numpy format: {version}")
        self.shape, self.dtype = shape, dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        return self.cache.read(self.source, self.digest, self.field, key)


def load_sharded_bc(manifest_path: Path, cache_root: Path, cache_gib=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != "crrl-bc-shards-v1":
        raise ValueError("Unsupported BC shard manifest")
    fields = ("observations", "masks", "actions", "replay_ids", "seats", "flags", "ticks")
    pieces = {key: [] for key in fields}
    cache = (
        ShardArrayCache(
            cache_root,
            cache_gib * 1024**3,
            minimum_free_bytes=16 * 1024**3 if cache_gib >= 32 else 0,
        )
        if cache_gib
        else None
    )
    for shard in manifest["shards"]:
        source = manifest_path.parent / shard["path"]
        if file_sha256(source) != shard["sha256"]:
            raise ValueError(f"BC shard checksum mismatch: {source.name}")
        cached = cache_root / shard["sha256"]
        cached.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for key in fields:
                if cache is not None and key in ("observations", "masks"):
                    pieces[key].append(CachedShardArray(source, shard["sha256"], key, cache))
                    continue
                target = cached / f"{key}.npy"
                record = archive.getinfo(target.name)
                if not target.exists() or target.stat().st_size != record.file_size:
                    temporary = target.with_suffix(".partial")
                    with archive.open(record) as src, temporary.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=4 * 1024**2)
                    temporary.replace(target)
                array = np.load(target, mmap_mode="r", allow_pickle=False)
                if key == "replay_ids":
                    array = array.astype(np.int32) + int(shard["replay_offset"])
                pieces[key].append(array)
    result = {
        key: MappedShardArray(value) if key in ("observations", "masks") else np.concatenate(value)
        for key, value in pieces.items()
    }
    metadata_path = manifest_path.parent / manifest["metadata"]
    if file_sha256(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError("BC dataset metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return result, metadata
