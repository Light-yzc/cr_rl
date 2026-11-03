"""Compact exact validation histories, without materializing whole unused state shards."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cr_rl.data.sharded_bc import CachedShardArray
from cr_rl.training.persistence import atomic_json, file_sha256


class SelectedRows:
    """Retain original row indexing while refusing reads outside the frozen cohort."""

    def __init__(self, rows, values, shape):
        self.rows, self.values, self.shape = rows, values, shape
        self.dtype = values.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        first, *rest = key if isinstance(key, tuple) else (key,)
        requested = np.asarray(first)
        if requested.dtype.kind not in "iu":
            raise IndexError("Compact validation expects integer row indices")
        requested = np.where(requested < 0, requested + len(self), requested)
        indices = np.searchsorted(self.rows, requested)
        safe = np.minimum(indices, max(0, len(self.rows) - 1))
        if (
            not len(self.rows)
            or np.any(indices >= len(self.rows))
            or np.any(self.rows[safe] != requested)
        ):
            raise IndexError("Read outside the frozen validation histories")
        return self.values[(indices, *rest)]


def read_npz_rows(source, field, indices, target, destinations):
    """Read selected C-order rows in one forward pass; skipped bytes never go to disk."""
    with zipfile.ZipFile(source) as archive, archive.open(f"{field}.npy") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
        else:
            raise ValueError("Unsupported compact validation numpy format")
        if fortran or dtype.hasobject or dtype != target.dtype or shape[1:] != target.shape[1:]:
            raise ValueError("Unsupported validation shard layout or dtype")
        if len(indices) and (indices[0] < 0 or indices[-1] >= shape[0]):
            raise IndexError("Validation rows outside source shard")
        stride = int(np.prod(shape[1:])) * dtype.itemsize
        origin = member.tell()
        # Bound each read, even when almost the whole shard is selected.
        maximum_rows = max(1, (8 * 1024**2) // stride)
        left = 0
        while left < len(indices):
            right = left + 1
            while (
                right < len(indices)
                and right - left < maximum_rows
                and indices[right] == indices[right - 1] + 1
            ):
                right += 1
            member.seek(origin + int(indices[left]) * stride)
            count = (right - left) * stride
            raw = member.read(count)
            if len(raw) != count:
                raise ValueError("Truncated validation state array")
            target[destinations[left:right]] = np.frombuffer(raw, dtype).reshape(
                right - left, *shape[1:]
            )
            left = right


def compact_field(array, rows, path, progress=None):
    target = np.lib.format.open_memmap(
        path, mode="w+", dtype=array.dtype, shape=(len(rows), *array.shape[1:])
    )
    shards = np.searchsorted(array.offsets[1:], rows, side="right")
    touched = np.unique(shards)

    def copy_shard(index):
        destination = np.flatnonzero(shards == index)
        local = rows[destination] - array.offsets[index]
        source = array.arrays[index]
        if isinstance(source, CachedShardArray):
            cache_root = getattr(source.cache, "readonly", getattr(source.cache, "root", None))
            full = cache_root / source.digest / f"{source.field}.npy" if cache_root else None
            if full is not None and full.exists():
                mapped = np.load(full, mmap_mode="r", allow_pickle=False)
                try:
                    # Avoid a large temporary if one shard supplies many history rows.
                    for left in range(0, len(local), 256):
                        target[destination[left : left + 256]] = mapped[local[left : left + 256]]
                finally:
                    mapped._mmap.close()
            else:
                read_npz_rows(source.source, source.field, local, target, destination)
        else:
            for left in range(0, len(local), 256):
                target[destination[left : left + 256]] = source[local[left : left + 256]]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(copy_shard, index) for index in touched]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if progress and (done % 16 == 0 or done == len(touched)):
                progress(done, len(touched))
    target.flush()
    target._mmap.close()


def prepare_validation_arrays(
    obs,
    masks,
    rows,
    replay_ids,
    seats,
    groups,
    positions,
    context,
    root,
    dataset_sha256,
    progress=None,
):
    """Freeze exact history row membership; original labels and validation draws are unchanged."""
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    history = []
    for row in rows:
        key = (int(replay_ids[row]), int(seats[row]))
        pos = positions[key][int(row)]
        history.extend(groups[key][max(0, pos - context + 1) : pos + 1])
    histories = np.unique(np.asarray(history, dtype=np.int64))
    signature = hashlib.sha256(
        dataset_sha256.encode() + str(context).encode() + rows.tobytes() + histories.tobytes()
    ).hexdigest()
    root = Path(root) / signature
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "complete.json"
    fields = [("observations", obs, histories), ("masks", masks, rows)]
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record["signature"] != signature:
            raise ValueError("Compact validation identity mismatch")
        for name, _, _ in fields:
            if file_sha256(root / f"{name}.npy") != record["sha256"][name]:
                raise ValueError("Compact validation checksum mismatch")
    else:
        required = (
            sum(
                len(selected) * int(np.prod(array.shape[1:])) * array.dtype.itemsize
                for _, array, selected in fields
            )
            + 1024**2
        )
        cache = getattr(obs.arrays[0], "cache", None)
        reserve = getattr(cache, "minimum_free_bytes", 16 * 1024**3 if cache else 0)
        if shutil.disk_usage(root).free < required + reserve and callable(
            getattr(cache, "_evict", None)
        ):
            cache._evict(required)
        if shutil.disk_usage(root).free < required + reserve:
            raise OSError("Insufficient Colab scratch for compact validation; source data retained")
        hashes = {}
        for name, array, selected in fields:
            temporary = root / f"{name}.building.npy"
            compact_field(
                array,
                selected,
                temporary,
                (lambda done, total, field=name: progress(field, done, total))
                if progress
                else None,
            )
            target = root / f"{name}.npy"
            temporary.replace(target)
            hashes[name] = file_sha256(target)
        atomic_json(
            manifest,
            dict(
                signature=signature,
                dataset_sha256=dataset_sha256,
                context=context,
                validation_rows=len(rows),
                history_rows=len(histories),
                sha256=hashes,
            ),
        )
    return tuple(
        SelectedRows(selected, np.load(root / f"{name}.npy", mmap_mode="r"), array.shape)
        for name, array, selected in fields
    )
