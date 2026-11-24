"""Release resume mappings and unused host allocations without changing training state."""

import ctypes
import gc
import hashlib
import io
import os
import pickle
import sys
import shutil
from pathlib import Path

import torch
import torch.serialization as _ts


def cache_frozen_checkpoint(source: Path, directory: Path, *, max_bytes: int) -> Path:
    """Bounded local read cache for immutable opponents; originals stay intact."""
    source = source.resolve()
    stat = source.stat()
    if stat.st_size > max_bytes:
        return source
    directory.mkdir(parents=True, exist_ok=True)
    identity = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
    target = directory / (hashlib.sha256(identity.encode()).hexdigest() + ".pt")
    if target.exists() and target.stat().st_size == stat.st_size:
        os.utime(target, None)
        return target
    files = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime_ns)
    total = sum(p.stat().st_size for p in files)
    for stale in files:
        if total + stat.st_size <= max_bytes:
            break
        total -= stale.stat().st_size
        stale.unlink()
    # Leave filesystem headroom; an optional cache must not fill a data volume.
    if shutil.disk_usage(directory).free < stat.st_size + 2 * 1024**3:
        return source
    temporary = target.with_suffix(".tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    except OSError:
        temporary.unlink(missing_ok=True)
        return source
    return target


def detach_optimizer_counters(optimizer):
    # Adam's CPU scalar steps otherwise retain the entire checkpoint mmap.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.ndim == 0:
                state[key] = value.clone()


def release_host_memory():
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass


def streaming_torch_save(obj, path: Path, *, fsync: bool = False) -> None:
    """Write a standard CPU-tagged PyTorch zip checkpoint by streaming one tensor storage at a time.

    Avoids materializing multi-GiB model + optimizer state dicts in host RAM all at once.
    """
    serialized_storages: dict[str, torch.UntypedStorage] = {}
    id_map: dict[int, str] = {}
    storage_dtypes: dict[int, torch.dtype] = {}

    def persistent_id(item):
        if isinstance(item, torch.storage.TypedStorage) or torch.is_storage(item):
            if isinstance(item, torch.storage.TypedStorage):
                storage = item._untyped_storage
                storage_dtype = item.dtype
                storage_type = getattr(torch, item._pickle_storage_type())
                storage_numel = item._size()
            else:
                storage = item
                storage_dtype = torch.uint8
                storage_type = _ts.normalize_storage_type(type(item))
                storage_numel = storage.nbytes()

            if str(storage.device) != "meta" and storage.data_ptr() != 0:
                prev_dtype = storage_dtypes.get(storage.data_ptr())
                if prev_dtype is not None and prev_dtype != storage_dtype:
                    raise RuntimeError(
                        "Cannot save multiple tensors or storages that view the same data as different types"
                    )
                storage_dtypes.setdefault(storage.data_ptr(), storage_dtype)

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            serialized_storages[storage_key] = storage
            return ("storage", storage_type, storage_key, "cpu", storage_numel)
        return None

    data_buf = io.BytesIO()

    class _Pickler(pickle.Pickler):
        def persistent_id(self, item):
            return persistent_id(item)

    _Pickler(data_buf, protocol=_ts.DEFAULT_PROTOCOL).dump(obj)
    data_value = data_buf.getvalue()
    del data_buf

    with path.open("wb", buffering=8 * 1024 * 1024) as stream:
        with _ts._open_zipfile_writer(stream) as zip_file:
            zip_file.write_record("data.pkl", data_value, len(data_value))
            del data_value
            zip_file.write_record("byteorder", sys.byteorder, len(sys.byteorder))
            for key in sorted(serialized_storages.keys()):
                storage = serialized_storages.pop(key)
                num_bytes = storage.nbytes()
                cpu_storage = (
                    storage.cpu()
                    if (storage.device.type not in ("cpu", "meta") and num_bytes > 0)
                    else storage
                )
                zip_file.write_record(f"data/{key}", cpu_storage, num_bytes)
                del cpu_storage, storage
        if fsync:
            stream.flush()
            os.fsync(stream.fileno())
    storage_dtypes.clear()
    id_map.clear()


def streaming_torch_load(path: Path, *, target_device: torch.device | str | None = None, min_gpu_bytes: int = 65536):
    """Load a checkpoint while streaming large tensor storages directly to `target_device` VRAM.

    Avoids faulting multi-gigabyte checkpoints into host page-cache RSS (`VmHWM`) all at once.
    """
    dev = None if target_device is None else torch.device(target_device)
    if dev is None or dev.type != "cuda":
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)

    def _map_to_vram(storage, _location):
        if storage.nbytes() <= min_gpu_bytes:
            return storage
        return storage.to(device=dev)

    return torch.load(path, map_location=_map_to_vram, mmap=False, weights_only=False)
