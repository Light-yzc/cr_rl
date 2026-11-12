"""Release resume mappings and unused host allocations without changing training state."""

import ctypes
import gc
import sys

import torch


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
