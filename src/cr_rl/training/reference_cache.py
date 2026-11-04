"""Lazy, round-local cache of frozen reference outputs, never trainable features."""

import numpy as np
import torch


class ReferenceDistributionCache:
    def __init__(self, capacity, max_bytes=128 * 1024**2):
        if capacity < 1 or max_bytes < 1:
            raise ValueError("Positive cache capacity and budget required")
        self.capacity, self.max_bytes = capacity, max_bytes
        self.valid = np.zeros(capacity, dtype=bool)
        self.values = None
        self.forward_calls = self.hit_rows = 0
        self.disabled = False

    @torch.no_grad()
    def evaluate(self, indices, compute):
        indices = np.asarray(indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or not len(indices)
            or (indices < 0).any()
            or (indices >= self.capacity).any()
        ):
            raise IndexError("Reference rows are outside the immutable rollout")
        if self.values is not None and self.valid[indices].all():
            self.hit_rows += len(indices)
            return self.values[torch.as_tensor(indices, device=self.values.device)]
        # Misses use the original complete microbatch and ordering; never eagerly
        # run an entire rollout that may be discarded by an early KL stop.
        result = compute().detach().float()
        self.forward_calls += 1
        if result.ndim != 2 or len(result) != len(indices):
            raise ValueError("Frozen reference outputs must be [samples, actions]")
        if self.values is None and not self.disabled:
            self.disabled = self.capacity * result.shape[1] * result.element_size() > self.max_bytes
            if not self.disabled:
                self.values = result.new_empty((self.capacity, result.shape[1]))
        if self.values is not None:
            self.values[torch.as_tensor(indices, device=result.device)] = result
            self.valid[indices] = True
        return result
