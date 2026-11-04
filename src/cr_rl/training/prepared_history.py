"""Decode immutable rollout inputs once, while recomputing trainable features.

Only raw normalized observations are cached. Entity/temporal embeddings always
participate in the current forward/backward pass; no stale training features.
"""

import time

import numpy as np
import torch


class PreparedHistoryBatches:
    def __init__(self, refs, context_len, adapter, device, cuda_budget_gib=2):
        began = time.perf_counter()
        if not refs or context_len < 1:
            raise ValueError("Nonempty references and a positive context length are required")
        self.device = torch.device(device)
        self.adapter = adapter
        groups = {}
        for trace, position, seat in refs:
            if not 0 <= position < len(trace.observations):
                raise IndexError("Transition is outside its episode")
            key = id(trace), seat
            if key not in groups:
                groups[key] = [trace, position, position]
            else:
                groups[key][1] = min(groups[key][1], position)
                groups[key][2] = max(groups[key][2], position)
        offsets, total = {}, 1  # Index zero is the same raw-zero left padding as EpisodeTrace.
        for key, (_trace, first, last) in groups.items():
            first = max(0, first - context_len + 1)
            offsets[key] = total, first
            total += last - first + 1
        raw = np.zeros((total, adapter.schema.seat_ints), dtype=refs[0][0].observations[0].dtype)
        for key, (trace, _, last) in groups.items():
            offset, first = offsets[key]
            raw[offset : offset + last - first + 1] = np.stack(
                trace.observations[first : last + 1]
            )[:, key[1]]
        windows = np.zeros((len(refs), context_len), dtype=np.int64)
        legal = np.stack([trace.masks[position][seat] for trace, position, seat in refs])
        delta = np.arange(1 - context_len, 1)
        for index, (trace, position, seat) in enumerate(refs):
            offset, first = offsets[id(trace), seat]
            positions = position + delta
            windows[index] = np.where(positions >= 0, positions - first + offset, 0)
        count_index = adapter.schema.header["entity_count"][0]
        counts = np.clip(raw[:, count_index], 0, adapter.schema.max_entities)
        self.limits = counts[windows].max(axis=1)
        if adapter.entity_bucket:
            self.limits = np.minimum(
                adapter.schema.max_entities,
                ((np.maximum(1, self.limits) + adapter.entity_bucket - 1) // adapter.entity_bucket)
                * adapter.entity_bucket,
            )
        else:
            self.limits[:] = adapter.schema.max_entities
        header, entities, entity_mask = adapter(raw)
        current = raw[windows[:, -1]]
        arrays = [
            header,
            entities,
            entity_mask,
            legal.astype(bool),
            windows,
            current[:, list(adapter.schema.header["hand_card"])],
            current[:, list(adapter.schema.header["deck"])],
            current[:, list(adapter.schema.header["enemy_seen"])],
        ]
        nbytes = sum(value.nbytes for value in arrays)
        # Keep headroom for parameters, optimizer states and larger training batches.
        use_cuda = self.device.type == "cuda" and nbytes <= cuda_budget_gib * 2**30
        if use_cuda:
            free, _ = torch.cuda.mem_get_info(self.device)
            use_cuda = free - nbytes >= 8 * 2**30
        self.storage_device = self.device if use_cuda else torch.device("cpu")
        self.arrays = tuple(torch.as_tensor(value, device=self.storage_device) for value in arrays)
        self.metadata = dict(
            unique_frames=total - 1,
            samples=len(refs),
            bytes=nbytes,
            storage_device=str(self.storage_device),
            seconds=time.perf_counter() - began,
        )

    def __call__(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        selected = torch.as_tensor(indices, device=self.storage_device)
        header, entities, entity_mask, legal, windows, hand, deck, seen = self.arrays
        ids = windows[selected]
        limit = int(self.limits[indices].max())
        values = (
            header[ids],
            entities[:, :limit][ids],
            entity_mask[:, :limit][ids],
            legal[selected],
            ids != 0,
            hand[selected],
            deck[selected],
            seen[selected],
        )
        return tuple(value.to(self.device, non_blocking=True) for value in values)
