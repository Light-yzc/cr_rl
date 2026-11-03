"""Bounded inference-only cache before temporal position encoding.

The temporal Transformer still recomputes the exact rolling window. This is not
a rolling KV cache. Parameter version changes invalidate all frame features.
"""

from collections import OrderedDict

import numpy as np
import torch


class FrameFeatureCache:
    def __init__(self, model, adapter, device, capacity=None, encode_batch=16):
        self.automatic_capacity = capacity is None
        if capacity is None:
            capacity = max(512, model.cfg.context_len)
        if capacity < model.cfg.context_len or encode_batch < 1:
            raise ValueError("Cache must hold a full history; encode_batch must be positive")
        self.model, self.adapter, self.device = model, adapter, device
        self.capacity, self.encode_batch = capacity, encode_batch
        self.entries = OrderedDict()
        self.version = None
        self.encoded_frames = 0
        self.invalidations = 0

    def clear(self):
        self.entries.clear()
        self.version = None

    @torch.inference_mode()
    def encode(self, windows, time_mask):
        if self.model.training:
            raise ValueError("Frame caching requires eval mode")
        if windows.shape[:2] != time_mask.shape or not time_mask[:, -1].all():
            raise ValueError("Cache expects left-padded histories ending in a valid current frame")
        if self.automatic_capacity:
            # All lanes' rolling histories plus their next frames must fit.
            # A fixed 512-frame cache thrashes once B*T exceeds 512.
            self.capacity = max(self.capacity, windows.shape[0] * (windows.shape[1] + 1))
        version = tuple(p._version for p in self.model.parameters())
        if self.version != version:
            self.entries.clear()
            self.version = version
            self.invalidations += 1
        needed = {}
        keys = {}
        for batch, step in np.argwhere(time_mask):
            key = np.ascontiguousarray(windows[batch, step]).tobytes()
            keys[int(batch), int(step)] = key
            if key not in self.entries:
                needed[key] = windows[batch, step]
        # Keep all features used by this call alive even if the bounded LRU evicts
        # earlier frames while filling a larger multi-environment request.
        active = {key: self.entries[key] for key in keys.values() if key in self.entries}
        missing = list(needed)
        for left in range(0, len(missing), self.encode_batch):
            selected = missing[left : left + self.encode_batch]
            raw = np.stack([needed[key] for key in selected])
            h, e, mask = self.adapter(raw)
            features, tokens = self.model.encode_frames(
                torch.as_tensor(h, device=self.device),
                torch.as_tensor(e, device=self.device),
                torch.as_tensor(mask, device=self.device),
            )
            for i, key in enumerate(selected):
                count = max(1, int(mask[i].sum()))
                value = (
                    features[i].clone(),
                    tokens[i, :count].clone(),
                    torch.as_tensor(mask[i, :count], device=self.device).clone(),
                )
                self.entries[key] = active[key] = value
                self.encoded_frames += 1
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
        batch_size, steps = time_mask.shape
        first = next(iter(active.values()))[0]
        zero = first.new_zeros(self.model.cfg.entity_dim)
        frames = torch.stack(
            [
                active[keys[batch, step]][0] if (batch, step) in keys else zero
                for batch in range(batch_size)
                for step in range(steps)
            ]
        ).reshape(batch_size, steps, -1)
        for key in keys.values():
            if key in self.entries:
                self.entries.move_to_end(key)
        current = [active[keys[batch, steps - 1]] for batch in range(batch_size)]
        memory = torch.nn.utils.rnn.pad_sequence([value[1] for value in current], batch_first=True)
        memory_mask = torch.nn.utils.rnn.pad_sequence(
            [value[2] for value in current], batch_first=True
        )
        raw = windows[:, -1]
        schema = self.adapter.schema
        context = self.model.encode_temporal(
            frames,
            torch.as_tensor(time_mask, device=self.device),
            deck_card_ids=torch.as_tensor(raw[:, list(schema.header["deck"])], device=self.device),
            enemy_seen_card_ids=torch.as_tensor(
                raw[:, list(schema.header["enemy_seen"])], device=self.device
            ),
        )
        return context, memory, memory_mask
