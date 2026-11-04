"""Inference-only keyed ring buffers; temporal windows are still fully recomputed."""

from dataclasses import dataclass

import numpy as np
import torch


class IncrementalFrameHistory:
    def __init__(self, streams, context_len, seat_ints):
        self.streams, self.context_len = streams, context_len
        self.raw = np.empty((streams, context_len, seat_ints), dtype=np.int32)
        self.episodes = np.full(streams, -1, dtype=np.int64)
        self.positions = np.full(streams, -1, dtype=np.int64)
        self.generation = 0

    def append(self, raw, episodes):
        raw, episodes = np.asarray(raw), np.asarray(episodes, dtype=np.int64)
        if raw.shape != (self.streams, self.raw.shape[-1]) or episodes.shape != (self.streams,):
            raise ValueError("One current frame and episode ID are required for each stream")
        positions = np.where(episodes == self.episodes, self.positions + 1, 0)
        self.raw[np.arange(self.streams), positions % self.context_len] = raw
        self.episodes[:] = episodes
        self.positions[:] = positions
        self.generation += 1

    def request(self, streams=None, *, following=None, episodes=None):
        streams = (
            np.arange(self.streams) if streams is None else np.asarray(streams, dtype=np.int64)
        )
        if following is None:
            if (self.positions[streams] < 0).any():
                raise ValueError("Append a real frame before requesting history")
            positions = self.positions[streams].copy()
            episode_ids = self.episodes[streams].copy()
            current = self.raw[streams, positions % self.context_len]
        else:
            if episodes is None:
                raise ValueError("Preview requires actual current episode identities")
            episode_ids = np.asarray(episodes, dtype=np.int64)[streams]
            positions = np.where(
                episode_ids == self.episodes[streams], self.positions[streams] + 1, 0
            )
            current = np.asarray(following)[streams]
        indices = positions[:, None] + np.arange(1 - self.context_len, 1)
        return IncrementalFrameBatch(
            self, streams, episode_ids, indices, current, following is not None, self.generation
        )


@dataclass
class IncrementalFrameBatch:
    history: IncrementalFrameHistory
    streams: np.ndarray
    episodes: np.ndarray
    indices: np.ndarray
    current: np.ndarray
    preview: bool
    generation: int

    def select(self, indices):
        return IncrementalFrameBatch(
            self.history,
            self.streams[indices],
            self.episodes[indices],
            self.indices[indices],
            self.current[indices],
            self.preview,
            self.generation,
        )

    def frame(self, row, index):
        if self.preview and index == self.indices[row, -1]:
            return self.current[row]
        return self.history.raw[self.streams[row], index % self.history.context_len]


class IncrementalFrameEncoder:
    def __init__(self, model, adapter, device, encode_batch=16):
        self.model, self.adapter, self.device = model, adapter, device
        self.encode_batch = encode_batch
        self.version = None
        self.features = None
        self.current_memory = {}
        self.encoded_frames = 0

    @torch.inference_mode()
    def encode(self, batch):
        if self.model.training:
            raise ValueError("Incremental frame features require frozen eval inference")
        history = batch.history
        if batch.generation != history.generation:
            raise ValueError("History request became stale after another append")
        if history.context_len != self.model.cfg.context_len:
            raise ValueError("Rolling history length differs from the model")
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        precision = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else None
        )
        version = precision, tuple(p._version for p in self.model.parameters())
        if version != self.version or self.features is None:
            self.version = version
            self.features = None
            self.episodes = np.full((history.streams, history.context_len), -1, dtype=np.int64)
            self.positions = np.full_like(self.episodes, -1)
            self.current_memory.clear()
        missing = []
        for row, stream in enumerate(batch.streams):
            episode = int(batch.episodes[row])
            latest = int(batch.indices[row, -1])
            indices = batch.indices[row]
            slots = np.maximum(indices, 0) % history.context_len
            absent = (indices >= 0) & (
                (self.episodes[stream, slots] != episode)
                | (self.positions[stream, slots] != indices)
            )
            if self.current_memory.get(int(stream), (None,))[0] != (episode, latest):
                absent[-1] = True
            for index in indices[absent]:
                missing.append(
                    (row, int(stream), episode, int(index), int(index % history.context_len))
                )
        for left in range(0, len(missing), self.encode_batch):
            selected = missing[left : left + self.encode_batch]
            raw = np.stack([batch.frame(row, index) for row, _, _, index, _ in selected])
            header, entities, mask = self.adapter(raw)
            frames, tokens = self.model.encode_frames(
                torch.as_tensor(header, device=self.device),
                torch.as_tensor(entities, device=self.device),
                torch.as_tensor(mask, device=self.device),
            )
            streams = [item[1] for item in selected]
            slots = [item[4] for item in selected]
            if self.features is None:
                self.features = frames.new_zeros(
                    (history.streams, history.context_len, self.model.cfg.entity_dim)
                )
            self.features[streams, slots] = frames
            for i, (row, stream, episode, index, slot) in enumerate(selected):
                self.episodes[stream, slot], self.positions[stream, slot] = episode, index
                if index == batch.indices[row, -1]:
                    count = max(1, int(mask[i].sum()))
                    self.current_memory[stream] = (
                        (episode, index),
                        tokens[i, :count].clone(),
                        torch.as_tensor(mask[i, :count], device=self.device).clone(),
                    )
            self.encoded_frames += len(selected)
        valid = batch.indices >= 0
        stream_ids = torch.as_tensor(batch.streams[:, None], device=self.device)
        slots = torch.as_tensor(
            np.maximum(batch.indices, 0) % history.context_len, device=self.device
        )
        time_mask = torch.as_tensor(valid, device=self.device)
        frames = self.features[stream_ids, slots].masked_fill(~time_mask[..., None], 0)
        memory = [self.current_memory[int(stream)] for stream in batch.streams]
        entities = torch.nn.utils.rnn.pad_sequence([item[1] for item in memory], batch_first=True)
        entity_mask = torch.nn.utils.rnn.pad_sequence(
            [item[2] for item in memory], batch_first=True
        )
        schema = self.adapter.schema
        context = self.model.encode_temporal(
            frames,
            time_mask,
            deck_card_ids=torch.as_tensor(
                batch.current[:, list(schema.header["deck"])], device=self.device
            ),
            enemy_seen_card_ids=torch.as_tensor(
                batch.current[:, list(schema.header["enemy_seen"])], device=self.device
            ),
        )
        return context, entities, entity_mask
