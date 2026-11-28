"""Frozen actor/value inference with bounded, weight-aware frame caches."""

import numpy as np
import torch

from cr_rl.models import ActorCritic, HistoryValueNetwork
from cr_rl.models.frame_cache import FrameFeatureCache
from cr_rl.training.separate import autocast_for


def history_tensors(windows, time_mask, masks, adapter, device):
    header, entities, entity_mask = adapter(windows)
    raw, schema = windows[:, -1], adapter.schema
    values = (
        header,
        entities,
        entity_mask,
        masks.astype(bool),
        time_mask,
        raw[:, list(schema.header["hand_card"])],
        raw[:, list(schema.header["deck"])],
        raw[:, list(schema.header["enemy_seen"])],
    )
    return tuple(torch.as_tensor(value, device=device) for value in values)


class FrozenInference:
    def __init__(self, model, adapter, device, capacity=None, *, forced_wait_fastpath=False):
        self.model, self.adapter, self.device = model.eval(), adapter, device
        self.cache = FrameFeatureCache(model, adapter, device, capacity=capacity)
        self.incremental_cache = None
        self.forced_wait_fastpath = forced_wait_fastpath
        self.forced_wait_batches = self.forced_wait_rows = 0

    def _forced_wait_logits(self, legal_masks, rows):
        if not self.forced_wait_fastpath or not isinstance(self.model, ActorCritic):
            return None
        if self.model.cfg.action_count != 2307:
            return None
        if self.model.training:
            raise ValueError("Frozen inference requires eval mode")
        if legal_masks is None:
            return None
        legal = np.asarray(legal_masks, dtype=bool)
        if legal.shape != (rows, self.model.cfg.action_count):
            raise ValueError("One complete legal action mask is required per history")
        if not rows or not legal[:, 0].all() or legal[:, 1:].any():
            return None
        # Only probabilities on legal actions are observable here. Invalid
        # logits remain finite and underflow to zero, as in the regular codec.
        # sample()/sample_incremental() still draw the same full-shape Categorical
        # so skipping a deterministic turn does not change future sampling RNG.
        logits = torch.full(legal.shape, -1e9, dtype=torch.float32, device=self.device)
        logits[:, 0] = 0
        self.forced_wait_batches += 1
        self.forced_wait_rows += rows
        return logits

    @torch.inference_mode()
    def evaluate_incremental(self, batch, legal_masks=None):
        from cr_rl.models.incremental_frames import IncrementalFrameEncoder

        if batch.generation != batch.history.generation:
            raise ValueError("History request became stale after another append")
        if batch.history.context_len != self.model.cfg.context_len:
            raise ValueError("Rolling history length differs from the model")
        if batch.current.shape != (len(batch.streams), self.adapter.schema.seat_ints):
            raise ValueError("Current observation shape differs from the schema")
        forced = self._forced_wait_logits(legal_masks, len(batch.streams))
        if forced is not None:
            # The caller keeps appending every raw frame to its ring. At the next
            # real choice, encode() catches up on all missing in-window frames.
            return forced
        if self.incremental_cache is None:
            self.incremental_cache = IncrementalFrameEncoder(self.model, self.adapter, self.device)
        with autocast_for(self.device):
            encoded = self.incremental_cache.encode(batch)
            if isinstance(self.model, HistoryValueNetwork):
                return self.model.critic(encoded[0]).squeeze(-1).float()
            if legal_masks is None:
                raise ValueError("Actor inference requires current legal masks")
            header, entities, mask = self.adapter(batch.current)
            schema = self.adapter.schema
            return self.model(
                torch.as_tensor(header, device=self.device),
                torch.as_tensor(entities, device=self.device),
                torch.as_tensor(mask, device=self.device),
                torch.as_tensor(legal_masks, dtype=torch.bool, device=self.device),
                hand_card_ids=torch.as_tensor(
                    batch.current[:, list(schema.header["hand_card"])], device=self.device
                ),
                encoded_history=encoded,
            )[0].float()

    @torch.inference_mode()
    def sample_incremental(self, batch, legal_masks, *, collect_stats=True):
        logits = self.evaluate_incremental(batch, legal_masks)
        return self._sample_logits(logits, collect_stats=collect_stats)

    @staticmethod
    def _sample_logits(logits, *, collect_stats):
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        return (
            actions.cpu().numpy().astype(np.int32),
            distribution.log_prob(actions).cpu().numpy(),
            {
                "entropy": float(distribution.entropy().mean()),
                "max_action_probability": float(distribution.probs.max(-1).values.mean()),
            }
            if collect_stats
            else {},
        )

    @torch.inference_mode()
    def evaluate(self, windows, time_mask, legal_masks=None):
        if self.forced_wait_fastpath:
            if windows.ndim != 3 or windows.shape[-1] != self.adapter.schema.seat_ints:
                raise ValueError("History observation shape differs from the schema")
            if time_mask.shape != windows.shape[:2]:
                raise ValueError("Time mask must match the history window")
            if not windows.shape[1] or not time_mask[:, -1].all():
                raise ValueError("History must end in a valid current frame")
            forced = self._forced_wait_logits(legal_masks, len(windows))
            if forced is not None:
                return forced
        with autocast_for(self.device):
            encoded = self.cache.encode(windows, time_mask)
            if isinstance(self.model, HistoryValueNetwork):
                return self.model.critic(encoded[0]).squeeze(-1).float()
            if legal_masks is None:
                raise ValueError("Actor inference requires the current legal action mask")
            raw = windows[:, -1]
            header, entities, entity_mask = self.adapter(raw)
            schema = self.adapter.schema
            return self.model(
                torch.as_tensor(header, device=self.device),
                torch.as_tensor(entities, device=self.device),
                torch.as_tensor(entity_mask, device=self.device),
                torch.as_tensor(legal_masks, dtype=torch.bool, device=self.device),
                hand_card_ids=torch.as_tensor(
                    raw[:, list(schema.header["hand_card"])], device=self.device
                ),
                encoded_history=encoded,
            )[0].float()

    @torch.inference_mode()
    def sample(self, windows, time_mask, legal_masks, *, collect_stats=True):
        logits = self.evaluate(windows, time_mask, legal_masks)
        return self._sample_logits(logits, collect_stats=collect_stats)


class StateInference:
    """Frozen architecture-v3 inference with one explicit memory per stream."""

    def __init__(self, model, adapter, device, streams, *, compute_value: bool = True, precision: str | None = None):
        if getattr(model, "architecture_version", None) != 3:
            raise ValueError("StateInference requires architecture-v3")
        self.model = model.eval()
        self.adapter = adapter
        self.device = torch.device(device)
        self.streams = int(streams)
        self.compute_value = bool(compute_value)
        self.precision = precision
        self._hand_indices = np.asarray(adapter.schema.header["hand_card"], dtype=np.int64)
        self.state = model.initial_state(self.streams, device=self.device)

    def _autocast(self):
        if self.precision == "fp32":
            return torch.autocast(device_type=self.device.type, enabled=False)
        if self.precision == "fp16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        return autocast_for(self.device)

    @torch.inference_mode()
    def evaluate(self, raw, legal_masks, *, compute_value: bool | None = None):
        raw = np.asarray(raw)
        if raw.shape != (self.streams, self.adapter.schema.seat_ints):
            raise ValueError("State inference observation shape differs from stream count")
        header, entities, entity_mask = self.adapter(raw)
        hand = raw[:, self._hand_indices]
        use_value = self.compute_value if compute_value is None else bool(compute_value)
        with self._autocast():
            logits, values, self.state = self.model.step(
                torch.as_tensor(header, device=self.device),
                torch.as_tensor(entities, device=self.device),
                torch.as_tensor(entity_mask, device=self.device),
                legal_action_mask=torch.as_tensor(legal_masks, dtype=torch.bool, device=self.device),
                state=self.state,
                hand_card_ids=torch.as_tensor(hand, device=self.device),
                compute_value=use_value,
            )
        return logits.float(), values.float()

    @torch.inference_mode()
    def sample(self, raw, legal_masks, *, collect_stats: bool = True, compute_value: bool | None = None):
        logits, values = self.evaluate(raw, legal_masks, compute_value=compute_value)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        stats = (
            {
                "entropy": float(distribution.entropy().mean()),
                "max_action_probability": float(distribution.probs.max(-1).values.mean()),
            }
            if collect_stats
            else {}
        )
        return (
            actions.cpu().numpy().astype(np.int32),
            distribution.log_prob(actions).cpu().numpy(),
            values.cpu().numpy(),
            stats,
        )

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.streams, dtype=torch.bool, device=self.device)
        self.state = type(self.state)(
            self.state.temporal.reset(torch.as_tensor(mask, device=self.device)),
            id(self.model),
            self.model.memory_generation,
        )
