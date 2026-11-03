"""Frozen actor/value inference with bounded, weight-aware frame caches."""

import numpy as np
import torch

from cr_rl.models import HistoryValueNetwork
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
    def __init__(self, model, adapter, device, capacity=None):
        self.model, self.adapter, self.device = model.eval(), adapter, device
        self.cache = FrameFeatureCache(model, adapter, device, capacity=capacity)

    @torch.inference_mode()
    def evaluate(self, windows, time_mask, legal_masks=None):
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
    def sample(self, windows, time_mask, legal_masks):
        logits = self.evaluate(windows, time_mask, legal_masks)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        return (
            actions.cpu().numpy().astype(np.int32),
            distribution.log_prob(actions).cpu().numpy(),
            {
                "entropy": float(distribution.entropy().mean()),
                "max_action_probability": float(distribution.probs.max(-1).values.mean()),
            },
        )
