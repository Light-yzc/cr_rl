"""Instance-local action-query ablation for frozen BC checkpoint comparisons.

The old frame/history/tile paths are retained. Install only after old weights
are loaded and before constructing the diagnostic optimizer. The new last
projection starts at zero, preserving the original action distribution at
installation (subject to the caller's ordinary floating-point tolerance).
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn


def install_diagnostic_action_readout(model):
    """Add seven learned action queries over current entity tokens.

    WAIT and the two ability queries have learned role embeddings. Four hand
    queries additionally receive the same card/slot features as the old slot
    head. All queries receive projected history context. A query residual keeps
    role/context information when all entity values are identical (empty or
    uniform boards), and avoids a shared-logit dead branch in that case.

    The branch runs without autocast in its parameter dtype. With the current
    FP32-parameter BC path this is an FP32 safety boundary, like the tile head;
    it does not alter autocast in the frame or temporal Transformer.
    """
    if getattr(model, "diagnostic_action_readout_installed", False):
        raise ValueError("Install the action-query diagnostic exactly once")
    cfg = model.cfg
    if cfg.action_count != 2307:
        raise ValueError("This diagnostic requires WAIT + four cards + two abilities")
    weight = model.card_embedding.weight
    factory = {"device": weight.device, "dtype": weight.dtype}
    before = sum(p.numel() for p in model.parameters())
    model.diagnostic_action_queries = nn.Parameter(torch.empty(1, 7, cfg.entity_dim, **factory))
    nn.init.normal_(model.diagnostic_action_queries, std=0.02)
    model.diagnostic_action_context = nn.Linear(cfg.temporal_dim, cfg.entity_dim, **factory)
    model.diagnostic_action_attention = nn.MultiheadAttention(
        cfg.entity_dim, cfg.entity_heads, batch_first=True, dropout=0.0, **factory
    )
    model.diagnostic_action_output = nn.Sequential(
        nn.LayerNorm(cfg.entity_dim, **factory),
        nn.Linear(cfg.entity_dim, cfg.entity_dim // 2, **factory),
        nn.GELU(),
        nn.Linear(cfg.entity_dim // 2, 1, **factory),
    )
    nn.init.zeros_(model.diagnostic_action_output[-1].weight)
    nn.init.zeros_(model.diagnostic_action_output[-1].bias)

    def forward(
        self, header, entities, entity_mask, legal_action_mask=None, time_mask=None,
        target_index=-1, hand_card_ids=None, deck_card_ids=None,
        enemy_seen_card_ids=None, compute_value=True, encoded_history=None,
    ):
        batch = entities.shape[0]
        if encoded_history is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Cached history is only valid for frozen inference")
            context, current_entities, current_entity_mask = encoded_history
        else:
            context, current_entities, current_entity_mask = self.encode_history(
                header, entities, entity_mask, time_mask, target_index,
                deck_card_ids, enemy_seen_card_ids,
            )
        if hand_card_ids is None:
            hand_card_ids = torch.full((batch, 4), -1, dtype=torch.long, device=context.device)
        card_indices = (hand_card_ids.long() + 1).clamp(0, self.cfg.vocab_size)
        card_features = self.card_embedding(card_indices) + self.card_slot_embedding
        context_slots = context.unsqueeze(1).expand(-1, 4, -1)
        slot_logits = self.slot_actor(torch.cat((context_slots, card_features), dim=-1)).squeeze(-1)
        card_logits = torch.cat(
            (self.wait_actor(context), slot_logits, self.ability_actor(context)), dim=-1
        )
        dtype = self.diagnostic_action_context.weight.dtype
        with torch.autocast(device_type=context.device.type, enabled=False):
            queries = self.diagnostic_action_queries.expand(batch, -1, -1)
            queries = queries + self.diagnostic_action_context(context.to(dtype))[:, None, :]
            card_condition = queries.new_zeros(batch, 7, self.cfg.entity_dim)
            card_condition[:, 1:5] = card_features.to(dtype)
            queries = queries + card_condition
            features, _ = self.diagnostic_action_attention(
                queries, current_entities.to(dtype), current_entities.to(dtype),
                key_padding_mask=~current_entity_mask.bool(), need_weights=False,
            )
            delta = self.diagnostic_action_output(features + queries).squeeze(-1)
        card_logits = card_logits + delta
        tile_logits = self.placement_logits(
            context, current_entities, current_entity_mask, card_features
        )
        if legal_action_mask is None:
            legal_action_mask = torch.ones(
                batch, self.cfg.action_count, dtype=torch.bool, device=context.device
            )
        joint_log_probs = self._joint_log_probs(card_logits, tile_logits, legal_action_mask)
        value = self.critic(context).squeeze(-1) if compute_value else context.new_zeros(batch)
        return joint_log_probs, value

    model.forward = MethodType(forward, model)
    model.diagnostic_action_readout_installed = True
    return {
        "diagnostic_only": True,
        "variant": "action_query_readout",
        "queries": 7,
        "query_dimension": cfg.entity_dim,
        "new_parameters": sum(p.numel() for p in model.parameters()) - before,
        "output_preserving_at_install": True,
        "new_output_initialization": "zero weight and bias; no additional zero gate",
        "query_residual": True,
        "branch_parameter_dtype": str(weight.dtype),
        "branch_autocast": False,
        "branch_fp32": weight.dtype == torch.float32,
        "old_frame_temporal_tile_parameters_changed": False,
    }
