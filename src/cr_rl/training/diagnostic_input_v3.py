"""Local-only input ablations; never silently reinterpret a production checkpoint.

Install after loading the original weights and before making the optimizer.
The categorical ablation deliberately changes the old raw-header path: a zero
new projection does NOT make this an output-preserving migration. This module
is for controlled fit probes, not for checkpoint publication or normal resume.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn


VARIANTS = ("bounded_timers", "categorical_headers", "both")
CARD_GROUP_NAMES = (
    "hand_card", "next_card", "deck", "enemy_seen", "pending_card",
    "ability_champion", "enemy_play_card",
)
HEADER_ID_NAMES = ("queue", "enemy_play_cast")
TIMER_NAMES = ("buff_ms_0", "buff_ms_1", "buff_ms_2")


def bounded_buff_times(entities, entity_fields, cap_ms=360000.0):
    """Cap only the three adapted buff durations; retain negative unknowns.

    CrForgeObservationAdapter has already divided these durations by 10000.
    A 360000ms cap therefore means 36 in the model tensor. This is a bounded
    duration ablation, not a claim that every long effect is semantically equal.
    """
    if cap_ms <= 0:
        raise ValueError("Buff timer cap must be positive")
    columns = [i for i, name in enumerate(entity_fields) if name in TIMER_NAMES]
    if not columns:
        raise ValueError("The diagnostic requires buff duration schema fields")
    result = entities.clone()
    result[..., columns] = result[..., columns].clamp_max(cap_ms / 10000.0)
    return result


def _card_context(model, header):
    """Exact existing readout-v2 card context, before diagnostic additions."""
    summaries = []
    for name, indices in zip(CARD_GROUP_NAMES, model.card_groups, strict=True):
        raw = header[..., list(indices)].float()
        ids = (raw * model.cfg.vocab_size).round().long() + 1
        valid = (raw >= 0) & (ids <= model.cfg.vocab_size)
        ids = torch.where(valid, ids, 0)
        embedded = model.card_embedding(ids)
        pooled = (embedded * valid.unsqueeze(-1)).sum(-2)
        pooled = pooled / valid.sum(-1, keepdim=True).clamp_min(1)
        if model.cfg.readout_version == 2 and name in model.ordered_card_context:
            ordered = (embedded * valid.unsqueeze(-1)).flatten(-2)
            pooled = pooled + model.ordered_card_context[name](ordered)
        summaries.append(pooled)
    return model.early_card_context(torch.cat(summaries, dim=-1))


def install_diagnostic_input_variant(model, variant, *, timer_cap_ms=360000.0):
    """Attach one explicit, instance-local diagnostic path and return its protocol.

    New parameters have the ``diagnostic_`` prefix and are automatically included
    in make_bc_optimizer's backbone group. No weights are removed or changed.
    Existing production encoder classes and ModelConfig are left untouched.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown diagnostic input variant: {variant}")
    if getattr(model, "diagnostic_input_variant", None):
        raise ValueError("Install exactly one input diagnostic per model instance")
    if model.cfg.representation_version != 2:
        raise ValueError("Input diagnostics require the representation-v2 schema")
    timer = variant in ("bounded_timers", "both")
    categorical = variant in ("categorical_headers", "both")
    fields = tuple(model.cfg.header_fields)
    id_columns = [i for i, name in enumerate(fields) if name in HEADER_ID_NAMES]
    handle_columns = [i for i, name in enumerate(fields) if name == "ability_unit"]
    timer_columns = [i for i, name in enumerate(model.cfg.entity_fields) if name in TIMER_NAMES]
    if timer and (timer_cap_ms <= 0 or not timer_columns):
        raise ValueError("Positive cap and buff timer schema fields required")
    if categorical:
        if not id_columns:
            raise ValueError("No queue/enemy_play_cast columns in this checkpoint")
        weight = model.card_embedding.weight
        model.diagnostic_header_projection = nn.Linear(
            len(id_columns) * model.cfg.entity_dim, model.cfg.entity_dim,
            bias=False, device=weight.device, dtype=weight.dtype,
        )
        # The new path gets gradients immediately, while its shared embeddings
        # begin receiving extra-path gradients after this projection moves.
        nn.init.zeros_(model.diagnostic_header_projection.weight)
    original_encode = model._encode_frames

    def encode(self, header, entities, entity_mask):
        if timer:
            entities = bounded_buff_times(entities, self.cfg.entity_fields, timer_cap_ms)
        if not categorical:
            return original_encode(header, entities, entity_mask)
        context = _card_context(self, header)
        raw_ids = header[..., id_columns].float()
        rounded = raw_ids.round().long()
        valid = (raw_ids >= 0) & (rounded >= 0) & (rounded < self.cfg.vocab_size)
        indices = torch.where(valid, rounded + 1, 0)
        embedded = self.card_embedding(indices)
        ordered = (embedded * valid.unsqueeze(-1)).flatten(-2)
        context = context + self.diagnostic_header_projection(ordered)
        numeric = header.clone()
        numeric[..., [i for group in self.card_groups for i in group]] = 0
        numeric[..., id_columns] = 0
        if handle_columns:
            numeric[..., handle_columns] = (header[..., handle_columns] >= 0).to(numeric.dtype)
        return self.frame(numeric, entities, entity_mask, context)

    model._encode_frames = MethodType(encode, model)
    model.diagnostic_input_variant = variant
    return {
        "diagnostic_only": True,
        "variant": variant,
        "output_preserving_migration": False if categorical else "only when no timer exceeds cap",
        "buff_timer_cap_ms": timer_cap_ms if timer else None,
        "buff_timer_columns": timer_columns if timer else [],
        "categorical_header_columns": id_columns if categorical else [],
        "categorical_header_fields": [fields[i] for i in id_columns] if categorical else [],
        "ability_handle_presence_columns": handle_columns if categorical else [],
        "new_parameters": sum(p.numel() for n, p in model.named_parameters() if n.startswith("diagnostic_")),
    }
