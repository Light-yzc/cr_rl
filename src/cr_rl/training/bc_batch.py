"""One BC history-batch builder shared by training and throughput tuning."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch


def attention_context(backend):
    if backend == "auto":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


def build_bc_batch(
    rows, obs, masks, actions, replay_ids, seats, groups, positions, adapter, context, device
):
    schema = adapter.schema
    windows = np.zeros((len(rows), context, schema.seat_ints), dtype=np.int32)
    time_mask = np.zeros((len(rows), context), dtype=np.bool_)
    for j, row in enumerate(rows):
        key = (int(replay_ids[row]), int(seats[row]))
        indices = groups[key]
        pos = positions[key][int(row)]
        seq = indices[max(0, pos - context + 1) : pos + 1]
        pad = context - len(seq)
        windows[j, pad:] = obs[seq]
        time_mask[j, pad:] = True
    header, entities, entity_mask = adapter(windows)
    current = obs[rows]
    values = (
        header,
        entities,
        entity_mask,
        masks[rows].astype(bool),
        time_mask,
        current[:, list(schema.header["hand_card"])],
        actions[rows],
        current[:, list(schema.header["deck"])],
        current[:, list(schema.header["enemy_seen"])],
    )
    return tuple(torch.as_tensor(v, device=device) for v in values)
