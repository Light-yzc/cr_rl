from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.training.episode_trace import EpisodeTrace
from cr_rl.training.inference import history_tensors
from cr_rl.training.prepared_history import PreparedHistoryBatches


def fixture(bucket):
    header = ("entity_count", *(["hand_card"] * 4), *(["deck"] * 3), "enemy_seen")
    fields = ("id", "kind", "side", "row", "x", "y", "state", "flags", "buff_0", "buff_1", "buff_2")
    schema = SimpleNamespace(
        header_fields=header,
        entity_fields=fields,
        header_ints=len(header),
        entity_ints=len(fields),
        max_entities=5,
        seat_ints=len(header) + 5 * len(fields),
        header={
            name: tuple(i for i, field in enumerate(header) if field == name)
            for name in set(header)
        },
    )
    adapter = CrForgeObservationAdapter(schema, entity_bucket=bucket)
    traces = []
    for eid, length in [(0, 12), (1, 3)]:
        trace = EpisodeTrace(eid, eid + 1)
        for step in range(length):
            raw = np.zeros((2, schema.seat_ints), dtype=np.int32)
            raw[:, 0] = (step + eid) % 6
            raw[:, 1:9] = np.arange(8) + step + eid * 20
            raw[1, 2] = -1
            raw[:, schema.header_ints :] = np.arange(5 * len(fields)) + step
            mask = np.ones((2, 2307), dtype=np.uint8)
            mask[:, step + 1] = 0
            trace.append(raw, mask, np.zeros(2))
        traces.append(trace)
    return adapter, [
        (traces[0], 9, 0),
        (traces[0], 7, 0),
        (traces[1], 0, 1),
        (traces[1], 2, 1),
        (traces[0], 0, 1),
    ]


@pytest.mark.parametrize("bucket", [0, 2])
def test_prepared_inputs_match_original_exactly_across_episodes_padding_and_crops(bucket):
    adapter, refs = fixture(bucket)
    prepared = PreparedHistoryBatches(refs, 4, adapter, "cpu")
    for indices in [[0, 1], [2], [3, 0, 4], [4, 1, 1]]:
        windows, times, legal = [], [], []
        for index in indices:
            trace, position, seat = refs[index]
            w, t = trace.window(position, seat, 4)
            windows.append(w)
            times.append(t)
            legal.append(trace.masks[position][seat])
        expected = history_tensors(
            np.stack(windows), np.stack(times), np.stack(legal), adapter, "cpu"
        )
        for a, b in zip(prepared(indices), expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert a.requires_grad is False
    # Overlapping windows reuse decoded raw frames rather than caching a B*T expansion.
    assert prepared.metadata["unique_frames"] == 10


def test_prepared_rejects_empty_refs_and_caches_values_not_live_mutable_traces():
    adapter, refs = fixture(2)
    with pytest.raises(ValueError):
        PreparedHistoryBatches([], 4, adapter, "cpu")
    prepared = PreparedHistoryBatches(refs, 4, adapter, "cpu")
    before = prepared([0])
    refs[0][0].observations[9][:] = 999
    for a, b in zip(before, prepared([0]), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cached_inputs_preserve_real_critic_gradients():
    import copy

    from cr_rl.models import HistoryValueNetwork, ModelConfig
    from cr_rl.training.separate import forward_inputs

    adapter, refs = fixture(2)
    schema = adapter.schema
    cfg = ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=16,
        temporal_heads=2,
        temporal_layers=1,
        context_len=4,
        header_dim=schema.header_ints,
        entity_feature_dim=schema.entity_ints,
    ).for_v2(schema)
    torch.manual_seed(9)
    original = HistoryValueNetwork(cfg).train()
    cached = copy.deepcopy(original).train()
    windows, times, legal = [], [], []
    for trace, position, seat in refs:
        window, time_mask = trace.window(position, seat, cfg.context_len)
        windows.append(window)
        times.append(time_mask)
        legal.append(trace.masks[position][seat])
    old_inputs = history_tensors(
        np.stack(windows), np.stack(times), np.stack(legal), adapter, "cpu"
    )
    new_inputs = PreparedHistoryBatches(refs, cfg.context_len, adapter, "cpu")(np.arange(len(refs)))
    old_value = forward_inputs(original, old_inputs)
    new_value = forward_inputs(cached, new_inputs)
    torch.testing.assert_close(old_value, new_value, atol=0, rtol=0)
    old_value.square().mean().backward()
    new_value.square().mean().backward()
    for old, new in zip(original.parameters(), cached.parameters(), strict=True):
        if old.grad is None:
            assert new.grad is None
        else:
            torch.testing.assert_close(old.grad, new.grad, atol=0, rtol=0)
    assert any(p.grad is not None and p.grad.norm() > 0 for p in cached.frame.parameters())
