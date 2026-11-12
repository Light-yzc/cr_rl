import copy

import numpy as np
import pytest
import torch
from test_prepared_history import fixture

from cr_rl.models import ActorPolicy, HistoryValueNetwork, ModelConfig
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.separate import forward_inputs


def test_frame_indices_reconstruct_raw_windows_without_merging_seats_or_episodes():
    adapter, refs = fixture(2)
    prepared = PreparedHistoryBatches(refs, 4, adapter, 'cpu', deduplicate_frames=True)
    indices = np.array([0, 1, 2, 3, 4, 1])
    batch = prepared(indices)
    first, inverse = batch[8]
    ids = prepared.windows_numpy[indices].reshape(-1)
    np.testing.assert_array_equal(ids[first][inverse], ids)
    for raw in batch[:3]:
        flat = raw.flatten(0, 1)
        torch.testing.assert_close(flat[first][inverse], flat, atol=0, rtol=0)
    assert len(first) < len(inverse)
    assert prepared.metadata['frames_encoded'] < prepared.metadata['frames_requested']


@pytest.mark.parametrize('model_type', [ActorPolicy, HistoryValueNetwork])
def test_dedup_preserves_outputs_gradients_and_fresh_graphs_after_updates(model_type):
    torch.set_num_threads(1)
    adapter, refs = fixture(2)
    schema = adapter.schema
    cfg = ModelConfig(entity_dim=16, entity_heads=2, temporal_dim=16, temporal_heads=2,
                      temporal_layers=1, context_len=4, header_dim=schema.header_ints,
                      entity_feature_dim=schema.entity_ints).for_v2(schema)
    torch.manual_seed(37)
    baseline = model_type(cfg).train()
    dedup = copy.deepcopy(baseline).train()
    ordinary = PreparedHistoryBatches(refs, 4, adapter, 'cpu')
    packed = PreparedHistoryBatches(refs, 4, adapter, 'cpu', deduplicate_frames=True)
    optimizers = [torch.optim.SGD(m.parameters(), lr=.001) for m in (baseline, dedup)]
    for indices in ([0, 1, 1, 2, 3, 4], [4, 3, 0, 1, 1]):
        outputs = []
        for model, optimizer, prepared in zip(
            (baseline, dedup), optimizers, (ordinary, packed), strict=True
        ):
            optimizer.zero_grad()
            result = forward_inputs(model, prepared(indices))
            value = result[0][:, 0] if isinstance(result, tuple) else result
            outputs.append(value)
            weights = torch.arange(1, len(indices)+1, dtype=value.dtype)
            (value * weights).mean().backward()
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
        for old, new in zip(baseline.parameters(), dedup.parameters(), strict=True):
            if old.grad is None:
                assert new.grad is None
            else:
                torch.testing.assert_close(old.grad, new.grad, atol=5e-6, rtol=2e-4)
        for optimizer in optimizers:
            optimizer.step()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in dedup.frame.parameters())


def test_disjoint_windows_keep_original_forward_path():
    adapter, refs = fixture(2)
    prepared = PreparedHistoryBatches(refs, 4, adapter, 'cpu', deduplicate_frames=True)
    assert len(prepared([0])) == 8
