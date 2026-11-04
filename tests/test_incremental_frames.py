from collections import deque

import numpy as np
import pytest
import torch
from test_prepared_history import fixture

from cr_rl.models import ActorCritic, ActorPolicy, HistoryValueNetwork, ModelConfig
from cr_rl.models.incremental_frames import IncrementalFrameHistory
from cr_rl.training.inference import FrozenInference


@pytest.mark.parametrize("critic", [False, True])
def test_incremental_matches_legacy_across_wrap_reset_preview_and_weight_change(critic):
    adapter, refs = fixture(2)
    cfg = ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=16,
        temporal_heads=2,
        temporal_layers=1,
        context_len=4,
        header_dim=adapter.schema.header_ints,
        entity_feature_dim=adapter.schema.entity_ints,
    ).for_v2(adapter.schema)
    torch.manual_seed(18)
    model = HistoryValueNetwork(cfg) if critic else ActorPolicy.from_shared(ActorCritic(cfg))
    model.eval()
    legacy = FrozenInference(model, adapter, "cpu")
    incremental = FrozenInference(model, adapter, "cpu")
    history = IncrementalFrameHistory(2, 4, adapter.schema.seat_ints)
    queues = [deque(maxlen=4), deque(maxlen=4)]
    episode_ids = np.array([10, 10])
    trace = refs[0][0]

    def full_windows(frames, episodes=None):
        windows = np.zeros((2, 4, adapter.schema.seat_ints), dtype=np.int32)
        time_mask = np.zeros((2, 4), dtype=bool)
        for seat in range(2):
            seq = list(queues[seat])
            if episodes is not None:
                seq = ([] if episodes[seat] != episode_ids[seat] else seq) + [frames[seat]]
                seq = seq[-4:]
            windows[seat, -len(seq) :] = seq
            time_mask[seat, -len(seq) :] = True
        return windows, time_mask

    with torch.inference_mode():
        for step in range(10):
            if step == 5:
                episode_ids[0] = 11
                queues[0].clear()
            if step == 7:
                next(model.parameters()).add_(0.001)
            raw, legal = trace.observations[step], trace.masks[step]
            for seat in range(2):
                queues[seat].append(raw[seat])
            history.append(raw, episode_ids)
            windows, time_mask = full_windows(raw)
            expected = legacy.evaluate(windows, time_mask, None if critic else legal)
            actual = incremental.evaluate_incremental(history.request(), None if critic else legal)
            torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)
            if step == 2:
                before = history.raw.copy()
                following = trace.observations[step + 1]
                preview_episodes = episode_ids.copy()
                preview_episodes[0] = 12
                w, t = full_windows(following, preview_episodes)
                expected = legacy.evaluate(w, t, None if critic else legal)
                actual = incremental.evaluate_incremental(
                    history.request(following=following, episodes=preview_episodes),
                    None if critic else legal,
                )
                torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)
                np.testing.assert_array_equal(history.raw, before)
    assert incremental.incremental_cache.encoded_frames < 60
    stale = history.request()
    history.append(trace.observations[10], episode_ids)
    with pytest.raises(ValueError, match="stale"):
        incremental.evaluate_incremental(stale, None if critic else legal)
