from collections import deque
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_prepared_history import fixture

from cr_rl.models import ActorCritic, ActorPolicy, HistoryValueNetwork, ModelConfig
from cr_rl.models.incremental_frames import IncrementalFrameHistory
from cr_rl.training.inference import FrozenInference


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup_model(kind="actor"):
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
    cfg = replace(cfg, frame_encoder_fp32=True)
    torch.manual_seed(28)
    model = {
        "actor": ActorPolicy,
        "shared": ActorCritic,
        "critic": HistoryValueNetwork,
    }[kind](cfg).eval()
    return model, adapter, refs[0][0]


@pytest.mark.parametrize("incremental", [False, True])
def test_forced_wait_preserves_sampling_rng_history_wrap_reset_and_weight_change(incremental):
    model, adapter, trace = setup_model()
    regular = FrozenInference(model, adapter, "cpu")
    fast = FrozenInference(model, adapter, "cpu", forced_wait_fastpath=True)
    history = IncrementalFrameHistory(2, 4, adapter.schema.seat_ints)
    queues = [deque(maxlen=4), deque(maxlen=4)]
    episodes = np.array([10, 20])
    forced = np.zeros((2, 2307), dtype=np.uint8)
    forced[:, 0] = 1
    for step in range(14):
        if step == 5:
            episodes[0] += 1
            queues[0].clear()
        if step == 7:
            with torch.no_grad():
                next(model.parameters()).add_(0.001)
        raw = trace.observations[step % len(trace.observations)]
        history.append(raw, episodes)
        for seat in (0, 1):
            queues[seat].append(raw[seat])
        windows = np.zeros((2, 4, adapter.schema.seat_ints), dtype=np.int32)
        times = np.zeros((2, 4), dtype=bool)
        for seat in (0, 1):
            windows[seat, -len(queues[seat]) :] = np.stack(queues[seat])
            times[seat, -len(queues[seat]) :] = True
        legal = forced.copy()
        if step >= 10:
            legal[:, 1:6] = 1

        def evaluate(engine):
            if incremental:
                return engine.evaluate_incremental(history.request(), legal)
            return engine.evaluate(windows, times, legal)

        expected = evaluate(regular)
        actual = evaluate(fast)
        torch.testing.assert_close(actual.exp(), expected.exp(), atol=5e-5, rtol=1e-5)
        # Use the production sampling wrappers, retaining draws on forced turns.
        before = torch.get_rng_state()
        if incremental:
            expected_sample = regular.sample_incremental(history.request(), legal)
        else:
            expected_sample = regular.sample(windows, times, legal)
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        if incremental:
            actual_sample = fast.sample_incremental(history.request(), legal)
        else:
            actual_sample = fast.sample(windows, times, legal)
        torch.testing.assert_close(torch.get_rng_state(), after, rtol=0, atol=0)
        np.testing.assert_array_equal(actual_sample[0], expected_sample[0])
        np.testing.assert_allclose(actual_sample[1], expected_sample[1], atol=5e-5)
        if step < 10:
            assert (actual_sample[0] == 0).all()
            assert actual_sample[2] == {"entropy": 0.0, "max_action_probability": 1.0}
    assert fast.forced_wait_batches == 20 and fast.forced_wait_rows == 40
    if incremental:
        assert fast.incremental_cache.encoded_frames < regular.incremental_cache.encoded_frames
        stale = history.request()
        history.append(raw, episodes)
        with pytest.raises(ValueError, match="stale"):
            fast.evaluate_incremental(stale, forced)
    else:
        assert fast.cache.encoded_frames < regular.cache.encoded_frames


@pytest.mark.parametrize("kind", ["shared", "critic"])
def test_shared_bc_policy_can_skip_but_critic_values_are_always_computed(kind):
    model, adapter, trace = setup_model(kind)
    raw = np.stack(trace.observations[:4]).transpose(1, 0, 2)
    times = np.ones((2, 4), dtype=bool)
    legal = np.zeros((2, 2307), dtype=np.uint8)
    legal[:, 0] = 1
    regular = FrozenInference(model, adapter, "cpu")
    fast = FrozenInference(model, adapter, "cpu", forced_wait_fastpath=True)
    expected = regular.evaluate(raw, times, legal)
    actual = fast.evaluate(raw, times, legal)
    if kind == "critic":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert fast.forced_wait_batches == 0 and fast.cache.encoded_frames > 0
    else:
        torch.testing.assert_close(actual.exp(), expected.exp(), rtol=0, atol=0)
        assert fast.forced_wait_batches == 1 and fast.cache.encoded_frames == 0


def test_fastpath_is_disabled_by_default_and_keeps_mask_and_eval_guards():
    model, adapter, trace = setup_model()
    windows = np.stack(trace.observations[:4]).transpose(1, 0, 2)
    times = np.ones((2, 4), dtype=bool)
    legal = np.zeros((2, 2307), dtype=np.uint8)
    legal[:, 0] = 1
    default = FrozenInference(model, adapter, "cpu")
    default.evaluate(windows, times, legal)
    assert default.forced_wait_batches == 0 and default.cache.encoded_frames > 0
    fast = FrozenInference(model, adapter, "cpu", forced_wait_fastpath=True)
    with pytest.raises(ValueError, match="complete legal"):
        fast.evaluate(windows, times, legal[:1])
    with pytest.raises(ValueError, match="Time mask"):
        fast.evaluate(windows, times[:, :2], legal)
    invalid_current = times.copy()
    invalid_current[0, -1] = False
    with pytest.raises(ValueError, match="valid current"):
        fast.evaluate(windows, invalid_current, legal)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        fast.evaluate(windows, times, legal)
