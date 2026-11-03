import numpy as np
import pytest

from cr_rl.training.episode_trace import EpisodeTrace, on_policy_seats


def test_trace_keeps_history_and_complete_mc_targets_across_collection_chunks():
    trace = EpisodeTrace(7, 123)
    for frame in range(6):
        trace.append(np.full((2, 4), frame), np.ones((2, 3)), [0, 0])
        if frame == 2:
            with pytest.raises(ValueError, match="entire valid"):
                trace.monte_carlo_returns(0.9)
    trace.rewards[-1] = np.array([1, -1], dtype=np.float32)
    trace.finish(terminated=True)
    result = trace.monte_carlo_returns(0.9)
    np.testing.assert_allclose(result[:, 0], 0.9 ** np.arange(5, -1, -1), atol=1e-7)
    np.testing.assert_allclose(result[:, 0], -result[:, 1])
    window, time_mask = trace.window(4, 1, 3)
    np.testing.assert_array_equal(window[:, 0], [2, 3, 4])
    assert time_mask.all()
    following = EpisodeTrace(8, 456)
    following.append(np.full((2, 4), 99), np.ones((2, 3)), [0, 0])
    window, time_mask = following.window(0, 0, 3)
    np.testing.assert_array_equal(window[:, 0], [0, 0, 99])
    np.testing.assert_array_equal(time_mask, [False, False, True])


def test_fault_or_time_limit_never_enters_mc_warmup():
    for terminal, fault in ((False, False), (True, True)):
        trace = EpisodeTrace(0, 1)
        trace.append(np.zeros((2, 4)), np.ones((2, 3)), [1, -1])
        trace.finish(terminated=terminal, faulted=fault)
        with pytest.raises(ValueError):
            trace.monte_carlo_returns(0.99)


def test_only_same_behavior_opponent_contributes_second_seat():
    versions = np.array([[3, 3], [3, 1], [0, 3]])
    np.testing.assert_array_equal(on_policy_seats(versions, 3, [0, 0, 1]), [[1, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match="current behavior"):
        on_policy_seats(versions, 3, [0, 1, 1])
