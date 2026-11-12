import numpy as np
import pytest

from loc_transfer.audit_credit_assignment import (
    complete_episode_mask,
    reconstruct_chunk,
    trace_decay,
)


def test_complete_mc_selection_rejects_missing_faulted_truncated_and_mixed_episodes():
    count = 10
    data = {
        "episode_ids": np.repeat(np.arange(1, 6), 2)[:, None],
        "episode_start": np.tile([1, 0], 5)[:, None].astype(np.uint8),
        "episode_steps": np.tile([1, 2], 5)[:, None],
        "terminated": np.tile([0, 1], 5)[:, None].astype(np.uint8),
        "dones": np.tile([0, 1], 5)[:, None].astype(np.uint8),
        "truncated": np.zeros((count, 1), dtype=np.uint8),
        "fault_codes": np.zeros((count, 1), dtype=np.uint8),
        "learner_mask": np.zeros((count, 1, 2), dtype=np.uint8),
        "valid": np.ones((count, 1, 2), dtype=np.uint8),
        "policy_versions": np.full((count, 1, 2), 182, dtype=np.int32),
    }
    data["learner_mask"][:, :, 0] = 1
    data["episode_start"][0] = 0
    data["terminated"][5] = 0
    data["truncated"][5] = 1
    data["fault_codes"][6] = 1
    data["policy_versions"][8, 0, 0] = 181
    selected, summary = complete_episode_mask(data)
    expected = np.zeros_like(selected)
    expected[2:4, 0, 0] = True
    np.testing.assert_array_equal(selected, expected)
    assert summary["eligible_games"] == summary["eligible_episode_seats"] == 1
    assert summary["target_count"] == 2
    assert summary["excluded"]["missing_prefix"] == 1
    assert summary["excluded"]["unfinished_or_nonterminal"] == 1
    assert summary["excluded"]["fault_or_truncation"] == 1
    assert summary["excluded"]["mixed_or_frozen_behavior"] == 1


def test_actor_lambda_keeps_critic_targets_and_real_truncation_bootstrap():
    def seats(values):
        return np.array(values, dtype=np.float32)[:, None, None].repeat(2, axis=2)

    data = {
        "training_rewards": seats([1, 2, 7, 3]),
        "old_values": seats([5, 6, 11, 13]),
        "bootstrap_values": seats([6, 100, 13, 0]),
        "terminated": np.array([[0], [0], [0], [1]], dtype=np.uint8),
        "dones": np.array([[0], [1], [0], [1]], dtype=np.uint8),
        "groups": {"choice": seats([1, 0, 1, 1]).astype(bool)},
    }
    before = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
    short = reconstruct_chunk(data, 0.9, 0, 0.5)
    long = reconstruct_chunk(data, 0.9, 1, 0.5)
    np.testing.assert_array_equal(short["critic_target"], long["critic_target"])
    # The truncated first episode keeps V(final)=100; the next episode cannot
    # leak into it. The forced-WAIT row still contributes on the physical grid.
    expected = seats([83.8, 92, 9.7, 3])
    np.testing.assert_allclose(long["actor_return"], expected, rtol=2e-6, atol=2e-6)
    assert not np.array_equal(short["actor_return"], long["actor_return"])
    for name, expected in before.items():
        np.testing.assert_array_equal(data[name], expected)


def test_trace_scales_are_physical_seconds_and_not_hard_cutoffs():
    short = trace_decay(0.9995, 0.993, 0.25)
    longer = trace_decay(0.9995, 0.997, 0.25)
    assert short["e_fold_seconds"] == pytest.approx(33.2237393863)
    assert longer["e_fold_seconds"] == pytest.approx(71.3341237854)
    assert longer["residual_trace_weights"]["120"] > short["residual_trace_weights"]["120"] > 0
    assert short["half_life_seconds"] == pytest.approx(short["e_fold_seconds"] * np.log(2))
    with pytest.raises(ValueError):
        trace_decay(0.9995, 1.1, 0.25)
