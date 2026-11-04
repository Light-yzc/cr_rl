import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "critic_audit", Path(__file__).parents[1] / "scripts/audit_critic_returns.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def chunk(step, *, start=False, terminal=False, version=5, truncated=False):
    return dict(
        episode_ids=np.array([[2]]),
        episode_start=np.array([[start]]),
        episode_steps=np.array([[step]]),
        training_rewards=np.array([[[1.0, -1.0]]]) if terminal else np.zeros((1, 1, 2)),
        old_values=np.zeros((1, 1, 2)),
        policy_versions=np.full((1, 1, 2), version),
        learner_mask=np.ones((1, 1, 2)),
        valid=np.ones((1, 1, 2)),
        terminated=np.array([[terminal]]),
        truncated=np.array([[truncated]]),
        fault_codes=np.zeros((1, 1)),
    )


def test_complete_mc_uses_stored_values_and_cross_chunk_prefix():
    data = [chunk(1, start=True), chunk(2, terminal=True)]
    result = module().analyze_chunks(data, 0.5)
    assert result["eligible_episode_seats"] == 2 and result["eligible_games"] == 1
    assert result["target_count"] == 4 and result["explained_variance"] == 0
    assert result["mse"] == pytest.approx((0.25 + 1) / 2)
    assert result["episodes"][0]["initial_mc_return"] == 0.5
    assert data[0]["old_values"].sum() == 0


def test_value_diagnostic_excludes_mixed_policy_partial_and_truncated_games():
    audit = module()
    result = audit.analyze_chunks([chunk(1, start=True), chunk(2, terminal=True, version=6)], 0.9)
    assert not result["target_count"] and result["excluded"]["mixed_behavior_versions"] == 2
    result = audit.analyze_chunks([chunk(8), chunk(9, terminal=True)], 0.9)
    assert result["excluded"]["missing_prefix"] == 1
    result = audit.analyze_chunks([chunk(1, start=True, truncated=True)], 0.9)
    assert result["excluded"]["fault_or_truncation"] == 1
    result = audit.analyze_chunks([chunk(1, start=True), chunk(3, terminal=True)], 0.9)
    assert result["excluded"]["noncontiguous_steps"] == 1
