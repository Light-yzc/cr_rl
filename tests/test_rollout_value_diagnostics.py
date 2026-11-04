import json

import numpy as np
import pytest

from cr_rl.training.telemetry import rollout_value_diagnostics


def test_value_diagnostics_preserve_terminal_trace_boundaries_and_original_arrays(tmp_path):
    path = tmp_path / "rollout.npz"
    rewards = np.repeat(np.array([1, 2, 3], dtype=np.float32)[:, None, None], 2, axis=2)
    values = np.repeat(np.array([1, 5, 3], dtype=np.float32)[:, None, None], 2, axis=2)
    next_values = np.array([0, 3, 0], dtype=np.float32)[:, None, None].repeat(2, axis=2)
    flags = np.ones((3, 1, 2), dtype=np.uint8)
    np.savez(
        path,
        training_rewards=rewards,
        old_values=values,
        bootstrap_values=next_values,
        learner_mask=flags,
        valid=flags,
        bootstrap_valid=flags,
        terminated=np.array([[1], [0], [1]], dtype=np.uint8),
        dones=np.array([[1], [0], [1]], dtype=np.uint8),
    )
    path.with_suffix(".npz.json").write_text(
        json.dumps(dict(training_driver="persistent-v2", gamma=1, gae_lambda=1))
    )
    before = path.read_bytes()
    result = rollout_value_diagnostics(path)
    assert result["return_mean"] == 3
    assert result["advantage_mean"] == 0
    assert result["explained_variance"] == 1
    assert result["target_count"] == 6
    assert path.read_bytes() == before
    path.with_suffix(".npz.json").write_text(json.dumps(dict(training_driver="other")))
    with pytest.raises(ValueError, match="provenance"):
        rollout_value_diagnostics(path)
