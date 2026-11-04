import importlib.util
from pathlib import Path

import numpy as np


def test_real_cohort_never_fabricates_missing_history_and_covers_lanes_seats():
    spec = importlib.util.spec_from_file_location(
        "cuda_parity_audit", Path(__file__).parents[1] / "scripts/validate_cuda_policy_parity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    steps = 12
    data = {
        "obs": np.zeros((steps, 2, 2, 3), dtype=np.int32),
        "masks": np.ones((steps, 2, 2, 5), dtype=np.uint8),
        "actions": np.ones((steps, 2, 2), dtype=np.int32),
        "episode_ids": np.zeros((steps, 2), dtype=np.int64),
        "episode_start": np.zeros((steps, 2), dtype=np.uint8),
        "learner_mask": np.ones((steps, 2, 2), dtype=np.uint8),
        "valid": np.ones((steps, 2, 2), dtype=np.uint8),
    }
    data["episode_ids"][6:] = 1
    data["episode_start"][6] = 1
    data["actions"][::2] = 0
    data["valid"][8, 1, 1] = 0
    rows = module.select_histories(data, 4, 0, per_group=100)
    assert {(r[0]["lane"], r[0]["seat"]) for r in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(row["step"] >= 3 for row, _, _ in rows)
    assert any(row["history_length"] == 1 for row, _, _ in rows)
    assert not any(row["step"] == 8 and row["lane"] == row["seat"] == 1 for row, _, _ in rows)
    assert {row["action"] for row, _, _ in rows} == {0, 1}
