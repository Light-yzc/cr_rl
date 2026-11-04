import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_audit_rejects_frozen_actions_in_ppo_and_reports_conditional_wait():
    spec = importlib.util.spec_from_file_location(
        "history_audit", Path(__file__).parents[1] / "scripts/audit_history_rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = dict(
        policy_versions=np.array([[[14, (1 << 30) + 14]], [[14, 14]]]),
        learner_mask=np.array([[[1, 0]], [[1, 1]]]),
        valid=np.ones((2, 1, 2)),
        old_log_probs=np.zeros((2, 1, 2)),
        old_values=np.zeros((2, 1, 2)),
        actions=np.array([[[1, 0]], [[0, 0]]]),
        masks=np.ones((2, 1, 2, 2307), dtype=bool),
    )
    report = module.audit_arrays(data, 14)
    assert report["valid_learner_samples"] == 3 and report["frozen_opponent_decisions"] == 1
    assert report["card_action_decisions"] == 1
    assert report["wait_when_card_legal"] == pytest.approx(2 / 3)
    data["learner_mask"][0, 0, 1] = 1
    with pytest.raises(ValueError, match="frozen"):
        module.audit_arrays(data, 14)
