import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cr_rl.training.calibration import (
    adjusted_act_probability,
    audit_validation_provenance,
    calibrate_timing,
    fit_wait_bias,
    timing_metrics,
)


def test_constant_odds_has_analytic_nll_optimum_and_exact_prior_shift():
    odds = np.zeros(100)
    actions = np.zeros(100, np.int64)
    actions[:2] = 1
    fitted = fit_wait_bias(odds, actions)
    assert fitted == pytest.approx(math.log(49), abs=1e-10)
    assert adjusted_act_probability(odds, fitted).mean() == pytest.approx(0.02)
    optimum = timing_metrics(odds, actions, fitted)
    assert (
        optimum["binary_timing_nll"]
        < timing_metrics(odds, actions, fitted + 0.1)["binary_timing_nll"]
    )
    assert optimum["sampled_act_recall_probability"] == pytest.approx(0.02)
    assert optimum["greedy_act_recall"] == 0
    assert optimum["expected_false_actions_per_wait_second"] == pytest.approx(0.08)


def test_wait_only_states_are_fixed_and_change_the_fitted_bias():
    odds = np.r_[np.zeros(50), np.full(50, -np.inf)]
    legal = np.r_[np.ones(50, bool), np.zeros(50, bool)]
    actions = np.zeros(100, np.int64)
    actions[:2] = 1
    fitted = fit_wait_bias(odds, actions, legal_act=legal)
    assert fitted == pytest.approx(math.log(24), abs=1e-10)
    probability = adjusted_act_probability(odds, fitted, legal)
    np.testing.assert_array_equal(probability[~legal], np.zeros(50))
    assert probability.mean() == pytest.approx(0.02)
    report = calibrate_timing(odds, actions, legal_act=legal)
    assert report["theoretical_prior_bias"] == pytest.approx(math.log(49))
    assert report["nll_fitted"]["bias"] != report["theoretical_prior_bias"]
    assert len(report["grid"]) == 5


def test_bounded_fit_remains_finite_and_reports_unattainable_within_bounds():
    actions = np.tile(np.array([0, 1]), 10)
    report = calibrate_timing(np.full(20, 1000.0), actions)
    assert report["nll_fitted"]["bias"] == 12
    assert report["nll_fitted"]["at_bound"]
    assert math.isfinite(report["nll_fitted"]["binary_timing_nll"])
    assert report["nll_fitted"]["predicted_act_probability"] > 0.99
    assert fit_wait_bias(np.zeros(20), actions, target_rate=0.1) == pytest.approx(math.log(9))


def test_invalid_or_impossible_targets_are_rejected():
    with pytest.raises(ValueError, match="no legal ACT"):
        fit_wait_bias([0.0], np.array([1]), legal_act=[False])
    with pytest.raises(ValueError, match="non-finite"):
        fit_wait_bias([float("nan")], np.array([0]))
    with pytest.raises(ValueError, match="finite odds"):
        fit_wait_bias([-np.inf], np.array([0]))
    with pytest.raises(ValueError, match="attainable"):
        fit_wait_bias([0.0], np.array([0]), legal_act=[False], target_rate=0.1)
    with pytest.raises(ValueError, match="finite"):
        adjusted_act_probability([0], float("inf"))
    result = calibrate_timing([-np.inf], np.array([0]), legal_act=[False])
    assert not result["identifiable"]
    assert result["nll_fitted"]["predicted_act_probability"] == 0


def provenance():
    return {
        "partition": "validation",
        "natural_sampling": True,
        "checkpoint_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "validation_replays": [2, 3],
        "training_replays": [0, 1],
        "heldout_deck_replays_excluded": [4],
        "excluded_replay_ids": [5],
    }


def test_provenance_rejects_balanced_sampling_train_and_excluded_rows():
    source = provenance()
    assert audit_validation_provenance(source, [2, 3])["observed_validation_replays"] == 2
    for bad in ([0], [4], [5], [9]):
        with pytest.raises(ValueError, match="outside validation"):
            audit_validation_provenance(source, bad)
    source["natural_sampling"] = False
    with pytest.raises(ValueError, match="natural"):
        audit_validation_provenance(source, [2])


def test_probability_cli_generates_small_provenance_bound_report(tmp_path):
    data = tmp_path / "samples.npz"
    actions = np.array([0] * 18 + [1, 1], dtype=np.int64)
    np.savez_compressed(
        data,
        act_log_odds=np.zeros(20),
        actions=actions,
        replay_ids=np.full(20, 2),
        legal_act=np.ones(20, bool),
        provenance_json=json.dumps(provenance()),
    )
    output = tmp_path / "report.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/calibrate_wait_bias.py"),
            "--input",
            str(data),
            "--out",
            str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        check=True,
        capture_output=True,
    )
    result = json.loads(output.read_text())
    assert result["provenance"]["checkpoint_sha256"] == "a" * 64
    assert result["nll_fitted"]["bias"] == pytest.approx(math.log(9))
    assert result["partition_audit"]["observed_validation_replays"] == 1


def test_existing_training_cache_is_reused_without_writes_or_lru_changes(tmp_path):
    from scripts.calibrate_wait_bias import ReadOnlyReuseCache

    class Private:
        def read(self, *args):
            raise AssertionError("Existing cached data should not be extracted again")

    path = tmp_path / ("c" * 64) / "observations.npy"
    path.parent.mkdir()
    np.save(path, np.arange(30).reshape(5, 6))
    original, stamp = path.read_bytes(), path.stat().st_mtime_ns
    cache = ReadOnlyReuseCache(Private(), tmp_path)
    result = cache.read(None, "c" * 64, "observations", [3, 1])
    np.testing.assert_array_equal(result, np.arange(30).reshape(5, 6)[[3, 1]])
    result[:] = -1
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == stamp
