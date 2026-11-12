"""Ranking/calibration arithmetic requires neither Torch nor model resources."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('state_calibration_test', Path(__file__).resolve().parents[1] / 'scripts/diagnose_state_bc_calibration.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def test_auroc_mann_whitney_ties_and_order_invariance():
    y = np.array([0, 1, 0, 1])
    assert helper.binary_auroc(y, [.1, .8, .3, .9]) == 1
    assert helper.binary_auroc(y, [.9, .3, .8, .1]) == 0
    assert helper.binary_auroc(y, [.2, .2, .2, .2]) == .5
    assert helper.binary_auroc([0, 1], [.5, .5]) == .5
    assert helper.binary_auroc([0, 0], [.2, .8]) is None


def test_average_precision_groups_exact_ties():
    y = np.array([0, 1, 0, 1])
    assert helper.binary_average_precision(y, [.1, .8, .3, .9]) == 1
    assert helper.binary_average_precision(y, [.2, .2, .2, .2]) == .5
    assert helper.binary_average_precision(y[::-1], [.2, .2, .2, .2]) == .5
    assert helper.binary_average_precision([0, 0], [.1, .9]) is None


def test_legal_constant_baseline_has_finite_loss_and_choice_auc_half():
    y = np.array([0, 0, 0, 1, 1], dtype=bool)
    choice = np.array([0, 1, 1, 1, 1], dtype=bool)
    p = np.where(choice, .2, 0)
    result = helper.timing_report(y, p, choice, sampled_prior=.4)
    assert result['forced_wait_rows'] == 1 and result['choice_wait_rows'] == 2
    assert result['choice_auroc'] == .5
    assert result['choice_average_precision'] == .5
    assert result['importance_weighted_timing_nll'] > 0 and np.isfinite(result['natural_timing_nll'])
    assert result['thresholds']['0.1']['action_recall'] == 1
    assert result['thresholds']['0.5']['action_recall'] == 0


def test_scalar_bias_preserves_choice_rank_and_forced_wait_legality():
    y = np.array([0, 0, 0, 1, 1], dtype=bool)
    choice = np.array([0, 1, 1, 1, 1], dtype=bool)
    p = np.array([0, .1, .3, .2, .5])
    shifted = helper.logit_bias(p, 1., choice)
    assert shifted[0] == 0
    assert helper.binary_auroc(y[choice], p[choice]) == helper.binary_auroc(y[choice], shifted[choice])
    assert helper.binary_average_precision(y[choice], p[choice]) == helper.binary_average_precision(y[choice], shifted[choice])


def test_training_only_bias_selects_better_weighted_timing_loss():
    y = np.array([0] * 8 + [1] * 2, dtype=bool)
    choice = np.ones(10, dtype=bool)
    p = np.full(10, .05)
    before = helper.timing_report(y, p, choice, sampled_prior=.2)['importance_weighted_timing_nll']
    fit = helper.fit_scalar_bias(y, p, choice, sampled_prior=.2)
    assert 1 < fit['bias'] < 2
    assert fit['training_weighted_timing_nll'] < before
    assert helper.binary_auroc(y, p) == .5


def test_one_class_is_reported_as_unavailable_and_nonfinite_is_rejected():
    with pytest.warns(RuntimeWarning, match='one timing class'):
        result = helper.timing_report([0, 0], [0, .2], [0, 1], sampled_prior=.2)
    assert result['balanced_timing_nll'] is None and result['choice_auroc'] is None
    with pytest.raises(ValueError, match='finite'):
        helper.timing_report([0, 1], [.2, float('nan')], [1, 1], sampled_prior=.2)


def test_subset_manifest_is_bound_to_original_metadata_and_shard_identity(tmp_path):
    parent_path, subset_path = tmp_path / 'parent.json', tmp_path / 'subset.json'
    record = {'path': 'shards/replay_000005.npz', 'sha256': 'a' * 64, 'rows': 1024,
              'replays': 1, 'replay_offset': 5}
    parent = {'metadata_sha256': 'b' * 64, 'shards': [record, {**record, 'path': 'shards/replay_000013.npz', 'replay_offset': 13}]}
    subset = {'metadata_sha256': 'b' * 64, 'shards': [record]}
    parent_path.write_text(json.dumps(parent))
    subset_path.write_text(json.dumps(subset))
    fingerprint = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    helper.check_subset(parent_path, subset_path, fingerprint)
    subset['shards'][0] = {**record, 'replay_offset': 13}
    subset_path.write_text(json.dumps(subset))
    with pytest.raises(ValueError, match='immutable'):
        helper.check_subset(parent_path, subset_path, fingerprint)
    with pytest.raises(ValueError, match='fingerprint'):
        helper.check_subset(parent_path, subset_path, '0' * 64)
