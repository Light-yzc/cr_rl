"""Held-out scalar WAIT calibration; conditional action probabilities are unchanged."""

from __future__ import annotations

import math

import numpy as np

DEFAULT_BIAS_GRID = (0.0, 1.0, 1.5, 2.0, 2.5)


def _inputs(act_log_odds, actions, legal_act=None):
    odds = np.asarray(act_log_odds, dtype=np.float64)
    actions = np.asarray(actions)
    if odds.ndim != 1 or actions.shape != odds.shape or not len(odds):
        raise ValueError("Expected nonempty one-dimensional odds and action arrays")
    if actions.dtype.kind not in "biu" or ((actions < 0) | (actions > 2306)).any():
        raise ValueError("Actions must be integer crforge action labels in [0, 2306]")
    legal = np.ones(len(odds), dtype=bool) if legal_act is None else np.asarray(legal_act, bool)
    if legal.shape != odds.shape or np.isnan(odds).any() or np.isposinf(odds).any():
        raise ValueError("Invalid legal-action mask or non-finite ACT/WAIT log odds")
    if ((actions != 0) & ~legal).any():
        raise ValueError("An ACT target has no legal ACT action")
    if np.isneginf(odds[legal]).any():
        raise ValueError("Legal ACT rows require finite odds; mark WAIT-only rows legal_act=False")
    return odds, actions, legal


def adjusted_act_probability(act_log_odds, bias, legal_act=None):
    """Adding b to WAIT subtracts b from ACT/WAIT log odds after legal masking."""
    if not math.isfinite(bias):
        raise ValueError("WAIT bias must be finite")
    odds = np.asarray(act_log_odds, dtype=np.float64)
    probability = np.exp(-np.logaddexp(0.0, -(odds - bias)))
    if legal_act is not None:
        probability = np.where(np.asarray(legal_act, dtype=bool), probability, 0.0)
    return probability


def fit_wait_bias(act_log_odds, actions, *, legal_act=None, bounds=(-12.0, 12.0), target_rate=None):
    """Fit a finite bias by monotone bisection, without a scipy dependency.

    With target_rate=None this minimizes unweighted binary timing NLL on a
    natural held-out cohort: dNLL/db = mean(y - p_b), so its interior optimum
    also matches the empirical ACT rate. Matching an external target is a
    separate rate constraint, not necessarily an NLL optimum.
    """
    odds, actions, legal = _inputs(act_log_odds, actions, legal_act)
    lower, upper = map(float, bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Bias bounds must be finite and increasing")
    target = float(np.mean(actions != 0)) if target_rate is None else float(target_rate)
    if not math.isfinite(target) or not 0 <= target <= float(legal.mean()):
        raise ValueError("Target ACT rate is outside the attainable legal-action range")
    if not legal.any():
        return min(upper, max(lower, 0.0))
    if adjusted_act_probability(odds, lower, legal).mean() <= target:
        return lower
    if adjusted_act_probability(odds, upper, legal).mean() >= target:
        return upper
    for _ in range(96):
        middle = (lower + upper) / 2.0
        if adjusted_act_probability(odds, middle, legal).mean() > target:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def timing_metrics(act_log_odds, actions, bias, *, legal_act=None, decision_seconds=0.25):
    odds, actions, legal = _inputs(act_log_odds, actions, legal_act)
    if not math.isfinite(decision_seconds) or decision_seconds <= 0:
        raise ValueError("Decision interval must be finite and positive")
    probability = adjusted_act_probability(odds, bias, legal)
    act = actions != 0
    wait = ~act
    play = (actions > 0) & (actions < 2305)
    # Avoid 0 * (-inf) on WAIT-only rows and stable exp/log at extreme logits.
    nll = np.zeros(len(actions), np.float64)
    shifted = odds[legal] - bias
    nll[legal] = np.logaddexp(0.0, shifted) - act[legal] * shifted

    def average(values, mask):
        return float(np.mean(values[mask])) if mask.any() else None

    false_probability = average(probability, wait)
    return {
        "bias": float(bias),
        "rows": len(actions),
        "act_rows": int(act.sum()),
        "play_rows": int(play.sum()),
        "ability_rows": int((actions >= 2305).sum()),
        "wait_rows": int(wait.sum()),
        "wait_only_legal_rows": int((~legal).sum()),
        "natural_act_fraction": float(act.mean()),
        "predicted_act_probability": float(probability.mean()),
        "binary_timing_nll": float(nll.mean()),
        "brier_score": float(np.mean((probability - act) ** 2)),
        "sampled_act_recall_probability": average(probability, act),
        "greedy_act_recall": average(probability > 0.5, act),
        "greedy_timing_recall_on_play_rows": average(probability > 0.5, play),
        "wait_false_action_probability": false_probability,
        "greedy_wait_false_action_rate": average(probability > 0.5, wait),
        "expected_false_actions_per_wait_second": (
            false_probability / decision_seconds if false_probability is not None else None
        ),
    }


def calibrate_timing(
    act_log_odds,
    actions,
    *,
    legal_act=None,
    decision_seconds=0.25,
    training_act_fraction=0.5,
    bias_bounds=(-12.0, 12.0),
    grid=DEFAULT_BIAS_GRID,
    target_rate=None,
):
    odds, actions, legal = _inputs(act_log_odds, actions, legal_act)
    if not 0 < training_act_fraction < 1:
        raise ValueError("Training ACT fraction must be strictly between zero and one")
    natural = float(np.mean(actions != 0))
    theoretical = (
        math.log(training_act_fraction / (1 - training_act_fraction))
        - math.log(natural / (1 - natural))
        if 0 < natural < 1
        else None
    )
    fitted = fit_wait_bias(odds, actions, legal_act=legal, bounds=bias_bounds)
    matched = fit_wait_bias(
        odds, actions, legal_act=legal, bounds=bias_bounds, target_rate=target_rate
    )

    def metrics(bias):
        return timing_metrics(
            odds, actions, bias, legal_act=legal, decision_seconds=decision_seconds
        )

    return {
        "protocol": "natural_heldout_binary_timing_bias_v1",
        "training_act_fraction": float(training_act_fraction),
        "natural_act_fraction": natural,
        "theoretical_prior_bias": theoretical,
        "theoretical_prior_assumptions": (
            "Pure class-prior shift only; WAIT-margin sampling, legality and covariate shift "
            "can invalidate this formula. It is diagnostic, not a deployment choice."
        ),
        "bias_bounds": list(map(float, bias_bounds)),
        "identifiable": bool(legal.any()),
        "nll_fitted": {**metrics(fitted), "at_bound": fitted in tuple(bias_bounds)},
        "rate_matched": {
            **metrics(matched),
            "target_rate": natural if target_rate is None else float(target_rate),
        },
        "grid": [metrics(float(bias)) for bias in grid],
        "limitations": (
            "Offline probabilities and greedy timing decisions are not executed card counts, "
            "card-choice accuracy or closed-loop win rate. Validate bias in complete games."
        ),
    }


def audit_validation_provenance(provenance, replay_ids):
    """Fail closed when a calibration artifact is not declared natural validation data."""
    if (
        provenance.get("partition") != "validation"
        or provenance.get("natural_sampling") is not True
    ):
        raise ValueError("Calibration requires natural, unbalanced validation sampling")
    for key in ("checkpoint_sha256", "dataset_sha256"):
        value = provenance.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing or invalid {key}")
    replay_ids = np.asarray(replay_ids)
    if replay_ids.ndim != 1 or replay_ids.dtype.kind not in "iu":
        raise ValueError("Calibration rows require integer replay ids")
    observed = set(map(int, replay_ids))
    validation = set(map(int, provenance.get("validation_replays", [])))
    excluded = set(map(int, provenance.get("heldout_deck_replays_excluded", [])))
    excluded |= set(map(int, provenance.get("excluded_replay_ids", [])))
    training = set(map(int, provenance.get("training_replays", [])))
    if not observed or not observed <= validation or observed & (excluded | training):
        raise ValueError("Calibration rows leak outside validation or into excluded/train replays")
    if validation & (excluded | training):
        raise ValueError("Validation assignment overlaps excluded/train assignments")
    return {
        "observed_validation_replays": len(observed),
        "assigned_validation_replays": len(validation),
        "scope": "Validated supplied provenance; input-mode hashes are asserted by its producer",
    }
