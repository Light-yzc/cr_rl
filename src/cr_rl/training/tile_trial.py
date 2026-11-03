"""Conservative automatic selection for the matched placement-head comparison."""

import math


def select_tile_trial(comparison):
    for key in ("same_parent", "same_training_samples", "same_validation_samples"):
        if not comparison.get(key):
            raise ValueError(f"Unmatched comparison: {key}")
    control, fixed = [
        comparison["cases"][name]["metrics"]["history"][-1]["validation"]
        for name in ("control", "fixed")
    ]
    keys = (
        "position_loss",
        "play_tile_mean_l1_distance",
        "global_loss",
        "play_slot_accuracy",
        "balanced_wait_play_accuracy",
    )
    finite = all(math.isfinite(x[k]) for x in (control, fixed) for k in keys)
    checks = {
        "finite": finite,
        "position_ce_improved_2pct": fixed["position_loss"] <= 0.98 * control["position_loss"],
        "tile_l1_improved_half_tile": fixed["play_tile_mean_l1_distance"]
        <= control["play_tile_mean_l1_distance"] - 0.5,
        "global_loss_not_worse": fixed["global_loss"] <= control["global_loss"] + 0.02,
        "slot_accuracy_not_materially_worse": fixed["play_slot_accuracy"]
        >= control["play_slot_accuracy"] - 0.03,
        "decision_accuracy_not_materially_worse": fixed["balanced_wait_play_accuracy"]
        >= control["balanced_wait_play_accuracy"] - 0.03,
    }
    return {
        "choice": "fixed" if all(checks.values()) else "control",
        "automatic": True,
        "checks": checks,
        "limitation": "Matched BC validation only; not a battle-strength gate.",
    }
