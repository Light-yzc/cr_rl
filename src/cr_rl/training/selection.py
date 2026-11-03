"""Select among completed BC epochs without overwriting the resumable last epoch."""

import json
import math
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified


def select_bc_checkpoint(checkpoint: Path, backup: Path | None = None) -> tuple[Path, dict]:
    report_path = checkpoint.parent / "bc.json"
    if not report_path.exists():
        return checkpoint, {"selection": "explicit checkpoint"}
    report = json.loads(report_path.read_text())
    full_path = checkpoint.parent / "bc_selection_full.json"
    v2 = report.get("representation_version", 1) == 2
    data_report = checkpoint.parent / "bc.data.json"
    settings = json.loads(data_report.read_text()) if data_report.exists() else {}
    conditional = v2 and report.get("bc_objective", settings.get("bc_objective")) == "conditional"
    if full_path.exists() and not v2:
        full = json.loads(full_path.read_text())
        report["history"] = full["epochs"]
        report["baseline"] = full["baseline"]
    eligible = [row for row in report["history"] if row["validation"]["play_slot_accuracy"] > 0]
    if conditional:
        baseline = report.get("natural_baseline", {})
        baseline_score = report["baseline"].get("conditional_validation_score", float("nan"))
        baseline_tile = report["baseline"].get("position_loss", float("nan"))
        if not all(math.isfinite(v) for v in (baseline_score, baseline_tile)):
            raise ValueError("Conditional BC selection requires finite component baselines")
        eligible = [
            row
            for row in eligible
            if math.isfinite(row["validation"].get("conditional_validation_score", float("nan")))
            and math.isfinite(row["validation"].get("position_loss", float("nan")))
            and row["validation"]["conditional_validation_score"] < baseline_score
            and row["validation"]["position_loss"] < baseline_tile
            and all(
                math.isfinite(row.get("natural_validation", {}).get(key, float("nan")))
                for key in ("nonwait_probability_on_action_rows", "wait_false_action_probability")
            )
            and row["natural_validation"]["nonwait_probability_on_action_rows"]
            > row["natural_validation"]["wait_false_action_probability"] + 0.01
        ]
    elif v2:
        baseline = report.get("natural_baseline", {})
        baseline_tile = report["baseline"].get("position_loss", float("nan"))
        baseline_loss = baseline.get("global_loss", float("nan"))
        baseline_play_fraction = baseline.get("label_nonwait_fraction", float("nan"))
        if not all(
            math.isfinite(v) for v in (baseline_loss, baseline_tile, baseline_play_fraction)
        ):
            raise ValueError("v2 selection requires finite natural and placement baselines")
        eligible = [
            row
            for row in eligible
            if math.isfinite(row.get("natural_validation", {}).get("global_loss", float("nan")))
            and math.isfinite(row["validation"].get("position_loss", float("nan")))
            and row["natural_validation"]["global_loss"] < baseline_loss
            and row["validation"]["position_loss"] < baseline_tile
        ]
        # Natural loss alone rewards predicting WAIT for this highly imbalanced
        # dataset. Keep epochs whose predicted action rate is at least within 2x
        # of the held-out label prior, and whose action rows score above WAIT rows.
        eligible = [
            row
            for row in eligible
            if all(
                math.isfinite(row["natural_validation"].get(key, float("nan")))
                for key in (
                    "policy_nonwait_probability",
                    "nonwait_probability_on_action_rows",
                    "wait_false_action_probability",
                )
            )
            and 0.5 * baseline_play_fraction
            <= row["natural_validation"]["policy_nonwait_probability"]
            <= 2.0 * baseline_play_fraction
            and row["natural_validation"]["nonwait_probability_on_action_rows"]
            > row["natural_validation"]["wait_false_action_probability"]
        ]
    if not eligible:
        raise ValueError("No completed BC epoch passes held-out action/placement improvement gates")
    if conditional:
        best = min(eligible, key=lambda row: row["validation"]["conditional_validation_score"])
    elif v2:
        best = min(eligible, key=lambda row: row["natural_validation"]["global_loss"])
    else:
        best = max(eligible, key=lambda row: row["validation"]["balanced_wait_play_accuracy"])
    baseline_score = report["baseline"].get("balanced_wait_play_accuracy", float("-inf"))
    if not v2 and best["validation"]["balanced_wait_play_accuracy"] <= baseline_score:
        raise ValueError("BC selection did not improve held-out imitation over initialization")
    backup = backup or Path("/content/drive/MyDrive/cr_rl/runs") / checkpoint.parent.name / "bc"
    source = backup / f"bc_epoch_{best['epoch']:04d}.pt"
    if not source.exists():
        raise FileNotFoundError(f"Selected BC epoch is unavailable: {source}")
    selected = checkpoint.parent / "bc_selected.pt"
    record = copy_verified(source, selected)
    selection = {
        "protocol": (
            "conditional_components_with_placement_and_timing_gate"
            if conditional
            else "v2_natural_global_loss_with_placement_gate"
            if v2
            else "balanced_accuracy"
        ),
        "selected_epoch": best["epoch"],
        "trained_epochs": len(report["history"]),
        "validation": best["validation"],
        "baseline": report["baseline"],
        **record,
    }
    if v2:
        selection.update(natural_validation=best["natural_validation"], natural_baseline=baseline)
    if full_path.exists() and not v2:
        selection.update(
            validation_play_labels=full["play_labels"],
            validation_wait_labels=full["wait_labels"],
            protocol=full["protocol"],
        )
        copy_verified(full_path, backup / "validation_full.json")
    atomic_json(checkpoint.parent / "bc_selection.json", selection)
    copy_verified(selected, backup / "bc_selected.pt")
    copy_verified(checkpoint.parent / "bc_selection.json", backup / "selection.json")
    return selected, selection
