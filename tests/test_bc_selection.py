import json

import pytest

from cr_rl.training.selection import select_bc_checkpoint


def test_selection_keeps_last_resume_and_ignores_zero_play_collapse(tmp_path):
    last, backup = tmp_path / "bc.pt", tmp_path / "drive"
    backup.mkdir()
    last.write_bytes(b"last resumable checkpoint")
    (backup / "bc_epoch_0001.pt").write_bytes(b"first better policy")
    (backup / "bc_epoch_0002.pt").write_bytes(last.read_bytes())
    (tmp_path / "bc.json").write_text(
        json.dumps(
            {
                "baseline": {},
                "history": [
                    {
                        "epoch": 1,
                        "validation": {
                            "play_slot_accuracy": 0.2,
                            "balanced_wait_play_accuracy": 0.55,
                        },
                    },
                    {
                        "epoch": 2,
                        "validation": {
                            "play_slot_accuracy": 0.0,
                            "balanced_wait_play_accuracy": 0.5,
                        },
                    },
                ],
            }
        )
    )
    selected, report = select_bc_checkpoint(last, backup)
    assert selected.read_bytes() == b"first better policy"
    assert last.read_bytes() == b"last resumable checkpoint"
    assert report["selected_epoch"] == 1
    assert report["trained_epochs"] == 2
    assert (backup / "bc_selected.pt").read_bytes() == selected.read_bytes()
    (backup / "bc_epoch_0001.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Selected BC epoch"):
        select_bc_checkpoint(last, backup)


def test_v2_selects_natural_loss_without_rewarding_wait_or_placement_collapse(tmp_path):
    last, backup = tmp_path / "bc.pt", tmp_path / "drive"
    backup.mkdir()
    last.write_bytes(b"resume")
    history = []
    # A high balanced accuracy must not beat a better calibrated natural loss.
    # A later low natural loss must still demonstrate placement learning.
    for epoch, natural_loss, tile_loss, balanced in [
        (1, 1.4, 4.9, 0.8),
        (2, 1.1, 4.5, 0.5),
        (3, 0.9, 5.8, 0.9),
    ]:
        (backup / f"bc_epoch_{epoch:04d}.pt").write_bytes(str(epoch).encode())
        history.append(
            {
                "epoch": epoch,
                "validation": {
                    "play_slot_accuracy": 0.4,
                    "position_loss": tile_loss,
                    "balanced_wait_play_accuracy": balanced,
                },
                "natural_validation": {
                    "global_loss": natural_loss,
                    "policy_nonwait_probability": 0.02,
                    "nonwait_probability_on_action_rows": 0.03,
                    "wait_false_action_probability": 0.01,
                },
            }
        )
    report = {
        "representation_version": 2,
        "baseline": {"position_loss": 5.5},
        "natural_baseline": {"global_loss": 2.0, "label_nonwait_fraction": 0.02},
        "history": history,
    }
    (tmp_path / "bc.json").write_text(json.dumps(report))
    selected, result = select_bc_checkpoint(last, backup)
    assert selected.read_bytes() == b"2"
    assert result["selected_epoch"] == 2
    assert result["natural_validation"]["global_loss"] == 1.1
    assert last.read_bytes() == b"resume"
    report["natural_baseline"] = {}
    (tmp_path / "bc.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="requires finite"):
        select_bc_checkpoint(last, backup)


def test_conditional_selection_uses_component_losses_without_sparse_prior_gate(tmp_path):
    last, backup = tmp_path / "bc.pt", tmp_path / "drive"
    backup.mkdir()
    last.write_bytes(b"resumable")
    history = []
    for epoch, score, tile, natural, action_p, wait_p in [
        (1, 5.0, 4.6, 1.7, 0.7, 0.4),
        (2, 4.8, 4.4, 2.1, 0.65, 0.4),
        (3, 4.5, 4.3, 0.001, 0.004, 0.004),
    ]:
        (backup / f"bc_epoch_{epoch:04d}.pt").write_bytes(str(epoch).encode())
        history.append(
            {
                "epoch": epoch,
                "validation": {
                    "play_slot_accuracy": 0.4,
                    "position_loss": tile,
                    "conditional_validation_score": score,
                },
                "natural_validation": {
                    "global_loss": natural,
                    "policy_nonwait_probability": wait_p,
                    "nonwait_probability_on_action_rows": action_p,
                    "wait_false_action_probability": wait_p,
                },
            }
        )
    report = {
        "representation_version": 2,
        "bc_objective": "conditional",
        "baseline": {"position_loss": 5.0, "conditional_validation_score": 5.8},
        "natural_baseline": {"label_nonwait_fraction": 0.02},
        "history": history,
    }
    (tmp_path / "bc.json").write_text(json.dumps(report))
    selected, result = select_bc_checkpoint(last, backup)
    assert selected.read_bytes() == b"2"
    assert result["protocol"] == "conditional_components_with_placement_and_timing_gate"
    assert last.read_bytes() == b"resumable"
    report["baseline"].pop("conditional_validation_score")
    (tmp_path / "bc.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="component baselines"):
        select_bc_checkpoint(last, backup)
