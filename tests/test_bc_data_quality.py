import errno
import json

import numpy as np
import pytest
import torch

from cr_rl.data.bc_sampling import class_prior_weights, epoch_rows
from cr_rl.data.bc_snapshot import freeze_datasets
from cr_rl.data.replay_supervision import prefix_cutoff
from cr_rl.training.bc import hierarchical_bc_loss
from cr_rl.training.persistence import file_sha256
from cr_rl.training.tile_trial import select_tile_trial


def test_first_missed_command_invalidates_both_seats_from_its_window():
    assert prefix_cutoff([(17, 1), (23, 0)]) == 15
    ticks = np.repeat([0, 5, 10, 15, 20], 2)
    np.testing.assert_array_equal(ticks[ticks < prefix_cutoff([(17, 1)])], [0, 0, 5, 5, 10, 10])
    assert prefix_cutoff([(0, 0)]) == 0
    assert prefix_cutoff([]) is None


def test_ability_stratum_preserves_full_card_coverage_and_meets_requested_fraction():
    cards, abilities, waits = np.arange(100), np.arange(100, 102), np.arange(102, 1102)
    samples = epoch_rows(
        np.r_[cards, abilities],
        waits,
        np.random.default_rng(4),
        0.5,
        ability_rows=abilities,
        ability_fraction=0.1,
    )
    assert set(cards).issubset(samples)
    assert set(abilities).issubset(samples)
    nonwait = samples[samples < 102]
    assert np.isin(nonwait, abilities).mean() >= 0.1


def test_card_importance_weights_recover_natural_prior_without_inference_bias():
    natural = np.r_[np.zeros(98, dtype=int), np.ones(2, dtype=int)]
    sampled = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    weights = torch.tensor(class_prior_weights(natural, sampled), dtype=torch.float32)
    torch.testing.assert_close(weights, torch.tensor([1.96, 0.04, 0.0]))
    logits = torch.tensor([np.log(0.98), np.log(0.02)], requires_grad=True)
    logp = torch.full((100, 2307), -1e9)
    logp[:, :2] = torch.log_softmax(logits, -1)
    legal = torch.zeros_like(logp, dtype=torch.bool)
    legal[:, :2] = True
    loss, _ = hierarchical_bc_loss(logp, torch.tensor(sampled), legal, card_class_weights=weights)
    expected = -0.98 * np.log(0.98) - 0.02 * np.log(0.02)
    assert abs(float(loss.detach()) - expected) < 1e-6
    loss.backward()
    assert logits.grad.abs().max() < 1e-6


def test_gaussian_tile_targets_never_assign_mass_to_illegal_cells():
    logits = torch.randn(1, 2307, requires_grad=True)
    legal = torch.zeros_like(logits, dtype=torch.bool)
    legal[:, 0] = True
    for tile in (100, 101, 119):
        legal[:, 1 + tile] = True
    joint = torch.log_softmax(logits.masked_fill(~legal, -1e9), -1)
    loss, parts = hierarchical_bc_loss(joint, torch.tensor([101]), legal, tile_label_sigma=1.0)
    assert torch.isfinite(loss) and torch.isfinite(parts["position_hard_loss"])
    loss.backward()
    assert (logits.grad[~legal] == 0).all()
    assert logits.grad[legal].abs().sum() > 0


@pytest.mark.parametrize("cross_device", [False, True])
def test_snapshot_deduplicates_canonical_games_without_rewriting_history(
    tmp_path, monkeypatch, cross_device
):
    if cross_device:

        def cross_device_link(*args):
            raise OSError(errno.EXDEV, "cross-filesystem link")

        monkeypatch.setattr("cr_rl.data.bc_snapshot.os.link", cross_device_link)
    sources = []
    for i in range(2):
        folder = tmp_path / str(i)
        folder.mkdir()
        data = folder / "data.npz"
        np.savez_compressed(data, observations=np.zeros((2, 3)))
        metadata = folder / "meta.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema": {},
                    "vocabulary": {},
                    "backend_commit": "a",
                    "content_sha": "b",
                    "level": 11,
                    "replays": [
                        {"source": str(i), "replay_tag": str(i), "canonical_game_signature": "same"}
                    ],
                }
            )
        )
        (folder / "dataset.manifest.json").write_text(
            json.dumps(
                {
                    "metadata": "meta.json",
                    "metadata_sha256": file_sha256(metadata),
                    "shards": [
                        {"path": "data.npz", "sha256": file_sha256(data), "replays": 1, "rows": 2}
                    ],
                }
            )
        )
        sources.append((folder, folder))
    result = freeze_datasets(sources, tmp_path / "frozen", deduplicate=True)
    metadata = json.loads((tmp_path / "frozen/dataset_metadata.json").read_text())
    assert result["supervised_replays"] == 1 and result["duplicates_excluded"] == 1
    assert metadata["excluded_replay_ids"] == [1]
    assert len(metadata["replays"]) == 2
    original_locations = json.loads((tmp_path / "frozen/source_locations.json").read_text())
    freeze_datasets([(tmp_path / "frozen", tmp_path / "nonexistent_cloud")], tmp_path / "stage2")
    restored_locations = json.loads((tmp_path / "stage2/source_locations.json").read_text())
    assert [item["source"] for item in restored_locations["files"]] == [
        item["source"] for item in original_locations["files"]
    ]
    manifest = json.loads((tmp_path / "frozen/dataset.manifest.json").read_text())
    for record in manifest["shards"]:
        assert file_sha256(tmp_path / "frozen" / record["path"]) == record["sha256"]


def test_trial_selection_never_waits_for_approval_and_falls_back_on_regression():
    control = {
        "position_loss": 5.5,
        "play_tile_mean_l1_distance": 16.0,
        "global_loss": 2.3,
        "play_slot_accuracy": 0.4,
        "balanced_wait_play_accuracy": 0.65,
    }
    fixed = {
        **control,
        "position_loss": 4.5,
        "play_tile_mean_l1_distance": 10.0,
        "global_loss": 2.1,
    }
    report = {
        "same_parent": True,
        "same_training_samples": True,
        "same_validation_samples": True,
        "cases": {
            name: {"metrics": {"history": [{"validation": value}]}}
            for name, value in [("control", control), ("fixed", fixed)]
        },
    }
    assert select_tile_trial(report)["choice"] == "fixed"
    fixed["play_slot_accuracy"] = 0.2
    assert select_tile_trial(report)["choice"] == "control"
