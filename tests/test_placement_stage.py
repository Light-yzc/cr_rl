"""Small CPU checks for a dataset-bound spatial-only training stage."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ModelConfig
from cr_rl.models.state_policy import StateActorCritic, state_model_from_checkpoint
from cr_rl.models.state_temporal import StateTemporalConfig
from cr_rl.training.placement_stage import (
    build_training_prior, initialize_placement_stage, set_placement_training_mode,
    validate_training_prior,
)
from cr_rl.training.state_bc import (ReplaySequenceDataset, StateEpisode, load_state_checkpoint,
                                    run_state_episode, save_state_checkpoint)
from cr_rl.training.state_bc_parallel import run_state_episode_group
from test_state_bc import synthetic_dataset


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fixture(tmp_path):
    manifest, _ = synthetic_dataset(tmp_path / "dataset")
    dataset = ReplaySequenceDataset(manifest, tmp_path / "scratch", ram_limit_gib=0)
    split = {0: "train", 1: "validation", 2: "heldout"}
    cfg = ModelConfig(entity_dim=8, entity_heads=2, entity_layers=1, temporal_dim=16,
        temporal_heads=2, temporal_layers=2, header_dim=dataset.schema.header_ints,
        entity_feature_dim=dataset.schema.entity_ints, max_entities=dataset.schema.max_entities, vocab_size=16)
    cfg = replace(cfg.for_v2(dataset.schema, "dynamic4"), readout_version=2)
    temporal = StateTemporalConfig(width=16, heads=2, kv_heads=1, head_dim=8,
        pattern=("kda", "gqa"), chunk_size=2)
    torch.manual_seed(12)
    model = StateActorCritic(cfg, temporal, decision_ticks=8, frame_microbatch=2)
    parent = tmp_path / "parent.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    save_state_checkpoint(parent, model, optimizer, None,
        dataset_fingerprint=dataset.fingerprint, split=split, epoch=2, next_episode=1,
        optimizer_steps=123, training_config={})
    args = SimpleNamespace(upgrade_placement_from=parent, decision_ticks=8, device="cpu",
        frame_microbatch=2, placement_prior=None, out=tmp_path / "newstage",
        tile_label_sigma=1., ram_limit_gib=0)
    args.out.mkdir()
    return model, dataset, split, args


def test_prior_is_built_only_from_original_train_split_and_bound(tmp_path):
    _, dataset, split, args = fixture(tmp_path)
    path = args.out / "prior.json"
    prior = build_training_prior(dataset, split, 1., path)
    assert prior["fit_replay_ids"] == [0]
    assert prior["fit_games"] == 1 and prior["training_play_rows"] == 4
    assert sum(prior["card_counts"].values()) == 4
    assert build_training_prior(dataset, split, 1., path) == prior
    for key, replacement in (("dataset_fingerprint", "different"), ("fit_partition", "validation"),
                             ("fit_replay_ids", [1]), ("sigma", 0.)):
        changed = {**prior, key: replacement}
        with pytest.raises(ValueError, match="bound"):
            validate_training_prior(changed, dataset, split, 1.)


def test_upgrade_keeps_all_parent_weights_but_freezes_their_optimizer_scope(tmp_path):
    original, dataset, split, args = fixture(tmp_path)
    expected = {name: value.detach().clone() for name, value in original.state_dict().items()}
    upgraded, report = initialize_placement_stage(args, dataset, split)
    for name, value in expected.items():
        torch.testing.assert_close(upgraded.state_dict()[name], value, rtol=0, atol=0)
    assert not any(parameter.requires_grad for name, parameter in upgraded.named_parameters()
                   if not name.startswith("learned_placement."))
    assert all(parameter.requires_grad for name, parameter in upgraded.learned_placement.named_parameters()
               if not name.endswith(".anchor"))
    assert report["source_progress"] == {"epoch": 2, "next_episode": 1, "optimizer_steps": 123}
    assert report["stage_epoch_start"] == report["stage_optimizer_steps_start"] == 0
    assert not report["parent_optimizer_restored"] and not report["parent_epoch_restored"]
    assert upgraded.learned_placement.definition["temperature"] == 1.
    assert torch.count_nonzero(upgraded.learned_placement.decoder.original.original.tile_actor[-1].weight) == 0
    assert not any((args.out / ".replay-cache").glob("*"))


def test_unbound_external_prior_and_repeat_upgrade_are_rejected(tmp_path):
    _, dataset, split, args = fixture(tmp_path)
    args.placement_prior = tmp_path / "unsafe_prior.json"
    args.placement_prior.write_text(json.dumps({"mass": {}, "fit_games": 1, "sigma": 1.}))
    with pytest.raises(ValueError, match="bound"):
        initialize_placement_stage(args, dataset, split)
    args.placement_prior = None
    upgraded, _ = initialize_placement_stage(args, dataset, split)
    parent = tmp_path / "already_upgraded.pt"
    optimizer = torch.optim.AdamW((p for p in upgraded.parameters() if p.requires_grad), lr=3e-4)
    save_state_checkpoint(parent, upgraded, optimizer, None, dataset_fingerprint=dataset.fingerprint,
        split=split, epoch=0, next_episode=1, optimizer_steps=1, training_config={})
    args.upgrade_placement_from = parent
    with pytest.raises(ValueError, match="already has"):
        initialize_placement_stage(args, dataset, split)


def cli_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_state_bc.py"
    spec = importlib.util.spec_from_file_location("placement_stage_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upgrade_cli_rejects_ambiguous_initialization_and_unsupported_freeze_modes(tmp_path):
    module = cli_module()
    base = ["dataset.json", "--out", str(tmp_path / "out")]
    for flags in (["--upgrade-placement-from", "old.pt", "--resume", "latest.pt"],
                  ["--placement-prior", "unbound.json"], ["--placement-training-only"],
                  ["--upgrade-placement-from", "old.pt", "--no-placement-training-only"]):
        with pytest.raises(SystemExit):
            module.parse_args(base + flags)


def test_explicit_parent_sidecar_allows_repacking_but_not_split_or_game_changes(tmp_path):
    _, dataset, split, args = fixture(tmp_path)
    args.init_state_dataset_manifest = dataset.path
    relocated_manifest = dataset.path.with_name("repacked.manifest.json")
    relocated_manifest.write_text(json.dumps({**dataset.manifest, "packing_note": "same-games-new-paths"}))
    repacked = ReplaySequenceDataset(relocated_manifest, tmp_path / "repacked-cache", ram_limit_gib=0)
    assert repacked.fingerprint != dataset.fingerprint
    _, audit = initialize_placement_stage(args, repacked, split)
    assert audit["source_binding"]["same_replay_identities_and_partitions"]
    assert audit["source_dataset_fingerprint"] == dataset.fingerprint
    assert audit["source_binding"]["new_dataset_fingerprint"] == repacked.fingerprint
    with pytest.raises(ValueError, match="same replay identities"):
        initialize_placement_stage(args, repacked, {**split, 1: "train"})
    with pytest.raises(ValueError, match="same replay identities"):
        initialize_placement_stage(args, repacked, {0: "train", 1: "validation"})
    args.init_state_dataset_manifest = relocated_manifest
    with pytest.raises(ValueError, match="fingerprint"):
        initialize_placement_stage(args, repacked, split)


def test_cli_upgrade_and_exact_resume_restore_only_decoder_optimizer_and_stage_cursor(tmp_path, monkeypatch):
    original, dataset, split, args = fixture(tmp_path)
    module = cli_module()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"assignments": {"game-0": "train", "game-1": "validation",
                                                   "game-2": "heldout"}}))
    original_weights = {name: value.detach().clone() for name, value in original.state_dict().items()}
    out = tmp_path / "training"
    base = [str(dataset.path), "--out", str(out), "--device", "cpu", "--precision", "fp32",
            "--chunk-size", "2", "--frame-microbatch", "2", "--epochs", "2",
            "--max-episodes", "1", "--max-validation-episodes", "1", "--cpu-threads", "1",
            "--validation-every-episodes", "0", "--checkpoint-every-episodes", "0",
            "--split-registry", str(registry), "--learning-rate", "0.0003", "--no-initial-validation"]
    observed_scopes = []
    runner = module.run_state_episode

    def observed_runner(model, *runner_args, **runner_kwargs):
        optimizer = runner_kwargs.get("optimizer")
        if optimizer is not None:
            names = {name for name, p in model.named_parameters() if p.requires_grad}
            assert names and all(name.startswith("learned_placement.") for name in names)
            params = {id(p) for group in optimizer.param_groups for p in group["params"]}
            assert params == {id(p) for p in model.parameters() if p.requires_grad}
            observed_scopes.append(len(optimizer.state))
        return runner(model, *runner_args, **runner_kwargs)

    monkeypatch.setattr(module, "run_state_episode", observed_runner)
    first = module.run(module.parse_args(base + ["--upgrade-placement-from", str(args.upgrade_placement_from)]))
    assert first["epoch"] == 0 and first["next_episode"] == 1 and first["optimizer_steps"] == 1
    state = load_state_checkpoint(out / "latest.pt", dataset_fingerprint=dataset.fingerprint, split=split)
    stage = state["training_config"]["placement_stage"]
    assert stage["source_progress"]["optimizer_steps"] == 123
    assert state["learned_placement_definition"]["variant"] == "card_prior_cnn_001"
    restored = state_model_from_checkpoint(state)
    for name, expected in original_weights.items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    # The original actor remains byte-identical while at least one decoder parameter changed.
    fresh, _ = initialize_placement_stage(args, dataset, split)
    assert any(not torch.equal(value, fresh.state_dict()[name])
               for name, value in restored.state_dict().items()
               if isinstance(value, torch.Tensor) and name.startswith("learned_placement."))
    second = module.run(module.parse_args(base + ["--resume", str(out / "latest.pt")]))
    assert second["optimizer_steps"] == 2 and second["epoch"] == 1 and second["next_episode"] == 0
    assert observed_scopes[0] == 0 and observed_scopes[1] > 0
    resumed = load_state_checkpoint(out / "latest.pt", dataset_fingerprint=dataset.fingerprint, split=split)
    assert resumed["training_config"]["placement_stage"] == stage
    assert max(int(values["step"]) for values in resumed["optimizer"]["state"].values()) == 2
    for name, expected in original_weights.items():
        torch.testing.assert_close(resumed["model"][name], expected, rtol=0, atol=0)
    assert not list((out / ".replay-cache").iterdir())


@pytest.mark.parametrize("grouped", [False, True])
def test_frozen_stage_all_wait_does_not_backward_or_step_or_advance_generation(tmp_path, grouped):
    _, dataset, split, args = fixture(tmp_path)
    model, _ = initialize_placement_stage(args, dataset, split)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    iterator = dataset.episode_groups(split, "train", batch_size=2)
    try:
        _, episodes = next(iterator)
        wait_episodes = []
        for episode in episodes:
            # This tiny fixture must own its arrays before closing the mmap
            # generator: pytest may repr exception locals after teardown.
            arrays = {key: np.array(value, copy=True) for key, value in episode.arrays.items()}
            arrays["actions"] = np.zeros_like(arrays["actions"])
            wait_episodes.append(StateEpisode(episode.replay_id, episode.seat, episode.rows, arrays, 8))
    finally:
        iterator.close()
    del episode, episodes, iterator
    generation = model.memory_generation
    options = dict(optimizer=optimizer, chunk_size=2, sampled_action_prior=6 / 11)
    adapter = CrForgeObservationAdapter(dataset.schema, vocabulary_size=16)
    report = (run_state_episode_group(model, wait_episodes, adapter, **options) if grouped
              else run_state_episode(model, wait_episodes[0], adapter, **options))
    assert not report["optimizer_stepped"] and not report["amp_overflow_skipped"]
    assert not optimizer.state and generation == model.memory_generation
    assert all(p.grad is None for p in model.parameters())


def test_parallel_frozen_stage_handles_mixed_lengths_and_eval_padding(tmp_path):
    _, dataset, split, args = fixture(tmp_path)
    model, _ = initialize_placement_stage(args, dataset, split)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    iterator = dataset.episode_groups(split, "train", batch_size=2)
    try:
        _, mapped_episodes = next(iterator)
        episodes = [StateEpisode(e.replay_id, e.seat, e.rows.copy(),
                                {key: np.array(value, copy=True) for key, value in e.arrays.items()}, 8)
                    for e in mapped_episodes]
    finally:
        iterator.close()
    del mapped_episodes, iterator
    episodes[1] = StateEpisode(episodes[1].replay_id, episodes[1].seat,
                              episodes[1].rows[:3], episodes[1].arrays, 8)
    adapter = CrForgeObservationAdapter(dataset.schema, vocabulary_size=16)
    options = dict(chunk_size=2, sampled_action_prior=6 / 11)
    report = run_state_episode_group(model, episodes, adapter, optimizer=optimizer, **options)
    assert report["optimizer_stepped"] and report["memory_positions"] == [6, 3]
    assert report["history_frames"] == 9 and report["padded_frames"] == 12
    report = run_state_episode_group(model, episodes, adapter, **options)
    assert not report["optimizer_stepped"] and report["memory_positions"] == [6, 3]
    assert all(np.isfinite(row["loss"]) for row in report["reports"])
    # Padding must preserve each real stream's loss, not only suppress NaNs.
    for episode, grouped_report in zip(episodes, report["reports"], strict=True):
        individual = run_state_episode(model, episode, adapter, **options)
        for key in ("loss", "timing_loss", "conditional_slot_loss", "position_loss"):
            assert grouped_report[key] == pytest.approx(individual[key], abs=2e-5), key
