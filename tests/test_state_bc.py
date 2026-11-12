"""CPU-only streaming, episode-update, normalization, and resume checks."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.bc import hierarchical_bc_loss
from cr_rl.training.persistence import file_sha256
from cr_rl.training.state_bc import (
    ReplaySequenceDataset,
    StateEpisode,
    check_ram_budget,
    fixed_replay_split,
    load_state_checkpoint,
    restore_state_rng,
    run_state_episode,
    save_state_checkpoint,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def synthetic_dataset(path, games=3, *, decision_ticks=8):
    path.mkdir(parents=True, exist_ok=True)
    header_fields = ["entity_count", "elixir", *(["hand_card"] * 4),
                     *(["deck"] * 8), *(["enemy_seen"] * 8)]
    entity_fields = ["id", "kind", "side", "row", "x", "y", "state", "flags",
                     "buff_0", "buff_1", "buff_2"]
    metadata = {
        "decision_ticks": decision_ticks, "training_ready": True, "smoke_only": False,
        "schema": {"header_fields": header_fields, "entity_fields": entity_fields,
                   "header_ints": len(header_fields), "entity_ints": len(entity_fields),
                   "max_entities": 2, "seat_ints": len(header_fields) + 2 * len(entity_fields)},
        "vocabulary": {"names": [f"card_{i}" for i in range(16)]},
        "replays": [{"canonical_game_signature": f"game-{i}", "replay_tag": str(i)} for i in range(games)],
    }
    metadata_path = path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    records = []
    for game in range(games):
        n = 12
        obs = np.zeros((n, metadata["schema"]["seat_ints"]), dtype=np.int32)
        obs[:, 0] = 1
        obs[:, 1] = np.repeat(np.arange(6), 2) * 10000
        obs[:, 2:6] = [1, 2, 3, 4]
        obs[:, 6:14] = np.arange(8)
        legal = np.zeros((n, 2307), dtype=bool)
        legal[:, :37] = True
        legal[:, 577:613] = True
        legal[:, 2305:] = True
        labels = np.array([0, 0, 1, 577, 0, 0, 2305, 2306, 0, 0, 18, 577])
        flags = np.zeros(n, dtype=np.int32)
        flags[4] = 1  # History survives even when its training label is excluded.
        shard = path / f"replay_{game}.npz"
        np.savez_compressed(shard, observations=obs, masks=legal, actions=labels,
                            flags=flags, ticks=np.repeat(np.arange(6) * decision_ticks, 2),
                            seats=np.tile([0, 1], 6), replay_ids=np.zeros(n, dtype=np.int32))
        records.append({"path": shard.name, "sha256": file_sha256(shard), "replay_offset": game})
    manifest = path / "dataset.manifest.json"
    manifest.write_text(json.dumps({"format": "crrl-bc-shards-v1", "shards": records,
                                    "metadata": metadata_path.name,
                                    "metadata_sha256": file_sha256(metadata_path)}), encoding="utf-8")
    return manifest, metadata


def test_reader_never_loads_whole_observations_and_reclaims_its_scratch(tmp_path, monkeypatch):
    manifest, _ = synthetic_dataset(tmp_path / "data")
    original = np.lib.npyio.NpzFile.__getitem__

    def no_observation_materialization(self, key):
        assert key != "observations", "Compressed observations must be streamed to a memmap"
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", no_observation_materialization)
    cache = tmp_path / "scratch"
    dataset = ReplaySequenceDataset(manifest, cache)
    assignments = {0: "train", 1: "validation", 2: "heldout"}
    episodes = dataset.episodes(assignments, "train")
    episode = next(episodes)
    assert isinstance(episode.arrays["observations"], np.memmap)
    assert episode.rows.tolist() == [0, 2, 4, 6, 8, 10]
    assert episode.normalization.rows == 5
    assert episode.normalization.act == 3
    assert episode.normalization.play == 2
    assert len(list(cache.iterdir())) == 1
    episodes.close()
    assert not list(cache.iterdir())
    assert manifest.exists()


def test_fixed_split_keeps_both_seats_and_canonical_duplicates_together(tmp_path):
    manifest, metadata = synthetic_dataset(tmp_path / "data")
    metadata["replays"].append({**metadata["replays"][0], "source": "another-copy"})
    registry = {"game-0": "train", "game-1": "validation", "game-2": "heldout"}
    split = fixed_replay_split(metadata, [0, 1, 2, 3], registry=registry)
    assert split == {0: "train", 1: "validation", 2: "heldout", 3: "train"}
    assert split == fixed_replay_split(metadata, [3, 2, 1, 0], registry=registry)
    dataset = ReplaySequenceDataset(manifest, tmp_path / "scratch")
    episodes = dataset.episodes(split, "validation")
    try:
        assert [(episode.replay_id, episode.seat) for episode in episodes] == [(1, 0), (1, 1)]
    finally:
        episodes.close()
    counts = dataset.partition_counts(split, "train")
    assert counts == {"rows": 11, "act": 6, "play": 4, "wait": 5, "action_prior": 6 / 11}


def test_reader_rejects_wrong_grid_and_changed_metadata(tmp_path):
    manifest, _ = synthetic_dataset(tmp_path / "data", decision_ticks=5)
    with pytest.raises(ValueError, match="decision_ticks"):
        ReplaySequenceDataset(manifest, tmp_path / "scratch")
    (manifest.parent / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata checksum"):
        ReplaySequenceDataset(manifest, tmp_path / "scratch")


def test_legacy_registry_identity_preserves_existing_default_json_spacing():
    replay = {"source": "original", "source_revision": 7, "replay_tag": "a"}
    legacy_key = json.dumps(["original", 7, "a"])
    split = fixed_replay_split({"replays": [replay]}, [0], registry={legacy_key: "heldout"})
    assert split == {0: "heldout"}
    replay["canonical_game_signature"] = "newly-added-canonical-key"
    assert fixed_replay_split({"replays": [replay]}, [0], registry={legacy_key: "heldout"}) == split
    with pytest.raises(ValueError, match="registry keys disagree"):
        fixed_replay_split({"replays": [replay]}, [0], registry={
            legacy_key: "heldout", "newly-added-canonical-key": "train",
        })


@pytest.mark.parametrize("ticks", [[8, 16, 24], [0, 8, 24], [0, 8, 8]])
def test_episode_rejects_missing_opening_gaps_and_duplicate_ticks(ticks):
    arrays = {"actions": np.zeros(3, dtype=np.int64), "ticks": np.array(ticks),
              "flags": np.zeros(3), "masks": np.ones((3, 2307), dtype=bool)}
    with pytest.raises(ValueError, match="opening|time steps"):
        StateEpisode(0, 0, np.arange(3), arrays, 8)


class TinyStatePolicy(torch.nn.Module):
    """Deterministic stateful policy whose recurrence does not depend on weights."""
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(19)
        self.weight = torch.nn.Parameter(torch.randn(2, 7 + 4 * 576, generator=generator) * 0.03)
        self.cfg = SimpleNamespace(vocab_size=16)
        self.generation = 0
        self.seen = []

    def initial_state(self, batch_size=1, device=None, dtype=None):
        assert batch_size == 1
        return {"sum": torch.zeros((), device=device), "position": 0, "generation": self.generation}

    def forward_state(self, header, entities, entity_mask, legal_action_mask, *, state=None, **kwargs):
        assert state["generation"] == self.generation
        self.seen.append(state["position"])
        scalar = header[:, 1]
        cumulative = scalar.cumsum(0) + state["sum"]
        raw = torch.stack((scalar + 1, cumulative + 1), -1) @ self.weight
        logp = ActorCritic._joint_log_probs(raw[:, :7], raw[:, 7:].reshape(-1, 4, 576), legal_action_mask)
        return logp, scalar * 0, {"sum": cumulative[-1], "position": state["position"] + len(header),
                                 "generation": self.generation}

    def detach_state(self, state):
        return {**state, "sum": state["sum"].detach()}

    def mark_weights_updated(self):
        self.generation += 1

    def checkpoint_config(self):
        return {"model_architecture_version": 3, "config": {"fake": True},
                "temporal_config": {"fake": True}, "decision_ticks": 8}


@pytest.mark.parametrize("timing", ["natural", "balanced"])
def test_optimizer_runs_once_after_all_frames_and_chunk_contributions_match(tmp_path, timing):
    manifest, _ = synthetic_dataset(tmp_path / "data")
    dataset = ReplaySequenceDataset(manifest, tmp_path / "scratch")
    episodes = dataset.episodes({0: "train", 1: "validation", 2: "heldout"}, "train")
    try:
        episode = next(episodes)
        adapter = CrForgeObservationAdapter(dataset.schema, vocabulary_size=16)
        whole, split = TinyStatePolicy(), TinyStatePolicy()
        from cr_rl.training.state_bc import prepare_state_chunk
        inputs = prepare_state_chunk(episode, episode.rows, adapter, "cpu")
        logp, _, _ = whole.forward_state(**inputs, state=whole.initial_state())
        valid = torch.tensor(episode.supervised)
        labels = torch.tensor(episode.labels)
        expected, _ = hierarchical_bc_loss(
            logp[valid], labels[valid], inputs["legal_action_mask"][valid],
            conditional_card_loss=True, position_coef=0.5, tile_label_sigma=1.0,
            timing_prior=0.2 if timing == "natural" else None, sampled_action_prior=6 / 11,
        )
        expected.backward()
        initial = split.weight.detach().clone()

        class CheckedOptimizer(torch.optim.SGD):
            steps = 0

            def step(self, closure=None):
                assert split.seen == [0, 2, 4]
                self.steps += 1
                return super().step(closure)

        optimizer = CheckedOptimizer(split.parameters(), lr=0.1)
        report = run_state_episode(
            split, episode, adapter, chunk_size=2, optimizer=optimizer,
            sampled_action_prior=6 / 11, timing_objective=timing, max_grad_norm=1e6,
        )
        torch.testing.assert_close(split.weight, initial - 0.1 * whole.weight.grad,
                                   atol=2e-6, rtol=2e-5)
        assert report["loss"] == pytest.approx(float(expected.detach()), abs=2e-6)
        assert optimizer.steps == 1 and split.generation == 1
        assert report["history_frames"] == 6 and report["rows"] == 5
        assert report["optimizer_stepped"] and not report["amp_overflow_skipped"]
    finally:
        episodes.close()


def test_checkpoint_is_versioned_dataset_bound_and_resumes_at_episode_boundary(tmp_path):
    model = TinyStatePolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    split = {0: "train", 1: "validation"}
    checkpoint = tmp_path / "latest.pt"
    save_state_checkpoint(checkpoint, model, optimizer, None, dataset_fingerprint="dataset-a", split=split,
                          epoch=2, next_episode=5, optimizer_steps=11, training_config={"chunk_size": 2})
    state = load_state_checkpoint(checkpoint, dataset_fingerprint="dataset-a", split=split)
    assert state["progress"] == {"epoch": 2, "next_episode": 5, "optimizer_steps": 11}
    assert state["model_architecture_version"] == 3
    assert "memory" not in state and "temporal_state" not in state
    restore_state_rng(state)
    with pytest.raises(ValueError, match="fingerprint"):
        load_state_checkpoint(checkpoint, dataset_fingerprint="dataset-b", split=split)
    with pytest.raises(ValueError, match="split"):
        load_state_checkpoint(checkpoint, dataset_fingerprint="dataset-a", split={0: "validation"})
    del state
    old = tmp_path / "old.pt"
    torch.save({"config": {}, "model": {}}, old)
    with pytest.raises(ValueError, match="independent state BC"):
        load_state_checkpoint(old, dataset_fingerprint="dataset-a", split=split)


def test_ram_guard_reports_and_rejects_over_budget():
    assert check_ram_budget(0) > 0
    with pytest.raises(MemoryError, match="RAM guard"):
        check_ram_budget(1e-9)


def test_real_tiny_model_continuous_training_and_independent_resume(tmp_path):
    from cr_rl.models.state_policy import StateActorCritic, state_model_from_checkpoint
    from cr_rl.models.state_temporal import StateTemporalConfig
    manifest, _ = synthetic_dataset(tmp_path / "data")
    dataset = ReplaySequenceDataset(manifest, tmp_path / "scratch")
    cfg = ModelConfig(entity_dim=8, entity_heads=2, entity_layers=1, temporal_dim=16,
                      temporal_heads=2, temporal_layers=2, header_dim=dataset.schema.header_ints,
                      entity_feature_dim=dataset.schema.entity_ints, vocab_size=16)
    cfg = cfg.for_v2(dataset.schema, "dynamic4")
    temporal = StateTemporalConfig(width=16, heads=2, kv_heads=1, head_dim=8,
                                   pattern=("kda", "gqa"), chunk_size=2)
    model = StateActorCritic(cfg, temporal, frame_microbatch=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    adapter = CrForgeObservationAdapter(dataset.schema, vocabulary_size=16)
    episodes = dataset.episodes({0: "train", 1: "validation", 2: "heldout"}, "train")
    try:
        episode = next(episodes)
        report = run_state_episode(model, episode, adapter, chunk_size=2, optimizer=optimizer,
                                   sampled_action_prior=6 / 11)
        assert np.isfinite(report["loss"]) and report["optimizer_stepped"]
        checkpoint = tmp_path / "latest.pt"
        split = {0: "train", 1: "validation", 2: "heldout"}
        save_state_checkpoint(checkpoint, model, optimizer, None, dataset_fingerprint=dataset.fingerprint,
                              split=split, epoch=0, next_episode=1, optimizer_steps=1)
        state = load_state_checkpoint(checkpoint, dataset_fingerprint=dataset.fingerprint, split=split)
        restored = state_model_from_checkpoint(state)
        evaluation = run_state_episode(restored, episode, adapter, chunk_size=3,
                                       sampled_action_prior=6 / 11)
        assert np.isfinite(evaluation["loss"]) and not evaluation["optimizer_stepped"]
        assert restored.memory_generation == 0
    finally:
        episodes.close()


def test_cli_cpu_tiny_smoke_and_resume_without_restarting_episode_zero(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_state_bc.py"
    spec = importlib.util.spec_from_file_location("state_bc_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest, _ = synthetic_dataset(tmp_path / "data")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"assignments": {"game-0": "train", "game-1": "validation",
                                                     "game-2": "heldout"}}), encoding="utf-8")
    monkeypatch.setattr(module.ModelConfig, "preset", lambda _: ModelConfig(
        entity_dim=8, entity_heads=2, entity_layers=1, temporal_dim=16, temporal_heads=2,
        temporal_layers=2,
    ))
    output = tmp_path / "run"
    argv = [str(manifest), "--out", str(output), "--model", "tiny", "--device", "cpu",
            "--precision", "fp32", "--chunk-size", "2", "--frame-microbatch", "2",
            "--temporal-width", "16", "--temporal-heads", "2", "--kv-heads", "1",
            "--epochs", "2", "--max-episodes", "1", "--max-validation-episodes", "1",
            "--split-registry", str(registry), "--validation-every-episodes", "0",
            "--checkpoint-every-episodes", "0"]
    first = module.run(module.parse_args(argv))
    assert first["optimizer_steps"] == 1 and first["next_episode"] == 1
    second = module.run(module.parse_args([*argv, "--resume", str(output / "latest.pt")]))
    assert second["optimizer_steps"] == 2 and second["epoch"] == 1 and second["next_episode"] == 0
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    training = [row for row in rows if row["phase"] == "state_bc_train"]
    assert [row["seat"] for row in training] == [0, 1]
    assert [row["episode_index"] for row in training] == [0, 1]
    assert not list((output / ".replay-cache").iterdir())
    assert not list(output.glob("*.tmp"))


@pytest.mark.parametrize("targets", [[0, 1, 2305, 577], [0, 2305, 0, 2306], [0, 0, 0, 0]],
                         ids=["mixed", "skills_no_placement", "all_wait"])
@pytest.mark.parametrize("sigma", [0.0, 1.0], ids=["hard", "gaussian"])
@pytest.mark.parametrize("prior", [None, 0.20], ids=["balanced", "natural"])
def test_selective_placement_preserves_supervised_loss_and_every_parameter_gradient(
    tmp_path, monkeypatch, targets, sigma, prior
):
    from cr_rl.models.state_policy import StateActorCritic
    from cr_rl.models.state_temporal import StateTemporalConfig
    from cr_rl.training.state_bc import prepare_state_chunk

    manifest, _ = synthetic_dataset(tmp_path / "data")
    dataset = ReplaySequenceDataset(manifest, tmp_path / "scratch")
    cfg = ModelConfig(entity_dim=8, entity_heads=2, entity_layers=1, temporal_dim=16,
                      temporal_heads=2, temporal_layers=2, header_dim=dataset.schema.header_ints,
                      entity_feature_dim=dataset.schema.entity_ints, vocab_size=16)
    cfg = cfg.for_v2(dataset.schema, "dynamic4")
    temporal = StateTemporalConfig(width=16, heads=2, kv_heads=1, head_dim=8,
                                   pattern=("kda", "gqa"), chunk_size=2)
    torch.manual_seed(41)
    whole = StateActorCritic(cfg, temporal, frame_microbatch=2).train()
    selected = StateActorCritic(cfg, temporal, frame_microbatch=2).train()
    selected.load_state_dict(whole.state_dict())
    adapter = CrForgeObservationAdapter(dataset.schema, vocabulary_size=16)
    episodes = dataset.episodes({0: "train", 1: "validation", 2: "heldout"}, "train")
    try:
        episode = next(episodes)
        inputs = prepare_state_chunk(episode, episode.rows[:4], adapter, "cpu")
        labels = torch.tensor(targets)
        play = (labels > 0) & (labels < 2305)
        calls = []
        original = selected.placement_logits

        def count_rows(context, entities, entity_mask, card_features):
            calls.append(len(context))
            return original(context, entities, entity_mask, card_features)

        monkeypatch.setattr(selected, "placement_logits", count_rows)
        logp_whole, _, memory_whole = whole.forward_state(**inputs, compute_value=False)
        logp_selected, _, memory_selected = selected.forward_state(
            **inputs, placement_mask=play, compute_value=False
        )
        assert sum(calls) == int(play.sum()), "Only placement-labeled rows should invoke the tile head"
        cards_whole = torch.cat((logp_whole[:, :1], torch.logsumexp(
            logp_whole[:, 1:2305].reshape(-1, 4, 576), -1), logp_whole[:, 2305:]), -1)
        cards_selected = torch.cat((logp_selected[:, :1], torch.logsumexp(
            logp_selected[:, 1:2305].reshape(-1, 4, 576), -1), logp_selected[:, 2305:]), -1)
        torch.testing.assert_close(cards_selected, cards_whole, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(logp_selected[play], logp_whole[play], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(memory_selected.temporal.positions, memory_whole.temporal.positions)
        assert memory_selected.temporal.length_bound == memory_whole.temporal.length_bound == 4
        options = {"conditional_card_loss": True, "position_coef": 0.7,
                   "tile_label_sigma": sigma, "timing_prior": prior, "sampled_action_prior": 0.4}
        loss_whole, _ = hierarchical_bc_loss(logp_whole, labels, inputs["legal_action_mask"], **options)
        loss_selected, _ = hierarchical_bc_loss(logp_selected, labels, inputs["legal_action_mask"], **options)
        torch.testing.assert_close(loss_selected, loss_whole, atol=2e-6, rtol=2e-6)
        loss_whole.backward()
        loss_selected.backward()
        for (name, full_parameter), (selected_name, selected_parameter) in zip(
            whole.named_parameters(), selected.named_parameters(), strict=True
        ):
            assert name == selected_name
            full_grad = full_parameter.grad if full_parameter.grad is not None else torch.zeros_like(full_parameter)
            selected_grad = selected_parameter.grad if selected_parameter.grad is not None else torch.zeros_like(selected_parameter)
            torch.testing.assert_close(selected_grad, full_grad, atol=3e-6, rtol=2e-4, msg=name)
        selected.eval()
        with pytest.raises(ValueError, match="supervised-training"):
            selected.forward_state(**inputs, placement_mask=play, compute_value=False)
    finally:
        episodes.close()


def state_cli_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_state_bc.py"
    spec = importlib.util.spec_from_file_location("state_transfer_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state_transfer_fixture(tmp_path):
    from cr_rl.models.state_policy import StateActorCritic
    from cr_rl.models.state_temporal import StateTemporalConfig

    module = state_cli_module()
    parent_manifest, _ = synthetic_dataset(tmp_path / "parent_data")
    new_manifest, metadata = synthetic_dataset(tmp_path / "new_data")
    metadata["supervision_protocol"] = "preserve_all_recorded_actions_new_stage"
    metadata_path = new_manifest.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest = json.loads(new_manifest.read_text())
    manifest["metadata_sha256"] = file_sha256(metadata_path)
    new_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    parent_dataset = ReplaySequenceDataset(parent_manifest, tmp_path / "parent_scratch")
    dataset = ReplaySequenceDataset(new_manifest, tmp_path / "new_scratch")
    cfg = ModelConfig(entity_dim=8, entity_heads=2, entity_layers=1, temporal_dim=16,
                      temporal_heads=2, temporal_layers=2, header_dim=dataset.schema.header_ints,
                      entity_feature_dim=dataset.schema.entity_ints, vocab_size=16,
                      max_entities=dataset.schema.max_entities).for_v2(dataset.schema, "dynamic4")
    temporal = StateTemporalConfig(width=16, heads=2, kv_heads=1, head_dim=8,
                                   pattern=("kda", "gqa"), chunk_size=2)
    torch.manual_seed(17)
    parent = StateActorCritic(cfg, temporal, frame_microbatch=2)
    optimizer = torch.optim.AdamW(parent.parameters(), lr=0.0007)
    # Populate parent optimizer moments without requiring an episode/GPU.
    for parameter in parent.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = tmp_path / "parent_state.pt"
    save_state_checkpoint(checkpoint, parent, optimizer, None,
                          dataset_fingerprint=parent_dataset.fingerprint,
                          split={0: "train", 1: "validation", 2: "heldout"},
                          epoch=2, next_episode=7, optimizer_steps=256,
                          training_config={"learning_rate": 0.0007})
    argv = [str(new_manifest), "--out", str(tmp_path / "new_run"), "--device", "cpu",
            "--precision", "fp32", "--chunk-size", "2", "--frame-microbatch", "2",
            "--init-from-state", str(checkpoint), "--init-state-dataset-manifest", str(parent_manifest),
            "--decision-ticks", "8", "--epochs", "1", "--max-episodes", "1",
            "--max-validation-episodes", "1", "--validation-every-episodes", "0",
            "--checkpoint-every-episodes", "0"]
    return module, parent, checkpoint, parent_manifest, dataset, argv


def test_state_transfer_preserves_every_weight_and_temporal_output_on_new_dataset(state_transfer_fixture):
    module, parent, checkpoint, parent_manifest, dataset, argv = state_transfer_fixture
    # Parent data are an identity sidecar only; no old observation shards are
    # needed or materialized to transfer a frozen state checkpoint.
    for shard in parent_manifest.parent.glob("replay_*.npz"):
        shard.unlink()
    args = module.parse_args(argv)
    torch.manual_seed(args.seed)
    transferred, report = module.build_model(args, dataset)
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(transferred.state_dict()[name], value, atol=0, rtol=0, msg=name)
    assert transferred.temporal_cfg == parent.temporal_cfg
    assert transferred.decision_ticks == 8
    assert report["parent_decision_ticks"] == 8 and report["optimizer_steps_base"] == 256
    assert report["temporal_weights_preserved"] and not report["exact_resume"]
    assert not report["optimizer_migrated"] and not report["scaler_migrated"] and not report["rng_restored"]
    assert report["parent_weight_sha256"] == file_sha256(checkpoint)
    assert report["parent_dataset_fingerprint"] == file_sha256(parent_manifest) != dataset.fingerprint
    assert report["new_dataset_fingerprint"] == dataset.fingerprint
    assert report["parent_vocabulary_sha256"] == report["new_vocabulary_sha256"]
    assert torch.initial_seed() == args.seed
    fresh = transferred.initial_state()
    assert fresh.temporal.length_bound == 0 and int(fresh.temporal.positions.max()) == 0
    from cr_rl.training.state_bc import prepare_state_chunk
    episodes = dataset.episodes({0: "train", 1: "validation", 2: "heldout"}, "train")
    try:
        episode = next(episodes)
        inputs = prepare_state_chunk(episode, episode.rows, CrForgeObservationAdapter(dataset.schema, 16), "cpu")
        parent.eval()
        transferred.eval()
        with torch.no_grad():
            expected = parent.forward_state(**inputs, compute_value=False)[0]
            actual = transferred.forward_state(**inputs, compute_value=False)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        episodes.close()
    with pytest.raises(ValueError, match="fingerprint"):
        load_state_checkpoint(checkpoint, dataset_fingerprint=dataset.fingerprint,
                              split={0: "train", 1: "validation", 2: "heldout"})


def test_state_transfer_resets_optimizer_and_epoch_but_preserves_global_counter(
    state_transfer_fixture, tmp_path, monkeypatch
):
    module, _, _, _, dataset, argv = state_transfer_fixture
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"assignments": {"game-0": "train", "game-1": "validation",
                                                     "game-2": "heldout"}}), encoding="utf-8")
    original = module.run_state_episode
    seen = []

    def verify_optimizer(*args, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            seen.append(len(optimizer.state))
            if len(seen) == 1:
                assert len(optimizer.state) == 0
            assert optimizer.param_groups[0]["lr"] == 3e-5
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_state_episode", verify_optimizer)
    args = module.parse_args([*argv, "--split-registry", str(registry)])
    report = module.run(args)
    assert report["optimizer_steps"] == 257 and report["epoch"] == 0 and report["next_episode"] == 1
    saved = torch.load(args.out / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["dataset_fingerprint"] == dataset.fingerprint
    assert saved["progress"] == {"epoch": 0, "next_episode": 1, "optimizer_steps": 257}
    assert saved["training_config"]["initialization"]["optimizer_steps_base"] == 256
    assert max(int(values["step"]) for values in saved["optimizer"]["state"].values()) == 1
    resume_argv = [str(dataset.path), "--out", str(args.out), "--device", "cpu", "--precision", "fp32",
                   "--chunk-size", "2", "--frame-microbatch", "2", "--epochs", "1",
                   "--max-episodes", "1", "--max-validation-episodes", "1",
                   "--validation-every-episodes", "0", "--checkpoint-every-episodes", "0",
                   "--split-registry", str(registry), "--resume", str(args.out / "latest.pt")]
    resumed = module.run(module.parse_args(resume_argv))
    assert resumed["optimizer_steps"] == 258 and resumed["epoch"] == 1 and resumed["next_episode"] == 0
    assert seen == [0, len(saved["optimizer"]["state"])]


@pytest.mark.parametrize("mismatch", ["width", "heads", "kv_heads", "pattern", "vocabulary", "schema", "fingerprint", "metadata_checksum"])
def test_state_transfer_rejects_incompatible_config_and_unbound_metadata(state_transfer_fixture, mismatch):
    module, _, _, parent_manifest, dataset, argv = state_transfer_fixture
    args = module.parse_args(argv)
    if mismatch == "width":
        args.temporal_width = 32
    elif mismatch == "heads":
        args.temporal_heads = 4
    elif mismatch == "kv_heads":
        args.kv_heads = 2
    elif mismatch == "pattern":
        args.pattern = "gqa,gqa"
    elif mismatch == "vocabulary":
        dataset.metadata["vocabulary"]["names"][1:3] = dataset.metadata["vocabulary"]["names"][2:0:-1]
    elif mismatch == "schema":
        dataset.metadata["schema"]["header_fields"][1] = "changed_elixir"
    elif mismatch == "fingerprint":
        parent_manifest.write_text(parent_manifest.read_text() + " ")
    elif mismatch == "metadata_checksum":
        metadata = parent_manifest.parent / "metadata.json"
        metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ValueError, match="temporal|vocabulary|schema|fingerprint|checksum"):
        module.build_model(args, dataset)


@pytest.mark.parametrize("flags", [
    ["--resume", "a.pt", "--init-from-state", "b.pt"],
    ["--init-from-bc", "a.pt", "--init-from-state", "b.pt"],
    ["--resume", "a.pt", "--init-from-bc", "b.pt"],
    ["--init-from-state", "a.pt"],
    ["--init-state-dataset-manifest", "parent.json"],
])
def test_state_initialization_modes_are_mutually_exclusive_and_sidecar_is_required(flags):
    with pytest.raises(SystemExit):
        state_cli_module().parse_args(["dataset.json", "--out", "newrun", *flags])
