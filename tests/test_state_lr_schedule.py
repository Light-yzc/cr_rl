"""Successful-update LR scheduling and exact BC handoff/resume, CPU only."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from cr_rl.training.lr_schedule import (
    WarmupCosineLR,
    cosine_config,
    learning_rate_at,
    remaining_episode_updates,
)


def configuration(**changes):
    return cosine_config(**{
        "peak_lr": 3e-5, "min_lr": 3e-6, "warmup_steps": 350, "total_steps": 34650,
        "anchor_step": 606, "total_epochs": 100, "start_progress": {"epoch": 1, "next_episode": 0},
        "train_seats": 350, **changes,
    })


def test_warmup_cosine_boundaries_and_monotonic_segments():
    config = configuration()
    assert learning_rate_at(config, 0) == pytest.approx(3e-6)
    assert learning_rate_at(config, 350) == pytest.approx(3e-5)
    middle = (config["total_steps"] + 350) // 2
    assert learning_rate_at(config, middle) == pytest.approx((3e-5 + 3e-6) / 2)
    assert learning_rate_at(config, 34650) == pytest.approx(3e-6)
    assert learning_rate_at(config, 40000) == pytest.approx(3e-6)
    warm = [learning_rate_at(config, index) for index in range(351)]
    decay = [learning_rate_at(config, index) for index in range(350, 34651, 100)]
    assert warm == sorted(warm)
    assert decay == sorted(decay, reverse=True)


def test_schedule_horizon_counts_only_remaining_seats_in_target_total_epochs():
    assert remaining_episode_updates(100, {"epoch": 1, "next_episode": 128}, 350) == 34522
    assert remaining_episode_updates(100, {"epoch": 0, "next_episode": 256}, 350) == 34744
    with pytest.raises(ValueError, match="No remaining"):
        remaining_episode_updates(100, {"epoch": 100, "next_episode": 0}, 350)
    with pytest.raises(ValueError, match="exceeds"):
        remaining_episode_updates(100, {"epoch": 1, "next_episode": 351}, 350)


def test_overflow_never_advances_schedule_and_restart_preserves_exact_next_lr():
    config = configuration()
    optimizer = SimpleNamespace(param_groups=[{"lr": 3e-5}])
    schedule = WarmupCosineLR(config)
    schedule.apply(optimizer)
    before = deepcopy(schedule.state_dict())
    schedule.after_attempt(optimizer, successful=False)
    assert schedule.state_dict() == before and optimizer.param_groups[0]["lr"] == 3e-6
    for _ in range(777):
        schedule.after_attempt(optimizer, successful=True)
    restored = WarmupCosineLR.from_state_dict(schedule.state_dict(), expected_config=config,
                                             optimizer_step=606 + 777, optimizer=optimizer)
    assert restored.state_dict() == schedule.state_dict()
    for success in (False, True, True, False, True):
        schedule.after_attempt(optimizer, successful=success)
        restored.after_attempt(optimizer, successful=success)
        assert restored.state_dict() == schedule.state_dict()
    with pytest.raises(ValueError, match="position"):
        WarmupCosineLR.from_state_dict(schedule.state_dict(), expected_config=config, optimizer_step=1)


@pytest.mark.parametrize("options", [
    {"min_lr": 4e-5}, {"min_lr": -1.}, {"warmup_steps": 34650}, {"total_steps": 0},
    {"anchor_step": -1}, {"peak_lr": float("nan")},
])
def test_invalid_schedule_bounds_are_rejected(options):
    with pytest.raises(ValueError):
        configuration(**options)


def cli():
    from test_state_bc import state_cli_module
    return state_cli_module()


@pytest.mark.parametrize("flags", [
    ["--lr-schedule", "constant", "--warmup-steps", "350"],
    ["--warmup-steps", "350"],
    ["--lr-schedule", "cosine", "--warmup-steps", "-1"],
    ["--lr-schedule", "cosine", "--schedule-total-steps", "0"],
    ["--lr-schedule", "cosine", "--min-learning-rate", "0.1"],
    ["--start-lr-schedule"],
    ["--resume", "a.pt", "--lr-schedule", "constant", "--start-lr-schedule"],
])
def test_cli_rejects_ambiguous_or_invalid_schedule_flags(flags):
    with pytest.raises(SystemExit):
        cli().parse_args(["dataset.json", "--out", "newrun", *flags])


def test_runtime_performance_flags_are_explicit_and_fused_is_cuda_only(tmp_path):
    module = cli()
    args = module.parse_args(["missing.json", "--out", str(tmp_path), "--device", "cpu", "--precision", "fp32",
                             "--fused-optimizer", "--defer-metrics", "--float32-matmul-precision", "high"])
    assert args.fused_optimizer and args.defer_metrics and args.float32_matmul_precision == "high"
    with pytest.raises(ValueError, match="requires CUDA"):
        module.run(args)


def test_constant_checkpoint_definition_stays_unchanged_and_only_explicit_handoff_is_allowed(tmp_path):
    module = cli()
    parent = tmp_path / "latest.pt"
    parent.write_bytes(b"parent checkpoint")
    args = module.parse_args(["dataset.json", "--out", str(tmp_path / "out"), "--resume", str(parent), "--epochs", "100"])
    definition = {"learning_rate": 3e-5, "position_coef": 1.0}
    state = {"training_config": deepcopy(definition), "progress": {"epoch": 1, "next_episode": 128, "optimizer_steps": 734}}
    assert module.resolve_lr_schedule(args, definition, state=state, train_seats=350) is None
    assert definition == state["training_config"]
    args.lr_schedule = "cosine"
    with pytest.raises(ValueError, match="explicit --start"):
        module.resolve_lr_schedule(args, definition, state=state, train_seats=350)
    args.start_lr_schedule = True
    config = module.resolve_lr_schedule(args, definition, state=state, train_seats=350)
    assert config["anchor_step"] == 734 and config["total_steps"] == 34522
    assert config["transition"]["model_optimizer_rng_preserved"]
    changed = {"learning_rate": 3e-5, "position_coef": .5}
    with pytest.raises(ValueError, match="training configuration differs"):
        module.resolve_lr_schedule(args, changed, state=state, train_seats=350)


def test_tiny_cpu_constant_handoff_and_cosine_resume_preserve_model_adam_rng_and_position(tmp_path, monkeypatch):
    from test_state_bc import synthetic_dataset
    module = cli()
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    manifest, _ = synthetic_dataset(tmp_path / "data")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"assignments": {"game-0": "train", "game-1": "validation", "game-2": "heldout"}}))
    monkeypatch.setattr(module.ModelConfig, "preset", lambda _: module.ModelConfig(
        entity_dim=8, entity_heads=2, entity_layers=1, temporal_dim=16, temporal_heads=2, temporal_layers=2))
    output = tmp_path / "run"
    common = [str(manifest), "--out", str(output), "--model", "tiny", "--device", "cpu", "--precision", "fp32",
              "--chunk-size", "2", "--frame-microbatch", "2", "--temporal-width", "16", "--temporal-heads", "2",
              "--kv-heads", "1", "--epochs", "3", "--max-episodes", "1", "--max-validation-episodes", "1",
              "--split-registry", str(registry), "--validation-every-episodes", "0", "--checkpoint-every-episodes", "0"]
    common += ["--defer-metrics"]
    try:
        module.run(module.parse_args(common))
        path = output / "latest.pt"
        parent = torch.load(path, map_location="cpu", weights_only=False)
        assert "lr_schedule" not in parent and "lr_schedule" not in parent["training_config"]
        initial_rng = parent["rng"]["torch"].clone()
        original_episode = module.run_state_episode
        first = True

        def inspect_handoff(model, episode, adapter, **kwargs):
            nonlocal first
            optimizer = kwargs.get("optimizer")
            if optimizer is not None and first:
                first = False
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, parent["model"][name], atol=0, rtol=0)
                assert len(optimizer.state) == len(parent["optimizer"]["state"])
                torch.testing.assert_close(torch.get_rng_state(), initial_rng, atol=0, rtol=0)
                assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)
                actual = optimizer.state_dict()["state"]
                for index, old in parent["optimizer"]["state"].items():
                    torch.testing.assert_close(actual[index]["exp_avg"], old["exp_avg"], atol=0, rtol=0)
            return original_episode(model, episode, adapter, **kwargs)

        monkeypatch.setattr(module, "run_state_episode", inspect_handoff)
        handoff = module.parse_args([*common, "--resume", str(path), "--lr-schedule", "cosine",
                                    "--start-lr-schedule", "--warmup-steps", "1"])
        second = module.run(handoff)
        assert second["optimizer_steps"] == 2 and second["epoch"] == 1 and second["next_episode"] == 0
        scheduled = torch.load(path, map_location="cpu", weights_only=False)
        assert scheduled["lr_schedule"]["successful_updates"] == 1
        assert scheduled["lr_schedule"]["config"]["anchor_step"] == 1
        assert scheduled["lr_schedule"]["config"]["total_steps"] == 5
        assert scheduled["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-5)
        third = module.run(module.parse_args([*common, "--resume", str(path), "--no-defer-metrics"]))
        assert third["optimizer_steps"] == 3
        restored = torch.load(path, map_location="cpu", weights_only=False)
        assert restored["lr_schedule"]["successful_updates"] == 2
        assert restored["lr_schedule"]["config"] == scheduled["lr_schedule"]["config"]
        assert "defer_metrics" not in restored["training_config"]
        runtime = json.loads((output / "run_config.json").read_text())["runtime_performance"]
        assert not runtime["defer_metrics"] and not runtime["fused_optimizer"]
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        train = [row for row in rows if row["phase"] == "state_bc_train"]
        assert [row["learning_rate"] for row in train] == pytest.approx([3e-5, 3e-6, 3e-5])
        with pytest.raises(ValueError, match="already occurred"):
            module.run(module.parse_args([*common, "--resume", str(path), "--start-lr-schedule"]))
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Existing CUDA required for fused Adam resume")
def test_tiny_cuda_fp16_resume_switches_old_adam_to_fused_with_cuda_step_tensors(tmp_path, monkeypatch):
    from test_state_bc import synthetic_dataset
    module = cli()
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    manifest, _ = synthetic_dataset(tmp_path / "data")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"assignments": {"game-0": "train", "game-1": "validation", "game-2": "heldout"}}))
    monkeypatch.setattr(module.ModelConfig, "preset", lambda _: module.ModelConfig(
        entity_dim=8, entity_heads=2, entity_layers=1, temporal_dim=16, temporal_heads=2, temporal_layers=2))
    output = tmp_path / "run"
    common = [str(manifest), "--out", str(output), "--model", "tiny", "--device", "cuda", "--precision", "fp16",
              "--chunk-size", "2", "--frame-microbatch", "2", "--temporal-width", "16", "--temporal-heads", "2",
              "--kv-heads", "1", "--epochs", "2", "--max-episodes", "1", "--max-validation-episodes", "1",
              "--split-registry", str(registry), "--validation-every-episodes", "0", "--checkpoint-every-episodes", "0",
              "--frame-encoder-fp32"]
    try:
        module.run(module.parse_args(common))
        checkpoint = output / "latest.pt"
        original_episode = module.run_state_episode
        checked = []

        def verify_fused(model, episode, adapter, **kwargs):
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                assert all(group["fused"] and group["foreach"] is False for group in optimizer.param_groups)
                assert optimizer.state
                for parameter, values in optimizer.state.items():
                    if "step" in values:
                        assert values["step"].device == parameter.device
                        assert values["step"].dtype == torch.float32
                checked.append(True)
            return original_episode(model, episode, adapter, **kwargs)

        monkeypatch.setattr(module, "run_state_episode", verify_fused)
        report = module.run(module.parse_args([*common, "--resume", str(checkpoint), "--fused-optimizer", "--defer-metrics"]))
        assert report["optimizer_steps"] == 2 and checked == [True]
        runtime = json.loads((output / "run_config.json").read_text())["runtime_performance"]
        assert runtime["fused_optimizer"] and runtime["defer_metrics"]
    finally:
        torch.set_num_threads(previous)
        torch.cuda.empty_cache()
