"""Small CPU tests; no production checkpoint, dataset extraction or GPU run."""
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_representation_v2 import configuration
from cr_rl.models import ActorCritic
from cr_rl.models.state_policy import PolicyMemory, StateActorCritic, state_model_from_checkpoint
from cr_rl.models.state_temporal import GQAState, KDAState, StateTemporalConfig


def test_slice_memory_batched_position_bound_matches_exact_trim():
    from cr_rl.models.state_policy import slice_memory
    from cr_rl.models.state_temporal import TemporalState

    key = torch.randn(3, 2, 16, 8)
    value = torch.randn_like(key)
    memory = PolicyMemory(TemporalState(
        [KDAState(torch.randn(3, 4, 8, 8)), GQAState(key, value)],
        torch.tensor([0, 3, 7]), 7,
    ), owner=123, generation=4)
    for i, bound in enumerate(memory.temporal.positions.tolist()):
        expected = slice_memory(memory, [i], trim_gqa=True)
        actual = slice_memory(memory, [i], trim_gqa=True, position_bound=bound)
        assert actual.temporal.length_bound == expected.temporal.length_bound
        torch.testing.assert_close(actual.temporal.positions, expected.temporal.positions)
        torch.testing.assert_close(actual.temporal.layers[0].matrix, expected.temporal.layers[0].matrix)
        torch.testing.assert_close(actual.temporal.layers[1].key, expected.temporal.layers[1].key)
        torch.testing.assert_close(actual.temporal.layers[1].value, expected.temporal.layers[1].value)
        actual.temporal.layers[1].key.zero_()
        assert torch.equal(memory.temporal.layers[1].key, key)
        assert actual.temporal.layers[1].key.data_ptr() != key.data_ptr()


def test_frozen_checkpoint_disk_cache_is_bounded_and_invalidates(tmp_path):
    from cr_rl.training.host_memory import cache_frozen_checkpoint

    source = tmp_path / "source.pt"
    source.write_bytes(b"first")
    cache = tmp_path / "cache"
    staged = cache_frozen_checkpoint(source, cache, max_bytes=8)
    assert staged != source and staged.read_bytes() == b"first"
    assert cache_frozen_checkpoint(source, cache, max_bytes=8) == staged
    source.write_bytes(b"second")
    updated = cache_frozen_checkpoint(source, cache, max_bytes=8)
    assert updated.read_bytes() == b"second" and updated != staged
    assert sum(p.stat().st_size for p in cache.glob("*.pt")) <= 8
    assert source.read_bytes() == b"second"
    assert cache_frozen_checkpoint(source, cache, max_bytes=1) == source.resolve()


def model_pair():
    torch.manual_seed(270)
    cfg = replace(configuration(), readout_version=2)
    temporal = StateTemporalConfig(width=32, heads=4, kv_heads=2, head_dim=8,
                                   pattern=("kda", "gqa"), ffn_multiplier=2,
                                   chunk_size=4, cache_initial_capacity=4)
    return cfg, StateActorCritic(cfg, temporal, frame_microbatch=3).eval()


def frames(batch=2, length=7):
    torch.manual_seed(39)
    header = torch.zeros(batch, length, 8)
    header[..., 0] = torch.rand(batch, length)
    header[..., 1:] = 4 / 1466
    entities = torch.zeros(batch, length, 4, 11)
    entities[..., 3] = 1 / 1466
    entities[..., 4:6] = torch.rand(batch, length, 4, 2)
    entities[..., 7] = 1 / 65535
    entities[..., 8:] = -1
    mask = torch.tensor([True, True, False, False]).expand(batch, length, 4).clone()
    legal = torch.ones(batch, length, 2307, dtype=torch.bool)
    legal[..., 2306] = False
    hands = torch.arange(4).expand(batch, length, 4)
    return header, entities, mask, legal, hands


def test_forward_state_matches_incremental_and_chunked_policy_distributions():
    _, model = model_pair()
    h, e, m, legal, hands = frames()
    with torch.inference_mode():
        expected, expected_value, memory = model.forward_state(h, e, m, legal, hand_card_ids=hands)
        state = model.initial_state(2)
        outputs, values = [], []
        for t in range(h.shape[1]):
            out, value, state = model.step(h[:, t], e[:, t], m[:, t], legal[:, t],
                                           state=state, hand_card_ids=hands[:, t], compute_value=True)
            outputs.append(out); values.append(value)
        torch.testing.assert_close(torch.stack(outputs, 1), expected, atol=4e-6, rtol=2e-5)
        torch.testing.assert_close(torch.stack(values, 1), expected_value, atol=3e-6, rtol=2e-5)
        prefix, _, state = model.forward_state(h[:, :3], e[:, :3], m[:, :3], legal[:, :3], hand_card_ids=hands[:, :3])
        suffix, _, _ = model.forward_state(h[:, 3:], e[:, 3:], m[:, 3:], legal[:, 3:],
                                           state=state, hand_card_ids=hands[:, 3:])
        torch.testing.assert_close(torch.cat((prefix, suffix), 1), expected, atol=4e-6, rtol=2e-5)
        assert memory.temporal.positions.tolist() == [7, 7]
        torch.testing.assert_close(expected.exp().sum(-1), torch.ones(2, 7), atol=2e-6, rtol=2e-6)


def test_single_stream_chronological_input_has_same_output():
    _, model = model_pair()
    h, e, m, legal, hands = frames(batch=1)
    with torch.inference_mode():
        batch = model.forward_state(h, e, m, legal, hand_card_ids=hands)
        single = model.forward_state(h[0], e[0], m[0], legal[0], hand_card_ids=hands[0])
    torch.testing.assert_close(single[0], batch[0][0], atol=0, rtol=0)
    torch.testing.assert_close(single[1], batch[1][0], atol=0, rtol=0)


def test_reset_and_policy_version_prevent_cross_game_or_weight_cache_reuse():
    _, model = model_pair()
    h, e, m, legal, hands = frames()
    with torch.inference_mode():
        _, _, memory = model.forward_state(h, e, m, legal, hand_card_ids=hands)
        reset = PolicyMemory(memory.temporal.reset(torch.tensor([True, False])),
                             memory.owner, memory.generation)
        output = model.step(h[:, 0], e[:, 0], m[:, 0], legal[:, 0], state=reset, hand_card_ids=hands[:, 0])[0]
        fresh = model.step(h[:1, 0], e[:1, 0], m[:1, 0], legal[:1, 0], hand_card_ids=hands[:1, 0])[0]
        torch.testing.assert_close(output[0], fresh[0], atol=4e-6, rtol=2e-5)
        other = model_pair()[1]
        with pytest.raises(ValueError, match="different policy"):
            other.step(h[:, 0], e[:, 0], m[:, 0], legal[:, 0], state=memory)
        model.mark_weights_updated()
        with pytest.raises(ValueError, match="stale"):
            model.step(h[:, 0], e[:, 0], m[:, 0], legal[:, 0], state=memory)


def test_import_copies_existing_representation_without_replacing_new_core():
    cfg, model = model_pair()
    old = ActorCritic(cfg)
    before = {name: value.clone() for name, value in model.state_dict().items() if name.startswith("temporal.")}
    source = {name: value.clone() for name, value in old.state_dict().items()}
    memory = model.initial_state(2)
    report = model.migrate_bc_representation(source)
    assert report["copied_tensors"] > 10
    assert report["initial_policy_equivalent"] is False
    assert report["optimizer_migrated"] is False
    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)
    for name, expected in source.items():
        if name in model.state_dict() and expected.shape == model.state_dict()[name].shape and not name.startswith(("temporal.", "position")):
            torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)
        torch.testing.assert_close(old.state_dict()[name], expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="stale"):
        model.detach_state(memory)


def test_checkpoint_roundtrip_rebuilds_nonpersistent_spatial_and_rope_buffers():
    _, model = model_pair()
    checkpoint = {**model.checkpoint_config(), "model": model.state_dict()}
    restored = state_model_from_checkpoint(checkpoint).eval()
    h, e, m, legal, hands = frames(batch=1, length=3)
    with torch.inference_mode():
        expected = model.forward_state(h, e, m, legal, hand_card_ids=hands)[0]
        actual = restored.forward_state(h, e, m, legal, hand_card_ids=hands)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="architecture-v3"):
        state_model_from_checkpoint({"model_architecture_version": 2})


def test_explicit_fp16_memory_uses_fp32_delta_matrix_and_fp16_kv():
    _, model = model_pair()
    memory = model.initial_state(2, dtype=torch.float16)
    for layer in memory.temporal.layers:
        if isinstance(layer, KDAState):
            assert layer.matrix.dtype == torch.float32
        elif isinstance(layer, GQAState):
            assert layer.key.dtype == layer.value.dtype == torch.float16


def test_implicit_memory_matches_active_autocast_precision():
    _,model=model_pair()
    h,e,m,legal,hands=frames(batch=1,length=3)
    with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
        output,_,memory=model.forward_state(h,e,m,legal,hand_card_ids=hands)
    assert torch.isfinite(output).all()
    for layer in memory.temporal.layers:
        if isinstance(layer,GQAState):
            assert layer.key.dtype==layer.value.dtype==torch.bfloat16
        else:
            assert layer.matrix.dtype==torch.float32


def test_fp32_ppo_actor_matches_streaming_under_outer_autocast():
    _, model = model_pair()
    model.ppo_actor_fp32 = True
    h, e, mask, legal, hands = frames(batch=2, length=9)
    original_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            state = None
            streamed = []
            for t in range(h.shape[1]):
                row, _, state = model.step(h[:, t], e[:, t], mask[:, t], legal[:, t],
                                           hand_card_ids=hands[:, t], state=state)
                streamed.append(row)
        model.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            sequence, _, _ = model.forward_state(h, e, mask, legal, hand_card_ids=hands)
        torch.testing.assert_close(sequence, torch.stack(streamed, dim=1), atol=1e-5, rtol=1e-6)
        assert all(layer.key.dtype == torch.float32 for layer in state.temporal.layers
                   if isinstance(layer, GQAState))
        (-sequence[..., 1].mean()).backward()
        assert model.wait_actor.weight.grad is not None
    finally:
        torch.backends.mha.set_fastpath_enabled(original_fastpath)


def test_streamed_fixture_reads_one_history_only_and_retains_one_frame(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_state_policy.py"
    spec = importlib.util.spec_from_file_location("benchmark_state_policy", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    windows = np.arange(4 * 5 * 12, dtype=np.int32).reshape(4, 5, 12)
    masks = np.ones((4, 7), dtype=bool)
    fixture = tmp_path / "probe.npz"
    np.savez_compressed(fixture, windows=windows, masks=masks)
    raw, legal, report = benchmark.read_frame_fixture(fixture)
    np.testing.assert_array_equal(raw, windows[0, -1])
    np.testing.assert_array_equal(legal, masks[0])
    assert report["decompressed_array_bytes_read_or_skipped"] == 5 * 12 * 4 + 7
    assert report["retained_raw_bytes"] == 12 * 4 + 7
    assert benchmark.statistics([1, 2, 3])["max_ms"] == 3


def benchmark_module():
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_state_policy.py"
    spec = importlib.util.spec_from_file_location("benchmark_state_policy_weights", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_benchmark_cli_disallows_mixing_legacy_and_trained_weights():
    benchmark = benchmark_module()
    args = benchmark.parse_args(["--state-checkpoint", "parent.pt", "--width", "1280", "--out", "report.json"])
    assert args.state_checkpoint == Path("parent.pt") and args.checkpoint is None
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--state-checkpoint", "parent.pt", "--checkpoint", "bc.pt", "--out", "report.json"])


def test_state_benchmark_strictly_restores_all_temporal_weights(tmp_path):
    benchmark = benchmark_module()
    fixture_cfg, model = model_pair()
    checkpoint = {**model.checkpoint_config(), "model": model.state_dict()}
    path = tmp_path / "parent_state.pt"
    torch.save(checkpoint, path)
    args = SimpleNamespace(state_checkpoint=path, checkpoint=None, width=32, streams=[1, 8])
    loaded, migration, report = benchmark.load_benchmark_model(args, fixture_cfg, "cpu")
    assert migration is None and report["strict"] is True
    assert report["loaded_temporal_tensors"] > 0 and loaded.frame_microbatch == 8
    for name, expected in checkpoint["model"].items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, atol=0, rtol=0)
    h, e, mask, legal, hands = frames(batch=1, length=3)
    with torch.inference_mode():
        expected = model.forward_state(h, e, mask, legal, hand_card_ids=hands)[0]
        actual = loaded.eval().forward_state(h, e, mask, legal, hand_card_ids=hands)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_state_benchmark_rejects_wrong_width_schema_interval_and_missing_core_key(tmp_path):
    benchmark = benchmark_module()
    fixture_cfg, model = model_pair()
    checkpoint = {**model.checkpoint_config(), "model": model.state_dict()}
    path = tmp_path / "parent_state.pt"
    torch.save(checkpoint, path)
    args = SimpleNamespace(state_checkpoint=path, checkpoint=None, width=512, streams=[1])
    with pytest.raises(ValueError, match="width"):
        benchmark.load_benchmark_model(args, fixture_cfg, "cpu")
    args.width = 32
    with pytest.raises(ValueError, match="schema mismatch"):
        benchmark.load_benchmark_model(args, replace(fixture_cfg, vocab_size=100), "cpu")
    interval_path = tmp_path / "wrong_interval.pt"
    torch.save({**checkpoint, "decision_ticks": 5}, interval_path)
    args.state_checkpoint = interval_path
    with pytest.raises(ValueError, match="decision_ticks=8"):
        benchmark.load_benchmark_model(args, fixture_cfg, "cpu")
    missing = dict(checkpoint["model"])
    missing.pop(next(name for name in missing if name.startswith("temporal.")))
    missing_path = tmp_path / "missing_core.pt"
    torch.save({**checkpoint, "model": missing}, missing_path)
    args.state_checkpoint = missing_path
    with pytest.raises(RuntimeError, match="Missing key"):
        benchmark.load_benchmark_model(args, fixture_cfg, "cpu")
