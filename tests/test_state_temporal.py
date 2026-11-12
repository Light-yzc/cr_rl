"""CPU correctness tests for explicit KDA/GQA state, independent of a policy."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from cr_rl.models.state_temporal import (
    GQAState, KDAState, StateTemporalConfig, StateTemporalCore,
    kda_chunk, kda_recurrent,
)


def configuration(pattern=("kda", "gqa")):
    return StateTemporalConfig(width=32, heads=4, kv_heads=2, head_dim=8,
                               pattern=pattern, chunk_size=4, cache_initial_capacity=4,
                               ffn_multiplier=2)


def kda_inputs(length=13, dtype=torch.float32, gradients=False):
    torch.manual_seed(215)
    shape = (2, 2, length, 8)
    tensors = [F.normalize(torch.randn(shape), dim=-1), F.normalize(torch.randn(shape), dim=-1),
               torch.randn(shape), -F.softplus(torch.randn(shape) - 2),
               torch.randn(shape[:-1]).sigmoid(), torch.randn(2, 2, 8, 8) * 0.1]
    return [x.to(dtype).requires_grad_(gradients) for x in tensors]


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 32])
@pytest.mark.parametrize("masked", [False, True])
def test_kda_chunk_matches_matrix_recurrence_and_all_input_gradients(chunk_size, masked):
    args = kda_inputs(13, gradients=True)
    mask = torch.tensor([[1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                         [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]], dtype=torch.bool) if masked else None
    expected, expected_state = kda_recurrent(*args, time_mask=mask)
    actual, actual_state = kda_chunk(*args, time_mask=mask, chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-6, rtol=1e-5)
    expected_grads = torch.autograd.grad(expected.square().mean() + expected_state.square().mean(), args)
    actual_grads = torch.autograd.grad(actual.square().mean() + actual_state.square().mean(), args)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=3e-5)


def test_kda_chunk_splits_and_initial_state_keep_the_same_gradient():
    args = kda_inputs(17, gradients=True)
    whole, final = kda_chunk(*args, chunk_size=5)
    state = args[-1]
    parts = []
    for a, b in [(0, 2), (2, 11), (11, 17)]:
        output, state = kda_chunk(*(x[:, :, a:b] for x in args[:5]), state, chunk_size=3)
        parts.append(output)
    torch.testing.assert_close(torch.cat(parts, 2), whole, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(state, final, atol=2e-6, rtol=1e-5)
    a = torch.autograd.grad(whole.sum() + final.sum(), args)
    b = torch.autograd.grad(torch.cat(parts, 2).sum() + state.sum(), args)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=3e-5)


def test_fp16_long_kda_remains_finite_with_fp32_state():
    args = kda_inputs(257, dtype=torch.float16, gradients=True)
    result, state = kda_chunk(*args, chunk_size=32)
    expected, expected_state = kda_recurrent(*args)
    assert result.dtype == state.dtype == torch.float32
    assert torch.isfinite(result).all() and torch.isfinite(state).all()
    torch.testing.assert_close(result, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(state, expected_state, atol=3e-6, rtol=3e-5)
    (result.square().mean() + state.square().mean()).backward()
    assert all(torch.isfinite(x.grad).all() for x in args)


def test_autocast_does_not_downcast_kda_matrix_math():
    args = kda_inputs(9)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, state = kda_chunk(*args)
    expected, expected_state = kda_chunk(*args)
    assert output.dtype == state.dtype == torch.float32
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


@pytest.mark.parametrize("pattern", [("kda",), ("gqa",), ("kda", "gqa", "kda")])
def test_forward_prefill_step_and_reference_match(pattern):
    torch.manual_seed(42)
    model = StateTemporalCore(configuration(pattern))
    x = torch.randn(2, 15, 32)
    full = model(x)
    reference = model.prefill(x, reference=True)[0]
    state = model.init_state(2)
    steps = []
    for t in range(x.shape[1]):
        output, state = model.step(x[:, t], state)
        steps.append(output)
    torch.testing.assert_close(torch.stack(steps, 1), full, atol=4e-6, rtol=2e-5)
    torch.testing.assert_close(reference, full, atol=4e-6, rtol=2e-5)
    prefix, state = model.prefill(x[:, :7])
    suffix, state = model.prefill(x[:, 7:], state)
    torch.testing.assert_close(torch.cat([prefix, suffix], 1), full, atol=4e-6, rtol=2e-5)
    assert state.positions.tolist() == [15, 15]
    for memory in state.layers:
        if isinstance(memory, GQAState):
            assert memory.key.shape[2] >= 15  # No sliding truncation.


def test_core_chunk_gradients_match_reference_with_full_history_gqa():
    torch.manual_seed(63)
    actual_model = StateTemporalCore(configuration())
    expected_model = deepcopy(actual_model)
    x1 = torch.randn(2, 9, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    expected = expected_model.prefill(x2, reference=True)[0]
    _, state = actual_model.prefill(x1[:, :3])
    actual = actual_model.prefill(x1[:, 3:], state)[0]
    expected[:, 3:].square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=3e-6, rtol=3e-4)
    for (name1, p1), (name2, p2) in zip(actual_model.named_parameters(), expected_model.named_parameters()):
        assert name1 == name2
        torch.testing.assert_close(p1.grad, p2.grad, atol=5e-6, rtol=4e-4)


def test_full_core_is_causal():
    torch.manual_seed(17)
    model = StateTemporalCore(configuration())
    x = torch.randn(2, 12, 32)
    changed = x.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 5
    first = model(x)
    second = model(changed)
    torch.testing.assert_close(first[:, :7], second[:, :7], atol=2e-6, rtol=1e-5)
    assert not torch.allclose(first[:, 7:], second[:, 7:])


def test_padding_matches_compact_streams_and_never_advances_memory():
    torch.manual_seed(30)
    model = StateTemporalCore(configuration())
    x = torch.randn(2, 11, 32)
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.bool)
    batched, state = model.prefill(x, time_mask=mask)
    assert not batched[~mask].count_nonzero()
    assert state.positions.tolist() == mask.sum(-1).tolist()
    for row in range(2):
        expected, memory = model.prefill(x[row:row+1, mask[row]])
        torch.testing.assert_close(batched[row, mask[row]], expected[0], atol=4e-6, rtol=2e-5)
        for a, b in zip(state.layers, memory.layers):
            if isinstance(a, KDAState):
                torch.testing.assert_close(a.matrix[row], b.matrix[0], atol=3e-6, rtol=2e-5)
    unchanged, after = model.step(torch.randn(2, 32), state, torch.zeros(2, dtype=torch.bool))
    assert not unchanged.count_nonzero()
    torch.testing.assert_close(after.positions, state.positions)
    for a, b in zip(after.layers, state.layers):
        if isinstance(a, KDAState):
            torch.testing.assert_close(a.matrix, b.matrix, atol=0, rtol=0)


def test_selective_reset_isolates_episode_and_preserves_other_stream():
    torch.manual_seed(50)
    model = StateTemporalCore(configuration())
    _, state = model.prefill(torch.randn(2, 13, 32))
    reset = state.reset(torch.tensor([True, False]))
    x = torch.randn(2, 32)
    actual, next_state = model.step(x, reset)
    fresh = model(x[:1, None])[0, 0]
    preserved = model.step(x, state)[0][1]
    torch.testing.assert_close(actual[0], fresh, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(actual[1], preserved, atol=3e-6, rtol=2e-5)
    assert next_state.positions.tolist() == [1, 14]
    assert state.positions.tolist() == [13, 13]  # Original state was not mutated.
    all_reset = state.reset()
    assert all_reset.positions.tolist() == [0, 0]
    assert all_reset.length_bound == 0


def test_detach_truncates_gradient_but_keeps_forward_history():
    torch.manual_seed(94)
    model = StateTemporalCore(configuration())
    prefix = torch.randn(2, 7, 32, requires_grad=True)
    suffix = torch.randn(2, 3, 32, requires_grad=True)
    _, state = model.prefill(prefix)
    expected = model.prefill(suffix, state)[0]
    actual, detached = model.prefill(suffix, state.detach(), detach_state=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    assert prefix.grad is None
    assert suffix.grad is not None
    assert all(not (s.matrix if isinstance(s, KDAState) else s.key).requires_grad for s in detached.layers)


def test_fp16_core_projects_half_and_keeps_memory_accumulation_float():
    torch.manual_seed(52)
    model = StateTemporalCore(configuration()).half()
    x = torch.randn(2, 9, 32, dtype=torch.float16, requires_grad=True)
    output, state = model.prefill(x)
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    for memory in state.layers:
        assert (memory.matrix.dtype if isinstance(memory, KDAState) else memory.key.dtype) == (
            torch.float32 if isinstance(memory, KDAState) else torch.float16)
    output.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_rope_frequencies_remain_exact_float32_after_half_conversion():
    model = StateTemporalCore(configuration(("gqa",)))
    expected = model.blocks[0].mixer.inv_frequency.clone()
    model.half()
    actual = model.blocks[0].mixer.inv_frequency
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_512_and_768_configuration_and_invalid_shapes():
    assert StateTemporalConfig(width=512, heads=8, kv_heads=2).width == 512
    assert StateTemporalConfig().width == 768
    for kwargs in [dict(width=31), dict(kv_heads=5), dict(chunk_size=0), dict(pattern=("gru",))]:
        with pytest.raises(ValueError):
            StateTemporalConfig(**kwargs)
    model = StateTemporalCore(configuration())
    with pytest.raises(ValueError, match="time_mask"):
        model(torch.randn(2, 3, 32), torch.ones(2, 4))
    empty, state = model.prefill(torch.randn(2, 0, 32))
    assert empty.shape == (2, 0, 32) and state.positions.tolist() == [0, 0]
