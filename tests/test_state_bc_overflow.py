"""A finite element gradient can still have an overflowing FP32 norm."""
import copy

import pytest
import torch

from cr_rl.training.state_bc import _state_bc_optimizer_step


@pytest.mark.parametrize('gradient', [1e20, float('inf')])
def test_rejected_norm_preserves_weights_adam_and_successful_step_count(gradient):
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=32)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    assert _state_bc_optimizer_step(model, optimizer, scaler, 1.)[0]
    before = model.weight.detach().clone()
    adam_before = copy.deepcopy(optimizer.state[model.weight])
    old_scale = scaler.get_scale()
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    model.weight.grad.fill_(gradient * old_scale)
    if gradient != float('inf'):
        assert torch.isfinite(model.weight.grad).all()
    stepped, overflow, norm = _state_bc_optimizer_step(model, optimizer, scaler, 1.)
    assert not stepped and overflow and norm == 0.
    assert torch.equal(model.weight, before)
    for key, value in adam_before.items():
        torch.testing.assert_close(optimizer.state[model.weight][key], value, rtol=0, atol=0)
    assert model.weight.grad is None and scaler.get_scale() == old_scale / 2
    # Recovery must not leave GradScaler in its previous optimizer-stage state.
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    assert _state_bc_optimizer_step(model, optimizer, scaler, 1.)[0]
    assert int(optimizer.state[model.weight]['step']) == 2
    assert not torch.equal(model.weight, before)


def test_without_amp_rejects_aggregate_overflow_before_weight_decay():
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, 1e20)
    with pytest.raises(FloatingPointError, match='Nonfinite state BC gradients'):
        _state_bc_optimizer_step(model, optimizer, None, 1.)
    assert torch.equal(model.weight, before) and not optimizer.state
    assert model.weight.grad is None
