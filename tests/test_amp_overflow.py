import pytest
import torch

from cr_rl.training.separate import _optimizer_step


def test_amp_overflow_preserves_parameters_and_adam_then_recovers():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.amp_scaler = torch.amp.GradScaler('cpu', init_scale=128)
    scaler = optimizer.amp_scaler
    before = model.weight.detach().clone()
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    model.weight.grad.fill_(float('inf'))
    assert _optimizer_step(model, optimizer, 1, 0.5) is None
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert not optimizer.state
    assert model.weight.grad is None
    assert scaler.get_scale() == 64
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    assert _optimizer_step(model, optimizer, 1, 0.5) == pytest.approx(1)
    assert int(optimizer.state[model.weight]['step']) == 1
    assert not torch.equal(model.weight, before)


def test_without_amp_nonfinite_gradients_still_raise():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model.weight.grad = torch.full_like(model.weight, float('inf'))
    with pytest.raises(RuntimeError, match='non-finite'):
        _optimizer_step(model, optimizer, 1, 0.5)
    assert not optimizer.state
