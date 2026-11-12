"""Hardware integration gates for concurrent actor/value optimization and AMP reload."""

import copy

import numpy as np
import pytest
import torch
from test_separate_critic import setup_models

from cr_rl.training.fp16 import configure_optimizer_amp
from cr_rl.training.separate import autocast_for, forward_inputs, optimize_separate


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_parallel_update_and_amp_restore(monkeypatch, tmp_path, precision):
    monkeypatch.setenv("CRRL_AMP_DTYPE", precision)
    torch.manual_seed(123)
    _, actor, critic, inputs = setup_models()
    actor, critic = actor.to("cuda:0"), critic.to("cuda:1")
    inputs = tuple(t.to("cuda:0") for t in inputs)
    actor.eval()
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    ao = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    co = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    configure_optimizer_amp(ao, "cuda:0")
    configure_optimizer_amp(co, "cuda:1")
    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]
    for _ in range(2):
        behavior.load_state_dict(actor.state_dict())
        with torch.no_grad(), autocast_for("cuda:0"):
            logp, _ = forward_inputs(actor, inputs)
        result = optimize_separate(
            actor,
            critic,
            behavior,
            ao,
            co,
            lambda rows: tuple(t[rows] for t in inputs),
            np.arange(4),
            np.arange(4),
            logp[:, 0],
            torch.zeros(4, dtype=torch.long, device="cuda:0"),
            torch.ones(4, device="cuda:0"),
            torch.ones(4, device="cuda:0"),
            rng=np.random.default_rng(42),
            device="cuda:0",
            critic_device="cuda:1",
            batch_size=2,
            accumulation=2,
            actor_epochs=1,
            critic_epochs=1,
            clip_coef=0.15,
            entropy_coef=0.005,
            reference_kl_coef=0.01,
            target_kl=0.02,
            max_grad_norm=0.5,
            defer_scalar_metrics=True,
            max_initial_logp_error=1e-3,
        )
        assert result["actor_steps"] == result["critic_steps"] == 1
        assert np.isfinite(result["metrics"]["value_loss"])
        assert max(result["initial_logp_errors"]) < 1e-3
        for model in (actor, critic):
            assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(actor.parameters(), actor_before, strict=True))
    assert any(
        not torch.equal(a, b) for a, b in zip(critic.parameters(), critic_before, strict=True)
    )
    path = tmp_path / "amp.pt"
    torch.save(
        {
            "model": actor.state_dict(),
            "optimizer": ao.state_dict(),
            "scaler": ao.amp_scaler.state_dict(),
        },
        path,
    )
    saved = torch.load(path, weights_only=False)
    actor.load_state_dict(saved["model"])
    restored = torch.optim.AdamW(actor.parameters())
    restored.load_state_dict(saved["optimizer"])
    configure_optimizer_amp(restored, "cuda:0", saved["scaler"])
    assert restored.amp_scaler.state_dict() == ao.amp_scaler.state_dict()
