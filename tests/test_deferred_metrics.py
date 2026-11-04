import copy

import numpy as np
import pytest
import torch
from test_separate_critic import setup_models

from cr_rl.training.separate import forward_inputs, optimize_separate


def test_deferred_scalar_metrics_keep_parameter_updates_and_kl_decisions_identical():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    _, actor, critic, inputs = setup_models()
    original_actor, original_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    with torch.no_grad():
        old, _ = forward_inputs(actor.eval(), inputs)
    results, weights = [], []
    for deferred, cached in ((False, False), (True, False), (True, True)):
        actor, critic = copy.deepcopy(original_actor), copy.deepcopy(original_critic)
        behavior = copy.deepcopy(actor).eval().requires_grad_(False)
        result = optimize_separate(
            actor,
            critic,
            behavior,
            torch.optim.AdamW(actor.parameters(), lr=1e-5),
            torch.optim.AdamW(critic.parameters(), lr=1e-4),
            lambda rows: tuple(t[rows] for t in inputs),
            np.arange(4),
            np.arange(4),
            old[:, 0],
            torch.zeros(4, dtype=torch.long),
            torch.ones(4),
            torch.ones(4),
            rng=np.random.default_rng(9),
            device="cpu",
            batch_size=2,
            accumulation=2,
            actor_epochs=2,
            critic_epochs=2,
            clip_coef=0.15,
            entropy_coef=0.005,
            reference_kl_coef=0.01,
            target_kl=0.02,
            max_grad_norm=0.5,
            defer_scalar_metrics=deferred,
            cache_reference=cached,
        )
        results.append(result)
        weights.append({k: v.clone() for k, v in actor.state_dict().items()})
    for index in (1, 2):
        assert results[0]["actor_steps"] == results[index]["actor_steps"]
        assert results[0]["actor_kl_stopped"] == results[index]["actor_kl_stopped"]
        # Cache hits reuse rows computed in a different batch order. CPU GEMM may
        # change the final bits; scalar deferral alone must remain bit-exact.
        if index == 1:
            assert results[0]["metrics"] == results[index]["metrics"]
        else:
            assert results[0]["metrics"] == pytest.approx(
                results[index]["metrics"], abs=1e-7, rel=1e-6
            )
        for key in weights[0]:
            torch.testing.assert_close(
                weights[0][key],
                weights[index][key],
                atol=0 if index == 1 else 1e-7,
                rtol=0,
            )
