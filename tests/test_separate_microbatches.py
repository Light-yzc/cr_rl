import copy

import numpy as np
import pytest
import torch
from test_separate_critic import setup_models

from cr_rl.training.separate import forward_inputs, optimize_separate


def test_independent_microbatches_keep_effective_batch_and_optimizer_counts():
    torch.set_num_threads(1)
    _, actor, critic, inputs = setup_models()
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    with torch.no_grad():
        old, _ = forward_inputs(actor.eval(), inputs)
    kwargs = dict(
        rng=np.random.default_rng(18),
        device="cpu",
        batch_size=2,
        accumulation=2,
        actor_epochs=1,
        critic_epochs=1,
        clip_coef=0.15,
        entropy_coef=0.005,
        reference_kl_coef=0.01,
        target_kl=1,
        max_grad_norm=0.5,
        actor_batch_size=1,
        critic_batch_size=4,
    )
    args = (
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
    )
    result = optimize_separate(*args, **kwargs)
    assert result["actor_steps"] == result["critic_steps"] == 1
    kwargs["critic_batch_size"] = 3
    with pytest.raises(ValueError, match="fixed effective batch"):
        optimize_separate(*args, **kwargs)
