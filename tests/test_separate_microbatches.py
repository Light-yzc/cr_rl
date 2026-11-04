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


def test_actor_effective_batch_changes_only_actor_updates_and_scales_partial_tail():
    torch.set_num_threads(1)
    _, initial_actor, initial_critic, inputs = setup_models()
    behavior = copy.deepcopy(initial_actor).eval().requires_grad_(False)
    with torch.no_grad():
        old, _ = forward_inputs(behavior, inputs)
    old = old[:, 0].clone().requires_grad_(True)
    advantages = torch.tensor([1.0, -0.5, 0.25, -1.0], requires_grad=True)
    targets = torch.ones(4, requires_grad=True)
    snapshots = [v.detach().clone() for v in (old, advantages, targets)]

    def run(actor_effective):
        actor, critic = copy.deepcopy(initial_actor), copy.deepcopy(initial_critic)
        events = []
        result = optimize_separate(
            actor,
            critic,
            behavior,
            torch.optim.AdamW(actor.parameters(), lr=1e-5),
            torch.optim.AdamW(critic.parameters(), lr=1e-4),
            lambda rows: tuple(t[rows] for t in inputs),
            np.arange(4),
            np.arange(4),
            old,
            torch.zeros(4, dtype=torch.long),
            advantages,
            targets,
            rng=np.random.default_rng(18),
            device="cpu",
            batch_size=1,
            accumulation=2,
            actor_effective_batch_size=actor_effective,
            actor_epochs=1,
            critic_epochs=1,
            clip_coef=0.15,
            entropy_coef=0.005,
            reference_kl_coef=0.01,
            target_kl=1,
            max_grad_norm=0.5,
            emit=lambda **event: events.append(event),
        )
        return result, actor, critic, events

    default, _, default_critic, _ = run(None)
    larger, larger_actor, larger_critic, larger_events = run(4)
    tail, tail_actor, tail_critic, _ = run(8)
    assert default["actor_steps"] == 2
    assert larger["actor_steps"] == tail["actor_steps"] == 1
    assert default["critic_steps"] == larger["critic_steps"] == tail["critic_steps"] == 2
    for key, tensor in default_critic.state_dict().items():
        torch.testing.assert_close(larger_critic.state_dict()[key], tensor, rtol=0, atol=0)
        torch.testing.assert_close(tail_critic.state_dict()[key], tensor, rtol=0, atol=0)
    # Four rows at a partial eight-row boundary must be divided by four, not eight.
    for key, tensor in larger_actor.state_dict().items():
        torch.testing.assert_close(tail_actor.state_dict()[key], tensor, rtol=0, atol=0)
    assert len([e for e in larger_events if e["phase"] == "optimization"]) == 1
    for tensor, snapshot in zip((old, advantages, targets), snapshots, strict=True):
        assert tensor.grad is None
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    assert all(p.grad is None for p in behavior.parameters())


def test_actor_effective_batch_kl_guard_waits_for_whole_actor_group():
    torch.set_num_threads(1)
    _, actor, critic, inputs = setup_models()
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    with torch.no_grad():
        old, _ = forward_inputs(behavior, inputs)
        actor.wait_actor.bias.add_(3)
    events = []
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
        rng=np.random.default_rng(18),
        device="cpu",
        batch_size=1,
        accumulation=2,
        actor_effective_batch_size=4,
        actor_epochs=1,
        critic_epochs=1,
        clip_coef=0.15,
        entropy_coef=0.005,
        reference_kl_coef=0.01,
        target_kl=1e-6,
        max_grad_norm=0.5,
        emit=lambda **event: events.append(event),
    )
    assert result["actor_steps"] == 0 and result["actor_kl_stopped"]
    assert result["critic_steps"] == 2
    stopped = next(e for e in events if e["phase"] == "kl_stop")
    assert stopped["kl_samples"] == stopped["discarded_microbatches"] == 4
    assert all(p.grad is None for p in actor.parameters())


def test_v2_checks_both_effective_batch_minimums_before_starting_simulator(monkeypatch):
    from types import SimpleNamespace

    import cr_rl.training.selfplay_v2 as training

    calls = []

    class Runner:
        def __init__(self, _args):
            calls.append("started")

        def run_update(self):
            calls.append("updated")

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(training, "PersistentSelfplay", Runner)
    args = SimpleNamespace(
        batch_size=8,
        grad_accumulation=8,
        actor_effective_batch_size=32,
        critic_mode="independent",
        updates=1,
    )
    with pytest.raises(ValueError, match="at least 64"):
        training.run(args)
    assert not calls
    args.actor_effective_batch_size = 256
    training.run(args)
    assert calls == ["started", "updated", "closed"]
    calls.clear()
    args.grad_accumulation = 4
    with pytest.raises(ValueError, match="at least 64"):
        training.run(args)
    assert not calls
