import copy

import numpy as np
import pytest
import torch

from cr_rl.models import ActorCritic, ActorPolicy, HistoryValueNetwork, ModelConfig
from cr_rl.training.separate import (
    complete_episode_returns,
    forward_inputs,
    optimize_separate,
)


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def setup_models():
    cfg = ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=32,
        temporal_heads=2,
        temporal_layers=1,
        context_len=4,
        max_entities=3,
        header_dim=8,
        entity_feature_dim=6,
        vocab_size=32,
    )
    shared = ActorCritic(cfg)
    actor = ActorPolicy.from_shared(shared)
    critic = HistoryValueNetwork(cfg)
    report = critic.initialize_encoder(actor)
    assert report["encoder_parameters_copied"] > 0
    h = torch.randn(4, 4, 8)
    e = torch.randn(4, 4, 3, 6)
    m = torch.ones(4, 4, 3, dtype=torch.bool)
    legal = torch.zeros(4, 2307, dtype=torch.bool)
    legal[:, :3] = True
    t = torch.ones(4, 4, dtype=torch.bool)
    hand = torch.zeros(4, 4, dtype=torch.long)
    deck = torch.zeros(4, 8, dtype=torch.long)
    return shared, actor, critic, (h, e, m, legal, t, hand, deck, deck)


def test_bc_actor_outputs_unchanged_and_export_contains_no_value_weights(tmp_path):
    shared, actor, critic, inputs = setup_models()
    with torch.no_grad():
        expected, _ = forward_inputs(shared, inputs)
        actual, placeholder = forward_inputs(actor, inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not placeholder.any()
    assert not any(k.startswith("critic.") for k in actor.state_dict())
    assert not (
        {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()}
    )
    target = tmp_path / "actor.pt"
    torch.save(
        {"model": actor.state_dict(), "config": actor.cfg.__dict__, "policy_only": True}, target
    )
    restored = ActorPolicy(actor.cfg)
    restored.load_state_dict(torch.load(target, weights_only=True)["model"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(forward_inputs(restored, inputs)[0], actual)


def test_complete_mc_targets_exclude_truncations_faults_and_partial_tails():
    rewards = np.array([[0], [1], [0], [-1], [0], [1], [0]], dtype=np.float32)
    terminal = np.array([[0], [1], [0], [0], [0], [1], [0]], dtype=bool)
    truncated = np.array([[0], [0], [0], [1], [0], [0], [0]], dtype=bool)
    valid = np.ones_like(terminal)
    valid[4] = False
    values, selected, games = complete_episode_returns(rewards, terminal, truncated, valid, 0.5)
    assert games == 1
    np.testing.assert_array_equal(selected[:, 0], [1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(values[:, 0], [0.5, 1, 0, 0, 0, 0, 0])


def test_critic_fits_while_actor_frozen_and_continues_when_actor_kl_stops():
    _, actor, critic, inputs = setup_models()
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    actor_before = {k: v.clone() for k, v in actor.state_dict().items()}
    critic_before = {k: v.clone() for k, v in critic.state_dict().items()}
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    indices = np.arange(4)
    with torch.no_grad():
        logp, _ = forward_inputs(behavior, inputs)
    targets = torch.ones(4, requires_grad=True)
    events = []

    def run(enabled):
        return optimize_separate(
            actor,
            critic,
            behavior,
            actor_optimizer,
            critic_optimizer,
            lambda rows: tuple(t[rows] for t in inputs),
            indices,
            indices,
            logp[:, 0],
            torch.zeros(4, dtype=torch.long),
            torch.ones(4),
            targets,
            rng=np.random.default_rng(42),
            device="cpu",
            batch_size=2,
            accumulation=2,
            actor_epochs=2,
            critic_epochs=2,
            clip_coef=0.15,
            entropy_coef=0.005,
            reference_kl_coef=0.01,
            target_kl=1e-6,
            max_grad_norm=0.5,
            actor_enabled=enabled,
            emit=lambda **event: events.append(event),
        )

    result = run(False)
    assert result["actor_steps"] == 0 and result["critic_steps"] == 2
    assert not actor_optimizer.state
    assert targets.grad is None
    assert all(p.grad is None for p in actor.parameters())
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in actor_before.items())
    assert any(not torch.equal(critic.state_dict()[k], v) for k, v in critic_before.items())
    with torch.no_grad():
        actor.wait_actor.bias.add_(3)
    result = run(True)
    assert result["actor_kl_stopped"] and result["actor_steps"] == 0
    assert result["critic_steps"] == 2
    stops = [e for e in events if e["phase"] == "kl_stop"]
    assert stops[-1]["kl_samples"] == 4 and stops[-1]["discarded_microbatches"] == 2
    assert len([e for e in events if e["phase"] == "critic_optimization"]) == 4
    assert all(p.grad is None for p in actor.parameters())


def test_independent_actor_can_update_and_checkpoint_both_optimizers(tmp_path):
    _, actor, critic, inputs = setup_models()
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    ao = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    co = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    indices = np.arange(4)
    with torch.no_grad():
        logp, _ = forward_inputs(actor, inputs)
    result = optimize_separate(
        actor,
        critic,
        behavior,
        ao,
        co,
        lambda rows: tuple(t[rows] for t in inputs),
        indices,
        indices,
        logp[:, 0],
        torch.zeros(4, dtype=torch.long),
        torch.ones(4),
        torch.ones(4),
        rng=np.random.default_rng(42),
        device="cpu",
        batch_size=2,
        accumulation=2,
        actor_epochs=1,
        critic_epochs=1,
        clip_coef=0.15,
        entropy_coef=0.005,
        reference_kl_coef=0.01,
        target_kl=0.02,
        max_grad_norm=0.5,
    )
    assert result["actor_steps"] == result["critic_steps"] == 1
    target = tmp_path / "resume.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "ao": ao.state_dict(),
            "co": co.state_dict(),
            "warmup": {"games": 4, "rounds": 2},
        },
        target,
    )
    state = torch.load(target, weights_only=True)
    _, a2, c2, _ = setup_models()
    a2.load_state_dict(state["actor"], strict=True)
    c2.load_state_dict(state["critic"], strict=True)
    a2o = torch.optim.AdamW(a2.parameters())
    c2o = torch.optim.AdamW(c2.parameters())
    a2o.load_state_dict(state["ao"])
    c2o.load_state_dict(state["co"])
    assert all(int(s["step"]) == 1 for s in a2o.state.values())
    assert all(int(s["step"]) == 1 for s in c2o.state.values())
    with torch.no_grad():
        torch.testing.assert_close(forward_inputs(a2, inputs)[0], forward_inputs(actor, inputs)[0])
        torch.testing.assert_close(forward_inputs(c2, inputs), forward_inputs(critic, inputs))
