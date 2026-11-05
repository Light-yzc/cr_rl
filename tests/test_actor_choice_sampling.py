import copy

import numpy as np
import pytest
import torch
from test_separate_critic import setup_models

from cr_rl.training.sample_selection import actor_critic_rows, normalize_actor_advantages
from cr_rl.training.separate import forward_inputs, optimize_separate


def test_choice_rows_keep_wait_and_ability_decisions_and_all_value_states():
    legal = np.zeros((4, 2307), dtype=bool)
    legal[:, 0] = True
    legal[1, 1] = legal[2, 2305] = legal[3, 2] = True
    actor, critic = actor_critic_rows([True, True, True, False], legal, choice_only=True)
    np.testing.assert_array_equal(actor, [1, 2])
    np.testing.assert_array_equal(critic, [0, 1, 2])
    normalized = normalize_actor_advantages(torch.tensor([100.0, 1.0, 3.0, -100.0]), actor)
    torch.testing.assert_close(normalized[actor], torch.tensor([-1.0, 1.0]))


def test_forced_only_rollout_fits_value_without_actor_updates_or_nan_advantage():
    torch.set_num_threads(1)
    _, actor, critic, inputs = setup_models()
    legal = np.zeros((4, 2307), dtype=bool)
    legal[:, 0] = True
    inputs = (*inputs[:3], torch.tensor(legal), *inputs[4:])
    actor_rows, critic_rows = actor_critic_rows(np.ones(4, bool), legal, choice_only=True)
    advantages = normalize_actor_advantages(torch.ones(4), actor_rows)
    assert torch.isfinite(advantages).all()
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    value_before = {k: v.clone() for k, v in critic.state_dict().items()}
    result = optimize_separate(
        actor,
        critic,
        copy.deepcopy(actor).eval().requires_grad_(False),
        torch.optim.AdamW(actor.parameters(), lr=1e-4),
        torch.optim.AdamW(critic.parameters(), lr=1e-3),
        lambda rows: tuple(x[rows] for x in inputs),
        actor_rows,
        critic_rows,
        torch.zeros(4),
        torch.zeros(4, dtype=torch.long),
        advantages,
        torch.ones(4),
        rng=np.random.default_rng(4),
        device="cpu",
        batch_size=1,
        accumulation=2,
        actor_epochs=2,
        critic_epochs=1,
        clip_coef=0.15,
        entropy_coef=0.005,
        reference_kl_coef=0.01,
        target_kl=0.02,
        max_grad_norm=0.5,
        target_card_kl=0.015,
        target_tile_kl=0.06,
    )
    assert result["actor_steps"] == 0 and result["critic_steps"] == 2
    assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
    assert any(not torch.equal(v, value_before[k]) for k, v in critic.state_dict().items())


def test_choice_actor_and_value_optimization_use_distinct_sample_counts():
    torch.set_num_threads(1)
    _, actor, critic, inputs = setup_models()
    legal = np.zeros((4, 2307), dtype=bool)
    legal[:, 0] = True
    legal[1, 1] = legal[2, 2305] = True
    inputs = (*inputs[:3], torch.tensor(legal), *inputs[4:])
    actor_rows, critic_rows = actor_critic_rows([True, True, True, False], legal, choice_only=True)
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    actions = torch.tensor([0, 1, 2305, 0])
    with torch.no_grad():
        old, _ = forward_inputs(behavior, inputs)
        old_logp = torch.distributions.Categorical(logits=old).log_prob(actions)
    result = optimize_separate(
        actor,
        critic,
        behavior,
        torch.optim.AdamW(actor.parameters(), lr=1e-5),
        torch.optim.AdamW(critic.parameters(), lr=1e-3),
        lambda rows: tuple(x[rows] for x in inputs),
        actor_rows,
        critic_rows,
        old_logp,
        actions,
        normalize_actor_advantages(torch.tensor([100.0, 1.0, 3.0, -100.0]), actor_rows),
        torch.ones(4),
        rng=np.random.default_rng(4),
        device="cpu",
        batch_size=1,
        accumulation=2,
        actor_epochs=1,
        critic_epochs=1,
        clip_coef=0.15,
        entropy_coef=0.005,
        reference_kl_coef=0.01,
        target_kl=0.02,
        max_grad_norm=0.5,
        target_card_kl=0.015,
        target_tile_kl=0.06,
        max_initial_logp_error=0.05,
    )
    assert result["actor_steps"] == 1 and result["critic_steps"] == 2
    assert "card_kl" in result["metrics"] and "tile_kl" in result["metrics"]


@pytest.mark.parametrize("first_card_kl,expected_steps", [(0.04, 1), (0.08, 0)])
def test_split_kl_judges_whole_accumulation_and_value_still_trains(
    monkeypatch, first_card_kl, expected_steps
):
    import cr_rl.training.separate as separate

    torch.set_num_threads(1)
    _, actor, critic, inputs = setup_models()
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    with torch.no_grad():
        old, _ = forward_inputs(behavior, inputs)
    calls = []

    def components(new, reference):
        value = first_card_kl if not calls else 0
        calls.append(True)
        return new.new_full((len(new),), value), new.new_zeros(len(new))

    monkeypatch.setattr(separate, "factorized_reverse_kl", components)
    events = []
    result = optimize_separate(
        actor,
        critic,
        behavior,
        torch.optim.AdamW(actor.parameters(), lr=1e-5),
        torch.optim.AdamW(critic.parameters(), lr=1e-3),
        lambda rows: tuple(x[rows] for x in inputs),
        np.arange(2),
        np.arange(4),
        old[:, 0],
        torch.zeros(4, dtype=torch.long),
        torch.ones(4),
        torch.ones(4),
        rng=np.random.default_rng(4),
        device="cpu",
        batch_size=1,
        accumulation=2,
        actor_epochs=1,
        critic_epochs=1,
        clip_coef=0.15,
        entropy_coef=0.005,
        reference_kl_coef=0.01,
        target_kl=0.02,
        max_grad_norm=0.5,
        target_card_kl=0.03,
        target_tile_kl=0.06,
        emit=lambda **e: events.append(e),
    )
    assert result["actor_steps"] == expected_steps and result["critic_steps"] == 2
    assert len(calls) == 2
    boundary = next(e for e in events if e["phase"] in ("optimization", "kl_stop"))
    assert boundary["card_kl"] == pytest.approx(first_card_kl / 2)


def test_initial_log_probability_guard_rejects_before_any_actor_step():
    torch.set_num_threads(1)
    _, actor, critic, inputs = setup_models()
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    with torch.no_grad():
        old, _ = forward_inputs(behavior, inputs)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    with pytest.raises(ValueError, match="initial Actor"):
        optimize_separate(
            actor,
            critic,
            behavior,
            optimizer,
            torch.optim.AdamW(critic.parameters(), lr=1e-3),
            lambda rows: tuple(x[rows] for x in inputs),
            np.arange(2),
            np.arange(4),
            old[:, 0] + 0.1,
            torch.zeros(4, dtype=torch.long),
            torch.ones(4),
            torch.ones(4),
            rng=np.random.default_rng(4),
            device="cpu",
            batch_size=1,
            accumulation=2,
            actor_epochs=1,
            critic_epochs=1,
            clip_coef=0.15,
            entropy_coef=0.005,
            reference_kl_coef=0.01,
            target_kl=0.02,
            max_grad_norm=0.5,
            target_card_kl=0.03,
            target_tile_kl=0.06,
            max_initial_logp_error=0.05,
        )
    assert not optimizer.state
    assert all(p.grad is None for p in actor.parameters())
    assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
