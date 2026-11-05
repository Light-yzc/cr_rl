import pytest
import torch

from cr_rl.training.advantages import compute_actor_critic_targets, compute_gae


def cohort():
    rewards = torch.tensor([[0.0], [1.0], [3.0]])
    values = torch.tensor([[0.2], [0.4], [0.6]], requires_grad=True)
    terminated = torch.tensor([[0], [1], [0]])
    following = torch.tensor([[0.4], [0.0], [2.0]], requires_grad=True)
    return rewards, values, terminated, following


def test_longer_value_target_preserves_actor_advantage_and_rollout_snapshots():
    rewards, values, terminated, following = cohort()
    options = dict(gamma=0.9, gae_lambda=0.5, next_values=following)
    actor, _ = compute_gae(rewards, values, terminated, following[-1], **options)
    advantage, targets = compute_actor_critic_targets(
        rewards, values, terminated, following[-1], critic_gae_lambda=1.0, **options
    )
    assert torch.equal(advantage, actor)
    # The terminal cuts the first episode; the unfinished third row bootstraps.
    torch.testing.assert_close(targets[:, 0], torch.tensor([0.9, 1.0, 4.8]))
    assert not advantage.requires_grad and not targets.requires_grad
    saved_advantage, saved_targets = advantage.clone(), targets.clone()
    with torch.no_grad():
        values.add_(100)
        following.add_(100)
    assert torch.equal(advantage, saved_advantage)
    assert torch.equal(targets, saved_targets)


@pytest.mark.parametrize("critic_lambda", [None, 0.5])
def test_default_and_equal_value_lambda_preserve_both_existing_outputs(critic_lambda):
    rewards, values, terminated, following = cohort()
    options = dict(gamma=0.9, gae_lambda=0.5, next_values=following)
    expected = compute_gae(rewards, values, terminated, following[-1], **options)
    actual = compute_actor_critic_targets(
        rewards, values, terminated, following[-1], critic_gae_lambda=critic_lambda, **options
    )
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_independent_value_target_retains_time_limit_bootstrap_without_crossing_reset():
    rewards = torch.tensor([[0.0], [99.0]])
    values = torch.tensor([[1.0], [10.0]])
    following = torch.tensor([[2.0], [0.0]])
    _, targets = compute_actor_critic_targets(
        rewards, values, torch.zeros(2, 1), following[-1], gamma=0.9,
        gae_lambda=0.5, critic_gae_lambda=1.0, next_values=following,
        episode_ends=torch.ones(2, 1),
    )
    torch.testing.assert_close(targets[:, 0], torch.tensor([1.8, 99.0]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_value_lambda_is_rejected(value):
    rewards, values, terminated, following = cohort()
    with pytest.raises(ValueError, match="finite"):
        compute_actor_critic_targets(
            rewards, values, terminated, following[-1], gamma=0.9, gae_lambda=0.5,
            critic_gae_lambda=value,
        )
