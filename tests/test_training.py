import torch

from cr_rl.training.advantages import compute_gae
from cr_rl.training.bc import behaviour_cloning_loss, hierarchical_bc_loss
from cr_rl.training.ppo import ppo_loss


def test_gae_terminal_masks_bootstrap():
    rewards = torch.tensor([[1.0], [0.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[1.0], [0.0]])
    adv, returns = compute_gae(
        rewards, values, dones, torch.tensor([2.0]), gamma=1.0, gae_lambda=1.0
    )
    assert adv[0, 0].item() == 1.0
    assert returns[0, 0].item() == 1.0
    assert adv[1, 0].item() == 2.0


def test_gae_truncation_bootstraps_but_stops_trace_at_episode_boundary():
    rewards = torch.tensor([[0.0], [0.0]])
    values = torch.tensor([[1.0], [100.0]])
    next_values = torch.tensor([[2.0], [0.0]])
    terminated = torch.tensor([[0.0], [0.0]])
    episode_ends = torch.tensor([[1.0], [1.0]])
    advantages, _ = compute_gae(
        rewards,
        values,
        terminated,
        torch.tensor([0.0]),
        gamma=1.0,
        gae_lambda=1.0,
        next_values=next_values,
        episode_ends=episode_ends,
    )
    assert advantages[0, 0].item() == 1.0  # 2.0 bootstrap minus 1.0 value


def test_bc_and_ppo_losses_are_finite():
    logits = torch.randn(3, 5, requires_grad=True)
    legal = torch.ones(3, 5, dtype=torch.bool)
    bc = behaviour_cloning_loss(logits, torch.tensor([0, 1, 2]), legal)
    bc.backward()
    loss, metrics = ppo_loss(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.zeros(3), torch.ones(3), torch.ones(3)
    )
    assert torch.isfinite(bc)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["approx_kl"])


def test_reference_kl_is_zero_for_identical_policy_distributions():
    from cr_rl.training.ppo import ppo_loss_with_reference

    logits = torch.log_softmax(torch.randn(3, 5), dim=-1).requires_grad_()
    action = torch.tensor([0, 2, 4])
    distribution = torch.distributions.Categorical(logits=logits)
    loss, metrics = ppo_loss_with_reference(
        logits,
        logits.detach(),
        distribution.log_prob(action),
        distribution.log_prob(action).detach(),
        torch.ones(3),
        torch.zeros(3),
        torch.ones(3),
        distribution.entropy(),
    )
    assert torch.isfinite(loss)
    assert abs(float(metrics["reference_kl"])) < 1e-6


def test_hierarchical_bc_separates_slot_and_position_supervision():
    from cr_rl.models.policy import ActorCritic

    card = torch.randn(2, 7, requires_grad=True)
    tile = torch.randn(2, 4, 576, requires_grad=True)
    legal = torch.zeros(2, 2307, dtype=torch.bool)
    legal[:, 0] = True
    action = 1 + 2 * 576 + 17
    legal[1, action] = True
    joint = ActorCritic._joint_log_probs(card, tile, legal)
    loss, metrics = hierarchical_bc_loss(joint, torch.tensor([0, action]), legal)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["card_loss"])
    assert torch.isfinite(metrics["position_loss"])
    assert torch.isfinite(card.grad).all()
    assert torch.isfinite(tile.grad).all()
