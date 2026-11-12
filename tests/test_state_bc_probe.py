import importlib.util
from pathlib import Path
import sys

import pytest
import torch

SCRIPTS = str(Path(__file__).resolve().parents[1] / 'scripts')
sys.path.insert(0, SCRIPTS)
from state_bc_probe_common import differentiable_components, position_loss_from_logits
from cr_rl.models.policy import ActorCritic
from cr_rl.training.bc import BCBatchNormalization, hierarchical_bc_loss


@pytest.mark.parametrize('sigma', [0., 1.])
def test_probe_objective_loss_and_gradients_match_production(sigma):
    torch.manual_seed(11)
    card = torch.randn(7, 7, requires_grad=True)
    tile = torch.randn(7, 4, 576, requires_grad=True)
    legal = torch.ones(7, 2307, dtype=torch.bool)
    legal[:, 600:640] = False
    labels = torch.tensor([0, 10, 1000, 2305, 2306, 2000, 0])
    normalization = BCBatchNormalization.from_actions(labels)
    lp = ActorCritic._joint_log_probs(card, tile, legal)
    components = differentiable_components(lp, labels, legal, normalization,
        sigma=sigma, target_prior=.2, sampled_prior=.07)
    actual = components['timing'] + components['card'] + components['tile']
    expected, _ = hierarchical_bc_loss(lp, labels, legal, position_coef=1., tile_label_sigma=sigma,
        conditional_card_loss=True, normalization=normalization, timing_prior=.2, sampled_action_prior=.07)
    assert torch.allclose(actual, expected, atol=1e-6)
    for a, b in zip(torch.autograd.grad(actual, (card, tile), retain_graph=True),
                    torch.autograd.grad(expected, (card, tile), retain_graph=True)):
        assert torch.allclose(a, b, atol=2e-7, rtol=1e-5)
    tile_only_card_grad = torch.autograd.grad(components['tile'], card)[0]
    assert tile_only_card_grad.abs().max() < 2e-7


def test_direct_placement_graph_matches_loss_without_card_graph():
    torch.manual_seed(12)
    card = torch.randn(3, 7, requires_grad=True)
    tile = torch.randn(3, 4, 576, requires_grad=True)
    legal = torch.ones(3, 2307, dtype=torch.bool)
    legal[:, 100:140] = False
    labels = torch.tensor([1, 1000, 2000])
    norm = BCBatchNormalization.from_actions(labels)
    lp = ActorCritic._joint_log_probs(card, tile, legal)
    parts = differentiable_components(lp, labels, legal, norm, sigma=1., target_prior=.2, sampled_prior=.07)
    direct = position_loss_from_logits(tile, labels, legal, sigma=1., play_denominator=norm.play)
    assert torch.allclose(direct, parts['tile'], atol=1e-6)
    a = torch.autograd.grad(direct, (tile, card), retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(parts['tile'], tile)[0]
    assert torch.allclose(a[0], b, atol=1e-7)
    assert a[1] is None
