import pytest
import torch

from cr_rl.models.policy import ActorCritic
from cr_rl.training.ppo import factorized_entropy


def distribution():
    torch.manual_seed(17)
    card = torch.randn(3, 7, requires_grad=True)
    tile = torch.randn(3, 4, 576, requires_grad=True)
    legal = torch.zeros(3, 2307, dtype=torch.bool)
    legal[:, 0] = True
    legal[0, 1:12] = True
    legal[1, 1:577] = True
    legal[2, 1:1153] = True
    legal[2, 2305] = True
    return card, tile, ActorCritic._joint_log_probs(card, tile, legal)


def test_split_preserves_joint_entropy_value_and_tile_gradient():
    card, tile, joint = distribution()
    actual = torch.distributions.Categorical(logits=joint).entropy()
    card_h, tile_h = factorized_entropy(joint)
    torch.testing.assert_close(card_h + tile_h, actual, atol=2e-6, rtol=1e-5)
    original = torch.autograd.grad(actual.sum(), tile, retain_graph=True)[0]
    changed = torch.autograd.grad(tile_h.sum(), tile)[0]
    torch.testing.assert_close(changed, original, atol=2e-7, rtol=2e-4)
    assert changed.abs().max() > 0


def test_tile_exploration_does_not_directly_push_card_logits():
    card, tile, joint = distribution()
    card_h, tile_h = factorized_entropy(joint)
    old = torch.distributions.Categorical(logits=joint).entropy()
    old_grad = torch.autograd.grad(old.sum(), card, retain_graph=True)[0]
    card_grad = torch.autograd.grad(card_h.sum(), card, retain_graph=True)[0]
    detached_grad = torch.autograd.grad(tile_h.sum(), card)[0]
    assert (old_grad - card_grad).abs().max() > .01
    torch.testing.assert_close(detached_grad, torch.zeros_like(card), atol=2e-6, rtol=0)


def test_wait_only_is_finite_zero_and_wrong_codec_is_rejected():
    logits = torch.full((2, 2307), -1e9)
    logits[:, 0] = 0
    logits.requires_grad_()
    card_h, tile_h = factorized_entropy(logits)
    assert torch.isfinite(card_h).all() and torch.isfinite(tile_h).all()
    assert (card_h == 0).all() and (tile_h == 0).all()
    (card_h.sum() + tile_h.sum()).backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):
        factorized_entropy(torch.zeros(1, 10))


def test_card_entropy_wait_gradient_is_symmetric_and_invariant_to_play_arity():
    """Having 4 legal cards vs 1 legal card must not push p(WAIT) down toward 20%."""
    tile = torch.zeros(1, 4, 576)
    grads = []
    for n_cards in (1, 4):
        legal = torch.zeros(1, 2307, dtype=torch.bool)
        legal[0, 0] = True
        for s in range(n_cards):
            legal[0, 1 + s * 576] = True
        # Fix p(WAIT) = 0.50 exactly: wait_logit = log(n_cards), play_logits = 0.0
        card = torch.zeros(1, 7, requires_grad=True)
        with torch.no_grad():
            card[0, 0] = torch.log(torch.tensor(float(n_cards)))
        joint = ActorCritic._joint_log_probs(card, tile, legal)
        card_h, _ = factorized_entropy(joint)
        grad_wait = torch.autograd.grad(card_h.sum(), card)[0][0, 0].item()
        grads.append(grad_wait)
    # At p(WAIT) = 0.50, binary wait entropy is stationary -> zero gradient on wait logit regardless of 1 vs 4 legal cards!
    assert abs(grads[0]) < 1e-6
    assert abs(grads[1]) < 1e-6
