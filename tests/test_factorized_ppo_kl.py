import math

import pytest
import torch

from cr_rl.training.ppo import factorized_reverse_kl, stop_for_factorized_kl


def joint(card, first_tile):
    result = torch.full((1, 2307), -1e9)
    result[0, 0] = math.log(card[0])
    for slot in range(4):
        tile = first_tile if slot == 0 else [0.5, 0.5]
        result[0, 1 + 576 * slot : 3 + 576 * slot] = torch.log(torch.tensor(tile) * card[slot + 1])
    result[0, 2305:] = torch.log(torch.tensor(card[5:]))
    return result


def test_exact_kl_chain_rule_and_probability_weighted_placement():
    card = [0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1]
    reference = [0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1]
    new, old = joint(card, [0.75, 0.25]), joint(reference, [0.5, 0.5])
    card_kl, tile_kl = factorized_reverse_kl(new, old)
    expected_card = sum(p * math.log(p / q) for p, q in zip(card, reference, strict=True))
    expected_tile = 0.25 * (0.75 * math.log(1.5) + 0.25 * math.log(0.5))
    torch.testing.assert_close(card_kl, torch.tensor([expected_card]), atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(tile_kl, torch.tensor([expected_tile]), atol=2e-7, rtol=1e-5)
    total = (new.exp() * (new - old)).sum(-1)
    torch.testing.assert_close(card_kl + tile_kl, total, atol=5e-7, rtol=1e-5)


def test_tile_only_change_does_not_consume_card_kl_budget():
    card = [0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1]
    card_kl, tile_kl = factorized_reverse_kl(joint(card, [0.8, 0.2]), joint(card, [0.5, 0.5]))
    assert abs(float(card_kl)) < 1e-6 and float(tile_kl) > 0.04
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.sum().backward()
    assert not stop_for_factorized_kl(optimizer, float(card_kl), float(tile_kl), 0.015, 0.06)
    assert parameter.grad is not None
    assert stop_for_factorized_kl(optimizer, 0.016, 0, 0.015, 0.06)
    assert parameter.grad is None and not optimizer.state


def test_forced_wait_has_zero_component_kl_and_no_invalid_tile_nan():
    lp = torch.full((2, 2307), -1e9)
    lp[:, 0] = 0
    lp.requires_grad_(True)
    card, tile = factorized_reverse_kl(lp, lp.detach().clone())
    torch.testing.assert_close(card + tile, torch.zeros(2), rtol=0, atol=0)
    (card + tile).sum().backward()
    assert torch.isfinite(lp.grad).all() and not lp.grad.any()


@pytest.mark.parametrize("card,tile", [(0, 0.06), (0.015, float("nan")), (-1, 0.06)])
def test_invalid_component_guard_limits_are_rejected(card, tile):
    parameter = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="finite and positive"):
        stop_for_factorized_kl(torch.optim.AdamW([parameter]), 0, 0, card, tile)
