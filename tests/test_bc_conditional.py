import torch

from cr_rl.training.bc import hierarchical_bc_loss


def joint(logits, count):
    p = logits.log_softmax(-1)
    tiles = p[1:5, None].expand(4, 576) - torch.tensor(576.0).log()
    return torch.cat((p[:1], tiles.flatten(), p[5:])).expand(count, -1)


def test_conditional_choice_and_balanced_timing_do_not_vanish_under_wait_imbalance():
    gradients = []
    for waits in (1, 99):
        logits = torch.zeros(7, requires_grad=True)
        logp = joint(logits, waits + 1)
        actions = torch.zeros(waits + 1, dtype=torch.long)
        actions[-1] = 1
        loss, _ = hierarchical_bc_loss(
            logp,
            actions,
            torch.ones_like(logp, dtype=torch.bool),
            position_coef=0,
            conditional_card_loss=True,
            card_class_weights=torch.tensor([2.0, 0.001, 0.00001]),
        )
        loss.backward()
        gradients.append(logits.grad.clone())
    torch.testing.assert_close(gradients[0], gradients[1])
    assert gradients[1][1] < -0.5, "The selected card must retain a substantial learning signal"


def test_conditional_loss_is_finite_for_a_forced_wait_only_batch():
    logits = torch.zeros(7, requires_grad=True)
    logp = joint(logits, 3)
    mask = torch.zeros_like(logp, dtype=torch.bool)
    mask[:, 0] = True
    legal_logp = logp.masked_fill(~mask, -1e9).log_softmax(-1)
    loss, _ = hierarchical_bc_loss(
        legal_logp, torch.zeros(3, dtype=torch.long), mask, conditional_card_loss=True
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
