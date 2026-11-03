import torch

from cr_rl.training.bc import hierarchical_action_predictions


def test_play_mass_is_not_compared_to_individual_slot_or_tile():
    logp = torch.full((1, 2307), -1e9)
    logp[:, 0] = torch.tensor(0.4).log()
    for slot, probability in enumerate((0.1, 0.1, 0.3, 0.1)):
        logp[:, 1 + slot * 576 : 1 + (slot + 1) * 576] = torch.tensor(probability / 576).log()
    card_logp, play, slot = hierarchical_action_predictions(logp)
    assert logp.argmax(-1).item() == 0
    assert card_logp.argmax(-1).item() == 0
    assert play.item(), "Total play probability is 0.6, greater than WAIT's 0.4"
    assert slot.item() == 2


def test_forced_wait_has_no_play_prediction():
    logp = torch.full((1, 2307), -1e9)
    logp[:, 0] = 0
    _, play, _ = hierarchical_action_predictions(logp)
    assert not play.item()
