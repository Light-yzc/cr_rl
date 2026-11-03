import random

import pytest

from cr_rl.data.opening_cycle import infer_opening_cycle, reorder_for_deal


def test_inference_reproduces_long_legal_card_sequence_without_known_deal():
    rng = random.Random(47)
    hand, queue = [6, 2, 4, 1], [7, 0, 5, 3]
    sequence = []
    for _ in range(80):
        slot = rng.randrange(4)
        sequence.append(hand[slot])
        queue.append(hand[slot])
        hand[slot] = queue.pop(0)
    inferred = infer_opening_cycle(sequence, maximum_events=80)
    assert inferred["consistent_prefix_plays"] == len(sequence)
    assert sorted(inferred["order"]) == list(range(8))


def test_inference_refuses_impossible_early_repeats_and_unknown_cards():
    assert infer_opening_cycle([0, 0, 1, 2, 3, 4, 5, 6]) is None
    assert infer_opening_cycle([9]) is None
    assert infer_opening_cycle([]) is None


def test_native_shuffle_inverse_preserves_cards_and_evolution_flags():
    deck, slots = list("abcdefgh"), [0, 1, 2, 0, 0, 0, 0, 0]
    permutation, desired = [2, 7, 0, 6, 4, 1, 3, 5], [4, 6, 0, 2, 1, 5, 7, 3]
    names, forms = reorder_for_deal(deck, slots, permutation, desired)
    assert [names[i] for i in permutation] == [deck[i] for i in desired]
    assert [forms[i] for i in permutation] == [slots[i] for i in desired]
    assert sorted(zip(names, forms, strict=False)) == sorted(zip(deck, slots, strict=False))
    with pytest.raises(ValueError, match="permutations"):
        reorder_for_deal(deck, slots, [0] * 8, desired)
