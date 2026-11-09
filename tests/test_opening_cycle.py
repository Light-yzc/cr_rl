import random

import pytest

from cr_rl.data.opening_cycle import infer_opening_cycle, infer_opening_cycles, reorder_for_deal


def test_inference_reproduces_long_legal_card_sequence_without_known_deal():
    rng = random.Random(47)
    hand, queue = [6, 2, 4, 1], [7, 0, 5, 3]
    sequence = []
    for _ in range(80):
        slot = rng.randrange(4)
        sequence.append(hand[slot])
        queue.append(hand[slot])
        hand[slot] = queue.pop(0)
    inferred = infer_opening_cycle(sequence)
    assert inferred["consistent_prefix_plays"] == len(sequence)
    assert inferred["evaluated_source_plays"] == 80
    assert sorted(inferred["order"]) == list(range(8))


def test_inference_refuses_impossible_early_repeats_and_unknown_cards():
    assert infer_opening_cycle([0, 0, 1, 2, 3, 4, 5, 6]) is None
    assert infer_opening_cycle([9]) is None
    assert infer_opening_cycle([]) is None


def test_ranked_search_is_deterministic_and_preserves_candidate_ambiguity():
    sequence = list(range(8)) * 6
    ranked = infer_opening_cycles(sequence, max_candidates=8)
    assert ranked == infer_opening_cycles(sequence, max_candidates=8)
    assert 1 < len(ranked) <= 8
    assert all(row["consistent_prefix_plays"] == len(sequence) for row in ranked)
    assert all(row["candidate_orders"] >= len(ranked) for row in ranked)
    assert ranked[0] == infer_opening_cycle(sequence)
    assert infer_opening_cycle([0, 1, 2, 3, 4, 5, 6, 7, 0, 0])["consistent_prefix_plays"] == 9
    with pytest.raises(ValueError, match="positive"):
        infer_opening_cycles(sequence, max_candidates=0)


def test_native_shuffle_inverse_preserves_cards_and_evolution_flags():
    deck, slots = list("abcdefgh"), [0, 1, 2, 0, 0, 0, 0, 0]
    permutation, desired = [2, 7, 0, 6, 4, 1, 3, 5], [4, 6, 0, 2, 1, 5, 7, 3]
    names, forms = reorder_for_deal(deck, slots, permutation, desired)
    assert [names[i] for i in permutation] == [deck[i] for i in desired]
    assert [forms[i] for i in permutation] == [slots[i] for i in desired]
    assert sorted(zip(names, forms, strict=False)) == sorted(zip(deck, slots, strict=False))
    with pytest.raises(ValueError, match="permutations"):
        reorder_for_deal(deck, slots, [0] * 8, desired)


def test_live_bridge_uses_runtime_deck_index_and_preserves_pending_slots(monkeypatch):
    from types import SimpleNamespace

    import jpype
    import numpy as np

    from cr_rl.data import opening_cycle

    # Exercise the bridge's mutation contract without requiring JVM startup in
    # ordinary unit tests. A real pinned-JVM smoke is recorded separately.
    class Hand:
        def __init__(self):
            self.s = [3, 0, 7, 2]
            self.q = [1, 4, 6, 5]

        def slots(self):
            return list(self.s)

        def queue(self):
            return list(self.q)

        def getCooldownMs(self):
            return 150

        def rearrange(self, places, cards):
            both = self.s + self.q
            assert sorted(both[p] for p in places) == sorted(cards)
            for p, c in zip(places, cards, strict=True):
                both[p] = c
            self.s, self.q = both[:4], both[4:]

    hand = Hand()
    deck_names = [
        "Knight", "Zap", "Giant", "Arrows", "Musketeer", "Minions", "MiniPekka", "ZapMachine"
    ]
    deck = SimpleNamespace(get=lambda i: SimpleNamespace(name=lambda: deck_names[i]))
    side = SimpleNamespace(getHand=lambda: hand, getElixir=lambda: 40000, deck=lambda: deck)
    obs = np.full((1, 2, 4), -1, np.int32)
    obs[0, 0] = hand.s
    observe = SimpleNamespace(
        setAccessible=lambda value: None,
        invoke=lambda env: obs.__setitem__((0, 0), hand.s),
    )
    env = SimpleNamespace(
        commands=lambda: SimpleNamespace(isPending=lambda seat, card: card == 3),
        getClass=lambda: SimpleNamespace(getDeclaredMethod=lambda name: observe),
    )
    vector = SimpleNamespace(
        dones=[False], observations=obs,
        info=lambda i: {"tick": 40},
        card_id=lambda name: deck_names.index(name),
        schema=SimpleNamespace(header={"hand_card": (0, 1, 2, 3)}),
    )
    monkeypatch.setattr(
        opening_cycle, "_native_env_match",
        lambda v, index: (env, SimpleNamespace(side=lambda seat: side)),
    )
    monkeypatch.setattr(jpype, "JArray", lambda dtype: list)
    result = opening_cycle.force_card_into_hand(vector, 0, 5, return_record=True)
    assert result["slot"] == 1
    assert result["requested_card_name"] == "Minions"
    assert result["after_slots"] == [3, 5, 7, 2]
    assert result["after_queue"] == [1, 4, 6, 0]
    assert result["pending_before"] == [True, False, False, False]
    assert obs[0, 0].tolist() == result["after_slots"]
    with pytest.raises(ValueError, match="pending"):
        opening_cycle.force_card_into_hand(vector, 0, 3)
    assert hand.s == [3, 5, 7, 2]
