"""Infer a plausible opening cycle from recorded card identities, not combat state.

The latent deal is not a recovered human hand. Keep this provenance explicit.
Only ordinary eight-card cycle rules are modelled; omitted/forced opening cards
and champion cycle exceptions are not silently approximated here.
"""

from itertools import combinations, permutations


def infer_opening_cycle(sequence, minimum_prefix=8, maximum_events=32):
    sequence = tuple(int(card) for card in sequence[:maximum_events])
    if not sequence or any(card < 0 or card >= 8 for card in sequence):
        return None
    best, best_length, candidates = None, -1, 0
    for opening in combinations(range(8), 4):
        rest = tuple(card for card in range(8) if card not in opening)
        for queued in permutations(rest):
            hand = set(opening)
            queue = list(queued)
            length = 0
            for card in sequence:
                if card not in hand:
                    break
                hand.remove(card)
                hand.add(queue.pop(0))
                queue.append(card)
                length += 1
            if length > best_length:
                best, best_length, candidates = (*opening, *queued), length, 1
            elif length == best_length:
                candidates += 1
    if best_length < min(minimum_prefix, len(sequence)):
        return None
    return {
        "order": list(best),
        "consistent_prefix_plays": best_length,
        "candidate_orders": candidates,
        "evaluated_source_plays": len(sequence),
    }


def native_opening_deals(vector):
    """Read the pinned simulator's deal without altering any battle state.

    crforge 7ee59b39: VectorEnv.envs -> LadderEnv.match -> MatchSide.hand.
    Reflection is deliberately isolated and fails closed if upstream changes.
    """
    envs_field = vector._env.getClass().getDeclaredField("envs")
    envs_field.setAccessible(True)
    env = envs_field.get(vector._env)[0]
    match_field = env.getClass().getDeclaredField("match")
    match_field.setAccessible(True)
    match = match_field.get(env)
    if match is None:
        raise ValueError("Simulator reset produced no Ladder match")
    deals = []
    for seat in (0, 1):
        side = match.side(seat)
        deck = side.deck()
        ordinary = all(
            not card.forceToStartingHand() and not card.omitFromStartingHand() for card in deck
        )
        hand = side.getHand()
        order = [int(card) for card in hand.slots()] + [int(card) for card in hand.queue()]
        if sorted(order) != list(range(8)):
            raise ValueError("Native initial deal is not an eight-card permutation")
        deals.append({"order": order, "ordinary": ordinary})
    return deals


def reorder_for_deal(deck, slots, native_order, desired_order):
    if sorted(native_order) != list(range(8)) or sorted(desired_order) != list(range(8)):
        raise ValueError("Both deals must be eight-card permutations")
    names, forms = [None] * 8, [0] * 8
    for place, card in zip(native_order, desired_order, strict=True):
        names[place], forms[place] = deck[card], slots[card]
    return names, forms
