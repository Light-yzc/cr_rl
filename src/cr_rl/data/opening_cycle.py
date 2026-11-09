"""Infer a plausible opening cycle from recorded card identities, not combat state.

The latent deal is not a recovered human hand. Keep this provenance explicit.
Only ordinary eight-card cycle rules are modelled; omitted/forced opening cards
and champion cycle exceptions are not silently approximated here.
"""

from itertools import combinations, permutations

from cr_rl.data.replay_repair import plan_hand_swap


def infer_opening_cycles(sequence, minimum_prefix=8, maximum_events=None, max_candidates=16):
    """Rank ordinary opening deals against the whole supplied card sequence.

    There are 70 initial hand sets × 24 queue orders. Initial slot permutations
    are observationally equivalent when the source records card names alone.
    Return up to ``max_candidates`` deals in deterministic prefix-length order;
    a simulator replay must still verify fees, pending commands and mechanics.
    """
    if minimum_prefix < 0 or (maximum_events is not None and maximum_events < 1):
        raise ValueError("Invalid opening-cycle search limits")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    sequence = tuple(int(card) for card in sequence[:maximum_events])
    if not sequence or any(card < 0 or card >= 8 for card in sequence):
        return []
    candidates = []
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
            if length >= min(minimum_prefix, len(sequence)):
                candidates.append((length, (*opening, *queued)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    counts = {}
    for length, _ in candidates:
        counts[length] = counts.get(length, 0) + 1
    return [
        {
            "order": list(order),
            "consistent_prefix_plays": length,
            "candidate_orders": counts[length],
            "evaluated_source_plays": len(sequence),
        }
        for length, order in candidates[:max_candidates]
    ]


def infer_opening_cycle(sequence, minimum_prefix=8, maximum_events=None):
    """Best ranked deal; metadata retains ambiguity rather than claiming truth."""
    candidates = infer_opening_cycles(sequence, minimum_prefix, maximum_events, max_candidates=1)
    return candidates[0] if candidates else None


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


def _native_env_match(vector, env_index=0):
    """Pinned JVM bridge; never advances the match or changes its cards."""
    if not 0 <= int(env_index) < int(vector.count):
        raise ValueError("Environment index is outside the vector")
    envs_field = vector._env.getClass().getDeclaredField("envs")
    envs_field.setAccessible(True)
    env = envs_field.get(vector._env)[int(env_index)]
    match_field = env.getClass().getDeclaredField("match")
    match_field.setAccessible(True)
    match = match_field.get(env)
    if match is None:
        raise ValueError("Cannot repair a battle without a match")
    return env, match


def force_card_into_hand(
    vector, seat, deck_index, preferred_slot=None, *, env_index=0, return_record=False
):
    """Opt-in diagnostic repair: swap a queued card into a live hand slot.

    This creates repaired BC data, not exact human-state BC. It preserves the
    live deck-card multiset by swapping
    the displaced slot card with the requested queued card, then refreshes the
    observation/mask buffers without advancing time.  The caller must record a
    repair ledger entry and treat all subsequent rows as repaired. ``deck_index``
    is an index into the *runtime* deck after opening-deal reordering, never an
    index into the source card list. Pending and empty slots cannot be replaced.

    With ``return_record=True`` return an audit dictionary; otherwise return the
    chosen slot for compatibility. The caller must record this mutation because
    the native replay JSON does not record external hand rearrangements.
    """
    seat, env_index, deck_index = int(seat), int(env_index), int(deck_index)
    if seat not in (0, 1):
        raise ValueError("Seat must be zero or one")
    env, match = _native_env_match(vector, env_index)
    if bool(vector.dones[env_index]):
        raise ValueError("Cannot repair a finished battle")
    side = match.side(seat)
    hand = side.getHand()
    slots = [int(value) for value in hand.slots()]
    queue = [int(value) for value in hand.queue()]
    pending = [card >= 0 and bool(env.commands().isPending(seat, card)) for card in slots]
    plan = plan_hand_swap(slots, queue, deck_index, pending, preferred_slot)
    record = {
        **plan,
        "seat": seat,
        "env_index": env_index,
        "pending_before": pending,
        "requested_card_name": str(side.deck().get(deck_index).name()),
    }
    if not plan["changed"]:
        return record if return_record else plan["slot"]

    # Resolve reflection before mutating, so an upstream API mismatch cannot
    # leave the live hand partly modified.
    observe = env.getClass().getDeclaredMethod("observe")
    observe.setAccessible(True)
    before_tick = int(vector.info(env_index)["tick"])
    before_cooldown = int(hand.getCooldownMs())
    before_elixir = int(side.getElixir())
    import jpype

    places = jpype.JArray(jpype.JInt)(plan["places"])
    try:
        hand.rearrange(places, jpype.JArray(jpype.JInt)(plan["cards"]))
        observe.invoke(env)
        if [int(value) for value in hand.slots()] != plan["after_slots"] or [
            int(value) for value in hand.queue()
        ] != plan["after_queue"]:
            raise RuntimeError("Hand repair did not preserve its declared permutation")
        if (
            int(vector.info(env_index)["tick"]) != before_tick
            or int(hand.getCooldownMs()) != before_cooldown
            or int(side.getElixir()) != before_elixir
        ):
            raise RuntimeError("Hand repair unexpectedly changed time, cooldown, or elixir")
        field = vector.schema.header["hand_card"][plan["slot"]]
        if int(vector.observations[env_index, seat, field]) != vector.card_id(
            record["requested_card_name"]
        ):
            raise RuntimeError("Hand repair observation disagrees with runtime card mapping")
    except Exception:
        # Rearrange is a same-card permutation; swapping the pair back restores
        # both queue and hand without replaying time or spending elixir.
        hand.rearrange(
            places, jpype.JArray(jpype.JInt)([plan["displaced_deck_index"], deck_index])
        )
        observe.invoke(env)
        raise
    return record if return_record else plan["slot"]


def reorder_for_deal(deck, slots, native_order, desired_order):
    if sorted(native_order) != list(range(8)) or sorted(desired_order) != list(range(8)):
        raise ValueError("Both deals must be eight-card permutations")
    names, forms = [None] * 8, [0] * 8
    for place, card in zip(native_order, desired_order, strict=True):
        names[place], forms[place] = deck[card], slots[card]
    return names, forms
