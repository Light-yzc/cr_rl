"""Explicit append-only ordinary-deck curriculum transitions on checkpoint resume."""

import copy
import hashlib
import json


def _specifications(pool):
    decks, slots = pool.get("decks", {}), pool.get("slots", {})
    if not decks or set(slots) - set(decks):
        raise ValueError("Deck pool must contain named decks and matching slot names")
    result = {}
    for name, cards in decks.items():
        forms = slots.get(name, [0] * 8)
        if (
            not isinstance(name, str)
            or not name
            or len(cards) != 8
            or any(not isinstance(card, str) or not card for card in cards)
            or len(set(cards)) != 8
            or len(forms) != 8
            or any(type(value) is not int or value < 0 for value in forms)
        ):
            raise ValueError("Each named deck needs eight unique cards and valid slot forms")
        result[name] = {"cards": list(cards), "slots": list(forms)}
    return result


def expand_deck_pool(previous, requested):
    """Keep old deck indices/cards/forms; append validated ordinary-form decks.

    The caller retains models, optimizer states, RNG and league statistics. This
    function consumes no RNG and mutates neither checkpoint nor requested config.
    Simulator card-name/provenance validation remains the runner's responsibility.
    """
    old, new = _specifications(previous), _specifications(requested)
    names, expanded = list(old), list(new)
    if len(expanded) <= len(names) or expanded[: len(names)] != names:
        raise ValueError(
            "Deck expansion must append names without removing or reordering old decks"
        )
    if any(new[name] != spec for name, spec in old.items()):
        raise ValueError("Deck expansion cannot change existing cards or slot forms")
    added = expanded[len(names) :]
    if any(any(new[name]["slots"]) for name in added):
        raise ValueError("New curriculum decks must use ordinary forms")

    def digest(specs):
        ordered = [{"name": name, **spec} for name, spec in specs.items()]
        return hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest()

    return copy.deepcopy(requested), {
        "old_decks": names,
        "new_decks": expanded,
        "added_decks": added,
        "old_spec_sha256": digest(old),
        "new_spec_sha256": digest(new),
        "semantics": (
            "Append-only ordinary-deck curriculum; source episodes reset on process resume"
        ),
    }
