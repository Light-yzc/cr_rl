"""Versioned, game-level partition rule for the reconstructed expert-BC pool.

This module only assigns dataset partitions; it does not approve demonstrations.
The historical prior registry is authoritative and evaluation-deck games never
enter Train (or Validation). New, non-heldout games use a 10% validation hash.
"""

VALID_PARTITIONS = frozenset({'train', 'validation', 'heldout'})
VALIDATION_HASH_MODULUS_V2 = 10


def quality_bc_partition(signature, decks, heldout_decks, prior_assignments):
    if (not isinstance(signature, str) or len(signature) != 64
            or any(ch not in '0123456789abcdef' for ch in signature)):
        raise ValueError('Invalid canonical SHA256 game signature')
    if (not isinstance(decks, (list, tuple)) or len(decks) != 2
            or any(not isinstance(deck, (list, tuple)) or not deck for deck in decks)):
        raise ValueError('Missing complete game decks')
    previous = prior_assignments.get(signature)
    if previous is not None and previous not in VALID_PARTITIONS:
        raise ValueError('Invalid legacy BC partition for ' + signature)
    eval_deck_match = any(tuple(sorted(deck)) in heldout_decks for deck in decks)
    if eval_deck_match and previous in ('train', 'validation'):
        raise ValueError('Prior split conflicts with fixed heldout deck: ' + signature)
    if eval_deck_match or previous == 'heldout':
        return 'heldout'
    if previous is not None:
        return previous
    return ('validation' if int(signature[:8], 16) % VALIDATION_HASH_MODULUS_V2 == 0
            else 'train')
