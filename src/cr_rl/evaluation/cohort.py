"""Unique deck pairings with stable global indices across evaluation batches."""

from itertools import combinations

import numpy as np


def heldout_pairings(names, count, start=0, seed=20261103):
    names = list(names)
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("Need at least two distinct held-out deck names")
    if count < 1 or start < 0:
        raise ValueError("Pair count must be positive and start nonnegative")
    shuffled = np.random.default_rng(seed).permutation(names).tolist()
    # Preserve the original disjoint cohort for old comparisons, then add
    # different matchups instead of cycling through those same deck pairs.
    pairs = list(zip(shuffled[::2], shuffled[1::2], strict=False))
    seen = {frozenset(pair) for pair in pairs}
    remaining = [pair for pair in combinations(shuffled, 2) if frozenset(pair) not in seen]
    appearances = {name: sum(name in pair for pair in pairs) for name in shuffled}
    while remaining:
        pair = min(
            remaining,
            key=lambda pair: (
                appearances[pair[0]] + appearances[pair[1]],
                max(appearances[pair[0]], appearances[pair[1]]),
            ),
        )
        remaining.remove(pair)
        pairs.append(pair)
        for name in pair:
            appearances[name] += 1
    if start + count > len(pairs):
        raise ValueError(f"Requested cohort exceeds {len(pairs)} unique held-out deck pairs")
    return [(i, *pairs[i]) for i in range(start, start + count)]


def paired_match_specs(
    names, count, start=0, *, simulator_seed_base=1000000, pairing_seed=20261103
):
    """Repeat a fixed deck cohort with fresh simulator seeds, preserving seat swaps.

    The simulator uses positive signed 32-bit seeds. Policy sampling has its own
    independent seed in the evaluator; neither seed is inferred from outcomes.
    """
    pairs = heldout_pairings(names, count, start, seed=pairing_seed)
    if not isinstance(simulator_seed_base, int) or not 0 < simulator_seed_base <= 2**31 - 1:
        raise ValueError("Simulator seed base must be a positive signed 32-bit integer")
    if simulator_seed_base + pairs[-1][0] > 2**31 - 1:
        raise ValueError("Simulator seed range exceeds signed 32-bit integers")
    return [
        dict(pair=pair, seed=simulator_seed_base + pair, learner_seat=seat, deck0=a, deck1=b)
        for pair, a, b in pairs
        for seat in (0, 1)
    ]
