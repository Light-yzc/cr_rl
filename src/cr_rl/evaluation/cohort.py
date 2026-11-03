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
