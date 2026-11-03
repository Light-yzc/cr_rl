import numpy as np
import pytest

from cr_rl.evaluation.cohort import heldout_pairings


def test_eight_pair_cohort_is_unique_and_preserves_legacy_initial_pairs():
    names = [f"deck{i}" for i in range(10)]
    pairs = heldout_pairings(names, 8)
    assert len({frozenset((a, b)) for _, a, b in pairs}) == 8
    legacy = np.random.default_rng(20261103).permutation(names).tolist()
    assert pairs[:3] == [(i, legacy[i * 2], legacy[i * 2 + 1]) for i in range(3)]
    assert heldout_pairings(names, 4, start=4) == pairs[4:]
    appearances = [sum(name in (a, b) for _, a, b in pairs) for name in names]
    assert max(appearances) - min(appearances) <= 1


def test_pairing_rejects_repeated_or_impossible_cohorts():
    with pytest.raises(ValueError, match="distinct"):
        heldout_pairings(["a", "a"], 1)
    with pytest.raises(ValueError, match="unique"):
        heldout_pairings(["a", "b"], 2)
    with pytest.raises(ValueError, match="positive"):
        heldout_pairings(["a", "b"], 0)
