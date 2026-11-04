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


def test_new_simulator_seed_base_changes_only_seeds_and_keeps_swap_pairs():
    from cr_rl.evaluation.cohort import paired_match_specs

    names = [f"deck{i}" for i in range(10)]
    legacy = paired_match_specs(names, 4, 16)
    fresh = paired_match_specs(names, 4, 16, simulator_seed_base=2000000)
    assert len(legacy) == len(fresh) == 8
    for before, after in zip(legacy, fresh, strict=True):
        assert after["seed"] - before["seed"] == 1000000
        assert {k: v for k, v in before.items() if k != "seed"} == {
            k: v for k, v in after.items() if k != "seed"
        }
    for left, right in zip(fresh[::2], fresh[1::2], strict=True):
        assert left["learner_seat"] == 0 and right["learner_seat"] == 1
        assert left["seed"] == right["seed"]
        assert (left["deck0"], left["deck1"]) == (right["deck0"], right["deck1"])
    defaults = heldout_pairings(names, 4, 16)
    assert [(row["pair"], row["deck0"], row["deck1"]) for row in legacy[::2]] == defaults
    assert [row["seed"] for row in legacy[::2]] == [1000000 + i for i in range(16, 20)]


@pytest.mark.parametrize("base", [0, -1, 2**31, 2**31 - 1])
def test_evaluation_seed_range_cannot_wrap_or_use_invalid_seed(base):
    from cr_rl.evaluation.cohort import paired_match_specs

    with pytest.raises(ValueError, match="seed"):
        paired_match_specs(["a", "b", "c"], 2, simulator_seed_base=base)
