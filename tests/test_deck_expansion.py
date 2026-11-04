import copy

import pytest

from cr_rl.training.deck_expansion import expand_deck_pool


def pool(names):
    return {"decks": {name: [f"{name}_{i}" for i in range(8)] for name in names}}


def test_expansion_preserves_old_indices_cards_and_config_ownership():
    old = pool(["a", "b"])
    new = pool(["a", "b", "c", "d"])
    new["slots"] = {name: [0] * 8 for name in new["decks"]}
    before = copy.deepcopy((old, new))
    expanded, report = expand_deck_pool(old, new)
    assert report["added_decks"] == ["c", "d"]
    assert report["old_decks"] == ["a", "b"]
    assert report["new_decks"] == ["a", "b", "c", "d"]
    assert report["old_spec_sha256"] != report["new_spec_sha256"]
    assert (old, new) == before
    new["decks"]["c"][0] = "changed"
    assert expanded["decks"]["c"][0] == "c_0"


@pytest.mark.parametrize("names", [["a"], ["a", "b"], ["b", "a", "c"]])
def test_expansion_rejects_removal_reordering_or_no_new_deck(names):
    with pytest.raises(ValueError, match="append names"):
        expand_deck_pool(pool(["a", "b"]), pool(names))


def test_expansion_rejects_mutated_old_deck_new_forms_and_invalid_cards():
    old, new = pool(["a", "b"]), pool(["a", "b", "c"])
    new["decks"]["a"][0] = "different"
    with pytest.raises(ValueError, match="existing"):
        expand_deck_pool(old, new)
    new = pool(["a", "b", "c"])
    new["slots"] = {"a": [1] + [0] * 7}
    with pytest.raises(ValueError, match="existing"):
        expand_deck_pool(old, new)
    new["slots"] = {"c": [1] + [0] * 7}
    with pytest.raises(ValueError, match="ordinary"):
        expand_deck_pool(old, new)
    new = pool(["a", "b", "c"])
    new["decks"]["c"][0] = new["decks"]["c"][1]
    with pytest.raises(ValueError, match="eight unique"):
        expand_deck_pool(old, new)
