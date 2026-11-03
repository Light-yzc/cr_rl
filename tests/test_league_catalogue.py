from cr_rl.config import load_yaml


def test_default_league_is_unique_and_holdout_has_no_card_combination_leak():
    train, holdout = load_yaml("decks.yaml"), load_yaml("decks_eval.yaml")
    base_keys = []
    for catalogue in (train, holdout):
        canonical = set()
        bases = set()
        for name, cards in catalogue["decks"].items():
            forms = catalogue["slots"][name]
            assert len(cards) == len(set(cards)) == len(forms) == 8
            assert all(form in (0, 1, 2, 3) for form in forms)
            key = tuple(sorted(zip(cards, forms, strict=True)))
            assert key not in canonical, f"Duplicate deck or source alias: {name}"
            canonical.add(key)
            bases.add(tuple(sorted(cards)))
            assert name in catalogue["provenance"]
        base_keys.append(bases)
    assert not base_keys[0] & base_keys[1]
