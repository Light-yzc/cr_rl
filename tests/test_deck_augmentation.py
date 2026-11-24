import copy

import numpy as np
import pytest

from cr_rl.training.deck_augmentation import augment_forms, deck_probabilities


def test_only_explicit_enabled_forms_and_preserve_source():
    pool = {'decks': {'a': list('ABCDEFGH')}, 'slots': {'a': [1]+[0]*7}}
    tables = {'A': {'EvolvedSpells': ['A_EV1', 'A_hero']},
              'B': {'EvolvedSpells': ['B_hero']},
              'C': {'EvolvedSpells': ['C_EV1']},
              'A_EV1': {}, 'A_hero': {'CardForm': 'HeroForm'},
              'B_hero': {'CardForm': 'HeroForm'},
              'C_EV1': {'NotInUse': True},
              'D_hero': {'CardForm': 'HeroForm'}}
    # Nonempty real rows; unused/unlinked rows must not become candidates.
    tables['A_EV1'] = {'SummonCharacter': 'A_Evolved'}
    original = copy.deepcopy(pool)
    result = augment_forms(pool, tables)
    assert pool == original
    assert list(result['decks'])[0] == 'a'
    assert result['slots']['a'] == pool['slots']['a']
    flags = list(result['slots'].values())
    assert [0]*8 in flags
    assert [2]+[0]*7 in flags
    assert [1, 2]+[0]*6 in flags
    assert all(not any(row[2:]) for row in flags)
    assert augment_forms(result, tables) == result


def test_family_sampling_preserves_ordinary_share_despite_variant_count():
    pool = {'decks': {'a': list('ABCDEFGH'), 'a2': list('ABCDEFGH'),
                      'a3': list('ABCDEFGH'), 'b': list('IJKLMNOP')},
            'slots': {'a2': [1]+[0]*7, 'a3': [2]+[0]*7},
            'deck_sampling': {'mode': 'family_balanced', 'ordinary_fraction': .5}}
    p = deck_probabilities(pool, list(pool['decks']))
    assert np.allclose(p, [.25, .125, .125, .5])
    assert sum(p) == 1
    pool['slots']['a'] = [1]+[0]*7
    with pytest.raises(ValueError, match='ordinary'):
        deck_probabilities(pool, list(pool['decks']))
    assert deck_probabilities({'decks': {}}, []) is None
