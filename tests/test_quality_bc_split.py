import pytest

from cr_rl.data.quality_bc_split import quality_bc_partition


def signature(n):
    return f'{n:08x}' + '0'*56


def test_later_rebalance_halves_new_validation_without_touching_prior_or_heldout():
    hold={('H','J')}
    decks=[['A','B'],['C','D']]
    assert quality_bc_partition(signature(5),decks,hold,{})=='train'
    assert quality_bc_partition(signature(10),decks,hold,{})=='validation'
    assert quality_bc_partition(signature(1),decks,hold,{})=='train'
    assert quality_bc_partition(signature(5),decks,hold,
                                {signature(5):'validation'})=='validation'
    assert quality_bc_partition(signature(1),decks,hold,
                                {signature(1):'heldout'})=='heldout'
    assert quality_bc_partition(signature(1),[['J','H'],['C','D']],hold,{})=='heldout'


def test_bad_identity_and_legacy_heldout_conflict_fail_closed():
    with pytest.raises(ValueError, match='signature'):
        quality_bc_partition('broken', [['A'],['B']], set(), {})
    with pytest.raises(ValueError, match='conflicts'):
        quality_bc_partition(signature(1), [['H','J'],['C','D']],
                             {('H','J')},{signature(1):'train'})
