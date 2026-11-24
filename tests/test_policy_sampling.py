import numpy as np
import pytest

from cr_rl.client.action_selection import select_live_action
from cr_rl.client.schema_v4 import SchemaV4 as S


def inputs():
    obs = np.zeros(S.SEAT_INTS, dtype=np.int32)
    p = np.zeros(S.ACTIONS)
    return obs, p


def test_sampling_matches_joint_including_nonwinning_card_and_tiles():
    obs, p = inputs()
    actions = [0, S.play(0, 2, 4), S.play(0, 3, 4), S.play(1, 6, 8), S.FIRST_ABILITY]
    weights = [.50, .22, .13, .10, .05]
    p[actions] = weights
    rng = np.random.default_rng(735)
    counts = np.zeros(S.ACTIONS, dtype=int)
    for _ in range(12000):
        action, info = select_live_action(p, p > 0, obs, ('Giant', 'Knight', None, None),
                                         selection_mode='policy_sample', rng=rng,
                                         play_threshold=1, slot_threshold=1)
        counts[action] += 1
        assert info['proposed_action'] == action
    assert counts.sum() == counts[actions].sum()
    np.testing.assert_allclose(counts[actions] / 12000, weights, atol=.015, rtol=0)
    assert counts[actions[3]] > 0  # This card never wins the greedy ranking.


def test_sampling_respects_mask_and_replays_seed():
    obs, p = inputs()
    action = S.play(0, 2, 4)
    p[0] = .2
    p[action] = .8
    mask = np.zeros(S.ACTIONS, dtype=bool)
    mask[0] = True
    assert select_live_action(p, mask, obs, ('Giant',) * 4,
                              selection_mode='policy_sample')[0] == 0
    def sequence():
        rng = np.random.default_rng(17)
        return [select_live_action(p, p > 0, obs, ('Giant',) * 4,
                                   selection_mode='policy_sample', rng=rng)[0] for _ in range(50)]
    assert sequence() == sequence()


def test_sampled_spell_veto_retains_proposal_for_audit():
    obs, p = inputs()
    action = S.play(0, 2, 4)
    p[action] = 1
    selected, info = select_live_action(p, p > 0, obs, ('Fireball', None, None, None),
                                        selection_mode='policy_sample')
    assert selected == 0
    assert info['proposed_action'] == info['vetoed_action'] == action
    assert info['proposed_action_probability'] == 1
    assert info['selection_reason'] == 'spell_guard_veto'
    with pytest.raises(ValueError, match='selection mode'):
        select_live_action(p, p > 0, obs, ('Giant',) * 4, selection_mode='typo')


def test_argmax_adverse_defense_lane_lock_and_quiet_banking():
    # 1) Quiet-board 4-elixir banking guard when holding unaffordable 6-cost core (ZapMachine / Elixir Collector)
    obs, p = inputs()
    obs[S.H_ELIXIR] = 42000
    mask = np.zeros(S.ACTIONS, dtype=bool)
    mask[0] = True
    mp_bridge = S.play(3, 2, 13)
    mask[1 + 3 * S.TILES : 1 + 4 * S.TILES] = True
    p[0] = 0.30
    p[mp_bridge] = 0.70
    act, info = select_live_action(
        p, mask, obs, ('ZapMachine', 'Elixir Collector', 'Rage', 'MiniPekka'), selection_mode='argmax'
    )
    assert act == 0 and info['selection_reason'] == 'quiet_bank_wait'

    # 2) Heavy unilateral left-lane push: troop defender must stay on left/center instead of right-bridge deformation
    obs2, p2 = inputs()
    obs2[S.H_ELIXIR] = 52000
    obs2[S.H_ENTITY_COUNT] = 1
    rows = obs2[S.HEADER_INTS:].reshape(S.MAX_ENTITIES, S.ENTITY_INTS)
    rows[0, S.E_KIND] = S.KIND_CHARACTER
    rows[0, S.E_SIDE] = 1
    rows[0, S.E_HP] = 3100
    rows[0, S.E_X] = 3500
    rows[0, S.E_Y] = 10500
    mask2 = np.zeros(S.ACTIONS, dtype=bool)
    mask2[0] = True
    mask2[1 + 1 * S.TILES : 1 + 2 * S.TILES] = True
    wrong_right = S.play(1, 15, 12)
    left_def = S.play(1, 4, 10)
    p2[0] = 0.25
    p2[wrong_right] = 0.50
    p2[left_def] = 0.25
    act2, info2 = select_live_action(
        p2, mask2, obs2, ('ZapMachine', 'MiniPekka', 'Elixir Collector', 'Rage'), selection_mode='argmax'
    )
    dec2 = S.decode_action(act2)
    assert dec2['kind'] == 'play' and dec2['slot'] == 1 and dec2['col'] == 4 and dec2['row'] == 10
    assert info2['defend_lane'] == 'left'
