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
    weights = [0.50, 0.22, 0.13, 0.10, 0.05]
    p[actions] = weights
    rng = np.random.default_rng(735)
    counts = np.zeros(S.ACTIONS, dtype=int)
    for _ in range(12000):
        action, info = select_live_action(
            p,
            p > 0,
            obs,
            ("Giant", "Knight", None, None),
            selection_mode="policy_sample",
            rng=rng,
            play_threshold=1,
            slot_threshold=1,
        )
        counts[action] += 1
        assert info["proposed_action"] == action
    assert counts.sum() == counts[actions].sum()
    np.testing.assert_allclose(counts[actions] / 12000, weights, atol=0.015, rtol=0)
    assert counts[actions[3]] > 0  # This card never wins the greedy ranking.


def test_sampling_respects_mask_and_replays_seed():
    obs, p = inputs()
    action = S.play(0, 2, 4)
    p[0] = 0.2
    p[action] = 0.8
    mask = np.zeros(S.ACTIONS, dtype=bool)
    mask[0] = True
    assert select_live_action(p, mask, obs, ("Giant",) * 4, selection_mode="policy_sample")[0] == 0

    def sequence():
        rng = np.random.default_rng(17)
        return [
            select_live_action(p, p > 0, obs, ("Giant",) * 4, selection_mode="policy_sample", rng=rng)[0]
            for _ in range(50)
        ]

    assert sequence() == sequence()


def test_sampled_spell_veto_retains_proposal_for_audit():
    obs, p = inputs()
    action = S.play(0, 2, 4)
    p[action] = 1
    selected, info = select_live_action(p, p > 0, obs, ("Fireball", None, None, None), selection_mode="policy_sample")
    assert selected == 0
    assert info["proposed_action"] == info["vetoed_action"] == action
    assert info["proposed_action_probability"] == 1
    assert info["selection_reason"] == "spell_guard_veto"
    with pytest.raises(ValueError, match="selection mode"):
        select_live_action(p, p > 0, obs, ("Giant",) * 4, selection_mode="typo")


def test_pure_hierarchical_argmax_and_leak_guard():
    obs, p = inputs()
    obs[S.H_ELIXIR] = 50000
    mask = np.zeros(S.ACTIONS, dtype=bool)
    mask[0] = True
    act_slot1 = S.play(1, 4, 10)
    mask[act_slot1] = True
    p[0] = 0.60
    p[act_slot1] = 0.40
    act_w, info_w = select_live_action(
        p, mask, obs, ("ZapMachine", "MiniPekka", "Elixir Collector", "Rage"), selection_mode="argmax"
    )
    assert act_w == 0 and info_w["selection_reason"] == "policy_wait"

    # At >=9.7 elixir, leak guard plays the highest-mass legal slot instead of overflowing elixir forever
    obs[S.H_ELIXIR] = 98000
    act_p, info_p = select_live_action(
        p, mask, obs, ("ZapMachine", "MiniPekka", "Elixir Collector", "Rage"), selection_mode="argmax"
    )
    assert act_p == act_slot1 and info_p["selection_reason"] == "leak_guard_play"
