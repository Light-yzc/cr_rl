import numpy as np
import pytest

from cr_rl.training.sparky_curriculum import SparkyCurriculum


@pytest.mark.parametrize(('seat', 'arm'), [(0, 'reserve_wait'), (1, 'sparky_play')])
def test_native_sparky_intervention_only_changes_primary_mask(seat, arm):
    from test_army_elixir_shaping import native_env

    deck = ['ZapMachine', 'Knight', 'Archer', 'MinionHorde', 'Cannon', 'Golem', 'Tombstone', 'Graveyard']
    with native_env() as env:
        vector = env.vector
        schema = vector.schema
        card_id = vector.card_id('ZapMachine')
        for seed in range(10, 40):
            vector.reset_env(0, seed, deck, deck)
            if card_id in schema.field(env.observations[0, seat], 'hand_card'):
                break
        else:
            pytest.fail('Could not find a native opening hand with Sparky')

        curriculum = SparkyCurriculum(schema, card_id, vector.max_elixir,
                                     float(arm == 'reserve_wait'), float(arm == 'sparky_play'))
        curriculum.reset_lane(0, seed, 0, env.observations[0, seat])
        stats = curriculum.new_stats()
        active = [True]
        seat_ids = [seat]
        actions = np.zeros((1, 2), dtype=np.int32)
        if arm == 'reserve_wait':
            # Native battles start with six elixir and accrue more before the gate
            # opens. Spend once to exercise a later natural recovery through six.
            for _ in range(40):
                observation = env.observations[0, seat]
                costs = schema.field(observation, 'hand_cost')
                plays = env.masks[0, seat, 1:schema.first_ability].reshape(-1, curriculum.tiles)
                slots = np.flatnonzero((costs == 3) & plays.any(axis=1))
                if len(slots):
                    slot = int(slots[0])
                    actions[0, seat] = 1 + slot * curriculum.tiles + np.flatnonzero(plays[slot])[0]
                    env.step(actions)
                    actions[:] = schema.wait
                    break
                env.step(actions)
            else:
                pytest.fail('No native three-elixir play to open the reserve test')
        for _ in range(100):
            raw, native = env.observations.copy(), env.masks.copy()
            behavior, candidates = curriculum.prepare(raw, native, active, seat_ids, stats)
            if behavior is not native:
                break
            env.step(actions)
        else:
            pytest.fail('No native intervention opportunity reached')

        np.testing.assert_array_equal(native, env.masks)
        np.testing.assert_array_equal(behavior[0, 1 - seat], native[0, 1 - seat])
        assert np.all(behavior <= native)
        slot = int(np.flatnonzero(schema.field(raw[0, seat], 'hand_card') == card_id)[0])
        first = 1 + slot * curriculum.tiles
        sparky = slice(first, first + curriculum.tiles)
        if arm == 'reserve_wait':
            assert np.flatnonzero(behavior[0, seat]).tolist() == [schema.wait]
            assert not native[0, seat, sparky].any()
        else:
            np.testing.assert_array_equal(behavior[0, seat, sparky], native[0, seat, sparky])
            assert behavior[0, seat].sum() == behavior[0, seat, sparky].sum() > 1
            actions[0, seat] = np.flatnonzero(behavior[0, seat])[0]
        assert curriculum.record_choices(candidates, actions, raw, stats) == {(0, seat)}

        env.step(actions)
        assert not vector.fault(0)
        assert schema.field(env.observations[0, seat], 'last_refusal') == 0
        if arm == 'reserve_wait':
            for _ in range(3):
                next_raw = env.observations.copy()
                after, candidates = curriculum.prepare(next_raw, env.masks, active, seat_ids, stats)
                if after is env.masks:
                    break
                assert curriculum.record_choices(candidates, actions, next_raw, stats) == {(0, seat)}
                env.step(actions)
            assert after is env.masks
            assert after[0, seat, sparky].any()
            assert stats[arm]['forced_wait'] == 3
            assert stats[arm]['reserve_windows_started'] == stats[arm]['reserve_windows_reached_legal'] == 1
            assert stats[arm]['reserve_windows_aborted'] == 0
        else:
            after, _ = curriculum.prepare(env.observations, env.masks, active, seat_ids, stats)
            assert after is env.masks
            actions[:] = schema.wait
            for _ in range(40):
                env.step(actions)
                curriculum.record_followups(env.observations, env.dones, np.zeros(1, dtype=bool), stats)
            assert stats[arm]['forced_sparky_hp_windows'] == 1
            assert schema.field(env.observations[0, 1 - seat], 'enemy_play_count') == 1

        curriculum.reset_lane(0, seed, 1, raw[0, seat])
        if arm == 'reserve_wait':
            started, _ = curriculum.prepare(raw, native, active, seat_ids, stats)
            assert started is not native
        threatened = raw.copy()
        observation = threatened[0, seat]
        count_idx = schema.header['entity_count'][0]
        offset = schema.header_ints + schema.entity_ints * observation[count_idx]
        observation[count_idx] += 1
        observation[offset + schema.entity['side']] = 1
        observation[offset + schema.entity['kind']] = schema.kinds['character']
        observation[offset + schema.entity['flags']] = schema.flags['alive']
        observation[offset + schema.entity['y']] = schema.height_units // 2
        guarded, _ = curriculum.prepare(threatened, native, active, seat_ids, stats)
        assert guarded is native
        if arm == 'reserve_wait':
            assert stats[arm]['reserve_windows_aborted'] == 1
            recovered, _ = curriculum.prepare(raw, native, active, seat_ids, stats)
            assert recovered is native


def test_intervention_keeps_critic_target_and_masks_only_one_actor_row():
    from cr_rl.training.state_ppo import prepare_state_rollout_targets
    from cr_rl.training.state_selfplay import _finalize_segment, _new_segment

    segment = _new_segment(0, 'sparky', 'other', prefix_raw=np.empty((0, 1)),
                           h0_actor=None, h0_behavior=None)
    segment.update(
        _actor_excluded=[0], _raw_obs=[np.zeros(1), np.ones(1)],
        _legal_masks=[np.ones(4, dtype=bool)] * 2, _actions=[1, 2],
        _old_log_probs=[-0.7, -1.2], _rewards=[0, 1], _raw_rewards=[0, 1],
        _reward_hp=[0, 0], _reward_crown=[0, 0], _reward_overflow=[0, 0], _reward_army=[0, 0],
        _terminated=[False, True], _truncated=[False, False], _valid=[True, True],
    )
    segment = _finalize_segment(segment)
    segment['old_values'] = np.zeros(2)
    prepare_state_rollout_targets([segment], gamma=0.99, gae_lambda=1.0, choice_only=True)
    assert segment['actor_mask'].tolist() == [False, True]
    assert segment['critic_mask'].tolist() == [True, True]
    assert segment['returns'].tolist() == pytest.approx([0.99, 1.0])
