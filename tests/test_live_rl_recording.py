import gzip
import json
import math

import numpy as np
import pytest

from cr_rl.client.rl_recording import CONTRACT, as_ppo_segment, export_episode, load_episode, set_result
from cr_rl.client.schema_v4 import SchemaV4 as S


def test_vetoed_proposal_keeps_its_probability_and_loss(tmp_path):
    obs = [0] * S.SEAT_INTS
    obs[S.H_TICK] = 1
    p = [0.] * S.ACTIONS
    p[0], p[1] = .7, .3
    terminal = {'tick': 9, 'session_id': tmp_path.name, 'clock_ended': True,
                'match_id': 'game', 'obs': [9, *obs[1:]]}
    manifest = {'arguments': {'selection_mode': 'policy_sample', 'execute': True},
                'rl_contract': {'version': CONTRACT, 'action_semantics': 'policy_proposal',
                                'initial_memory_reset': True, 'actor_fingerprint': 'actor'},
                'checkpoint_sha256': 'checkpoint', 'frames': 1,
                'stop_reason': 'battle_end', 'collection_complete': True}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'terminal.json').write_text(json.dumps(terminal))
    events = [{'event':'battle_start','tick':1},
              {'event':'decision','tick':1,'decision_id':1,'action':0}]
    (tmp_path/'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    (tmp_path/'diagnostics').mkdir()
    with gzip.open(tmp_path/'diagnostics/frames_0001.jsonl.gz', 'wt') as stream:
        json.dump({'tick':1,'decision_id':1,'match_id':'game','action':0,
                   'action_selection':{'proposed_action':1,'spell_vetoed':True},
                   'proposal_log_probability':math.log(.3),'obs':obs,
                   'probabilities':p,'legal_actions':[0,1]}, stream)
    with pytest.raises(ValueError, match='No verified terminal outcome'):
        load_episode(tmp_path)
    set_result(tmp_path, 'loss', evidence='Observed defeat screen')
    export_episode(tmp_path)
    with np.load(tmp_path/'training/rl_episode.npz') as data:
        assert data['actions'].tolist() == [1]
        assert data['post_selection_actions'].tolist() == [0]
        assert data['old_logp'][0] == pytest.approx(math.log(.3))
        assert data['rewards'].tolist() == [-1.]
        assert data['terminated'].tolist() == [True]
        assert data['step_ticks'].tolist() == [8]
    terminal['tick'] = 10
    (tmp_path/'terminal.json').write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match='Result provenance mismatch'):
        load_episode(tmp_path)


def _record_execution_episode(run, execution='confirmed', first_action=1):
    """A complete two-frame record using the real schema and provenance checks."""
    manifest = {'arguments': {'selection_mode': 'policy_sample', 'execute': True},
                'rl_contract': {'version': CONTRACT, 'action_semantics': 'policy_proposal',
                                'initial_memory_reset': True, 'actor_fingerprint': 'actor'},
                'checkpoint_sha256': 'checkpoint', 'frames': 2,
                'stop_reason': 'battle_end', 'collection_complete': True}
    terminal_obs = [0] * S.SEAT_INTS
    terminal_obs[S.H_TICK] = 17
    terminal = {'tick': 17, 'session_id': run.name, 'clock_ended': True,
                'match_id': 'game', 'obs': terminal_obs}
    probabilities = [0.] * S.ACTIONS
    probabilities[0], probabilities[1] = .7, .3
    frames = []
    hand = ['Knight', 'Fireball', 'Skeletons', 'IceSpirit']
    decisions = []
    for idx, (tick, action) in enumerate(((1, first_action), (9, 0)), start=1):
        obs = [0] * S.SEAT_INTS
        obs[S.H_TICK] = tick
        frames.append({'tick': tick, 'decision_id': idx, 'match_id': 'game', 'action': action,
                       'action_selection': {'proposed_action': action, 'spell_vetoed': False},
                       'proposal_log_probability': math.log(probabilities[action]), 'obs': obs,
                       'probabilities': probabilities, 'legal_actions': [0, 1]})
        decisions.append({'event': 'decision', 'tick': tick, 'decision_id': idx,
                          'action': action, 'played_count': 0, 'hand': hand})
    events = [{'event': 'battle_start', 'tick': 1}, decisions[0]]
    if execution == 'stale':
        events.append({'event': 'stale_action_skipped', 'decision_id': 1, 'action': first_action})
    elif execution in ('confirmed', 'timeout'):
        receipt = {'match_id': 'game', 'action': first_action, 'status': 'sent_unconfirmed'}
        events.append({'event': 'submitted', 'decision_id': 1, 'receipt': receipt})
        final = dict(receipt, status='confirmed' if execution == 'confirmed' else 'timed_out')
        events.append({'event': 'receipt', 'decision_id': 1, 'receipt': final, 'played_count': 0})
    events.append(decisions[1])
    (run/'manifest.json').write_text(json.dumps(manifest))
    (run/'terminal.json').write_text(json.dumps(terminal))
    (run/'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    (run/'diagnostics').mkdir()
    with gzip.open(run/'diagnostics/frames_0001.jsonl.gz', 'wt') as stream:
        for frame in frames:
            stream.write(json.dumps(frame)+'\n')
    set_result(run, 'loss', evidence='Observed defeat screen')
    return events


def _write_events(run, events):
    (run/'events.jsonl').write_text('\n'.join(map(json.dumps, events)))


def test_stale_proposal_retains_context_but_not_actor_label(tmp_path):
    _record_execution_episode(tmp_path, execution='stale')
    ep = load_episode(tmp_path)
    assert ep['actions'].tolist() == [1, 0]
    assert ep['post_selection_actions'].tolist() == [1, 0]
    assert ep['confirmed_actions'].tolist() == [0, 0]
    assert ep['policy_loss_mask'].tolist() == [False, True]
    assert ep['raw_obs'].shape == (2, S.SEAT_INTS)
    assert ep['step_ticks'].tolist() == [8, 8]
    assert ep['old_logp'][0] == pytest.approx(math.log(.3))
    assert ep['rewards'].tolist() == [0., -1.]
    segment = as_ppo_segment(ep)
    assert segment['policy_loss_mask'].tolist() == [False, True]
    assert segment['valid'].tolist() == [True, True]
    from cr_rl.training.awbc_playbuffer import read_game
    assert read_game(tmp_path)['actor_keep'].tolist() == [False, True]


@pytest.mark.parametrize('decision_id', [0, 3, None, True])
def test_orphan_stale_event_is_rejected(tmp_path, decision_id):
    events = _record_execution_episode(tmp_path, execution='stale')
    events[2]['decision_id'] = decision_id
    _write_events(tmp_path, events)
    with pytest.raises(ValueError, match='no matching decision'):
        load_episode(tmp_path)


@pytest.mark.parametrize('action', [0, 2, True])
def test_mismatched_stale_action_is_rejected(tmp_path, action):
    events = _record_execution_episode(tmp_path, execution='stale')
    events[2]['action'] = action
    _write_events(tmp_path, events)
    with pytest.raises(ValueError, match='Stale skip does not match'):
        load_episode(tmp_path)


def test_stale_event_before_its_decision_is_rejected(tmp_path):
    events = _record_execution_episode(tmp_path, execution='stale')
    events[1], events[2] = events[2], events[1]
    _write_events(tmp_path, events)
    with pytest.raises(ValueError, match='no matching decision'):
        load_episode(tmp_path)


def test_duplicate_stale_event_is_rejected(tmp_path):
    events = _record_execution_episode(tmp_path, execution='stale')
    events.insert(3, dict(events[2]))
    _write_events(tmp_path, events)
    with pytest.raises(ValueError, match='duplicate stale skip'):
        load_episode(tmp_path)


@pytest.mark.parametrize('stale_before_submit', [False, True])
def test_stale_and_submitted_evidence_cannot_coexist(tmp_path, stale_before_submit):
    events = _record_execution_episode(tmp_path)
    events.insert(2 if stale_before_submit else 4,
                  {'event': 'stale_action_skipped', 'decision_id': 1, 'action': 1})
    _write_events(tmp_path, events)
    with pytest.raises(ValueError, match='Conflicting'):
        load_episode(tmp_path)


def test_nonwait_without_submission_or_skip_is_rejected(tmp_path):
    _record_execution_episode(tmp_path, execution='missing')
    with pytest.raises(ValueError, match='no submission or verified skip'):
        load_episode(tmp_path)


@pytest.mark.parametrize('conflicting', [False, True])
def test_duplicate_decision_events_are_rejected_before_dictionary_coalescing(tmp_path, conflicting):
    events = _record_execution_episode(tmp_path)
    duplicate = dict(events[1])
    if conflicting:
        duplicate['action'] = 0
    events.insert(2, duplicate)
    _write_events(tmp_path, events)
    with pytest.raises(ValueError, match='Invalid or duplicate decision events'):
        load_episode(tmp_path)


@pytest.mark.parametrize('identity', [True, None, []])
def test_decision_ids_require_actual_integers(tmp_path, identity):
    events = _record_execution_episode(tmp_path)
    events[1]['decision_id'] = identity
    _write_events(tmp_path, events)
    with pytest.raises(ValueError, match='Invalid or duplicate decision events'):
        load_episode(tmp_path)


@pytest.mark.parametrize('execution, action, expected_mask, expected_confirmed', [
    ('missing', 0, [True, True], [0, 0]),
    ('confirmed', 1, [True, True], [1, 0]),
    ('timeout', 1, [False, True], [0, 0]),
])
def test_wait_confirmed_and_verified_opening_timeout_remain_compatible(
        tmp_path, execution, action, expected_mask, expected_confirmed):
    _record_execution_episode(tmp_path, execution=execution, first_action=action)
    ep = load_episode(tmp_path)
    assert ep['policy_loss_mask'].tolist() == expected_mask
    assert ep['confirmed_actions'].tolist() == expected_confirmed
