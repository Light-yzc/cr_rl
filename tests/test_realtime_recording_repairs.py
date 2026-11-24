import gzip
import json
import math
import threading

import pytest

from cr_rl.client.async_recording import OrderedRecorder
from cr_rl.client.rl_recording import load_episode, as_ppo_segment, sha256_file
from cr_rl.client.schema_v4 import SchemaV4 as S
from test_live_rl_recording import _record_execution_episode, _write_events


def frames(run):
    with gzip.open(run/'diagnostics/frames_0001.jsonl.gz','rt') as stream:
        return [json.loads(line) for line in stream]


def save_frames(run,values):
    with gzip.open(run/'diagnostics/frames_0001.jsonl.gz','wt') as stream:
        for row in values:stream.write(json.dumps(row)+'\n')


def save_terminal(run,value):
    (run/'terminal.json').write_text(json.dumps(value))
    result=json.loads((run/'result.json').read_text())
    result['terminal_sha256']=sha256_file(run/'terminal.json')
    (run/'result.json').write_text(json.dumps(result))


@pytest.mark.parametrize('gap',[17,24,91,160])
def test_awbc_irregular_times_keep_actual_context_and_mask_boundaries(tmp_path,gap):
    events=_record_execution_episode(tmp_path,execution='missing',first_action=0)
    rows=frames(tmp_path);rows[1]['tick']=1+gap;rows[1]['obs'][S.H_TICK]=1+gap
    events[-1]['tick']=1+gap;_write_events(tmp_path,events);save_frames(tmp_path,rows)
    terminal=json.loads((tmp_path/'terminal.json').read_text())
    terminal['tick']=9+gap;terminal['obs'][S.H_TICK]=9+gap;save_terminal(tmp_path,terminal)
    with pytest.raises(ValueError,match='0.8s'):load_episode(tmp_path)
    episode=load_episode(tmp_path,purpose='awbc')
    assert episode['step_ticks'].tolist()==[gap,8]
    assert episode['raw_obs'].shape[0]==2 and not episode['policy_loss_mask'].any()
    assert episode['rewards'].tolist()==[0.,-1.]
    with pytest.raises(ValueError,match='not an admitted PPO'):as_ppo_segment(episode)


def native(count):
    return dict(own_seat=1,summoners=[dict(played_count=0,hand=[0,1,2,3]),
                                     dict(played_count=count,hand=[0,1,2,3],pending_commands=[])])


@pytest.mark.parametrize('status',['timed_out','rejected'])
def test_terminal_pending_cancel_does_not_discard_full_game(tmp_path,status):
    events=_record_execution_episode(tmp_path,execution='missing',first_action=0)
    rows=frames(tmp_path)
    rows[1].update(action=1,proposal_log_probability=math.log(.3),snapshot=native(7))
    rows[1]['action_selection']['proposed_action']=1
    events[-1].update(action=1,played_count=7)
    sent=dict(match_id='game',action=1,status='sent_unconfirmed')
    events += [dict(event='submitted',decision_id=2,receipt=sent),
               dict(event='receipt',decision_id=2,receipt=dict(sent,status=status))]
    _write_events(tmp_path,events);save_frames(tmp_path,rows)
    terminal=json.loads((tmp_path/'terminal.json').read_text());terminal['snapshot']=native(7)
    terminal['snapshot']['summoners'][1]['pending_commands']=[dict(seq=8)]
    save_terminal(tmp_path,terminal)
    episode=load_episode(tmp_path)
    assert episode['actions'].tolist()==[0,1] and episode['confirmed_actions'].tolist()==[0,0]
    assert episode['policy_loss_mask'].tolist()==[True,False]
    assert episode['execution_exclusions'][0]['proof']=='ended_battle_unchanged_native_count_and_hand'
    terminal['snapshot']['summoners'][1]['played_count']=8;save_terminal(tmp_path,terminal)
    with pytest.raises(ValueError,match='ambiguous'):load_episode(tmp_path)


def test_midgame_no_execution_requires_no_server_pending(tmp_path):
    events=_record_execution_episode(tmp_path,execution='timeout')
    for event in events:
        if event.get('event') in ('decision','receipt'):event['played_count']=7
    _write_events(tmp_path,events)
    rows=frames(tmp_path)
    for row in rows:row['snapshot']=native(7)
    save_frames(tmp_path,rows)
    assert load_episode(tmp_path)['policy_loss_mask'].tolist()==[False,True]
    rows[1]['obs'][S.H_PENDING_COUNT]=1;save_frames(tmp_path,rows)
    with pytest.raises(ValueError,match='ambiguous'):load_episode(tmp_path)


def test_ordered_background_io_never_drops_or_reorders_records():
    writer=OrderedRecorder(capacity=4);seen=[];main=threading.get_ident()
    for i in range(4):writer.submit(lambda n:seen.append((n,threading.get_ident())),i)
    writer.close()
    assert [row[0] for row in seen]==list(range(4))
    assert all(row[1]!=main for row in seen)


def test_background_error_is_reported_and_thread_closes():
    writer=OrderedRecorder()
    def fail():raise OSError('disk failure')
    writer.submit(fail)
    with pytest.raises(RuntimeError,match='recording failed'):writer.flush()
    with pytest.raises(RuntimeError,match='recording failed'):writer.close()
    assert not writer.thread.is_alive()
