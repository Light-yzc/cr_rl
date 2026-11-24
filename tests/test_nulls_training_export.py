import gzip
import importlib.util
import json
from pathlib import Path


spec=importlib.util.spec_from_file_location('nulls_export',Path(__file__).parents[1]/'scripts/export_nulls_training.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_only_confirmed_plays_and_explicit_policy_wait_get_targets(tmp_path):
    events=[{'event':'decision','decision_id':1,'tick':8},
            {'event':'submitted','decision_id':1,'receipt':{'status':'sent_unconfirmed'}},
            {'event':'receipt','decision_id':1,'receipt':{'status':'confirmed'}},
            {'event':'decision','decision_id':2,'tick':16},
            {'event':'decision','decision_id':3,'tick':24},
            {'event':'submitted','decision_id':3,'receipt':{'status':'sent_unconfirmed'}},
            {'event':'receipt','decision_id':3,'receipt':{'status':'timed_out'}}]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
    (tmp_path/'diagnostics').mkdir()
    rows=[dict(decision_id=i,tick=i*8,action=action,reason=reason)
          for i,action,reason in [(1,15,'action_ready'),(2,0,'policy_wait'),(3,20,'action_ready')]]
    with gzip.open(tmp_path/'diagnostics/frames_0001.jsonl.gz','wt') as stream:
        stream.write('\n'.join(json.dumps(x) for x in rows))
    result=list(module.training_rows(tmp_path))
    assert [x['behavior_cloning_target'] for x in result]==[15,0,None]
    assert all(x['reward'] is None and not x['ppo_eligible'] for x in result)
    assert all(not x['outcome_verified'] for x in result)


def test_legacy_tail_without_frame_ids_is_not_mislabeled(tmp_path):
    (tmp_path/'events.jsonl').write_text(json.dumps({'event':'decision','tick':8}))
    (tmp_path/'diagnostics').mkdir()
    with gzip.open(tmp_path/'diagnostics/frames_0002.jsonl.gz','wt') as stream:
        stream.write(json.dumps({'tick':800,'action':0,'reason':'policy_wait'}))
    row=next(module.training_rows(tmp_path))
    assert row['execution_status']=='unlinked'
    assert row['behavior_cloning_target'] is None


def test_spell_veto_wait_is_context_not_model_wait_target(tmp_path):
    events=[{'event':'decision','decision_id':i,'tick':i*8,
             **({'spell_vetoed':True,'selection_reason':'spell_guard_veto'} if i==1 else {})}
            for i in (1,2,3)]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
    (tmp_path/'diagnostics').mkdir()
    rows=[dict(decision_id=i,tick=i*8,action=0,reason='policy_wait',
               **({'action_selection':{'spell_vetoed':True,'proposed_action':42}} if i==2 else {}))
          for i in (1,2,3)]
    with gzip.open(tmp_path/'diagnostics/frames_0001.jsonl.gz','wt') as stream:
        stream.write('\n'.join(json.dumps(x) for x in rows))
    result=list(module.training_rows(tmp_path))
    assert len(result)==3  # Do not discard recurrent context.
    assert [r['behavior_cloning_target'] for r in result]==[None,None,0]
    assert all(not r['ppo_eligible'] and r['reward'] is None for r in result)
