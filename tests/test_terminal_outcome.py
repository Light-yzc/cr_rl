import copy
import json

import pytest

from cr_rl.client.outcome import finalize_result, infer_terminal_outcome, screen_score_bounds
from cr_rl.client.schema_v4 import SchemaV4 as S


def terminal(own=1,enemy=0):
    obs=[0]*S.SEAT_INTS;obs[S.H_TICK]=3600
    obs[S.H_CROWNS]=own;obs[S.H_CROWNS_ENEMY]=enemy
    obs[S.H_TOWERS:S.H_TOWERS+2]=[1000,7000]
    obs[S.H_TOWERS+6:S.H_TOWERS+8]=[1000,7000]
    return dict(tick=3600,clock_ended=True,own_seat=1,obs=obs,
                snapshot=dict(tick=3600,clock_ended=True,own_seat=1))


@pytest.mark.parametrize('own,enemy,outcome',[(3,1,'win'),(0,1,'loss'),(2,1,'win'),(1,3,'loss')])
def test_ended_status_produces_evidenced_terminal_outcome(own,enemy,outcome):
    result,evidence=infer_terminal_outcome(terminal(own,enemy))
    assert result==outcome and evidence['own_crowns']==own and evidence['clock_ended']


def test_unknown_tie_unended_or_unobserved_towers_do_not_invent_rewards():
    for changed in ('tie','unended','unknown','seat','tick'):
        t=terminal()
        if changed=='tie':t=terminal(1,1)
        if changed=='unended':t['snapshot']['clock_ended']=False
        if changed=='unknown':t['obs'][S.H_TOWERS+1]=0
        if changed=='seat':t['snapshot']['own_seat']=0
        if changed=='tick':t['obs'][S.H_TICK]=3599
        with pytest.raises(ValueError):infer_terminal_outcome(t)


def test_result_screen_conflict_rejects_status(tmp_path):
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(900,1600),'black')
    ImageDraw.Draw(image).rectangle((160,210,319,359),fill=(240,180,20))
    path=tmp_path/'result.png';image.save(path)
    assert screen_score_bounds(path)['outcome']=='loss'
    with pytest.raises(ValueError,match='disagree'):
        infer_terminal_outcome(terminal(3,1),path)


def test_backfill_writes_reward_and_integrity_provenance_idempotently(tmp_path):
    t=terminal(0,1);t.update(session_id=tmp_path.name,match_id='match')
    (tmp_path/'terminal.json').write_text(json.dumps(t))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(stop_reason='battle_end',
        collection_complete=True,checkpoint_sha256='source-model')))
    first=finalize_result(tmp_path);second=finalize_result(tmp_path)
    assert first==second and first['reward']==-1 and first['outcome']=='loss'
    assert first['terminal_sha256'] and first['source']=='terminal_status_inference_v1'
    assert json.loads((tmp_path/'manifest.json').read_text())['outcome_verified']
    bad=dict(first,reward=123)
    (tmp_path/'result.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='integrity'):
        finalize_result(tmp_path)


def test_recording_seal_captures_final_result_without_requiring_a_screenshot(tmp_path):
    from cr_rl.client.outcome import seal_recording
    t=terminal(1,0);t.update(session_id=tmp_path.name,match_id='match')
    (tmp_path/'terminal.json').write_text(json.dumps(t))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(stop_reason='battle_end',
        collection_complete=True,checkpoint_sha256='source-model')))
    seal_recording(tmp_path)
    marker=json.loads((tmp_path/'collection_sealed.json').read_text())
    assert marker['format']=='crrl-sealed-session-v1'
    assert marker['manifest']==json.loads((tmp_path/'manifest.json').read_text())
    assert marker['result']==json.loads((tmp_path/'result.json').read_text())
    assert marker['result']['reward']==1
