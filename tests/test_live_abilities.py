from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cr_rl.client.action_selection import select_live_action
from cr_rl.client.adb_actions import ActionObservation, AdbActionBridge, mumu_900x1600_calibration
from cr_rl.client.live_control import PendingAbility,reconcile_ability,pending_targets_for
from cr_rl.client.schema_v4 import SchemaV4 as S


def test_skill_selection_only_when_legal_and_preferred():
    obs=np.zeros(S.SEAT_INTS,dtype=np.int32)
    p=np.zeros(S.ACTIONS);p[0]=.1;p[1]=.2;p[2305]=.7
    mask=p>0
    assert select_live_action(p,mask,obs,('Giant',None,None,None))[0]==2305
    mask[2305]=False
    assert select_live_action(p,mask,obs,('Giant',None,None,None))[0]!=2305
    p[0]=.8;p[2305]=.1
    mask[2305]=True
    assert select_live_action(p,mask,obs,('Giant',None,None,None))[0]!=2305


def test_skill_confirmation_requires_same_champion_and_execution_evidence():
    pending=PendingAbility('game',0,2,55,1,0,10.)
    ab=SimpleNamespace(slot=0,deck_idx=2,unit_eid=55,charges=1,cooldown_ms=0,state=2,pending_seq=100)
    def check(ab,now=10.5):return reconcile_ability(pending,match_id='game',abilities=(ab,),now=now)
    assert check(ab) is None  # READY or sequence changes alone do not confirm.
    for name,value in [('state',7),('charges',0),('cooldown_ms',1000)]:
        changed=SimpleNamespace(**{**vars(ab),name:value})
        assert check(changed)=='confirmed'
    wrong=SimpleNamespace(**{**vars(ab),'state':7,'unit_eid':99})
    assert check(wrong) is None
    assert check(wrong,now=12.)=='timed_out'
    assert pending_targets_for(pending,(0,1,2,3))==(None,)*4


def test_skill_uses_one_shell_tap_and_missing_slot_never_sends():
    calls=[]
    state=ActionObservation('game',10.,(1,2,3,4),(True,)*2307,900,1600,True,(55,None))
    calibration=mumu_900x1600_calibration(validated=True,ability_points=((820,1215),None))
    def pair_only(_):raise AssertionError('Single skill must not be duplicated into a tap pair')
    bridge=AdbActionBridge(calibration,adb='adb',state_source=lambda:state,
                           clock=lambda:10.1,transport=calls.append,tap_transport=pair_only)
    bridge.submit(2305,match_id='game',expected_ability_unit_id=55,execute=True)
    assert calls[0][-1]=='input tap 820 1215'
    bridge.resolve('confirmed')
    with pytest.raises(ValueError):bridge.submit(2306,match_id='game',execute=True)
    with pytest.raises(ValueError):bridge.submit(2305,match_id='game',expected_ability_unit_id=99,execute=True)
    assert len(calls)==1
