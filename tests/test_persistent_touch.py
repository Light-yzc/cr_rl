import io
from types import SimpleNamespace

import pytest

from cr_rl.client.persistent_touch import PersistentAdbTouch
from cr_rl.client.adb_actions import ActionObservation,AdbActionBridge,mumu_900x1600_calibration


def test_command_framing_and_no_retry_after_partial_failure():
    touch=PersistentAdbTouch.__new__(PersistentAdbTouch)
    touch.gap_ms=20
    touch.process=SimpleNamespace(stdin=io.BytesIO(),stdout=io.BytesIO(b'SENT 42\nERROR InvocationTargetException\n'))
    touch._bounded=lambda operation:operation()
    assert touch(((280,1448),(500,800)))=={'android_touch_ms':42}
    with pytest.raises(RuntimeError,match='unconfirmed'):
        touch(((280,1448),(500,800)))
    assert touch.process.stdin.getvalue().count(b'PLAY ')==2
    with pytest.raises(ValueError):touch(((280,1448),(999,800)))
    assert touch.process.stdin.getvalue().count(b'PLAY ')==2


def test_fast_transport_keeps_freshness_and_pending_guards():
    calls=[]
    state=ActionObservation('game',10.,(1,2,3,4),(True,)*2307,900,1600,True)
    bridge=AdbActionBridge(mumu_900x1600_calibration('test',validated=True),adb='adb',
                           state_source=lambda:state,clock=lambda:10.1,tap_transport=calls.append)
    receipt=bridge.submit(1,match_id='game',expected_card_id=1,execute=True)
    assert receipt['status']=='sent_unconfirmed'
    assert len(calls)==1
    with pytest.raises(RuntimeError):bridge.submit(1,match_id='game',expected_card_id=1,execute=True)
    assert len(calls)==1
    bridge.resolve('confirmed')
    bridge.clock=lambda:11.
    with pytest.raises(ValueError):bridge.submit(1,match_id='game',expected_card_id=1,execute=True)
    assert len(calls)==1
