from types import SimpleNamespace

import pytest

from cr_rl.client.runtime import prepare_private_device
from cr_rl.client.runtime import ensure_transport, capture_screen
import subprocess


def test_root_is_restored_only_for_selected_emulator():
    calls=[];identities=iter(['2000','0'])
    def run(argv,**kwargs):
        calls.append(argv)
        assert kwargs['timeout']==10
        return SimpleNamespace(stdout=next(identities) if argv[-2:]==['id','-u'] else '')
    result=prepare_private_device('127.0.0.1:16416',adb='adb',run=run)
    assert result['uid']==0 and result['restarted_adbd']
    assert ['adb','-s','127.0.0.1:16416','root'] in calls


def test_no_restart_if_root_and_no_fake_success_on_denial():
    calls=[]
    def root(argv,**kwargs):calls.append(argv);return SimpleNamespace(stdout='0')
    assert not prepare_private_device('usb-specific',run=root)['restarted_adbd']
    assert all('root' not in command for command in calls)
    def denied(argv,**kwargs):return SimpleNamespace(stdout='2000')
    with pytest.raises(RuntimeError,match='no root ADB'):
        prepare_private_device('127.0.0.1:16416',run=denied)


def test_reconnects_only_requested_transport_and_verifies_state():
    calls=[]
    connected=False
    def run(argv,**kwargs):
        nonlocal connected
        calls.append(argv)
        if argv[1]=='connect':
            connected=True
            return SimpleNamespace(stdout='connected')
        if not connected:
            raise subprocess.CalledProcessError(1,argv)
        return SimpleNamespace(stdout='device\n')
    ensure_transport('127.0.0.1:16416',adb='adb',run=run,sleep=lambda _:None)
    assert ['adb','connect','127.0.0.1:16416'] in calls
    assert all('kill-server' not in a for a in calls)


def test_connect_success_text_does_not_hide_unavailable_device():
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='connected' if argv[1]=='connect' else 'offline')
    with pytest.raises(RuntimeError,match='unavailable after 2 attempts'):
        ensure_transport('127.0.0.1:16416',run=run,sleep=lambda _:None,attempts=2)
    assert len([x for x in calls if x[1]=='connect'])==2


def test_wake_request_is_temporary_and_released_after_failure():
    from cr_rl.client.runtime import keep_system_awake
    calls=[]
    def set_state(flags):calls.append(flags);return 1
    with pytest.raises(RuntimeError):
        with keep_system_awake(set_state=set_state) as enabled:
            assert enabled
            raise RuntimeError('collector stopped')
    assert calls==[0x80000001,0x80000000]
