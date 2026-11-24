"""Startup checks for the explicitly configured root MuMu private-server device."""
import subprocess
import time
import os
from contextlib import contextmanager


@contextmanager
def keep_system_awake(*, set_state=None):
    """Temporary per-thread idle-sleep request; does not cancel scheduled shutdown."""
    if set_state is None and os.name=='nt':
        import ctypes
        set_state=ctypes.windll.kernel32.SetThreadExecutionState
    enabled=bool(set_state(0x80000001)) if set_state is not None else False
    try:
        yield enabled
    finally:
        if enabled:set_state(0x80000000)


def ensure_transport(serial, *, adb='adb', attempts=3, run=None, sleep=None):
    """Reconnect exactly one configured transport; never kill the shared ADB server."""
    run = run or subprocess.run
    sleep = sleep or time.sleep
    last = None
    for attempt in range(attempts):
        try:
            result = run([adb,'-s',serial,'get-state'],capture_output=True,text=True,timeout=5,check=True)
            if result.stdout.strip()=='device':
                return
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            last = error
        if serial.startswith('127.0.0.1:'):
            try:
                run([adb,'connect',serial],capture_output=True,text=True,timeout=5,check=True)
                result = run([adb,'-s',serial,'get-state'],capture_output=True,text=True,timeout=5,check=True)
                if result.stdout.strip()=='device':
                    return
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                last = error
        if attempt+1 < attempts:
            sleep(1)
    raise RuntimeError(f'ADB device {serial} unavailable after {attempts} attempts. '
                       'Start the configured MuMu instance and enable ADB; no other device was selected.') from last


def capture_screen(adb, serial):
    """A read-only screenshot may be retried once after transport reconnection."""
    command=[adb,'-s',serial,'exec-out','screencap','-p']
    try:
        return subprocess.check_output(command,timeout=5,stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
        ensure_transport(serial,adb=adb)
        return subprocess.check_output(command,timeout=5,stderr=subprocess.PIPE)


def prepare_private_device(serial, *, adb='adb', run=subprocess.run):
    def command(*args):
        result=run([adb,*args],capture_output=True,text=True,timeout=10,check=True)
        return result.stdout.strip()
    # Connect only explicitly named loopback emulator transports. USB devices
    # must already be connected; no discovery or modification of other devices.
    if serial.startswith('127.0.0.1:'):
        command('connect',serial)
    identity=command('-s',serial,'shell','id','-u')
    changed=False
    if identity!='0':
        command('-s',serial,'root')
        command('-s',serial,'wait-for-device')
        identity=command('-s',serial,'shell','id','-u')
        changed=True
    if identity!='0':
        raise RuntimeError('Selected emulator has no root ADB; enable its root setting before reading game memory')
    return {'serial':serial,'uid':0,'restarted_adbd':changed}
