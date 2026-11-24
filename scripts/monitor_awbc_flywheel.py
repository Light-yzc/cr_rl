"""One low-overhead health check, intended for Windows Task Scheduler every 30m."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cr_rl.training.awbc_inbox import atomic_json, exclusive_lock
from cr_rl.training.colab_transport import run_bounded


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {} if default is None else default


def tail(path, limit=16384):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                continue
    except OSError:
        pass
    return {}


def probe(root, session):
    now = time.time()
    issues = []
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
               "-and $_.CommandLine -match 'run_awbc_flywheel.py|collect_nulls_presets.py|"
               "run_nulls_policy.py|upload_awbc_batches.py' } | "
               "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    process_result = run_bounded(['powershell', '-NoProfile', '-Command', command], timeout=20)
    processes = json.loads(process_result.stdout.decode('utf-8-sig') or '[]')
    if isinstance(processes, dict):
        processes = [processes]
    owned = [p for p in processes if str(ROOT).lower() in (p.get('CommandLine') or '').lower()]
    roles = {role: [p['ProcessId'] for p in owned if name in (p.get('CommandLine') or '')]
             for role, name in [('supervisor', 'run_awbc_flywheel.py'),
                                ('collector', 'collect_nulls_presets.py'),
                                ('policy', 'run_nulls_policy.py'),
                                ('uploader', 'upload_awbc_batches.py')]}
    paused = (root / 'STOP').exists()
    adb = 'G:/sdk/platform-tools/adb.exe'
    try:
        result = run_bounded([adb, '-s', '127.0.0.1:16416', 'get-state'], timeout=8)
        connected = result.returncode == 0 and result.stdout.strip() == b'device'
    except (OSError, subprocess.TimeoutExpired):
        connected = False
    if not paused:
        if not connected:
            issues.append('Emulator ADB disconnected')
        for role in ('supervisor', 'uploader'):
            if not roles[role]:
                issues.append('Missing local ' + role)
    state = read(root / 'flywheel.json')
    active = state.get('active') or {}
    batch_path = Path(active['folder']) / 'batch.json' if active else None
    batch = read(batch_path) if batch_path else {}
    inflight = batch.get('inflight') or {}
    events = batch_path.parent / inflight['name'] / 'events.jsonl' if inflight else None
    last = tail(events) if events else {}
    event_age = max(0, time.time() - last['unix']) if last.get('unix') else None
    if not paused and event_age is not None and event_age > 900:
        issues.append('Trajectory has no new events for over 15 minutes')
    if not paused and event_age is None:
        last_control = tail(root / 'events.jsonl')
        if now - last_control.get('time', 0) > 900:
            issues.append('No trajectory or supervisor activity for over 15 minutes')
    complete = sum(len(read(p).get('completed', {}))
                   for p in (root / 'sessions').glob('*/batch.json'))
    uploaded = len(read(root / 'outbox/upload_state.json').get('known', {}))
    report = dict(time=datetime.datetime.now().astimezone().isoformat(), unix=now,
                  paused=paused, local=dict(processes=roles, adb_connected=connected,
                  actor_update=active.get('update'), completed_games=complete,
                  uploaded_games=uploaded, backlog_games=max(0, complete-uploaded),
                  inflight=inflight.get('name'), last_event_age_seconds=event_age))
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total', 'available', 'total_commit',
             'free_commit', 'total_virtual', 'free_virtual', 'extended')]
    memory = Memory(); memory.length = ctypes.sizeof(memory)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        report['local']['free_commit_gib'] = round(memory.free_commit / 2**30, 2)
    cloud = None
    for _ in range(2):
        try:
            result = run_bounded([sys.executable, '-S', str(ROOT/'scripts/colab_cli_checked.py'),
                'exec', '-s', session, '-f', str(ROOT/'scripts/colab_flywheel_health.py'),
                '--timeout', '25'], timeout=100, cwd=ROOT)
            for line in result.stdout.decode('utf-8', errors='replace').splitlines():
                if line.startswith('FLYWHEEL_HEALTH_JSON '):
                    cloud = json.loads(line.split(' ', 1)[1])
            if cloud is not None:
                break
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    if cloud is None:
        issues.append('Colab status unavailable after two attempts (not proof of a crash)')
    else:
        for role in ('trainer', 'receiver', 'controller'):
            if not cloud['processes'][role]:
                issues.append('Missing Colab ' + role)
        if cloud.get('activity_age_seconds') is None or cloud['activity_age_seconds'] > 1200:
            issues.append('Colab training logs stale for over 20 minutes')
    report.update(cloud=cloud, issues=issues, status='attention' if issues else 'ok')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/awbc_flywheel/live')
    parser.add_argument('--session', default='cr-train-l4-1117')
    parser.add_argument('--notify', action='store_true')
    args = parser.parse_args()
    folder = args.root/'health'; folder.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(folder/'monitor.lock'):
        previous = read(folder/'latest.json')
        try:
            report = probe(args.root, args.session)
        except Exception as error:
            report = dict(time=datetime.datetime.now().astimezone().isoformat(), unix=time.time(),
                          status='attention', issues=['Health probe failed: '+type(error).__name__])
        old_local = previous.get('local', {})
        local = report.get('local', {})
        if (not report.get('paused') and previous.get('unix') and local
                and report['unix'] - previous['unix'] > 1200
                and local['completed_games'] == old_local.get('completed_games')):
            report['issues'].append('No completed games since previous check')
        cloud = report.get('cloud') or {}
        old_cloud = previous.get('cloud') or {}
        if (cloud.get('update') is not None and cloud.get('update') == old_cloud.get('update')
                and report['unix'] - previous.get('unix', report['unix']) > 1200):
            report['issues'].append('Colab update unchanged since previous check')
        report['status'] = 'attention' if report['issues'] else 'ok'
        atomic_json(folder/'latest.json', report)
        with (folder/'history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(report, ensure_ascii=False)+'\n')
        if args.notify and report['issues']:
            try:
                result = run_bounded(['msg.exe', os.environ['USERNAME'], '/TIME:60',
                    'CR_RL monitor: '+ '; '.join(report['issues'])], timeout=8)
                report['notification_sent'] = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                report['notification_sent'] = False
            atomic_json(folder/'latest.json', report)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
