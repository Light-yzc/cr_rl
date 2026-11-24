"""Collect bounded batches of complete matches per existing Nulls deck preset."""
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from cr_rl.client.menu_navigation import guarded_tap, ensure_lobby
from cr_rl.client.rl_recording import atomic_json, sha256_file
from cr_rl.client.runtime import ensure_transport

ROOT = Path(__file__).resolve().parents[1]


def next_slot(plan, completed, repeat=False):
    counts = Counter(row['slot'] for row in completed.values())
    cycle = min(counts[slot] // target for slot, target in plan) if repeat else 0
    return next((slot for slot, target in plan if counts[slot] < (cycle+1)*target), None)


def interrupted_attempt(run, sha):
    """Return a reason only for an incomplete attempt; identity errors stay fatal."""
    path = run/'manifest.json'
    if not path.exists():
        return 'interrupted_before_manifest'
    manifest = json.loads(path.read_text(encoding='utf8'))
    if manifest.get('checkpoint_sha256') != sha:
        raise ValueError(f'{run}: changed checkpoint')
    if manifest.get('collection_complete') or manifest.get('stop_reason') == 'battle_end':
        return None  # Even contradictory terminal evidence must be checked by finish().
    return manifest.get('stop_reason') or 'interrupted_without_terminal'


def read_completed(run, sha):
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf8'))
    if (manifest.get('checkpoint_sha256') != sha or not manifest.get('collection_complete')
            or manifest.get('stop_reason') != 'battle_end'):
        raise ValueError(f'{run}: incomplete session or changed checkpoint')
    from cr_rl.client.outcome import finalize_result
    finalize_result(run)
    if not (run/'terminal.json').is_file() or not (run/'terminal.json').stat().st_size:
        raise ValueError(f'{run}: missing terminal snapshot')
    events = [json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf8').splitlines()]
    decks = [r for r in events if r['event']=='own_deck']
    if len(decks)!=1 or len(decks[0]['cards'])!=8:
        raise ValueError(f'{run}: missing deck evidence')
    cards = decks[0]['cards']
    if len({int(c['scid']) for c in cards if int(c['scid']) > 0}) != 8:
        raise ValueError(f'{run}: invalid card identities')
    from cr_rl.client.outcome import seal_recording
    seal_recording(run)
    return sorted((int(c['scid']), c['slot_flags']) for c in cards)


def switch(adb, serial, slot):
    ensure_transport(serial,adb=adb)
    ensure_lobby(adb,serial,timeout=30,failure_dir=ROOT/'artifacts/nulls_live/navigation')
    # On first run the operator may already be on the deck page. Otherwise
    # verify the main battle button before tapping the collection shortcut.
    try:
        guarded_tap(adb, serial, 'leave_decks', timeout=1.5, tap=False)
    except RuntimeError:
        guarded_tap(adb, serial, 'open_decks')
    guarded_tap(adb, serial, 'leave_decks', tap=False)
    subprocess.run([sys.executable, str(ROOT/'scripts/switch_nulls_deck.py'), str(slot),
                    '--adb', adb, '--serial', serial], check=True)
    # A tap immediately after preset selection can be swallowed while the
    # deck screen settles. Retry only if the deck page is still recognized.
    for attempt in range(3):
        guarded_tap(adb, serial, 'leave_decks', timeout=2)
        try:
            guarded_tap(adb, serial, 'battle', tap=False, timeout=4)
            return
        except RuntimeError as error:
            if attempt == 2:
                raise RuntimeError('Could not return to the main battle page. Open it manually, then rerun the same command.') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, help='Preset:completed matches, e.g. 2:10,1:5')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/ppo_v3_v100/actor_export.pt')
    parser.add_argument('--adb', default='G:/sdk/platform-tools/adb.exe')
    parser.add_argument('--serial', default='127.0.0.1:16416')
    parser.add_argument('--verify-switches', action='store_true', help='Switch presets and return to lobby without playing')
    parser.add_argument('--stop-after-total', type=int, help='Pause after this many completed recordings in the batch')
    parser.add_argument('--repeat', action='store_true', help='Repeat the preset quotas until interrupted or STOP file exists')
    args = parser.parse_args()
    if args.stop_after_total is not None and args.stop_after_total < 1:
        parser.error('--stop-after-total must be positive')
    try:
        plan = [list(map(int, v.split(':'))) for v in args.plan.split(',')]
        slots = [s for s, _ in plan]
        if not plan or len(set(slots)) != len(slots) or any(s not in (1,2,3) or not 1<=n<=100 for s,n in plan):
            raise ValueError()
    except ValueError:
        parser.error('Use distinct presets 1-3 and 1-100 matches per preset, e.g. 2:10,1:5')
    if args.verify_switches:
        for slot, _ in plan:
            switch(args.adb,args.serial,slot)
        return
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if sys.platform == 'win32':
        import msvcrt
        batch_lock = (out/'batch.lock').open('a+b')
        if batch_lock.tell() == 0:
            batch_lock.write(b'0'); batch_lock.flush()
        batch_lock.seek(0)
        try:
            msvcrt.locking(batch_lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            parser.error('This collection batch is already running')
    path = out/'batch.json'
    checkpoint = args.checkpoint.resolve(strict=True)
    identity = sha256_file(checkpoint)
    state = (json.loads(path.read_text(encoding='utf8')) if path.exists() else
             {'plan':plan,'checkpoint_sha256':identity,'completed':{},'rosters':{},'inflight':None})
    if state['plan']!=plan or state['checkpoint_sha256']!=identity:
        parser.error('Resume requires the original plan and frozen checkpoint')

    def finish(run, slot):
        roster = [list(c) for c in read_completed(run, identity)]
        previous = state['rosters'].setdefault(str(slot),roster)
        if roster != previous or any(k != str(slot) and v == roster for k,v in state['rosters'].items()):
            raise ValueError(f'{run}: preset changed roster or duplicates another preset')
        state['completed'][run.name] = {'slot':slot,'frames':json.loads((run/'manifest.json').read_text())['frames']}
        state['inflight'] = None
        atomic_json(path,state)
        print(f'Collected {run.name} from preset {slot} ({len(state["completed"])} total)',flush=True)

    if state['inflight']:
        entry = state['inflight']
        run = out/entry['name']
        reason = interrupted_attempt(run,identity)
        if reason is None:
            finish(run,entry['slot'])
        else:
            state.setdefault('skipped', {})[run.name] = {'slot':entry['slot'],'reason':reason}
            state['inflight'] = None
            atomic_json(path,state)
            print(f'Skipping interrupted attempt {run.name}: {reason}. Logs kept; quota unchanged.',flush=True)
    current = None
    navigation_failures = 0
    runtime = None
    while (slot := next_slot(plan,state['completed'],repeat=args.repeat)) is not None:
        if (out/'STOP').exists():
            print(f'STOP requested; finished current match and stopped: {path}',flush=True)
            return
        if args.stop_after_total is not None and len(state['completed']) >= args.stop_after_total:
            print(f'Paused at {len(state["completed"])} completed recordings: {path}',flush=True)
            return
        if slot != current:
            switch(args.adb,args.serial,slot)
            current = slot
        run = out/f'{time.strftime("%Y%m%d_%H%M%S")}_slot{slot}_{uuid4().hex[:6]}'
        run.mkdir()
        state['inflight'] = {'name':run.name,'slot':slot}
        atomic_json(path,state)
        from run_nulls_policy import PolicyRuntime, parse_args, run_logged_battle
        policy_args = parse_args(['--adb', args.adb, '--serial', args.serial,
                                  '--checkpoint', str(checkpoint), '--out', str(run),
                                  '--seconds', '600', '--max-plays', '100', '--execute', '--auto-battle'])
        if runtime is None:
            runtime = PolicyRuntime(policy_args, checkpoint_sha=identity)
        failure = None
        with (run/'console.log').open('w', encoding='utf8', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    run_logged_battle(policy_args, runtime)
                except Exception as error:
                    failure = f'{type(error).__name__}: {error}'
        if failure is not None:
            manifest_path=run/'manifest.json'
            if manifest_path.exists():
                failed=json.loads(manifest_path.read_text(encoding='utf8'))
                if (failed.get('collection_complete') and failed.get('stop_reason')=='battle_end'
                        and (run/'result.json').is_file()):
                    # UI recovery is independent of a verified, fully recorded battle.
                    # Keep that game; the next iteration verifies the lobby before tapping.
                    finish(run,slot)
                    current=None
                    print(f'Kept completed battle after menu error: {run.name}',flush=True)
                    continue
                event_path=run/'events.jsonl'
                ev=[json.loads(l) for l in event_path.read_text(encoding='utf8').splitlines()] if event_path.exists() else []
                navigation_only = (failed.get('frames')==0 and failed.get('sent')==0
                    and not any(e['event'] in ('menu_battle_tapped','battle_start','submitted') for e in ev)
                    and any(e['event']=='error' and 'Could not verify the battle button' in e.get('traceback','') for e in ev))
                if navigation_only and navigation_failures<3:
                    navigation_failures+=1
                    state.setdefault('skipped',{})[run.name]={'slot':slot,'reason':'navigation_before_battle'}
                    state['inflight']=None
                    atomic_json(path,state)
                    print(f'Lobby not ready; retry {navigation_failures}/3 in 10 seconds. Evidence: {run}',flush=True)
                    time.sleep(10)
                    continue
            crash = run/'crash.log'
            tail = crash.read_text(encoding='utf8',errors='replace').splitlines() if crash.exists() else []
            parser.exit(1, f'{run.name}: controller failed: '
                            f'{tail[-1] if tail else failure}\nLog: {run / "console.log"}\n')
        finish(run,slot)
        navigation_failures=0
    print(f'Completed collection plan: {path}',flush=True)


if __name__=='__main__':
    main()
