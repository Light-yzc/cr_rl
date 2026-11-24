"""Continuously collect with frozen Actors, refresh between batches, upload in parallel."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import ctypes

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cr_rl.training.awbc_inbox import atomic_json, exclusive_lock
from cr_rl.training.colab_transport import terminate_tree
from cr_rl.training.process_watchdog import ProgressWatchdog
from cr_rl.client.menu_navigation import classify_menu, ensure_lobby
from cr_rl.client.runtime import capture_screen, ensure_transport, prepare_private_device, keep_system_awake


def read_json(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def latest_actor(models):
    receipt=read_json(models/'latest_actor.json')
    if not receipt:return None
    path=Path(receipt['checkpoint'])
    if path.is_file() and path.stat().st_size==receipt['bytes'] and receipt.get('actor_only'):
        return receipt
    return None


def prune_actors(models,active=None):
    """Only this cache's finished Actor files; never training checkpoints or recordings."""
    latest=latest_actor(models)
    keep={str(Path(active).resolve())} if active else set()
    if latest:keep.add(str(Path(latest['checkpoint']).resolve()))
    root=models.resolve()
    for path in models.glob('actor_u*.pt'):
        resolved=path.resolve()
        if resolved.parent==root and not path.is_symlink() and str(resolved) not in keep:
            path.unlink()
    partials=sorted(models.glob('actor_u*.partial'))
    for path in partials[:-1]:
        if path.resolve().parent==root and not path.is_symlink():path.unlink()


def free_commit_bytes():
    if os.name!='nt':return None
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
            ('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),
            ('total_commit',ctypes.c_ulonglong),('free_commit',ctypes.c_ulonglong),
            ('total_virtual',ctypes.c_ulonglong),('free_virtual',ctypes.c_ulonglong),
            ('extended',ctypes.c_ulonglong)]
    value=Memory();value.length=ctypes.sizeof(value)
    return int(value.free_commit) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) else None


class Flywheel:
    def __init__(self,args):
        self.args=args;self.root=args.out.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.models=args.models.resolve();self.models.mkdir(parents=True,exist_ok=True)
        self.sessions=self.root/'sessions';self.sessions.mkdir(exist_ok=True)
        self.state_path=self.root/'flywheel.json'
        self.state=read_json(self.state_path,dict(plan=args.plan,cycles=0,active=None))
        if self.state['plan']!=args.plan:raise ValueError('Resume requires the same deck plan')
        self.stopping=False;self.collector=None;self.uploader=None;self.downloader=None
        current=latest_actor(self.models)
        age=time.time()-current.get('downloaded_at',0) if current else float('inf')
        self.started_at=time.time()
        self.initial_wait=args.refresh_now and not self.state.get('active')
        self.next_download=0 if args.refresh_now else time.monotonic()+max(0,args.refresh_minutes*60-age)
        self.next_upload_retry=0
        self.env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
        self.env.setdefault('CUDA_MODULE_LOADING','LAZY')
        self.flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0

    def event(self,event,**data):
        row=dict(event=event,time=time.time(),**data)
        with (self.root/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

    def spawn(self,command,log):
        with log.open('a',encoding='utf-8') as output:
            return subprocess.Popen(command,cwd=ROOT,env=self.env,stdout=output,stderr=subprocess.STDOUT,
                                    creationflags=self.flags,start_new_session=os.name!='nt')

    def terminate_tree(self,process):
        if process is None or process.poll() is not None:return
        terminate_tree(process)

    def upload_command(self):
        return [sys.executable,'-u',str(ROOT/'scripts/upload_awbc_batches.py'),
            '--source',str(self.sessions),'--session',self.args.session,
            '--outbox',str(self.root/'outbox'),'--batch-games',str(self.args.upload_batch_games)]

    def tick(self):
        if (self.root/'STOP').exists():self.stopping=True
        if self.stopping:
            if self.state['active']:
                (Path(self.state['active']['folder'])/'STOP').touch()
            return
        now=time.monotonic()
        if self.uploader is None or (self.uploader.poll() is not None and now>=self.next_upload_retry):
            self.uploader=self.spawn(self.upload_command()+['--loop'],self.root/'uploader.log')
            self.next_upload_retry=now+60
            self.event('uploader_started',pid=self.uploader.pid)
        if self.downloader is not None and self.downloader.poll() is not None:
            code=self.downloader.returncode;self.downloader=None
            self.next_download=now+(self.args.refresh_minutes*60 if code==0 else 60)
            self.event('actor_sync_finished',exit_code=code,latest=latest_actor(self.models))
            prune_actors(self.models,(self.state.get('active') or {}).get('checkpoint'))
        if self.downloader is None and now>=self.next_download:
            candidate=latest_actor(self.models)
            active=self.state.get('active')
            if active and candidate and candidate['update']!=active['update']:
                # Do not spend bandwidth replacing an Actor that has not yet
                # been used. Finish this frozen batch, then refresh again.
                self.next_download=now+30
                return
            self.downloader=self.spawn([sys.executable,'-u',str(ROOT/'scripts/download_latest_actor.py'),
                '--session',self.args.session,'--out',str(self.models)],self.root/'model_sync.log')
            self.event('actor_sync_started',pid=self.downloader.pid)

    def recover(self):
        """Known overlays close by verified buttons. Active battles are never restarted."""
        from cr_rl.client.nulls_reader import NullsMemoryReader,NotInBattleError
        def probe():
            try:
                reader=NullsMemoryReader.connect(self.args.serial,adb=self.args.adb,
                                               require_running_clock=False)
                return not reader.read_snapshot().clock_ended
            except NotInBattleError:return False
            except Exception:return None
        deadline=time.monotonic()+660;inactive=0;unknown_since=time.monotonic()
        while time.monotonic()<deadline and not self.stopping:
            self.tick()
            try:
                ensure_transport(self.args.serial,adb=self.args.adb)
                screen=classify_menu(capture_screen(self.args.adb,self.args.serial))
                active=probe()
                self.event('navigation_recovery_probe',screen=screen,battle_active=active)
                if screen=='lobby':return
                if active is False:
                    inactive+=1
                    if screen!='unknown':
                        ensure_lobby(self.args.adb,self.args.serial,timeout=20,
                            failure_dir=self.root/'navigation',battle_probe=probe,allow_result=True)
                        return
                    if inactive>=3 and time.monotonic()-unknown_since>60:
                        # Only the configured private package, after repeated inactive memory reads.
                        cmd=[self.args.adb,'-s',self.args.serial,'shell']
                        foreground=subprocess.check_output(cmd+['dumpsys','activity','activities'],timeout=10).decode(errors='replace')
                        if any('nullsroyale.rel.free/' in line and ('ResumedActivity' in line or 'topResumedActivity' in line)
                               for line in foreground.splitlines()):
                            subprocess.run(cmd+['am','force-stop','nullsroyale.rel.free'],check=True,timeout=10)
                            subprocess.run(cmd+['monkey','-p','nullsroyale.rel.free','-c','android.intent.category.LAUNCHER','1'],
                                           check=True,timeout=10,stdout=subprocess.DEVNULL)
                            self.event('private_app_restarted_after_inactive_checks')
                            inactive=0;unknown_since=time.monotonic();time.sleep(5)
                else:inactive=0
            except (OSError,RuntimeError,subprocess.SubprocessError,ValueError) as error:
                self.event('navigation_retry',reason=str(error))
            time.sleep(3)
        raise RuntimeError('Could not verify a safe lobby; recordings retained, collection stopped')

    def run(self):
        self.event('private_device_ready',**prepare_private_device(self.args.serial,adb=self.args.adb))
        def stop(signum,frame):
            self.stopping=True
            if self.state['active']:(Path(self.state['active']['folder'])/'STOP').touch()
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        failures=0
        try:
            while not self.stopping:
                if self.args.cycles is not None and self.state['cycles']>=self.args.cycles:break
                self.tick()
                if self.stopping:break
                active=self.state['active']
                if not active:
                    actor=latest_actor(self.models)
                    if not actor:time.sleep(3);continue
                    if self.initial_wait and actor.get('downloaded_at',0)<self.started_at:
                        time.sleep(3);continue
                    self.initial_wait=False
                    folder=self.sessions/f'cycle_{self.state["cycles"]:05d}_u{actor["update"]:06d}'
                    folder.mkdir(exist_ok=True)
                    active=dict(folder=str(folder),checkpoint=actor['checkpoint'],update=actor['update'])
                    self.state['active']=active;atomic_json(self.state_path,self.state)
                    self.event('collection_batch_started',**active)
                if shutil.disk_usage(self.root).free<5*1024**3:
                    raise RuntimeError('Less than 5 GiB free; stopping before another battle')
                available=free_commit_bytes()
                if available is not None and available<self.args.min_free_commit_gib*1024**3:
                    self.event('waiting_for_host_memory',free_commit_gib=available/1024**3)
                    for _ in range(10):self.tick();time.sleep(3)
                    continue
                marker=Path(active['folder'])/'STOP'
                if marker.exists():marker.rename(marker.with_name('STOP_resumed_'+str(time.time_ns())))
                self.recover()
                command=[sys.executable,'-u',str(ROOT/'scripts/collect_nulls_presets.py'),
                    '--plan',self.args.plan,'--out',active['folder'],'--checkpoint',active['checkpoint'],
                    '--adb',self.args.adb,'--serial',self.args.serial]
                self.collector=self.spawn(command,Path(active['folder'])/'collector.log')
                self.event('collector_started',pid=self.collector.pid,update=active['update'])
                next_liveness_check=time.monotonic()+15
                watchdog=ProgressWatchdog([Path(active['folder'])/'collector.log'],300)
                while self.collector.poll() is None:
                    self.tick()
                    if time.monotonic()>=next_liveness_check:
                        next_liveness_check=time.monotonic()+15
                        current=read_json(Path(active['folder'])/'batch.json',{})
                        run_name=(current.get('inflight') or {}).get('name')
                        paths=[Path(active['folder'])/'collector.log']
                        if run_name:paths.append(Path(active['folder'])/run_name/'events.jsonl')
                        watchdog.paths=paths
                        changed=max((p.stat().st_mtime for p in paths if p.exists()),default=time.time())
                        if watchdog.stalled():
                            self.event('collector_stall_recovery',last_event_age_seconds=time.time()-changed)
                            self.terminate_tree(self.collector)
                            break
                    time.sleep(3)
                batch=read_json(Path(active['folder'])/'batch.json',{})
                if self.stopping:break
                if self.collector.returncode!=0:
                    log_path=Path(active['folder'])/'collector.log'
                    with log_path.open('rb') as log:
                        log.seek(max(0,log_path.stat().st_size-2600));tail=log.read().decode(errors='replace')
                    inflight=batch.get('inflight') or {}
                    manifest=read_json(Path(active['folder'])/inflight.get('name','')/'manifest.json',{})
                    if (any(text in tail for text in ('CUDA out of memory','WinError 1455','paging file is too small'))
                            and not manifest.get('frames',0) and not manifest.get('sent',0)):
                        self.event('model_startup_memory_retry',completed=len(batch.get('completed',{})))
                        for _ in range(20):self.tick();time.sleep(3)
                        continue
                    completed=len(batch.get('completed',{}))
                    if completed>active.get('last_completed',0):failures=0
                    active['last_completed']=completed;atomic_json(self.state_path,self.state)
                    failures+=1
                    self.event('collector_retry',exit_code=self.collector.returncode,
                        attempt=failures,completed=len(batch.get('completed',{})))
                    if failures>5:raise RuntimeError('Repeated collection failures; inspect collector.log')
                    time.sleep(3);continue
                failures=0
                self.event('collection_batch_completed',update=active['update'],games=len(batch.get('completed',{})))
                self.state['cycles']+=1;self.state['active']=None;atomic_json(self.state_path,self.state)
                prune_actors(self.models)
                if self.args.cycles is not None and self.state['cycles']>=self.args.cycles:break
        finally:
            self.terminate_tree(self.uploader)
            self.terminate_tree(self.downloader)
            # Send a final short batch too; incompletes remain excluded by the packager.
            if self.collector is None or self.collector.poll() is not None:
                try:
                    with (self.root/'uploader.log').open('a') as log:
                        subprocess.run(self.upload_command(),cwd=ROOT,env=self.env,stdout=log,
                                       stderr=subprocess.STDOUT,timeout=600)
                except subprocess.TimeoutExpired:self.event('final_upload_deferred')
            self.event('flywheel_stopped',cycles=self.state['cycles'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/awbc_flywheel/live')
    parser.add_argument('--models',type=Path,default=ROOT/'artifacts/awbc_flywheel/models')
    parser.add_argument('--session',default='cr-train-l4-1117')
    parser.add_argument('--plan',default='2:10,1:10')
    parser.add_argument('--refresh-minutes',type=float,default=60)
    parser.add_argument('--refresh-now',action='store_true',help='Fetch the latest committed Actor before a fresh batch')
    parser.add_argument('--upload-batch-games',type=int,default=10)
    parser.add_argument('--cycles',type=int)
    parser.add_argument('--min-free-commit-gib',type=float,default=6,
                        help='Windows host commit headroom before a model process starts')
    parser.add_argument('--adb',default='G:/sdk/platform-tools/adb.exe')
    parser.add_argument('--serial',default='127.0.0.1:16416')
    args=parser.parse_args()
    if args.refresh_minutes<1 or not 1<=args.upload_batch_games<=256 or (args.cycles is not None and args.cycles<1) or args.min_free_commit_gib<0:
        parser.error('Positive refresh interval/cycles and 1..256 upload batch size required')
    if (args.out/'STOP').exists():parser.error('STOP exists; remove this marker to resume collection')
    with exclusive_lock(ROOT/'artifacts/awbc_flywheel/collector.lock'):
        # Do not manipulate menus while a separately launched policy owns the device.
        with exclusive_lock(ROOT/'artifacts/nulls_live/policy-control.lock'):pass
        with keep_system_awake():
            Flywheel(args).run()


if __name__=='__main__':
    main()
