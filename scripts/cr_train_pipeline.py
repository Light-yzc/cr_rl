"""Single notebook/CLI entry point for a restartable PPO + BC + AWBC job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cr_rl.training.colab_pipeline import ppo_command, progress
from cr_rl.training.process_watchdog import ProgressWatchdog, restore_local_pair


def tracking_mode(config):
    if not config.get('wandb',True):
        return 'disabled'
    mode=os.environ.get('WANDB_MODE') or config.get('wandb_mode','online')
    if mode not in ('online','offline'):
        raise ValueError('WANDB_MODE must be online or offline for this pipeline')
    if mode=='online' and not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Load WANDB_API_KEY from Colab Secrets/environment, or explicitly select offline tracking')
    return mode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('start','status','stop','worker'))
    args=p.parse_args()
    config=json.loads((ROOT/'prepared/cr_train_config.json').read_text())
    out=ROOT/'runs'/config['run_name'];out.mkdir(parents=True,exist_ok=True)
    pid_path=out/'pipeline.pid'
    if args.action=='status':
        print(json.dumps(progress(ROOT,config['run_name']),ensure_ascii=False));return
    if args.action=='stop':
        pid=int(pid_path.read_text())
        if str(__file__).encode() not in Path(f'/proc/{pid}/cmdline').read_bytes():
            raise RuntimeError('Controller PID does not belong to this pipeline')
        os.kill(pid,signal.SIGINT);print('Graceful stop requested; checkpoint publication will finish.');return
    if args.action=='start':
        mode=tracking_mode(config)
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try: command=path.read_bytes()
            except OSError: continue
            if any((b'/'+s+b'\0') in command for s in
                   (b'train_selfplay.py',b'train_live_awbc.py',b'run_army_elixir_experiments.py')):
                raise RuntimeError('An existing training/evaluation experiment is active; refusing a second job')
        with (out/'controller.log').open('a') as log:
            process=subprocess.Popen([sys.executable,str(__file__),'worker'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(controller_pid=process.pid,log=str(out/'controller.log'),
                              wandb_mode=mode)));return
    lock=(out/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid_path.write_text(str(os.getpid()))
    (out/'exit.json').unlink(missing_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'.cache/vendor/crforge/python'),
             WANDB_MODE=tracking_mode(config),WANDB_DIR=str(out),
             WANDB_PROJECT=config.get('wandb_project','crrl-ppo-live-20261117'))
    if config.get('sparse_placement'):
        env['CRRL_SPARSE_PLACEMENT'] = '1'
    env.pop('CRRL_ALLOW_TF32', None)
    if config.get('gradient_checkpointing') is False:
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    with (out/'console.log').open('a') as log:
        trainer = None
        stopping = False
        syncer = None
        sync_restarts = 0
        def start_syncer():
            with (out/'awbc_inbox.log').open('a') as sync_log:
                return subprocess.Popen([sys.executable, str(ROOT/'scripts/sync_awbc_inbox.py'),
                    '--inbox', '/content/drive/MyDrive/cr_rl_ppo/preparation/awbc_inbox',
                    '--sessions', str(ROOT/'sessions'),
                    '--state', str(ROOT/'prepared/awbc_inbox_state.json'),
                    '--interval', str(config.get('awbc_inbox_poll_seconds',60)), '--loop'],
                    cwd=ROOT, stdout=sync_log, stderr=subprocess.STDOUT)
        if not config.get('ppo_only') and config.get('awbc',True) and config.get('awbc_inbox_sync',True):
            syncer = start_syncer()
        def _signal_trainer(proc, sig):
            if proc is None or proc.poll() is not None:
                return
            if hasattr(os, 'killpg'):
                try:
                    os.killpg(proc.pid, sig)
                    return
                except OSError:
                    pass
            proc.send_signal(sig)
        def _cleanup_trainer_group(pid):
            if hasattr(os, 'killpg') and pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    pass
        def stop(signum,frame):
            nonlocal stopping
            stopping = True
            _signal_trainer(trainer, signal.SIGINT)
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        recoveries = 0
        maximum = int(config.get('trainer_max_restarts',5))
        try:
            while True:
                command=ppo_command(ROOT,config)
                (out/'launch.json').write_text(json.dumps(command,indent=2))
                trainer=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=(os.name != 'nt'))
                watchdog=ProgressWatchdog([out/'progress.jsonl',out/'console.log',out/'summary.json',out/'validation'],
                    float(config.get('trainer_stall_seconds',1800)))
                forced = False
                while trainer.poll() is None:
                    if syncer is not None and syncer.poll() is not None and sync_restarts < 3:
                        sync_restarts += 1
                        print(json.dumps(dict(event='awbc_sync_restart', attempt=sync_restarts)),flush=True)
                        syncer = start_syncer()
                    print(json.dumps(progress(ROOT,config['run_name']),ensure_ascii=False),flush=True)
                    try: trainer.wait(timeout=30)
                    except subprocess.TimeoutExpired: pass
                    if trainer.poll() is None and watchdog.stalled():
                        print(json.dumps(dict(event='trainer_stall',pid=trainer.pid)),flush=True)
                        _signal_trainer(trainer, signal.SIGINT)
                        try: trainer.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            _signal_trainer(trainer, signal.SIGKILL)
                            trainer.wait();forced=True
                _cleanup_trainer_group(trainer.pid)
                if trainer.returncode == 0 and not forced:
                    break
                if recoveries >= maximum:
                    break
                restored_update=restore_local_pair(out)
                recoveries += 1
                print(json.dumps(dict(event='trainer_recovered',update=restored_update,
                    attempt=recoveries,discarded_uncommitted_work=True)),flush=True)
                (out/'watchdog_latest.json').write_text(json.dumps(dict(update=restored_update,
                    attempt=recoveries,exit_code=trainer.returncode,time=time.time())))
                if stopping:
                    break
        finally:
            if syncer is not None and syncer.poll() is None:
                syncer.terminate()
                try: syncer.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    syncer.kill();syncer.wait()
    (out/'exit.json').write_text(json.dumps(dict(exit_code=trainer.returncode,time=time.time(),recoveries=recoveries)))
    raise SystemExit(trainer.returncode)

if __name__=='__main__': main()
