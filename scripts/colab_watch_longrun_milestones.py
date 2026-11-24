"""Record significant changes while the current trainer is alive, without polling GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

out=Path('/content/crrl_training/runs/longrun')
backup=Path('/content/drive/MyDrive/cr_rl_ppo/training/longrun')
report=out/'milestones.jsonl'

def read(path):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return {}

def watch():
    pid=int((out/'pipeline.pid').read_text())
    def alive():
        try: return b'/cr_train_pipeline.py\0' in Path(f'/proc/{pid}/cmdline').read_bytes()
        except OSError: return False
    seen=set()
    def record(key,**values):
        if key in seen:return
        seen.add(key)
        with report.open('a') as stream:
            stream.write(json.dumps(dict(event=key,unix=time.time(),**values))+'\n')
    while True:
        summary=read(out/'summary.json')
        update=summary.get('update')
        if update is not None:
            actor=summary.get('actor_overflow_steps',0)
            critic=summary.get('critic_overflow_steps',0)
            if actor or critic or update%50==0:
                record(f'update_{update}',update=update,actor_overflows=actor,critic_overflows=critic,
                       samples_per_second=summary.get('samples_per_second'))
        for filename in ('awbc_latest_report.json','expert_bc_latest_report.json'):
            phase=read(out/filename)
            if phase:
                stamp=(out/filename).stat().st_mtime_ns
                record(f'{filename}:{stamp}',accepted=phase.get('accepted'),
                       actor_attempted=phase.get('actor_attempted'),reason=phase.get('reason'))
        receipt=read(backup/'latest.json')
        if receipt:
            record(f'drive_checkpoint_{receipt["update"]}',update=receipt['update'])
        if not alive():
            record('controller_exited',status=read(out/'exit.json'))
            break
        time.sleep(20)

if '--worker' in sys.argv:
    watch()
else:
    with (out/'milestones_monitor.log').open('a') as log:
        job=subprocess.Popen([sys.executable,'-u','/content/crrl_watch_longrun_milestones.py','--worker'],
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(watcher_pid=job.pid,report=str(report))))
