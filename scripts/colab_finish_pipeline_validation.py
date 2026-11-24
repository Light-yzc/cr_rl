"""Complete the BC transition from the verified AWBC checkpoint, when needed."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/content/crrl_training')
work=Path('/content/crrl_pipeline_validation')
run=root/'runs/awbc-smoke'

def read(path):
    return json.loads(path.read_text())

def set_value(command,option,value):
    command[command.index(option)+1]=str(value)

def worker():
    with (work/'finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid=int((work/'validation.pid').read_text())
        while not (work/'result.json').is_file():
            path=Path(f'/proc/{pid}/cmdline')
            if not path.is_file() or b'/content/crrl_queue_pipeline.py\0' not in path.read_bytes():
                raise RuntimeError('Original validation supervisor ended without a result')
            time.sleep(15)
        status=read(work/'result.json')
        awbc=read(run/'awbc_smoke_report.json')
        if status['exit_code']!=0 or not awbc['passed']:
            raise RuntimeError('Original PPO/AWBC validation failed; inspect its logs')
        bc=run/'expert_bc_latest_report.json'
        if bc.exists() and read(bc).get('actor_attempted'):
            print('BC already validated; existing throughput queue owns the sweep',flush=True)
            return
        out=root/'runs/bc-after-awbc-resume'
        out.mkdir(exist_ok=True)
        command=read(run/'exit.json')['argv']
        command.remove('--require-awbc-smoke')
        set_value(command,'--resume',run/'ppo_latest.pt')
        set_value(command,'--out',out)
        set_value(command,'--updates',2)
        # Keep the restored playbuffer, but reserve these two boundaries for BC.
        set_value(command,'--awbc-interval',50)
        env=dict(os.environ,PYTHONPATH=str(root/'src')+':'+str(root/'.cache/vendor/crforge/python'),
                 OMP_NUM_THREADS='2')
        with (out/'console.log').open('w') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        bc=read(out/'expert_bc_latest_report.json')
        summary=read(out/'summary.json')
        if not bc.get('actor_attempted') or summary.get('optimizer_steps',0)<1:
            raise RuntimeError('Restored BC/PPO failed to exercise optimizer steps')
        (work/'combined_passed.json').write_text(json.dumps(dict(
            awbc=awbc,bc_actor_attempted=True,bc_accepted=bc['accepted'],
            post_resume_update=summary['update']),indent=2))
        subprocess.run([sys.executable,'-u',str(root/'scripts/benchmark_colab_pipeline.py')],
                       cwd=root,check=True)

if '--worker' in sys.argv:
    worker()
else:
    with (work/'finish.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u','/content/crrl_finish_pipeline.py','--worker'],
             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,log=str(work/'finish.log'))))
