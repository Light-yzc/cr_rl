"""Launch the fastest stable measured configuration once its predecessor exits."""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

root=Path('/content/crrl_training')
drive=Path('/content/drive/MyDrive/cr_rl_ppo')
work=Path('/content/crrl_pipeline_validation')
result=root/'runs/throughput/results.json'

def read(path):
    return json.loads(path.read_text())

def checked_candidates():
    try: rows=read(result)
    except (OSError, ValueError): return []
    return rows

def processes(names):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            fields=p.read_bytes().split(b'\0')
        except OSError:
            continue
        if any(Path(os.fsdecode(arg)).name in names for arg in fields if arg.endswith(b'.py')):
            yield int(p.parent.name)

def worker():
    with (work/'longrun-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={'benchmark_colab_pipeline.py','train_selfplay.py','crrl_finish_pipeline.py',
                'crrl_queue_throughput.py','run_army_elixir_experiments.py'}
        while list(processes(active)):
            time.sleep(20)
        rows=checked_candidates()
        if len(rows)!=3 or {(r['envs'],r['frame_microbatch'],r['batch']) for r in rows} != {
                (4,32,1),(8,32,1),(8,64,2)}:
            raise RuntimeError('The planned throughput sweep did not finish')
        eligible=[r for r in rows if r.get('eligible') and r.get('exit_code')==0
                  and math.isfinite(r['samples_per_second'])]
        if not eligible:
            raise RuntimeError('No throughput configuration passed stability checks')
        best=max(eligible,key=lambda r:r['samples_per_second'])
        info=read(work/'combined_passed.json')
        if not (info['awbc']['passed'] and info['bc_actor_attempted']):
            raise RuntimeError('The combined pipeline has not passed')
        archive=Path('/content/crrl_pipeline_production.zip')
        with zipfile.ZipFile(archive) as z:
            if any(not (root/n).resolve().is_relative_to(root.resolve()) for n in z.namelist()):
                raise ValueError('Unsafe production archive member')
            z.extractall(root)
        target=drive/'preparation/crrl_training_update.zip'
        temporary=target.with_suffix('.partial')
        shutil.copyfile(archive,temporary)
        temporary.replace(target)
        decision=dict(envs=best['envs'],frame_microbatch=best['frame_microbatch'],
                      actor_batch_size=best['batch'],critic_batch_size=best['batch'],
                      wandb=False,validation_interval=200,validation_pairs=4)
        for target in (drive/'preparation/cr_train_config.json',root/'prepared/cr_train_config.json'):
            config=read(target)
            config.update(decision)
            temporary=target.with_suffix('.partial')
            temporary.write_text(json.dumps(config,indent=2))
            temporary.replace(target)
        snapshot=drive/'preparation/cr_train_throughput.json'
        snapshot.write_text(json.dumps(dict(selected=best,candidates=rows),indent=2))
        print(json.dumps(dict(stage='launching',selected=best,config=decision)),flush=True)
        subprocess.run([sys.executable,str(root/'scripts/cr_train_pipeline.py'),'start'],cwd=root,check=True)

if '--worker' in sys.argv:
    worker()
else:
    with (work/'longrun-launch.log').open('a') as log:
        job=subprocess.Popen([sys.executable,'-u','/content/crrl_queue_longrun.py','--worker'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=job.pid,log=str(work/'longrun-launch.log'))))
