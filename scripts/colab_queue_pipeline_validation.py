"""Stage the durable training inputs and queue validation on the existing L4."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import fcntl

root = Path('/content/crrl_training')
drive = Path('/content/drive/MyDrive/cr_rl_ppo')
work = Path('/content/crrl_pipeline_validation')
work.mkdir(exist_ok=True)

def busy():
    names = {'train_selfplay.py','run_army_elixir_experiments.py','train_live_awbc.py'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            if any(Path(os.fsdecode(a)).name in names for a in args if a.endswith(b'.py')):
                return True
        except OSError:
            continue
    return False

if '--worker' in sys.argv:
    lock = (work/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    archive = Path('/content/crrl_pipeline_candidate.zip')
    while busy():
        print('Waiting for isolated GPU experiment to finish',flush=True)
        time.sleep(30)
    with zipfile.ZipFile(archive) as z:
        if any(not (root/n).resolve().is_relative_to(root.resolve()) for n in z.namelist()):
            raise ValueError('Unsafe ZIP path')
        z.extractall(root)
    target = drive/'preparation/crrl_training_update.zip'
    temporary = target.with_suffix('.partial')
    shutil.copyfile(archive,temporary)
    temporary.replace(target)
    subprocess.run([sys.executable,'-u',str(root/'scripts/prepare_cr_train.py')],cwd=root,check=True)
    notebook = drive.parent/'Colab Notebooks/cr_train.ipynb'
    old = notebook.with_name('cr_train.before_pipeline.ipynb')
    if notebook.exists() and not old.exists():
        shutil.copyfile(notebook,old)
    temporary = notebook.with_suffix('.partial')
    shutil.copyfile(root/'notebooks/cr_train.ipynb',temporary)
    temporary.replace(notebook)
    env = dict(os.environ,CRRL_JOB='awbc-smoke')
    with (work/'validation.log').open('w') as log:
        result = subprocess.run([sys.executable,'-u',str(root/'scripts/colab_longrun_job.py'),'--worker'],
                                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    (work/'result.json').write_text(json.dumps(dict(exit_code=result.returncode)))
    sys.exit(result.returncode)
else:
    with (work/'queue.log').open('a') as log:
        job=subprocess.Popen([sys.executable,'-u','/content/crrl_queue_pipeline.py','--worker'],
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=job.pid,log=str(work/'queue.log'),gpu_busy=busy())))
