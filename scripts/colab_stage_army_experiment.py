"""Separate experimental workspace and job; refuse concurrent GPU training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

for path in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=path.read_bytes().split(b'\0')
    except OSError: continue
    if any(a.endswith(b'/train_selfplay.py') or a.endswith(b'/train_live_awbc.py') for a in args):
        raise RuntimeError('An existing trainer is running; experiment was not started')
root=Path('/content/crrl_army_elixir_1117')
if root.exists(): raise RuntimeError('Experiment workspace already exists; inspect status, do not overwrite')
root.mkdir()
with zipfile.ZipFile('/content/army_elixir_code.zip') as z:
    for member in z.infolist():
        if member.filename.startswith('.cache/'): continue
        if not (root/member.filename).resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe path')
        z.extract(member,root)
(root/'.cache').symlink_to('/content/crrl_training/.cache',target_is_directory=True)
backup=Path('/content/drive/MyDrive/cr_rl_ppo/preparation/army_elixir_experiment_code.zip')
shutil.copyfile('/content/army_elixir_code.zip',backup)
env=dict(os.environ,PYTHONPATH=str(root/'src')+':'+str(root/'.cache/vendor/crforge/python'))
with (root/'experiment.log').open('w') as log:
    job=subprocess.Popen([sys.executable,'-u',str(root/'scripts/run_army_elixir_experiments.py')],
        cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor.pid').write_text(str(job.pid))
print(json.dumps(dict(pid=job.pid,workspace=str(root),original_training_untouched=True)))
