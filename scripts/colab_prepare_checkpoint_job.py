"""Launch CPU checkpoint preparation and return without blocking notebook status."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/content/crrl_training')
out = root/'prepared'
out.mkdir(exist_ok=True)
env = dict(os.environ, PYTHONPATH=str(root/'src')+':'+str(root/'.cache/vendor/crforge/python'))
with (out/'checkpoint_preparation.log').open('a') as log:
    process = subprocess.Popen([sys.executable,'-u','scripts/prepare_single_ppo_checkpoint.py',
        '--checkpoint','/content/drive/MyDrive/cr_rl_ppo/ppo_latest.pt', '--out',str(out)],
        cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'stage':'checkpoint_preparation','pid':process.pid,'log':str(out/'checkpoint_preparation.log')}))
