"""Finalize the existing full-source worker after successful completion."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/content/crrl_training')
work = root/'public_bc_full'
if '--worker' in sys.argv:
    while True:
        state = json.loads((work/'state.json').read_text())
        if state['stage']=='failed':
            raise RuntimeError('Source worker failed: '+state.get('error','unknown'))
        if state['stage']=='stopped_between_shards':
            raise SystemExit('Source preparation was stopped')
        if state['stage']=='full_scan_completed':
            env = dict(os.environ,PYTHONPATH=str(root/'src'))
            subprocess.run([sys.executable,'scripts/catalog_quality_bc.py','--root',
                '/content/drive/MyDrive/cr_rl_ppo/preparation/full_quality_bc'],cwd=root,env=env,check=True)
            break
        time.sleep(30)
else:
    with (work/'catalog_finalizer.log').open('a') as log:
        process = subprocess.Popen([sys.executable,'-u',str(root/'scripts/colab_finalize_full_bc.py'),'--worker'],
            cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'catalog_finalizer_pid':process.pid}))
