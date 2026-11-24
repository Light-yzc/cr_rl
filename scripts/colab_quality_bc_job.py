"""Download one verified public shard and run a bounded CPU reconstruction probe."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

root = Path('/content/crrl_training')
work = root/'public_bc'
revision = '059d43a02138a34b1b3009cc2acc7630fb99a638'


def worker():
    manifest = json.loads((work/'source_manifest.json').read_text())
    record = next(row for row in manifest['files'] if row['path'].startswith('replays/'))
    target = work/'part-000000.parquet'
    h = hashlib.sha256()
    if target.exists():
        with target.open('rb') as source:
            for block in iter(lambda:source.read(1024**2),b''):
                h.update(block)
    else:
        with urlopen(f'https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/{revision}/'+record['path'],
                     timeout=90) as response, target.with_suffix('.partial').open('wb') as dest:
            for block in iter(lambda:response.read(1024**2),b''):
                dest.write(block);h.update(block)
    if h.hexdigest()!=record['sha256']:
        raise IOError('Public shard SHA256 mismatch')
    if not target.exists():
        target.with_suffix('.partial').replace(target)
    print(json.dumps({'stage':'source_verified','bytes':target.stat().st_size,
                      'sha256':h.hexdigest()}),flush=True)
    limit = int(os.environ.get('CRRL_BC_SCAN_ROWS','512'))
    output = work/f'native_probe_{limit}_{time.time_ns()}'
    command = [sys.executable,'-u','scripts/build_quality_bc_native.py',
        '--parquet',str(target),'--out',str(output),
        '--install',str(root/'.cache/vendor/crforge/env/build/install/env'),
        '--tables',str(Path.home()/'.crforge/tables/16.635.3'),
        '--max-source-rows',str(limit),'--target-games','32']
    status = subprocess.run(command,cwd=root).returncode
    backup = Path('/content/drive/MyDrive/cr_rl_ppo/preparation/public_bc_probe')
    if output.exists():
        shutil.copytree(output,backup,dirs_exist_ok=True)
    (work/'job_exit.json').write_text(json.dumps({'exit_code':status,'output':str(output)}))
    raise SystemExit(status)


if '--worker' in sys.argv:
    worker()
else:
    env = dict(os.environ,PYTHONPATH=str(root/'src')+':'+str(root/'.cache/vendor/crforge/python'))
    with (work/'quality_bc_probe.log').open('a') as log:
        process = subprocess.Popen([sys.executable,'-u',str(root/'scripts/colab_quality_bc_job.py'),'--worker'],
            cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'stage':'quality_bc_probe','pid':process.pid,
                      'source_rows_limit':int(os.environ.get('CRRL_BC_SCAN_ROWS','512')),
                      'log':str(work/'quality_bc_probe.log')}))
