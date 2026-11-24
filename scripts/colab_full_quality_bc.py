"""Resumable full-source cloud download and strict BC reconstruction.

All immutable source shards are fetched and hashed on the CPU VM first. Each
shard is then scanned in a separate bounded-memory process. No training starts.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path('/content/crrl_training')
WORK = ROOT/'public_bc_full'
DRIVE = Path('/content/drive/MyDrive/cr_rl_ppo/preparation/full_quality_bc')
REVISION = '059d43a02138a34b1b3009cc2acc7630fb99a638'
BASE = f'https://huggingface.co/datasets/VanguardX101/IL_Replay/resolve/{REVISION}/'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            value.update(block)
    return value.hexdigest()


def atomic(path,value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8')
    temporary.replace(path)


def fetch(record):
    target = WORK/'sources'/record['path']
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==record['bytes'] and sha(target)==record['sha256']:
        return target
    for attempt in range(3):
        try:
            temporary = target.with_suffix('.partial')
            with urlopen(BASE+record['path'],timeout=120) as src,temporary.open('wb') as dst:
                shutil.copyfileobj(src,dst,1024**2)
            if temporary.stat().st_size!=record['bytes'] or sha(temporary)!=record['sha256']:
                raise IOError('Source checksum mismatch: '+record['path'])
            temporary.replace(target)
            return target
        except Exception:
            if attempt==2:
                raise
            time.sleep(2)


def worker():
    WORK.mkdir(parents=True,exist_ok=True);DRIVE.mkdir(parents=True,exist_ok=True)
    lock = (WORK/'worker.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Full-source BC preparation is already running')
    state_path = WORK/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        revision=REVISION,downloaded={},completed={},training_started=False)
    if state['revision']!=REVISION:
        raise ValueError('Source revision changed; use a new workspace')
    def publish(stage,**values):
        state.update(stage=stage,updated_unix=time.time(),pid=os.getpid(),**values)
        atomic(state_path,state)
        atomic(DRIVE/'state.json',state)
        print(json.dumps(dict(stage=stage,**values)),flush=True)
    try:
        with urlopen(BASE+'manifest.json',timeout=60) as response:
            body = response.read(8*1024**2+1)
        if len(body)>8*1024**2:
            raise ValueError('Manifest exceeds bound')
        manifest = json.loads(body)
        files = [r for r in manifest['files'] if r['path'].startswith('replays/')]
        if any(Path(r['path']).is_absolute() or '..' in Path(r['path']).parts for r in files):
            raise ValueError('Unsafe upstream path')
        (WORK/'source_manifest.json').write_bytes(body)
        (DRIVE/'source_manifest.json').write_bytes(body)
        publish('downloading_all_sources',total_source_shards=len(files),
                total_source_rows=sum(r.get('rows',0) for r in files),
                total_source_bytes=sum(r['bytes'] for r in files))
        for index,record in enumerate(files):
            if (WORK/'STOP').exists():
                publish('stopped_between_shards');return
            fetch(record)
            state['downloaded'][record['path']] = record['sha256']
            publish('source_downloaded',downloaded_source_shards=index+1)
        for index,record in enumerate(files):
            if (WORK/'STOP').exists():
                publish('stopped_between_shards');return
            if record['path'] in state['completed']:
                continue
            target = WORK/'datasets'/f'shard_{index:03d}'
            if target.exists():
                target = target.with_name(target.name+'_'+str(time.time_ns()))
            publish('reconstructing',source_shard=record['path'],source_index=index)
            command = [sys.executable,'-u','scripts/build_quality_bc_native.py',
                '--parquet',str(WORK/'sources'/record['path']),'--out',str(target),
                '--install',str(ROOT/'.cache/vendor/crforge/env/build/install/env'),
                '--tables',str(Path.home()/'.crforge/tables/16.635.3'),
                '--max-source-rows',str(record.get('rows',5000)),
                '--target-games',str(record.get('rows',5000))]
            prior = ROOT/'configs/quality_bc_prior_split.json'
            if prior.exists():
                command += ['--prior-split',str(prior)]
            env = dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'.cache/vendor/crforge/python'),
                       JAVA_TOOL_OPTIONS='-Xmx2g',OMP_NUM_THREADS='2')
            with (WORK/f'shard_{index:03d}.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            summary = json.loads((target/'quality_summary.json').read_text())
            shutil.copytree(target,DRIVE/'datasets'/target.name,dirs_exist_ok=True)
            state['completed'][record['path']] = dict(source_sha256=record['sha256'],
                output=str(target),summary=summary)
            publish('shard_completed',completed_source_shards=len(state['completed']),
                accepted_games_before_cross_shard_dedup=sum(
                    row['summary']['accepted_games'] for row in state['completed'].values()))
        from catalog_quality_bc import catalog
        summary = catalog(DRIVE)
        publish('full_scan_completed',training_ready=False,catalog_summary=summary,
                remaining_gate='Cross-shard dedup, old split conflict audit, model contract and context quality review')
    except Exception as error:
        publish('failed',error=repr(error))
        raise


if '--worker' in sys.argv:
    worker()
else:
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'supervisor.log').open('a') as log:
        process = subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/colab_full_quality_bc.py'),'--worker'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'stage':'full_source_preparation_launched','pid':process.pid,
                      'status':str(WORK/'state.json'),'log':str(WORK/'supervisor.log')}))
