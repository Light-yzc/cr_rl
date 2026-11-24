"""Finish the current update, publish it, raise resident caches and resume once."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

root=Path('/content/crrl_training')
out=root/'runs/longrun'
drive=Path('/content/drive/MyDrive/cr_rl_ppo')
work=Path('/content/crrl_cache_upgrade')
work.mkdir(exist_ok=True)

def worker():
    lock=(work/'upgrade.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid=int((out/'pipeline.pid').read_text())
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'/cr_train_pipeline.py\0' not in command:
        raise RuntimeError('Controller identity mismatch')
    print(json.dumps(dict(stage='finishing_current_update',controller=pid)),flush=True)
    os.kill(pid,signal.SIGINT)
    while Path(f'/proc/{pid}/cmdline').exists():
        try: running=b'/cr_train_pipeline.py\0' in Path(f'/proc/{pid}/cmdline').read_bytes()
        except OSError: running=False
        if not running: break
        time.sleep(10)
    status=json.loads((out/'exit.json').read_text())
    if status['exit_code']!=0: raise RuntimeError('Trainer did not stop cleanly')
    sys.path.insert(0,str(root/'src'))
    from cr_rl.training.async_publication import freeze_publication_files
    from cr_rl.training.fast_publication import publish_snapshot
    summary=json.loads((out/'summary.json').read_text())
    snapshot=freeze_publication_files(out,summary['update'])
    publish_snapshot(snapshot,drive/'training/longrun',summary['update'],keep_generations=1)
    shutil.rmtree(snapshot)
    print(json.dumps(dict(stage='saved',update=summary['update'])),flush=True)
    source=Path('/content/crrl_colab_pipeline_cache4.py')
    target=root/'src/cr_rl/training/colab_pipeline.py'
    compile(source.read_text(),str(source),'exec')
    shutil.copyfile(source,target)
    for path in (drive/'preparation/cr_train_config.json',root/'prepared/cr_train_config.json'):
        cfg=json.loads(path.read_text())
        cfg.update(opponent_cache_size=4,training_opponent_cache_size=4)
        temp=path.with_suffix('.partial');temp.write_text(json.dumps(cfg,indent=2));temp.replace(path)
    archive=drive/'preparation/crrl_training_update.zip'
    local=work/'updated_code.zip'
    name='src/cr_rl/training/colab_pipeline.py'
    with zipfile.ZipFile(archive) as old,zipfile.ZipFile(local,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as new:
        for item in old.infolist():
            if item.filename!=name: new.writestr(item,old.read(item))
        new.write(target,name)
    temp=archive.with_suffix('.partial');shutil.copyfile(local,temp);temp.replace(archive)
    subprocess.run([sys.executable,str(root/'scripts/cr_train_pipeline.py'),'start'],cwd=root,check=True)
    (work/'result.json').write_text(json.dumps(dict(resumed=True,from_update=summary['update'],
        opponent_cache_size=4,training_opponent_cache_size=4)))

if '--worker' in sys.argv:
    worker()
else:
    with (work/'upgrade.log').open('a') as log:
        job=subprocess.Popen([sys.executable,'-u','/content/crrl_raise_opponent_cache.py','--worker'],
            cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=job.pid,log=str(work/'upgrade.log'))))
