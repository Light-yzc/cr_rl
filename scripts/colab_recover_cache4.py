"""Recover the verified disk checkpoint after a native collector deadlock."""
import json,os,signal,subprocess,sys,time,shutil,zipfile
from pathlib import Path
root=Path('/content/crrl_training')
out=root/'runs/longrun'
drive=Path('/content/drive/MyDrive/cr_rl_ppo')
work=Path('/content/crrl_cache_upgrade')
def worker():
    for pid,marker in ((42559,b'/crrl_raise_opponent_cache.py'),(26990,b'/train_selfplay.py')):
        path=Path(f'/proc/{pid}/cmdline')
        if path.exists() and marker in path.read_bytes():os.kill(pid,signal.SIGKILL)
    for _ in range(30):
        path=Path('/proc/26989/cmdline')
        if not path.exists() or b'/cr_train_pipeline.py' not in path.read_bytes():break
        time.sleep(1)
    else:raise RuntimeError('Old controller has not exited')
    import torch
    checkpoint=torch.load(out/'ppo_latest.pt',map_location='cpu',mmap=True,weights_only=False)
    update=int(checkpoint.get('updates',checkpoint.get('progress',{}).get('update',-1)))
    assert update>=930, update
    del checkpoint
    print(json.dumps(dict(stage='verified_durable_checkpoint',update=update)),flush=True)
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
    (work/'result.json').write_text(json.dumps(dict(resumed=True,from_update=update,
        opponent_cache_size=4,training_opponent_cache_size=4)))

if '--worker' in sys.argv:worker()
else:
    with (work/'recovery.log').open('a') as log:
        job=subprocess.Popen([sys.executable,'-u','/content/crrl_recover_cache4.py','--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('RECOVERY_PID',job.pid)
