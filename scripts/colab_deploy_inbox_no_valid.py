import fcntl,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
root=Path('/content/crrl_training');out=root/'runs/longrun';drive=Path('/content/drive/MyDrive/cr_rl_ppo')
work=Path('/content/crrl_inbox_upgrade');work.mkdir(exist_ok=True)
def worker():
 lock=(work/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for _ in range(90):
  active=[]
  for path in Path('/proc').glob('[0-9]*/cmdline'):
   try:args=path.read_bytes().split(b'\0')
   except OSError:continue
   if any(a.endswith(b'/train_selfplay.py') for a in args):active.append(path.parent.name)
  if not active:break
  time.sleep(5)
 else:raise RuntimeError('Old trainer still active; no forced termination')
 prior_exit=json.loads((out/'exit.json').read_text())['exit_code']
 with zipfile.ZipFile('/content/crrl_live_awbc_inbox_upgrade.zip') as z:
  names=z.namelist()
  for name in names:
   assert not name.startswith('/') and '..' not in Path(name).parts
   if name.endswith('.py'):compile(z.read(name),name,'exec')
  z.extractall(root)
 import torch
 checkpoint=torch.load(out/'ppo_latest.pt',map_location='cpu',mmap=True,weights_only=False)
 update=int(checkpoint['updates']);del checkpoint
 if prior_exit!=0:
  marker=json.loads((work/'recovered_native_hang.json').read_text())
  assert marker['update']==update==json.loads((out/'summary.json').read_text())['update'], 'Cannot discard unsaved completed updates'
 cfg_path=root/'prepared/cr_train_config.json';cfg=json.loads(cfg_path.read_text())
 cfg.update(validation_interval=0,awbc_inbox_sync=True,awbc_inbox_poll_seconds=60,trainer_stall_seconds=600,trainer_max_restarts=5)
 archive=drive/'preparation/crrl_training_update.zip';rebuilt=work/'updated_code.zip'
 with zipfile.ZipFile(archive) as old,zipfile.ZipFile(rebuilt,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as new:
  for item in old.infolist():
   if item.filename not in names:new.writestr(item,old.read(item))
  for name in names:new.write(root/name,name)
 tmp=archive.with_suffix('.partial');shutil.copyfile(rebuilt,tmp);tmp.replace(archive)
 for path in (cfg_path,drive/'preparation/cr_train_config.json'):
  tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(cfg,indent=2));tmp.replace(path)
 subprocess.run([sys.executable,str(root/'scripts/cr_train_pipeline.py'),'start'],cwd=root,check=True)
 result=dict(resumed=True,from_update=update,validation_interval=0,awbc_inbox_sync=True,poll_seconds=60)
 (work/'result.json').write_text(json.dumps(result));print(json.dumps(result),flush=True)
if '--worker' in sys.argv:worker()
else:
 with (work/'upgrade.log').open('a') as log:
  process=subprocess.Popen([sys.executable,'-u','/content/crrl_deploy_inbox_no_valid.py','--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('DEPLOY_PID',process.pid)
