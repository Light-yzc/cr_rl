"""Stage tested async saves, measure larger batches, persist settings and resume once."""
import fcntl,json,math,os,shutil,subprocess,sys,time,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root=Path('/content/crrl_training');out=root/'runs/longrun';drive=Path('/content/drive/MyDrive/cr_rl_ppo')
work=Path('/content/crrl_async_upgrade');work.mkdir(exist_ok=True)
def worker():
 lock=(work/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for _ in range(90):
  live=[]
  for p in Path('/proc').glob('[0-9]*/cmdline'):
   try:c=p.read_bytes().split(b'\0')
   except OSError:continue
   if any(a.endswith(b'/train_selfplay.py') for a in c):live.append(p.parent.name)
  if not live:break
  time.sleep(5)
 else:raise RuntimeError('Trainer did not stop; no forced termination')
 assert json.loads((out/'exit.json').read_text())['exit_code']==0
 with zipfile.ZipFile('/content/crrl_async_save_upgrade.zip') as z:
  names=z.namelist()
  for name in names:
   assert not name.startswith('/') and '..' not in Path(name).parts
   if name.endswith('.py'):compile(z.read(name),name,'exec')
  z.extractall(root)
 sys.path.insert(0,str(root/'src'))
 import torch
 from cr_rl.training.colab_pipeline import ppo_command
 from cr_rl.training.async_publication import freeze_publication_files
 from cr_rl.training.fast_publication import publish_snapshot
 checkpoint=torch.load(out/'ppo_latest.pt',map_location='cpu',mmap=True,weights_only=False)
 update=int(checkpoint['updates']);del checkpoint
 print(json.dumps(dict(stage='saved_start',update=update)),flush=True)
 publication=ThreadPoolExecutor(max_workers=1)
 frozen=freeze_publication_files(out,update)
 def upload():
  result=publish_snapshot(frozen,drive/'training/longrun',update,keep_generations=1)
  shutil.rmtree(frozen);return result
 upload_future=publication.submit(upload)
 config=json.loads((root/'prepared/cr_train_config.json').read_text())
 env=dict(os.environ,PYTHONPATH=str(root/'src')+':'+str(root/'.cache/vendor/crforge/python'),WANDB_MODE='disabled')
 pool=json.loads((out/'opponent_pool.json').read_text())
 for entry in pool['entries']:
  p=Path(entry['checkpoint']);entry['checkpoint']=str((out/p).resolve() if not p.is_absolute() else p)
 results=[]
 for lanes,frames,batch in ((8,128,4),(16,128,4)):
  dst=root/'runs'/f'async_profile_{int(time.time())}_{lanes}env';dst.mkdir()
  (dst/'opponent_pool.json').write_text(json.dumps(pool))
  cfg=dict(config,envs=lanes,frame_microbatch=frames,actor_batch_size=batch,critic_batch_size=batch,
   updates=2,checkpoint_interval=1,async_checkpoint=True,backup=False,wandb=False,awbc=False,expert_bc=False,
   validation_interval=200)
  command=ppo_command(root,cfg,out=dst,resume=out/'ppo_latest.pt')
  print(json.dumps(dict(stage='profile_started',envs=lanes,out=str(dst))),flush=True)
  with (dst/'console.log').open('w') as log:
   proc=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:status=proc.wait(timeout=1200)
   except subprocess.TimeoutExpired:proc.kill();proc.wait();status=-999
  row=dict(envs=lanes,frame_microbatch=frames,batch=batch,exit_code=status,out=str(dst))
  if status==0:
   data=json.loads((dst/'summary.json').read_text())
   for k in ('update','samples_per_second','peak_cuda_allocated_gib','initial_logp_max_error','optimizer_steps','actor_overflow_steps','critic_overflow_steps','faulted_samples','rollout_model_seconds','actor_update_seconds','critic_update_seconds'):
    row[k]=data.get(k)
   row['eligible']=bool(all(isinstance(row[k],(int,float)) and math.isfinite(row[k]) for k in ('samples_per_second','peak_cuda_allocated_gib','initial_logp_max_error')) and row['optimizer_steps']>0 and row['faulted_samples']==0 and row['initial_logp_max_error']<.002 and row['peak_cuda_allocated_gib']<20.5)
  else:row['eligible']=False;row['error_tail']=(dst/'console.log').read_text(errors='replace')[-2200:]
  results.append(row);(work/'profiles.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
  if not row['eligible'] and status==0:print('Profile numerical or memory gate did not pass',flush=True)
  if (dst/'checkpoints').exists():
   for p in (dst/'checkpoints').glob('*.pt'):p.unlink()
  for n in ('ppo_latest.pt','actor_export.pt'):(dst/n).unlink(missing_ok=True)
 good=[r for r in results if r['eligible']]
 if not good:raise RuntimeError('No enlarged profile passed; longrun remains safely stopped')
 best=max(good,key=lambda r:r['samples_per_second'])
 config.update(envs=best['envs'],frame_microbatch=best['frame_microbatch'],actor_batch_size=best['batch'],
  critic_batch_size=best['batch'],checkpoint_interval=5,async_checkpoint=True,opponent_cache_size=4,training_opponent_cache_size=4)
 for p in (root/'prepared/cr_train_config.json',drive/'preparation/cr_train_config.json'):
  temp=p.with_suffix('.partial');temp.write_text(json.dumps(config,indent=2));temp.replace(p)
 archive=drive/'preparation/crrl_training_update.zip';new=work/'updated_code.zip'
 with zipfile.ZipFile(archive) as old,zipfile.ZipFile(new,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for item in old.infolist():
   if item.filename not in names:z.writestr(item,old.read(item))
  for name in names:z.write(root/name,name)
 temp=archive.with_suffix('.partial');shutil.copyfile(new,temp);temp.replace(archive)
 upload_future.result();publication.shutdown()
 subprocess.run([sys.executable,str(root/'scripts/cr_train_pipeline.py'),'start'],cwd=root,check=True)
 (work/'result.json').write_text(json.dumps(dict(resumed=True,from_update=update,selected=best,checkpoint_interval=5,async_checkpoint=True,async_publication=True)))
 print('LONGRUN_RESUMED',flush=True)
if '--worker' in sys.argv:worker()
else:
 with (work/'upgrade.log').open('a') as log:
  job=subprocess.Popen([sys.executable,'-u','/content/crrl_apply_async_upgrade.py','--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('ASYNC_UPGRADE_PID',job.pid)
