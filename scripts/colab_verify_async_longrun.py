from pathlib import Path
import json,subprocess,zipfile
w=Path('/content/crrl_async_upgrade');r=Path('/content/crrl_training');o=r/'runs/longrun';d=Path('/content/drive/MyDrive/cr_rl_ppo')
p=w/'result.json'
if not p.exists():
 print('UPGRADE_PENDING', (w/'upgrade.log').read_text()[-1800:])
else:
 result=json.loads(p.read_text());print('RESULT',json.dumps(result))
 profiles=json.loads((w/'profiles.json').read_text())
 report=dict(result=result,profiles=profiles)
 dest=d/'preparation/async_checkpoint_performance_20261117.json';tmp=dest.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2));tmp.replace(dest)
 args=json.loads((o/'launch.json').read_text());keys=('--envs','--frame-microbatch','--actor-batch-size','--critic-batch-size','--checkpoint-interval','--opponent-cache-size','--training-opponent-cache-size')
 print('ACTUAL_ARGS',json.dumps({k:args[args.index(k)+1] for k in keys}));print('ASYNC_FLAGS','--async-checkpoint' in args,'--async-publication' in args,'--no-backup-hash' in args)
 assert args[args.index('--checkpoint-interval')+1]=='5' and '--async-checkpoint' in args and '--async-publication' in args
 cfg=json.loads((r/'prepared/cr_train_config.json').read_text());persisted=json.loads((d/'preparation/cr_train_config.json').read_text())
 assert cfg==persisted
 with zipfile.ZipFile(d/'preparation/crrl_training_update.zip') as z:assert z.read('src/cr_rl/training/async_checkpoint.py')==(r/'src/cr_rl/training/async_checkpoint.py').read_bytes()
 print('PERSISTENCE_OK')
 print('DRIVE', (d/'training/longrun/latest.json').read_text().strip())
 summary=json.loads((o/'summary.json').read_text());print('FORMAL_SUMMARY',{k:summary.get(k) for k in ('update','samples_per_second','actor_overflow_steps','critic_overflow_steps','faulted_samples')})
live=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:args=path.read_bytes().split(b'\0')
 except OSError:continue
 if any(a.endswith(b'/train_selfplay.py') for a in args):live.append(dict(pid=path.parent.name,out=args[args.index(b'--out')+1].decode()))
print('LIVE_TRAINERS',json.dumps(live))
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
