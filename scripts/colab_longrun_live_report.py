"""Report the live trainer and committed Drive checkpoint without reading model weights."""
import json
from pathlib import Path
import subprocess

root=Path('/content/crrl_training')
cfg=json.loads((root/'prepared/cr_train_config.json').read_text())
name=cfg['run_name']
out=root/'runs'/name
backup=Path('/content/drive/MyDrive/cr_rl_ppo/training')/name
pid_file=out/'pipeline.pid'
summary=out/'summary.json'
receipt=backup/'latest.json'
report={'run':name,'wandb':cfg.get('wandb',True)}
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True)
if gpu.returncode==0:
    used,total,util=map(int,gpu.stdout.splitlines()[0].split(','))
    report['gpu_now']=dict(used_mib=used,total_mib=total,utilization_percent=util)
if pid_file.is_file():
    pid=pid_file.read_text().strip()
    try: active=b'/cr_train_pipeline.py\0' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError: active=False
    report.update(controller_pid=int(pid),controller_active=active)
trainers=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=path.read_bytes().split(b'\0')
    except OSError: continue
    if any(a.endswith(b'/train_selfplay.py') for a in args) and bytes(out) in args:
        trainers.append(int(path.parent.name))
report['trainer_pids']=trainers
launch=out/'launch.json'
if launch.is_file():
    argv=json.loads(launch.read_text())
    def option(name):
        return argv[argv.index(name)+1] if name in argv else None
    bc=option('--expert-bc-dataset')
    live=option('--awbc-sessions')
    report['aux_inputs']=dict(
        bc_games=cfg['bc_snapshot']['games'] if bc and (Path(bc)/'dataset.manifest.json').is_file() else None,
        awbc_result_files=len(list(Path(live).glob('*/result.json'))) if live else None,
        bc_interval=option('--expert-bc-interval'),awbc_interval=option('--awbc-interval'))
if summary.is_file():
    data=json.loads(summary.read_text())
    keys=('update','valid_learner_samples','samples_per_second','peak_cuda_allocated_gib',
          'seconds','rollout_seconds','rollout_model_seconds','rollout_env_seconds',
          'critic_eval_seconds','actor_update_seconds','critic_update_seconds',
          'optimizer_steps','initial_logp_max_error','actor_overflow_steps','critic_overflow_steps',
          'faulted_samples','awbc_playbuffer','expert_bc')
    report['summary']={k:data.get(k) for k in keys}
if receipt.is_file():
    data=json.loads(receipt.read_text())
    ckpt=backup/data['generation']/'ppo_latest.pt'
    report['drive_checkpoint']=dict(update=data['update'],bytes=ckpt.stat().st_size,
        opponent_pool_present=(ckpt.parent/'opponent_pool.json').is_file())
else:
    pending=[]
    for p in sorted((backup/'generations').glob('update_*/ppo_latest.pt*')):
        try: pending.append(dict(generation=p.parent.name,file=p.name,bytes=p.stat().st_size))
        except FileNotFoundError: continue
    if pending:
        report['drive_publication_in_progress']=pending[-2:]
if (out/'exit.json').is_file():
    report['exit']=json.loads((out/'exit.json').read_text())
progress=out/'progress.jsonl'
if progress.is_file():
    with progress.open('rb') as stream:
        stream.seek(max(0,progress.stat().st_size-32768))
        tail=stream.read().splitlines()
    events=[]
    for line in tail[-5:]:
        try: event=json.loads(line)
        except ValueError: continue
        events.append({k:event[k] for k in ('phase','update','reward','episode_length','fault') if k in event})
    report['recent_events']=events
print(json.dumps(report,ensure_ascii=False))
