"""Measure end-to-end L4 full-episode throughput with the production command."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from cr_rl.training.colab_pipeline import ppo_command

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vram-sweep',action='store_true',
                    help='Compare bigger GPU batches without increasing simulator lanes')
args=parser.parse_args()
config=json.loads((root/'prepared/cr_train_config.json').read_text())
work=root/'runs'/('throughput-vram' if args.vram_sweep else 'throughput')
if args.vram_sweep:
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try: command=process.read_bytes().split(b'\0')
        except OSError: continue
        if any(arg.endswith((b'/train_selfplay.py',b'/crrl_queue_longrun.py',
                            b'/crrl_finish_pipeline.py')) for arg in command):
            parser.error('Existing trainer or launch queue owns this L4; do not run concurrent profiles')
work.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(root/'src')+':'+str(root/'.cache/vendor/crforge/python'))
results=[]
profiles=((8,64,2),(8,128,2),(8,128,4)) if args.vram_sweep else ((4,32,1),(8,32,1),(8,64,2))
for lanes,frames,batch in profiles:
    output=work/f'lanes{lanes}_frames{frames}_batch{batch}'
    output.mkdir(exist_ok=True)
    settings=dict(config,envs=lanes,frame_microbatch=frames,actor_batch_size=batch,
        critic_batch_size=batch,updates=2,wandb=False,expert_bc=False,awbc=False,backup=False)
    command=ppo_command(root,settings,out=output,resume=root/'prepared/input/ppo_latest.pt')
    with (output/'console.log').open('w') as log:
        status=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    row=dict(envs=lanes,frame_microbatch=frames,batch=batch,exit_code=status)
    if status==0:
        data=json.loads((output/'summary.json').read_text())
        row.update({key:data.get(key) for key in ('valid_learner_samples','samples_per_second',
            'rollout_samples_per_second','rollout_seconds','actor_update_seconds','critic_update_seconds',
            'peak_cuda_allocated_gib','initial_logp_max_error','optimizer_steps','faulted_samples',
            'actor_overflow_steps','critic_overflow_steps')})
        measurements=('samples_per_second','initial_logp_max_error','peak_cuda_allocated_gib')
        row['eligible']=bool(all(isinstance(row[k],(int,float)) and math.isfinite(row[k])
                                 for k in measurements)
            and row['optimizer_steps'] and row['samples_per_second']>0
            and row['actor_overflow_steps']==0 and row['critic_overflow_steps']==0
            and row['faulted_samples']==0
            and row['initial_logp_max_error']<=.002 and 0<row['peak_cuda_allocated_gib']<19)
    results.append(row)
    (work/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(row),flush=True)
