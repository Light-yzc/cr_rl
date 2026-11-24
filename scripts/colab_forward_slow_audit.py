"""Read specific completed rounds and instantaneous GPU diagnostics."""
import json
from pathlib import Path
import subprocess
out=Path('/content/crrl_training/runs/longrun')
with (out/'console.log').open('rb') as stream:
    stream.seek(max(0,(out/'console.log').stat().st_size-12*1024**2))
    for line in stream:
        try:r=json.loads(line)
        except ValueError:continue
        if r.get('phase')=='state_ppo' and r.get('update') in (941,942,943,944,947,948,949,950):
            keys=('update','seconds','rollout_model_seconds','max_frozen_opponent_groups',
                  'max_episode_frames','deck_sample_counts','role_draws','opponent_pool_size',
                  'actor_update_seconds','critic_update_seconds','peak_cuda_allocated_gib')
            print(json.dumps({k:r.get(k) for k in keys}))
pool=out/'opponent_pool.json'
if pool.exists():
    p=json.loads(pool.read_text())
    print('POOL',json.dumps(p)[:8000])
print('GPU',subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu',
    '--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
