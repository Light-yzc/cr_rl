from pathlib import Path
import json,subprocess
root=Path('/content/crrl_training');w=Path('/content/crrl_cache_upgrade');out=root/'runs/longrun'
print('RECOVERY', (w/'recovery.log').read_text()[-2000:])
print('CONFIG', {k:v for k,v in json.loads((root/'prepared/cr_train_config.json').read_text()).items() if k in ('updates','checkpoint_interval','wandb','envs','frame_microbatch','actor_batch_size','critic_batch_size','opponent_cache_size','training_opponent_cache_size')})
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip())
for name in ('summary.json','console.log'):
 p=out/name
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));print(name,f.read().decode(errors='replace'))
