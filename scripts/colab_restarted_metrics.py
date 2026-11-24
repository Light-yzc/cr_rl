import json,subprocess
from pathlib import Path
r=Path('/content/crrl_training');o=r/'runs/longrun'
d=json.loads((o/'summary.json').read_text())
print(json.dumps({k:d.get(k) for k in ('update','seconds','samples_per_second','rollout_model_seconds','actor_overflow_steps','critic_overflow_steps','faulted_samples','checkpoint','awbc_playbuffer','expert_bc')}))
p=o/'progress.jsonl'
with p.open('rb') as f:f.seek(max(0,p.stat().st_size-18000));rows=f.read().splitlines()
print('EVENTS',[{k:v for k,v in json.loads(line).items() if k in ('phase','update','episode_length')} for line in rows[-3:]])
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
