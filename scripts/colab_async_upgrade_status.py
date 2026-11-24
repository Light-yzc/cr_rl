from pathlib import Path
import json,subprocess
w=Path('/content/crrl_async_upgrade')
for name in ('upgrade.log','profiles.json','result.json'):
 p=w/name
 if p.exists():print(name,p.read_text()[-5500:])
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip())
for p in sorted(Path('/content/crrl_training/runs').glob('async_profile_*/progress.jsonl'))[-2:]:
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-18000));rows=f.read().splitlines()
 print(p.parent.name,[{k:v for k,v in json.loads(line).items() if k in ('phase','update','cpu_snapshot_seconds','local_write_seconds','episode_length')} for line in rows[-3:]])
