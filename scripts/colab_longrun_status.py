"""Bounded local log tail and small result files."""
from pathlib import Path
import subprocess
root = Path('/content/crrl_training/runs')
for mode in ('tests','ppo-smoke','bc-smoke','resume-smoke','awbc-smoke','longrun'):
    out = root/mode
    if not out.exists(): continue
    print('JOB',mode)
    for name in ('exit.json','expert_bc_latest_report.json','awbc_smoke_report.json'):
        p = out/name
        if p.exists():
            import json
            value=json.loads(p.read_text())
            value.pop('argv',None)
            print(name,json.dumps(value)[:2500])
    p = out/'console.log'
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-3500))
            print(f.read().decode(errors='replace'))
print(subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
    '--format=csv,noheader'],capture_output=True,text=True).stdout)
