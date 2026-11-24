from pathlib import Path
import subprocess,json
root=Path('/content/crrl_training/runs/longrun')
p=root/'progress.jsonl'
with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));print('EVENTS',f.read().decode(errors='replace')[-1600:])
print(subprocess.run(['py-spy','dump','--pid','58045'],capture_output=True,text=True,timeout=15).stdout[-6500:])
