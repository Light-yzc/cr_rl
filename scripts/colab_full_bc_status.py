"""Small status response for the full-source CPU job."""
import json
from pathlib import Path

root = Path('/content/crrl_training/public_bc_full')
state = json.loads((root/'state.json').read_text())
print(json.dumps({k:v for k,v in state.items() if k not in ('downloaded','completed')}))
for name,row in list(state.get('completed',{}).items())[-2:]:
    print(name,json.dumps(row['summary']))
index = state.get('source_index')
if index is not None:
    log = root/f'shard_{index:03d}.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-1200))
            print(stream.read().decode(errors='replace'))
