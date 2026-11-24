"""Audit actual SSD live-source admission without allocating GPU memory."""
from collections import Counter
import json
from pathlib import Path
import sys

root=Path('/content/crrl_training')
sys.path.insert(0,str(root/'src'))
from cr_rl.training.awbc_playbuffer import PlayBuffer

def audit():
    buffer=PlayBuffer(root/'sessions')
    refresh=buffer.refresh()
    selected=buffer.select(12,24)
    groups=Counter((row['behavior'],row['partition']) for row in buffer.state['entries'].values()
                   if row.get('active'))
    out=dict(audit=refresh,cohorts=[dict(behavior=behavior,partition=partition,count=count)
        for (behavior,partition),count in sorted(groups.items())],
        selected_games=len(selected),selection_ready=bool(selected))
    path=Path('/content/crrl_pipeline_validation/live_source_audit.json')
    path.write_text(json.dumps(out,indent=2))
    print(json.dumps({**out,'audit':{k:v for k,v in refresh.items() if k!='excluded'}}),flush=True)

if __name__=='__main__': audit()
