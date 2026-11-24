from pathlib import Path
import json
r=Path('/content/crrl_training');w=Path('/content/crrl_awbc_transport_smoke');w.mkdir(exist_ok=True)
from sys import path
path.insert(0,str(r/'src'))
from cr_rl.training.awbc_inbox import ingest_ready_batches
report=ingest_ready_batches('/content/drive/MyDrive/cr_rl_ppo/preparation/awbc_inbox',w/'sessions',w/'state.json')
print('TRANSPORT',json.dumps(report))
from cr_rl.training.awbc_playbuffer import read_game
for p in (w/'sessions').glob('*/result.json'):
 ep=read_game(p.parent);print('REAL_IMPORT',json.dumps(dict(session=p.parent.name,frames=len(ep['actions']),actor_labels=int(ep['actor_keep'].sum()),behavior=ep['behavior_version'][:12],reward=float(ep['rewards'].sum()))))
print('MAIN_SSD_GAMES',len(list((r/'sessions').glob('*/result.json'))))
state=r/'prepared/awbc_inbox_state.json'
print('MAIN_INBOX_STATE',state.read_text() if state.exists() else None)
