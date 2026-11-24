import json,sys
from pathlib import Path
root=Path('/content/crrl_training');sys.path.insert(0,str(root/'src'))
import torch
from cr_rl.training.awbc_playbuffer import PlayBuffer
checkpoint=torch.load(root/'runs/longrun/ppo_latest.pt',map_location='cpu',weights_only=False,mmap=True)
state=checkpoint.get('awbc_playbuffer');saved_update=checkpoint['updates'];del checkpoint
buffer=PlayBuffer(root/'sessions',state=state)
report=buffer.refresh();cohorts={}
for row in buffer.state['entries'].values():
 if row.get('active'):
  key=row['behavior'][:12];counts=cohorts.setdefault(key,dict(train=0,validation=0));counts[row['partition']]+=1
result=dict(saved_update=saved_update,refresh=report,cohorts=cohorts,consumption_changed=False)
print('READ_ONLY_BUFFER_AUDIT',json.dumps(result))
p=Path('/content/drive/MyDrive/cr_rl_ppo/preparation/flywheel_buffer_acceptance_20261117.json');p.write_text(json.dumps(result,indent=2))
