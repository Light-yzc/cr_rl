import sys,json
from pathlib import Path
root=Path('/content/crrl_training');sys.path.insert(0,str(root/'src'))
from cr_rl.training.awbc_playbuffer import read_game
name='20261117_214458_slot2_f807ea';run=root/'sessions'/name
if run.exists():
 ep=read_game(run);print('SEALED_REAL_GAME',json.dumps(dict(name=name,sealed=(run/'collection_sealed.json').exists(),frames=len(ep['actions']),actor_labels=int(ep['actor_keep'].sum()),choice_rows=int((ep['legal_masks'].sum(-1)>1).sum()),reward=float(ep['rewards'].sum()),max_gap=int(ep['max_observation_gap_ticks']),execution_exclusions=len(ep['execution_exclusions']),update=ep['manifest'].get('checkpoint_metadata',{}).get('updates'))))
else:print('NOT_IMPORTED_YET',name)
print('SSD_GAMES',len(list((root/'sessions').glob('*/result.json'))))
print('INBOX_STATUS',(root/'prepared/awbc_inbox_state.status.json').read_text())
s=json.loads((root/'runs/longrun/summary.json').read_text());print('TRAIN',json.dumps({k:s.get(k) for k in ('update','samples_per_second','awbc_playbuffer','expert_bc')}))
print('DRIVE',(Path('/content/drive/MyDrive/cr_rl_ppo/training/longrun/latest.json')).read_text())
