import json
from pathlib import Path
root=Path('/content/crrl_training');out=root/'runs/longrun'
print('NEW_INBOX_CODE','collection_sealed.json' in (root/'src/cr_rl/training/awbc_inbox.py').read_text())
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:args=path.read_bytes().split(b'\0')
 except OSError:continue
 if any(a.endswith((b'/sync_awbc_inbox.py',b'/train_selfplay.py')) for a in args):print('PROCESS',path.parent.name,[a.decode() for a in args if a.endswith(b'.py')])
import sys
sys.path.insert(0,str(root/'src'))
from cr_rl.training.awbc_playbuffer import read_game
for name in ('20261117_211041_slot2_dd14fa','20261117_211809_slot1_1b04bf'):
 run=root/'sessions'/name
 if not run.exists():print('NOT_IMPORTED_YET',name);continue
 try:
  ep=read_game(run)
  print('NEW_AWBC_GAME',json.dumps(dict(name=name,frames=len(ep['actions']),actor_labels=int(ep['actor_keep'].sum()),reward=float(ep['rewards'].sum()),max_gap=int(ep['max_observation_gap_ticks']),execution_exclusions=len(ep['execution_exclusions']),behavior=ep['behavior_version'][:12])))
 except Exception as error:print('IMPORT_ERROR',name,type(error).__name__,str(error))
s=json.loads((out/'summary.json').read_text());print('TRAIN',json.dumps({k:s.get(k) for k in ('update','samples_per_second','awbc_playbuffer','expert_bc')}))
print('SSD_GAME_COUNT',len(list((root/'sessions').glob('*/result.json'))))
