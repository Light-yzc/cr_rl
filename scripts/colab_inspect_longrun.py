"""Read bounded training metadata, never credentials or tensor contents."""
import json
from pathlib import Path

root = Path('/content/crrl_training')
drive = Path('/content/drive/MyDrive/cr_rl_ppo')
for name in ('prepared/source_metadata.json', 'prepared/single_checkpoint_preparation.json'):
    p = root/name
    if p.exists():
        data = json.loads(p.read_text())
        print(name, json.dumps(data, default=str)[:18000])
for name in ('smoke_snapshot.json','full_snapshot.json'):
    p = drive/'preparation/ssd_snapshots'/name
    if p.exists(): print(name, p.read_text())
for path in (drive/'sessions', root/'sessions', root/'prepared/anchor_actor.pt'):
    print('PATH', str(path), path.exists(), path.stat().st_size if path.is_file() else None)
p = Path('/content/drive/MyDrive/Colab Notebooks/cr_train.ipynb')
if p.exists():
    nb = json.loads(p.read_text())
    for i,c in enumerate(nb['cells']):
        source = ''.join(c.get('source',[]))
        # Report structure only: old cells can contain user-supplied secrets.
        print('CELL', i, c['cell_type'], len(source),
              [s for s in ('train_selfplay','run_live_awbc','launch_prepared','drive.mount','pip install') if s in source])
