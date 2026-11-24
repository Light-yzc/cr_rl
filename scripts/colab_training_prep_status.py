"""Report bounded CPU preparation output without loading models."""
import json
from pathlib import Path
import shutil

root = Path('/content/crrl_training')
for name in ('prepared/checkpoint_preparation.log','public_bc/quality_bc_probe.log'):
    path = root/name
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-3500))
            print(name,stream.read().decode(errors='replace'))
for name in ('prepared/single_checkpoint_preparation.json','public_bc/job_exit.json',
             'public_bc/native_probe_64/quality_summary.json','public_bc_full/state.json'):
    path = root/name
    if path.exists():
        value = json.loads(path.read_text())
        value.pop('saved_training_args',None)
        value.pop('smoke_argv',None)
        value.pop('completed',None)
        value.pop('downloaded',None)
        print(name,json.dumps(value))
        backup = Path('/content/drive/MyDrive/cr_rl_ppo/preparation')/path.name
        shutil.copyfile(path,backup)
