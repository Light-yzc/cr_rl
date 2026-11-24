"""Persist the user's tracking preference without touching any process."""
import json
from pathlib import Path
import shutil

drive = Path('/content/drive/MyDrive/cr_rl_ppo/preparation/cr_train_config.json')
local = Path('/content/crrl_training/prepared/cr_train_config.json')
data = json.loads(drive.read_text())
data['wandb'] = False
for path in (drive, local):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)
incoming = Path('/content/cr_train_updated.ipynb')
if incoming.exists():
    nb = json.loads(incoming.read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),str(incoming),'exec')
    for path in (local.parents[1]/'notebooks/cr_train.ipynb',
                 drive.parents[2]/'Colab Notebooks/cr_train.ipynb'):
        temporary = path.with_suffix('.partial')
        shutil.copyfile(incoming,temporary)
        temporary.replace(path)
print(json.dumps(dict(wandb=False,run=data['run_name'],configs='Drive and SSD')))
