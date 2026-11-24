"""Reusable preparation after notebook bootstrap; Drive is the durable source."""
import json
from pathlib import Path
import runpy
import sys

root=Path('/content/crrl_training')
sys.path.insert(0,str(root/'src'))
from cr_rl.training.colab_storage import stage_zip
from cr_rl.training.fast_publication import restore_latest, copy_once

drive=Path('/content/drive/MyDrive/cr_rl_ppo')
runpy.run_path(str(root/'scripts/colab_stage_crtrain_fast.py'),run_name='__main__')
config_path=drive/'preparation/cr_train_config.json'
if config_path.exists():
    config=json.loads(config_path.read_text())
else:
    report=json.loads((drive/'preparation/ssd_snapshots/full_snapshot.json').read_text())
    config=dict(bc_snapshot=report,run_name='longrun',updates=10000,
                envs=4,steps=128,expert_bc_interval=100,awbc_interval=50,
                checkpoint_interval=25,publication_interval=25)
    config_path.write_text(json.dumps(config,indent=2))
report=config['bc_snapshot']
dataset=stage_zip(report['archive'],root/'datasets'/report['snapshot_id'])
live_archive=drive/'preparation/ssd_snapshots/live_sessions.zip'
if live_archive.exists(): stage_zip(live_archive,root/'sessions')
backup=drive/'training'/config['run_name']
receipt=backup/'latest.json'
metadata=json.loads((root/'prepared/source_metadata.json').read_text())
staged=None
if receipt.exists():
    expected=backup/json.loads(receipt.read_text())['generation']/'ppo_latest.pt'
    if metadata.get('staged_source')==str(expected): staged=root/'prepared/input/ppo_latest.pt'
restored=restore_latest(backup,root/'runs'/config['run_name'],checkpoint_source=staged)
(root/'prepared/full_dataset.json').write_text(json.dumps(dict(report,local_path=str(dataset))))
(root/'prepared/cr_train_config.json').write_text(json.dumps(config,indent=2))
anchor=drive/'preparation/anchor_actor.pt'
if not anchor.exists(): copy_once(root/'prepared/anchor_actor.pt',anchor)
print(json.dumps(dict(stage='prepared',dataset=str(dataset),games=report['games'],
    restored=bool(restored),run=config['run_name'],ssd_disposable=True,
    durable_root=str(drive),checksum_scans=False)),flush=True)
