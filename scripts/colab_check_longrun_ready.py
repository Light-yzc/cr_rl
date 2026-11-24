"""Verify durable inputs and tracking configuration without reading large payloads."""
import importlib.util
import json
import netrc
import os
from pathlib import Path
import shutil
import zipfile

root=Path('/content/crrl_training')
drive=Path('/content/drive/MyDrive/cr_rl_ppo')
cfg=json.loads((drive/'preparation/cr_train_config.json').read_text())
with zipfile.ZipFile(drive/'preparation/crrl_training_update.zip') as archive:
    code=set(archive.namelist())
notebook=drive.parent/'Colab Notebooks/cr_train.ipynb'
nb=json.loads(notebook.read_text())
compile(''.join(nb['cells'][1]['source']),str(notebook),'exec')
try:
    netrc_ready=bool(netrc.netrc().authenticators('api.wandb.ai'))
except (OSError,netrc.NetrcParseError):
    netrc_ready=False
report=dict(
    drive_code_ready={'scripts/cr_train_pipeline.py','scripts/prepare_cr_train.py','notebooks/cr_train.ipynb'}<=code,
    drive_bc_bytes=Path(cfg['bc_snapshot']['archive']).stat().st_size,
    drive_awbc_bytes=(drive/'preparation/ssd_snapshots/live_sessions.zip').stat().st_size,
    drive_anchor_bytes=(drive/'preparation/anchor_actor.pt').stat().st_size,
    notebook_cells=len(nb['cells']),
    ssd_bc_games=cfg['bc_snapshot']['games'],
    ssd_disk_free_gib=round(shutil.disk_usage(root).free/1024**3,1),
    wandb_installed=importlib.util.find_spec('wandb') is not None,
    wandb_env_available=bool(os.environ.get('WANDB_API_KEY')),
    wandb_netrc_available=netrc_ready,
)
with zipfile.ZipFile(drive/'preparation/ssd_snapshots/live_sessions.zip') as archive:
    durable={Path(n).parent.name for n in archive.namelist() if n.endswith('/result.json')}
ssd={p.parent.name for p in (root/'sessions').glob('*/result.json')}
report.update(awbc_drive_games=len(durable),awbc_ssd_games=len(ssd),
              awbc_drive_ssd_match=durable==ssd)
print(json.dumps(report))
