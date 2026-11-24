"""Read-only status of CPU replay-hook validation."""
import json
from pathlib import Path
import shutil

root=Path('/content/crrl_training/awbc_validation')
log=root/'tests.log'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-5000))
        print(stream.read().decode(errors='replace'))
result=root/'result.json'
if result.exists():
    print(result.read_text())
    backup=Path('/content/drive/MyDrive/cr_rl_ppo/preparation/awbc_playbuffer_validation')
    backup.mkdir(exist_ok=True)
    shutil.copyfile(result,backup/'result.json');shutil.copyfile(log,backup/'tests.log')
