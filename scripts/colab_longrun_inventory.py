"""Small inventory of the existing GPU VM; no hashing or recursive Drive scan."""
import json
from pathlib import Path
import shutil
import subprocess

content=Path('/content');drive=content/'drive/MyDrive'
print('LOCAL',json.dumps(dict(free_gib=round(shutil.disk_usage(content).free/1024**3,1),
    directories=[p.name for p in content.iterdir() if p.is_dir()])))
print('GPU',subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used',
    '--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
print('DRIVE',drive.is_dir())
if drive.is_dir():
    for folder in (drive,drive/'Colab Notebooks',drive/'cr_rl_ppo',drive/'cr_rl_ppo/preparation'):
        if folder.is_dir():
            print('FILES',str(folder),json.dumps([p.name for p in folder.iterdir()][:65]))
for base in (Path('/content/crrl_training'),Path('/content/cr_train')):
    if base.is_dir():
        print('WORK',str(base),json.dumps([p.name for p in base.iterdir()][:30]))
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args=path.read_bytes().split(b'\0')
        script=next((Path(a.decode()).name for a in args if a.endswith(b'.py')
            and any(s in a for s in (b'train_selfplay',b'train_live_awbc',b'run_state_bc'))),None)
        if script:print('TRAIN_PROCESS',path.parent.name,script)
    except (OSError,UnicodeError):pass
