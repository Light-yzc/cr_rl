"""Prepare one mounted Colab CPU runtime; do not start training or replace Torch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path('/content/crrl_training')
DRIVE = Path('/content/drive/MyDrive/cr_rl_ppo')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if not DRIVE.is_dir():
        raise RuntimeError('Drive is not mounted')
    ROOT.mkdir(exist_ok=True)
    with zipfile.ZipFile('/content/crrl_training_code.zip') as archive:
        for row in json.loads(archive.read('CODE_MANIFEST.json')):
            target = (ROOT/row['path']).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError('Unsafe archive path')
            body = archive.read(row['path'])
            if len(body) != row['bytes'] or hashlib.sha256(body).hexdigest() != row['sha256']:
                raise IOError('Code checksum mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
    backup = DRIVE/'preparation'
    backup.mkdir(exist_ok=True)
    shutil.copyfile('/content/crrl_training_code.zip', backup/'crrl_training_code.zip')
    if digest(backup/'crrl_training_code.zip') != digest(Path('/content/crrl_training_code.zip')):
        raise IOError('Drive code backup checksum mismatch')
    packages = {'yaml':'PyYAML', 'numpy':'numpy', 'jpype':'jpype1', 'gymnasium':'gymnasium',
                'pettingzoo':'pettingzoo', 'pyarrow':'pyarrow', 'tqdm':'tqdm',
                'tensorboard':'tensorboard', 'zstandard':'zstandard', 'wandb':'wandb'}
    missing = [package for module,package in packages.items() if importlib.util.find_spec(module) is None]
    if importlib.util.find_spec('torch') is None:
        raise RuntimeError('Runtime lacks Torch; refusing automatic Torch installation')
    if missing:
        subprocess.run([sys.executable,'-m','pip','install','--quiet',*missing], check=True)
    subprocess.run(['java','-version'], check=True)
    tables = Path.home()/'.crforge/tables/16.635.3'
    shutil.copytree(ROOT/'assets/tables/16.635.3', tables, dirs_exist_ok=True)
    os.environ['PYTHONPATH'] = str(ROOT/'src')+':'+str(ROOT/'.cache/vendor/crforge/python')
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.cache/vendor/crforge/python')]
    status = dict(code_ready=True, drive_root=str(DRIVE), root=str(ROOT),
                  checkpoint_available=(DRIVE/'ppo_latest.pt').is_file(),
                  training_started=False, torch_replaced=False)
    (ROOT/'preparation_status.json').write_text(json.dumps(status, indent=2))
    shutil.copyfile(ROOT/'preparation_status.json', backup/'preparation_status.json')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
