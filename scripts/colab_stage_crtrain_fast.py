"""Stage CR-RL on the user's mounted GPU VM without checksum I/O passes."""
def main():
    import importlib.util
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import time
    import zipfile

    root = Path('/content/crrl_training')
    drive = Path('/content/drive/MyDrive/cr_rl_ppo')
    if not drive.is_dir():
        raise RuntimeError('Existing Drive mount is missing; no automatic remount')
    root.mkdir(exist_ok=True)
    def emit(stage, **values):
        print(json.dumps(dict(stage=stage, time=time.time(), **values)), flush=True)
    local_zip = Path('/content/crrl_training_code.zip')
    if not (root/'pyproject.toml').is_file():
        emit('copy_code_zip')
        shutil.copyfile(drive/'preparation/crrl_training_code.zip', local_zip)
        with zipfile.ZipFile(local_zip) as archive:
            for member in archive.infolist():
                target = (root/member.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Archive member escapes workspace')
            archive.extractall(root)
        emit('code_extracted', files=len(archive.infolist()))
    target = root/'prepared/input/ppo_latest.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    source = drive/'ppo_latest.pt'
    config_path = drive/'preparation/cr_train_config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        backup = drive/'training'/config['run_name']
        receipt = backup/'latest.json'
        if receipt.exists():
            generation = json.loads(receipt.read_text())['generation']
            candidate = (backup/generation/'ppo_latest.pt').resolve()
            if not candidate.is_relative_to(backup.resolve()):
                raise ValueError('Unsafe backup generation')
            source = candidate
    source_receipt = target.with_suffix('.source')
    staged_source = (source_receipt.read_text().strip() if source_receipt.is_file()
                     else str(drive/'ppo_latest.pt'))
    if not target.is_file():
        emit('copy_checkpoint', bytes=source.stat().st_size)
        temporary = target.with_suffix('.copying')
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise IOError('Incomplete checkpoint copy')
        temporary.replace(target)
        staged_source = str(source)
        source_receipt.write_text(staged_source)
        emit('checkpoint_local')
    packages = {'yaml':'PyYAML', 'jpype':'jpype1', 'gymnasium':'gymnasium',
                'pettingzoo':'pettingzoo', 'pyarrow':'pyarrow', 'tqdm':'tqdm',
                'tensorboard':'tensorboard', 'zstandard':'zstandard', 'pytest':'pytest'}
    missing = [p for m,p in packages.items() if importlib.util.find_spec(m) is None]
    if missing:
        emit('install_missing_dependencies', packages=missing)
        subprocess.run([sys.executable,'-m','pip','install','--quiet',*missing], check=True)
    if importlib.util.find_spec('torch') is None:
        raise RuntimeError('Runtime Torch missing; automatic replacement is forbidden')
    java = subprocess.run(['java','-version'], capture_output=True, text=True)
    emit('java', output=(java.stdout+java.stderr).strip())
    java.check_returncode()
    tables = Path.home()/'.crforge/tables/16.635.3'
    if not tables.is_dir():
        shutil.copytree(root/'assets/tables/16.635.3', tables)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Existing GPU runtime has no CUDA')
    ckpt = torch.load(target, map_location='cpu', weights_only=False, mmap=True)
    if ckpt.get('model_architecture_version') != 3 or ckpt.get('decision_ticks') != 8:
        raise ValueError('Expected original architecture-v3 / 8 ticks')
    info = {k:ckpt.get(k) for k in ['updates','model_architecture_version','decision_ticks',
                                    'training_args','reward_config','level_profile']}
    info['staged_source'] = staged_source
    (root/'prepared/source_metadata.json').write_text(json.dumps(info, indent=2, default=str))
    import yaml
    (root/'prepared/source_league.yaml').write_text(yaml.safe_dump(ckpt['league_config']))
    fields = ['model','config','temporal_config','learned_placement_definition',
              'model_architecture_version','decision_ticks','updates','backend_commit',
              'data_version','content_sha','level_profile']
    anchor = root/'prepared/anchor_actor.pt'
    if not anchor.is_file():
        saved_anchor = drive/'preparation/anchor_actor.pt'
        if saved_anchor.exists(): shutil.copyfile(saved_anchor, anchor)
        else: torch.save({k:ckpt[k] for k in fields if k in ckpt}, anchor)
    emit('staged', generation=ckpt.get('updates'), torch=torch.__version__,
         gpu=torch.cuda.get_device_name(0), checksum_scans=False, training_started=False)


if __name__ == '__main__':
    main()
