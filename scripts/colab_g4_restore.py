"""Restore the durable PPO + Sparky BC run to a fresh GPU runtime."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


ROOT = Path('/content/crrl_training')
DRIVE = Path('/content/drive/MyDrive/cr_rl_ppo')
PACKAGE = DRIVE / 'preparation/sparky_bc_20261120'
BC_SSD = Path('/content/crrl_sparky_bc_all_train_20261120')


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, unix=time.time(), **values)), flush=True)


def copy_once(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    if target.is_file() and target.stat().st_size == size:
        return
    pending = target.with_suffix(target.suffix + '.partial')
    shutil.copyfile(source, pending)
    if pending.stat().st_size != size:
        raise IOError(f'Incomplete copy to {target}')
    pending.replace(target)


def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if not (target / item.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError(f'Invalid archive member {item.filename}')
        bundle.extractall(target)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def assert_idle():
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = proc.read_bytes().split(b'\0')
        except OSError:
            continue
        if any(arg.endswith((b'/train_selfplay.py', b'/cr_train_pipeline.py')) for arg in args[:3]):
            raise RuntimeError(f'Training is already active (PID {proc.parent.name})')


def stage(*, gpu, start, profile=None):
    assert_idle()
    if not DRIVE.is_dir():
        raise RuntimeError('Mount the existing Drive before staging')
    if shutil.disk_usage('/content').free < 20 * 1024**3:
        raise RuntimeError('Insufficient SSD space for checkpoint, opponents and training writes')
    prep = ROOT / 'prepared'
    prep.mkdir(parents=True, exist_ok=True)
    config = read_json(DRIVE / 'preparation/cr_train_config.json')
    if (config.get('run_name') != 'longrun' or not config.get('expert_bc')
            or config.get('awbc') or config.get('awbc_inbox_sync')):
        raise ValueError('Drive config does not match the PPO → Sparky BC run')
    backup = DRIVE / 'training/longrun'
    receipt = read_json(backup / 'latest.json')
    generation = (backup / receipt['generation']).resolve()
    if not generation.is_relative_to(backup.resolve()):
        raise ValueError('Invalid checkpoint generation')
    update = int(receipt['update'])
    run = ROOT / 'runs/longrun'
    checkpoint = run / 'ppo_latest.pt'
    if checkpoint.exists() and read_json(run / 'summary.json').get('update') != update:
        raise RuntimeError('SSD already holds a different run generation')
    run.mkdir(parents=True, exist_ok=True)
    pool = read_json(generation / 'opponent_pool.json')
    copies = [(generation / 'ppo_latest.pt', checkpoint),
              (DRIVE / 'preparation/anchor_actor.pt', prep / 'anchor_actor.pt')]
    for row in pool['entries']:
        relative = Path(row['checkpoint'])
        source = (backup / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(backup.resolve()):
            raise ValueError(f'Invalid opponent path {relative}')
        target = run / relative
        copies.append((source, target))
        row['checkpoint'] = str(target)
    for name in ('summary.json', 'progress.jsonl', 'run_config.json'):
        if (generation / name).exists():
            copies.append((generation / name, run / name))
    if config.get('validation_history'):
        target = prep / 'validation_history.pt'
        copies.append((Path(config['validation_history']), target))
        config['validation_history'] = str(target)
    code, bc = Path('/content/crrl_g4_code.zip'), Path('/content/crrl_g4_sparky_bc.zip')
    copies.extend([(PACKAGE / 'g4_training_code.zip', code),
                   (PACKAGE / 'sparky_bc_all_train_20261120.zip', bc)])
    emit('copying_to_ssd', update=update, files=len(copies))
    packages = {'yaml':'PyYAML', 'jpype':'jpype1', 'gymnasium':'gymnasium',
                'pettingzoo':'pettingzoo', 'tqdm':'tqdm', 'tensorboard':'tensorboard',
                'zstandard':'zstandard'}
    missing = [name for module, name in packages.items() if importlib.util.find_spec(module) is None]
    deps = subprocess.Popen([sys.executable, '-m', 'pip', 'install', '--quiet', *missing]) if missing else None
    with ThreadPoolExecutor(max_workers=5) as transfer:
        for _ in transfer.map(lambda pair: copy_once(*pair), copies):
            pass
    extract(code, ROOT)
    if not (BC_SSD / 'dataset.manifest.json').exists():
        temporary = BC_SSD.with_name(BC_SSD.name + '.extracting')
        extract(bc, temporary)
        manifest = read_json(temporary / 'dataset.manifest.json')
        if sum(len(row['selected_seats']) for row in manifest['shards']) != 684:
            raise ValueError('Specialist archive has the wrong seat count')
        temporary.rename(BC_SSD)
    if deps is not None and deps.wait() != 0:
        raise RuntimeError('Pip dependency setup failed')
    tables = Path.home() / '.crforge/tables/16.635.3'
    if not tables.is_dir():
        shutil.copytree(ROOT / 'assets/tables/16.635.3', tables)
    subprocess.run(['java', '-version'], check=True)
    import torch
    import yaml

    snapshot = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert snapshot['updates'] == update and snapshot['model_architecture_version'] == 3
    assert snapshot['decision_ticks'] == 8
    bc_state = snapshot['expert_bc']
    assert bc_state['snapshot_id'] == read_json(BC_SSD / 'dataset.manifest.json')['snapshot_id']
    assert len(bc_state['train']) == 684
    summary = read_json(run / 'summary.json')
    assert summary['update'] == update
    metadata = {key: snapshot.get(key) for key in ('updates','model_architecture_version','decision_ticks',
                                                  'training_args','reward_config','level_profile')}
    metadata['staged_source'] = str(generation / 'ppo_latest.pt')
    write_json(prep / 'source_metadata.json', metadata)
    (prep / 'source_league.yaml').write_text(yaml.safe_dump(snapshot['league_config']))
    write_json(run / 'opponent_pool.json', pool)
    del snapshot
    config.update(expert_bc=True, expert_bc_dataset=str(BC_SSD), expert_bc_interval=1,
                  expert_bc_sequences=48, expert_bc_placement=False,
                  trainer_entry=str(PACKAGE / 'train_selfplay.py'))
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable on the requested runtime')
        device = torch.cuda.get_device_properties(0)
        l4 = device.name == 'NVIDIA L4'
        if not l4 and '6000' not in device.name and 'A100' not in device.name:
            raise RuntimeError(f'Expected L4, RTX PRO 6000, or A100, got {device.name}')
        probe = torch.ones((16, 16), device='cuda', dtype=torch.float16)
        assert float((probe @ probe)[0, 0]) == 16
        wide = device.total_memory >= 70 * 1024**3
        wide_a100 = wide and 'A100' in device.name
        if not l4:
            config.update(frame_microbatch=256 if wide else 128, actor_batch_size=8 if wide else 4,
                          critic_batch_size=8 if wide else 4, opponent_cache_size=8,
                          training_opponent_cache_size=8, env_threads=min(16, len(os.sched_getaffinity(0))))
        if wide_a100:
            config.update(frame_microbatch=512, actor_batch_size=8, critic_batch_size=16,
                          env_threads=min(10, len(os.sched_getaffinity(0))))
        overlay = PACKAGE / 'a100_runtime_overlay.zip'
        if overlay.is_file():
            overlay_ssd = Path('/content/crrl_a100_runtime_overlay.zip')
            copy_once(overlay, overlay_ssd)
            extract(overlay_ssd, ROOT)
            if wide_a100 and profile is None:
                profile = ROOT / 'prepared/a100_runtime_profile.json'
            if not wide_a100:
                config['gradient_checkpointing'] = True
            if l4:
                config.update(validation_argmax_lanes=8, native_step_diagnostic_seconds=8)
        elif l4 and not metadata['training_args'].get('gradient_checkpointing'):
            raise RuntimeError('The L4 restore needs the runtime overlay to restore gradient checkpointing')
        if profile is not None:
            change = read_json(profile)
            tunables = {'frame_microbatch', 'actor_batch_size', 'critic_batch_size',
                        'opponent_cache_size', 'training_opponent_cache_size', 'env_threads',
                        'validation_argmax_lanes', 'native_step_diagnostic_seconds'}
            switches = {'gradient_checkpointing', 'sparse_placement'}
            if not change or any(
                type(v) is not bool if k in switches else
                k not in tunables or type(v) is not int or v < 1
                for k, v in change.items()
            ):
                raise ValueError('Invalid GPU profile setting')
            config.update(change)
        config['env_threads'] = min(config.get('env_threads', 8), len(os.sched_getaffinity(0)))
        emit('gpu_checked', name=device.name, memory_gib=round(device.total_memory / 1024**3, 1),
             torch_version=torch.__version__)
    write_json(prep / 'cr_train_config.json', config)
    env = dict(os.environ, PYTHONPATH=f'{ROOT / "src"}:{ROOT / ".cache/vendor/crforge/python"}')
    if config.get('gradient_checkpointing') is False:
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if config.get('native_step_diagnostic_seconds'):
        env.setdefault('CRRL_NATIVE_STEP_DIAGNOSTIC_SECONDS',
                       str(config['native_step_diagnostic_seconds']))
    if config.get('sparse_placement'):
        env.setdefault('CRRL_SPARSE_PLACEMENT', '1')
    sys.path.insert(0, str(ROOT / 'src'))
    from cr_rl.training.colab_pipeline import ppo_command

    command = ppo_command(ROOT, config)
    assert command[1] == str(PACKAGE / 'train_selfplay.py')
    assert command[command.index('--expert-bc-sequences') + 1] == '48'
    assert command[command.index('--expert-bc-interval') + 1] == '1'
    assert '--awbc-sessions' not in command
    if config.get('gradient_checkpointing') is False:
        assert '--gradient-checkpointing' not in command
    if gpu and l4:
        assert '--gradient-checkpointing' in command
    if config.get('validation_argmax_lanes'):
        assert command[command.index('--validation-argmax-lanes') + 1] == str(config['validation_argmax_lanes'])
    emit('ready', update=update, bc_phase_count=bc_state['phase_count'], bc_cursor=bc_state['cursor'],
         actor_effective_batch_size=config.get('actor_effective_batch_size'),
         critic_effective_batch_size=config.get('critic_effective_batch_size'),
         frame_microbatch=config['frame_microbatch'], cpu_threads=config['env_threads'])
    if start:
        assert_idle()
        subprocess.run([sys.executable, str(ROOT / 'scripts/cr_train_pipeline.py'), 'start'],
                       cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', help='Verify restore on the CPU preparation VM')
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--profile', type=Path, help='JSON performance-only overrides')
    args = parser.parse_args()
    if args.cpu and args.start:
        parser.error('CPU preparation cannot start training')
    stage(gpu=not args.cpu, start=args.start, profile=args.profile)
