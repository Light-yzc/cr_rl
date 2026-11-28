"""Reuse immutable PPO snapshots; publish once without checksum readback scans."""
import json
import os
from pathlib import Path
import shutil

from cr_rl.training.persistence import atomic_json


def copy_once(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name+'.partial')
    shutil.copyfile(source, temp)
    try:
        with temp.open('rb+') as fh:
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass
    if temp.stat().st_size != source.stat().st_size:
        raise IOError('Incomplete copy: '+target.name)
    temp.replace(target)


def publish_snapshot(snapshot, destination, update, *, keep_generations=2):
    """Commit pointer last; keep two complete generations and their pool files."""
    snapshot, destination = Path(snapshot), Path(destination)
    generation = destination/'generations'/f'update_{update:06d}'
    generation.mkdir(parents=True, exist_ok=True)
    pool = json.loads((snapshot/'opponent_pool.json').read_text())
    for row in pool['entries']:
        source = Path(row['checkpoint'])
        relative = Path('opponents')/source.name
        target = destination/relative
        if not target.exists(): copy_once(source,target)
        row['checkpoint'] = str(relative)
    copy_once(snapshot/'ppo_latest.pt',generation/'ppo_latest.pt')
    actor = snapshot/'actor_export.pt'
    actor_receipt = {}
    if actor.is_file():
        copy_once(actor, generation/'actor_export.pt')
        actor_receipt = dict(actor=str((generation/'actor_export.pt').relative_to(destination)),
                             actor_bytes=actor.stat().st_size)
    atomic_json(generation/'opponent_pool.json',pool)
    for name in ('summary.json','progress.jsonl','wandb_run.json','run_config.json'):
        if (snapshot/name).exists(): copy_once(snapshot/name,generation/name)
    for source in (snapshot/'macro_search_buffer').glob('combo_*.npz'):
        copy_once(source, generation/'macro_search_buffer'/source.name)
    for source in (snapshot/'lookahead_distill_buffer').glob('lookahead_*.npz'):
        copy_once(source, generation/'lookahead_distill_buffer'/source.name)
    atomic_json(destination/'latest.json',dict(update=update,
        generation=str(generation.relative_to(destination)),checksum_scans=False,**actor_receipt))
    generations = sorted(p for p in (destination/'generations').glob('update_*')
                         if (p/'opponent_pool.json').is_file() and (p/'ppo_latest.pt').is_file()
                         and (p/'ppo_latest.pt').stat().st_size > 0)
    if keep_generations<1: raise ValueError('At least one restore generation is required')
    retained = generations[-keep_generations:]
    if generation not in retained:
        return dict(update=update,checksum_scans=False,generation=str(generation))
    active = set()
    for path in retained:
        for row in json.loads((path/'opponent_pool.json').read_text())['entries']:
            active.add(Path(row['checkpoint']).name)
    for path in generations[:-keep_generations]:
        if path.is_dir() and path.resolve().is_relative_to((destination/'generations').resolve()):
            shutil.rmtree(path)
    for path in (destination/'opponents').glob('*.pt'):
        if path.name not in active: path.unlink()
    return dict(update=update,checksum_scans=False,generation=str(generation))


def restore_latest(destination, local, *, checkpoint_source=None):
    destination, local = Path(destination), Path(local)
    receipt = destination/'latest.json'
    if not receipt.exists(): return None
    report = json.loads(receipt.read_text())
    generation = (destination/report['generation']).resolve()
    if not generation.is_relative_to(destination.resolve()): raise ValueError('Unsafe generation path')
    local.mkdir(parents=True,exist_ok=True)
    checkpoint = local/'ppo_latest.pt'
    # Caller uses this only for an empty/new runtime; do not overwrite a local run.
    if checkpoint.exists(): return checkpoint
    pool = json.loads((generation/'opponent_pool.json').read_text())
    for row in pool['entries']:
        relative = Path(row['checkpoint'])
        source = (destination/relative).resolve()
        if not source.is_relative_to(destination.resolve()): raise ValueError('Unsafe opponent path')
        copy_once(source,local/relative)
        row['checkpoint'] = str((local/relative).resolve())
    for name in ('summary.json','progress.jsonl','wandb_run.json','run_config.json'):
        if (generation/name).exists(): copy_once(generation/name,local/name)
    for source in (generation/'macro_search_buffer').glob('combo_*.npz'):
        copy_once(source, local/'macro_search_buffer'/source.name)
    for source in (generation/'lookahead_distill_buffer').glob('lookahead_*.npz'):
        copy_once(source, local/'lookahead_distill_buffer'/source.name)
    atomic_json(local/'opponent_pool.json',pool)
    if checkpoint_source:
        os.link(checkpoint_source, checkpoint)
    else:
        copy_once(generation/'ppo_latest.pt',checkpoint)
    upd = report.get('update')
    if isinstance(upd, int) and upd >= 0:
        ckpt_dir = local/'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        seed_ckpt = ckpt_dir/f'update_{upd:06d}.pt'
        seed_pool = ckpt_dir/f'update_{upd:06d}.pool.json'
        if not seed_ckpt.exists():
            os.link(checkpoint, seed_ckpt)
        if not seed_pool.exists():
            atomic_json(seed_pool, pool)
    return checkpoint
