"""Freeze committed BC games into one ZIP on CPU SSD, then copy once to Drive.

No checksum scans, no NPZ decompression, no re-running reconstruction. ZIP_STORED
avoids compressing already-compressed NPZ files. The snapshot remains explicitly
experimental: structural tests are not evidence of expert skill.
"""
def main():
    from collections import Counter
    import json
    import os
    from pathlib import Path
    import shutil
    import sys
    import time
    import zipfile
    import yaml

    root = Path('/content/crrl_training')
    work = root/'public_bc_full'
    sys.path.insert(0, str(root/'src'))
    from cr_rl.data.quality_bc_split import quality_bc_partition
    mode = os.environ.get('CRRL_PACK_MODE', 'smoke')
    if mode not in ('smoke','full'):
        raise ValueError('CRRL_PACK_MODE must be smoke or full')
    state = json.loads((work/'state.json').read_text())
    committed = dict(state.get('completed', {}))
    if not committed:
        raise RuntimeError('No committed reconstruction shards')
    prior = json.loads((root/'configs/quality_bc_prior_split.json').read_text())['assignments']
    heldout = {tuple(sorted(d)) for d in yaml.safe_load(
        (root/'configs/decks_eval.yaml').read_text())['decks'].values()}
    rows = []
    seen = {}
    template = None
    for source, item in sorted(committed.items()):
        folder = Path(item['output'])
        manifest = json.loads((folder/'dataset.manifest.json').read_text())
        meta = json.loads((folder/manifest['metadata']).read_text())
        if template is None:
            template = {k:v for k,v in meta.items() if k not in ('replays', 'source_file_sha256')}
        for key in ('schema','vocabulary','decision_ticks','mask_version','level','tower_level','import_policy'):
            if meta[key] != template[key]:
                raise ValueError('Incompatible completed metadata: ' + key)
        for record in manifest['shards']:
            replay = meta['replays'][record['replay_offset']]
            identity = replay['canonical_game_signature']
            split = quality_bc_partition(identity, replay['decks'], heldout, prior)
            if identity in seen:
                if seen[identity] != split:
                    raise ValueError('Duplicate game has conflicting partition')
                continue
            seen[identity] = split
            path = (folder/record['path']).resolve()
            if not path.is_relative_to(work.resolve()) or not path.is_file():
                raise ValueError('Missing or unsafe committed array path')
            if split != 'heldout':
                rows.append((path, record, replay, split))
    all_counts = dict(Counter(seen.values()))
    if mode == 'smoke':
        rows = ([r for r in rows if r[3] == 'train'][:6]
                + [r for r in rows if r[3] == 'validation'][:3])
    counts = Counter(r[3] for r in rows)
    if counts['train'] < 4 or counts['validation'] < 2:
        raise ValueError('Need at least four training and two validation games')
    stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    snapshot_id = f'quality_bc_{mode}_{stamp}_{len(committed)}shards'
    out = root/'exports'/f'{snapshot_id}.zip'
    out.parent.mkdir(exist_ok=True)
    partial = out.with_suffix('.zip.partial')
    dataset = dict(template, replays=[r[2] for r in rows], rows=sum(r[1]['rows'] for r in rows),
                   snapshot_id=snapshot_id, training_ready=False, experimental_auxiliary_bc=True,
                   expert_skill_verified=False, original_battlefield_parity_verified=False)
    manifest = dict(format='crrl-bc-shards-v1', dataset_kind='repaired', snapshot_id=snapshot_id,
                    metadata='dataset_metadata.json', decision_ticks=8,
                    training_ready=False, experimental_auxiliary_bc=True,
                    ppo_rollouts_allowed=False, shards=[], rows=dataset['rows'], replays=len(rows))
    splits = {'assignments':{r[2]['canonical_game_signature']:r[3] for r in rows},
              'version':'quality_bc_v2_10pct_nonheldout', 'snapshot_id':snapshot_id}
    print(json.dumps(dict(stage='pack_start', snapshot_id=snapshot_id,
                          completed_shards=len(committed), games=len(rows), partitions=dict(counts))), flush=True)
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for index, (path, record, replay, split) in enumerate(rows):
            name = f'shards/game_{index:06d}.npz'
            archive.write(path, name)
            manifest['shards'].append(dict(path=name, rows=record['rows'], replays=1,
                                            replay_offset=index, bytes=path.stat().st_size))
        for name, value in [('dataset.manifest.json',manifest),('dataset_metadata.json',dataset),
                            ('split_registry.json',splits)]:
            archive.writestr(name, json.dumps(value, separators=(',',':')))
    partial.replace(out)
    destination = Path('/content/drive/MyDrive/cr_rl_ppo/preparation/ssd_snapshots')
    destination.mkdir(exist_ok=True)
    target = destination/out.name
    temporary = target.with_suffix('.zip.partial')
    print(json.dumps(dict(stage='copy_single_zip_to_drive', bytes=out.stat().st_size)), flush=True)
    shutil.copyfile(out, temporary)
    if temporary.stat().st_size != out.stat().st_size:
        raise IOError('Incomplete ZIP copy')
    temporary.replace(target)
    report = dict(snapshot_id=snapshot_id, archive=str(target), bytes=target.stat().st_size,
                  completed_shards=len(committed), total_source_shards=state['total_source_shards'],
                  games=len(rows), partitions=dict(counts), full_candidate_partitions=all_counts,
                  source_stage=state['stage'], mode=mode, checksum_scans=False,
                  training_started=False, expert_skill_verified=False)
    (destination/f'{mode}_snapshot.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
