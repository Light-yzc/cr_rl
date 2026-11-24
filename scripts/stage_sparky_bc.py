"""Index Sparky seats in a committed quality-BC catalog and stage them on SSD."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from uuid import uuid4


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def catalog_records(path):
    """Read the catalog's indented JSON one replay at a time."""
    with path.open(encoding='utf-8') as stream:
        prefix = []
        for line in stream:
            if line.strip() == '"replays": [':
                break
            prefix.append(line)
        else:
            raise ValueError('Expected the indented quality BC catalog replay array')
        yield json.loads(''.join(prefix).rstrip().removesuffix(',') + '}')
        record = []
        for line in stream:
            if not record and line.startswith('  ]'):
                return
            record.append(line)
            if line.rstrip() in ('    }', '    },'):
                yield json.loads(''.join(record).rstrip().removesuffix(','))
                record.clear()
        raise ValueError('Quality BC catalog replay array is incomplete')


def build_index(source):
    manifest = read_json(source / 'dataset.manifest.json')
    splits = read_json(source / 'split_registry.json')
    if manifest.get('decision_ticks') != 8:
        raise ValueError('Sparky BC needs the 8-tick quality BC catalog')
    registry = splits['assignments']
    records = catalog_records(source / manifest['metadata'])
    header = next(records)
    schema = {key: header[key] for key in ('schema', 'vocabulary', 'decision_ticks')}
    shards = {row['replay_offset']: row['path'] for row in manifest['shards']}
    if set(shards) != set(range(len(manifest['shards']))):
        raise ValueError('Malformed quality BC manifest replay offsets')
    selected, counts, seat_counts = [], Counter(), Counter()
    seen = set()
    scanned = 0
    for offset, replay in enumerate(records):
        scanned += 1
        seats = [seat for seat, deck in enumerate(replay['decks']) if 'ZapMachine' in deck]
        if not seats:
            continue
        identity = replay['canonical_game_signature']
        partition = registry[identity]
        if identity in seen or partition not in ('train', 'validation', 'heldout'):
            raise ValueError('Invalid game identity or split in quality BC catalog')
        seen.add(identity)
        selected.append(dict(identity=identity, partition=partition, seats=seats,
                             decks=replay['decks'], path=shards[offset]))
        counts[partition] += 1
        seat_counts[partition] += len(seats)
    if scanned != len(shards) or not selected:
        raise ValueError('Catalog replay count disagrees with manifest or contains no Sparky seats')
    return dict(format='crrl-sparky-bc-index-v1', snapshot_id='sparky_bc_' + uuid4().hex,
                label_card='ZapMachine', source_games=scanned, selected_games=len(selected),
                games_by_partition=dict(counts), seats_by_partition=dict(seat_counts),
                split_version=splits.get('version'), metadata=schema, games=selected)


def write_json(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, separators=(',', ':'))
        stream.write('\n')


def stage_index(source, out, index, workers, *, all_train=False, array_cache=None):
    if index.get('format') != 'crrl-sparky-bc-index-v1' or index.get('label_card') != 'ZapMachine':
        raise ValueError('Expected a Sparky BC index')
    if out.exists():
        raise FileExistsError(out)
    if out.as_posix().startswith('/content/drive/'):
        raise ValueError('Stage arrays on instance SSD')
    games = index['games']
    if array_cache is not None:
        source = array_cache
        cached = read_json(source / 'dataset.manifest.json')
        if cached['snapshot_id'] != index['snapshot_id'] or len(cached['shards']) != len(games):
            raise ValueError('Cached arrays are from a different BC snapshot')
        paths = [row['path'] for row in cached['shards']]
    else:
        paths = [row['path'] for row in games]
    source = source.resolve()
    files = [(source / path).resolve() for path in paths]
    if any(not path.is_relative_to(source) or not path.is_file() for path in files):
        raise ValueError('Missing or unsafe quality BC array path')
    temporary = out.with_name(out.name + '.staging-' + uuid4().hex[:10])
    (temporary / 'games').mkdir(parents=True)
    descriptors = [f'games/{i:06d}.npz' for i in range(len(games))]

    def copy(task):
        path, relative = task
        dest = temporary / relative
        try:
            dest.hardlink_to(path)
        except OSError:
            shutil.copyfile(path, dest)
        if path.stat().st_size != dest.stat().st_size:
            raise OSError('Incomplete array copy: ' + relative)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(copy, zip(files, descriptors)):
            pass
    metadata = dict(index['metadata'], training_ready=False, focus_card=index['label_card'],
                    validation_enabled=not all_train,
                    replays=[dict(canonical_game_signature=r['identity'], decks=r['decks']) for r in games])
    shards = [dict(path=path, replay_offset=i, selected_seats=row['seats'])
              for i, (path, row) in enumerate(zip(descriptors, games))]
    write_json(temporary / 'dataset_metadata.json', metadata)
    assignments = {r['identity']: 'train' if all_train else r['partition'] for r in games}
    write_json(temporary / 'split_registry.json', dict(assignments=assignments))
    snapshot_id = index['snapshot_id'] + ('_all_train' if all_train else '')
    write_json(temporary / 'dataset.manifest.json', dict(snapshot_id=snapshot_id,
               metadata='dataset_metadata.json', format='crrl-bc-shards-v1', shards=shards))
    temporary.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write-index', type=Path)
    mode.add_argument('--index', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--all-train', action='store_true', help='Use every selected seat for BC; disable BC validation')
    parser.add_argument('--array-cache', type=Path, help='Reuse an already staged split snapshot on SSD')
    args = parser.parse_args()
    if args.workers < 1 or (args.index and not args.out) or (args.write_index and (args.out or args.all_train or args.array_cache)):
        parser.error('Supply --out only when staging with --index, with at least one worker')
    if args.write_index:
        index = build_index(args.source)
        args.write_index.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.write_index, index)
    else:
        index = read_json(args.index)
        stage_index(args.source, args.out, index, args.workers, all_train=args.all_train, array_cache=args.array_cache)
    report = {k: v for k, v in index.items() if k not in ('games', 'metadata')}
    if args.all_train:
        report.update(snapshot_id=index['snapshot_id'] + '_all_train', validation_enabled=False,
                      games_by_partition={'train': index['selected_games']},
                      seats_by_partition={'train': sum(index['seats_by_partition'].values())})
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
