"""Publish a non-destructive, versioned 10% validation registry for quality BC.

The active shard producer is never modified. This reads only committed shards;
the original per-shard registries and v1 catalog remain intact. A partial
snapshot is for audit only, not for training. Final publication additionally
requires full_scan_completed and a complete committed source-shard inventory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml

from cr_rl.data.quality_bc_split import quality_bc_partition
from cr_rl.training.persistence import file_sha256

FORMAT = 'crrl-quality-bc-split-v2'


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _digest(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')).hexdigest()


def build(root, prior_path, eval_decks_path, *, allow_partial=False):
    root = Path(root).resolve()
    state_file = root/'state.json'
    state = _read_json(state_file)
    completed = dict(state.get('completed', {}))
    total = state.get('total_source_shards')
    if not completed or type(total) is not int or not 0 < len(completed) <= total:
        raise ValueError('No valid committed BC shard inventory')
    is_final = (state.get('stage') == 'full_scan_completed'
                and len(completed) == total and state.get('training_started') is False)
    if not is_final and not allow_partial:
        raise ValueError('Full scan not complete; use --allow-partial for audit only')
    prior = _read_json(prior_path)
    assignments = prior.get('assignments', {})
    eval_cfg = yaml.safe_load(Path(eval_decks_path).read_text(encoding='utf-8'))
    heldout = {tuple(sorted(deck)) for deck in eval_cfg['decks'].values()}
    new_assignments, old_assignments = {}, {}
    transitions, prior_matches = Counter(), Counter()
    per_partition = Counter()
    shard_inputs = []
    total_rows = 0
    for source, info in sorted(completed.items()):
        folder = root/'datasets'/Path(info['output']).name
        manifest_path = folder/'dataset.manifest.json'
        manifest = _read_json(manifest_path)
        meta_path = folder/manifest['metadata']
        if file_sha256(meta_path) != manifest['metadata_sha256']:
            raise ValueError('Shard metadata checksum mismatch: ' + source)
        metadata = _read_json(meta_path)
        registry = _read_json(folder/'split_registry.json')['assignments']
        replays = metadata['replays']
        shards = manifest['shards']
        if len(replays) != len(shards) or len(replays) != info['summary']['accepted_games']:
            raise ValueError('Shard replay/manifest/producer count mismatch: ' + source)
        for shard in shards:
            if (type(shard.get('replay_offset')) is not int
                    or not 0 <= shard['replay_offset'] < len(replays)):
                raise ValueError('Invalid replay offset: ' + source)
            record = replays[shard['replay_offset']]
            signature = record['canonical_game_signature']
            original = registry[signature]
            result = quality_bc_partition(signature, record['decks'], heldout, assignments)
            if original not in ('train', 'validation', 'heldout'):
                raise ValueError('Unknown original split: ' + source)
            if original == 'heldout' and result != 'heldout':
                raise ValueError('Would promote original heldout game: ' + signature)
            if result == 'heldout' and original != 'heldout':
                raise ValueError('Original shard violated fixed heldout rule: ' + signature)
            if original == 'train' and result != 'train':
                raise ValueError('Would move existing Train game outside Train: ' + signature)
            if signature in old_assignments:
                if old_assignments[signature] != original or new_assignments[signature] != result:
                    raise ValueError('Cross-shard split conflict: ' + signature)
                continue
            old_assignments[signature] = original
            new_assignments[signature] = result
            transitions[f'{original}->{result}'] += 1
            per_partition[result] += 1
            if signature in assignments:
                prior_matches[assignments[signature]] += 1
            total_rows += shard['rows']
        shard_inputs.append({
            'source':source,
            'output_name':folder.name,
            'source_sha256':info['source_sha256'],
            'manifest_sha256':file_sha256(manifest_path),
        })
    latest = _read_json(state_file)
    if latest.get('completed') != completed:
        raise RuntimeError('Producer committed a shard while reading; retry snapshot')
    summary = {
        'format':FORMAT, 'completed_shards':len(completed), 'total_source_shards':total,
        'final_scan':is_final, 'training_ready':False, 'ppo_rollouts_allowed':False,
        'current_unique_games':len(new_assignments), 'accepted_frame_rows':total_rows,
        'old_partition_counts':dict(Counter(old_assignments.values())),
        'new_partition_counts':dict(per_partition), 'transitions':dict(transitions),
        'historical_prior_overlap':dict(prior_matches),
        'old_validation_to_train':transitions['validation->train'],
        'source_revision':state.get('revision'),
        'policy':'legacy prior preserved; eval decks heldout; new non-heldout sha256[:8] % 10 == 0 validation',
        'warning':'Changing a v1 provisional validation split invalidates comparison with v1 validation metrics. No new corpus training is approved.',
    }
    registry = {
        'format':FORMAT, 'version':'quality_bc_v2_10pct_nonheldout',
        'source_revision':state.get('revision'),
        'split_policy':summary['policy'],
        'prior_split_sha256':file_sha256(Path(prior_path)),
        'eval_decks_sha256':file_sha256(Path(eval_decks_path)),
        'committed_shards':shard_inputs, 'final_scan':is_final,
        'training_ready':False, 'ppo_rollouts_allowed':False,
        'assignments':dict(sorted(new_assignments.items())),
    }
    summary['registry_sha256_canonical'] = _digest(registry)
    return registry, summary


def publish(out_dir, registry, report):
    out_dir = Path(out_dir)
    suffix = ('final' if report['final_scan'] else 'preview')
    batch = f"{report['completed_shards']}of{report['total_source_shards']}"
    fingerprint = report['registry_sha256_canonical'][:12]
    folder = out_dir/f'quality_bc_v2_{suffix}_{batch}_{fingerprint}'
    folder.mkdir(parents=True, exist_ok=True)
    for name, payload in [('split_registry.json', registry), ('report.json', report)]:
        path = folder/name
        contents = json.dumps(payload, indent=2, ensure_ascii=False)+'\n'
        if path.is_file() and path.read_text(encoding='utf-8') != contents:
            raise ValueError('Existing versioned BC artifact is different: ' + str(path))
        if not path.is_file():
            path.write_text(contents, encoding='utf-8')
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prior-split', type=Path, required=True)
    parser.add_argument('--eval-decks', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    registry, summary = build(args.root, args.prior_split, args.eval_decks,
                              allow_partial=args.allow_partial)
    if args.publish:
        dest = publish(args.out_dir or args.root/'split_versions', registry, summary)
        summary['published_path'] = str(dest)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
