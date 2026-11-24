"""Merge completed strict BC shards without copying arrays or admitting training."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

from cr_rl.training.persistence import atomic_json, file_sha256


def _publish_final_validation_rebalance(root, state, original_assignments, summary,
                                       prior_split=None, eval_decks_path=None):
    """After all 52 shards finish, switch only the final catalog to split v2.

    Keep per-shard v1 registries unchanged; do not touch observations or labels.
    The immutable v1 consolidated registry is copied alongside the v2 report.
    """
    if (state is None or state.get('stage') != 'full_scan_completed'
            or len(state.get('completed', {})) != state.get('total_source_shards')):
        return summary
    script = Path(__file__).with_name('repartition_quality_bc_v2.py')
    spec = importlib.util.spec_from_file_location('quality_bc_v2_repartition', script)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load final BC repartition module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkout = Path(__file__).resolve().parents[1]
    prior = Path(prior_split) if prior_split is not None else checkout/'configs/quality_bc_prior_split.json'
    eval_decks = (Path(eval_decks_path) if eval_decks_path is not None
                  else checkout/'configs/decks_eval.yaml')
    registry, report = module.build(root, prior, eval_decks)
    if set(registry['assignments']) != set(original_assignments):
        raise ValueError('Final v2 registry does not match deduplicated catalog identities')
    version_dir = module.publish(root/'split_versions', registry, report)
    legacy = version_dir/'split_registry_v1.json'
    original = {'assignments':original_assignments}
    if legacy.is_file():
        if json.loads(legacy.read_text(encoding='utf-8')) != original:
            raise ValueError('Existing v1 BC split snapshot differs from this catalog')
    else:
        atomic_json(legacy, original)
    # The root registry is the only split the next training consumer will read.
    # It changes only after the original and new registries are fully published.
    atomic_json(root/'split_registry.json', registry)
    summary.update(partition_counts=report['new_partition_counts'],
                   original_partition_counts=report['old_partition_counts'],
                   validation_to_train=report['old_validation_to_train'],
                   split_registry_version=registry['version'],
                   split_versions_dir=str(version_dir.relative_to(root)))
    atomic_json(root/'catalog_summary.json', summary)
    return summary


def catalog(root, *, prior_split=None, eval_decks=None):
    root = Path(root).resolve()
    metadata, shards, seen, decisions = None, [], set(), []
    assignments = {}
    rows = 0
    state_path=root/'state.json'
    committed=None
    state=None
    if state_path.exists():
        state=json.loads(state_path.read_text())
        committed={Path(row['output']).name for row in state.get('completed',{}).values()}
    for folder in sorted((root/'datasets').iterdir()):
        if committed is not None and folder.name not in committed:
            continue
        if not (folder/'quality_summary.json').is_file():
            continue
        manifest = json.loads((folder/'dataset.manifest.json').read_text())
        source_metadata = folder/manifest['metadata']
        if file_sha256(source_metadata)!=manifest['metadata_sha256']:
            raise ValueError('Source metadata checksum mismatch')
        source = json.loads(source_metadata.read_text())
        split = json.loads((folder/'split_registry.json').read_text())['assignments']
        if metadata is None:
            metadata = {**source,'replays':[],'source_manifests':[]}
        for key in ('schema','vocabulary','decision_ticks','mask_version','level','tower_level','import_policy'):
            if source[key]!=metadata[key]:
                raise ValueError('Incompatible BC shards: '+key)
        for shard in manifest['shards']:
            replay = source['replays'][shard['replay_offset']]
            identity = replay['canonical_game_signature']
            partition = split[identity]
            if identity in seen:
                if assignments[identity]!=partition:
                    raise ValueError('Duplicate has conflicting split')
                decisions.append(dict(identity=identity,reason='cross_shard_duplicate',folder=folder.name))
                continue
            path = (folder/shard['path']).resolve()
            if not path.is_relative_to(root) or file_sha256(path)!=shard['sha256']:
                raise ValueError('Unsafe or corrupt BC array')
            seen.add(identity);assignments[identity]=partition
            shards.append(dict(shard,path=path.relative_to(root).as_posix(),
                               replay_offset=len(metadata['replays'])))
            metadata['replays'].append(dict(replay,source_dataset=folder.name))
            rows += shard['rows']
        metadata['source_manifests'].append(dict(path=str((folder/'dataset.manifest.json').relative_to(root)),
                                                sha256=file_sha256(folder/'dataset.manifest.json')))
    if metadata is None:
        raise ValueError('No completed source shard found')
    metadata.update(rows=rows,training_ready=False,ppo_rollouts_allowed=False)
    atomic_json(root/'dataset_metadata.json',metadata)
    atomic_json(root/'split_registry.json',dict(assignments=assignments))
    atomic_json(root/'dataset.manifest.json',dict(format='crrl-bc-shards-v1',dataset_kind='repaired',
        metadata='dataset_metadata.json',metadata_sha256=file_sha256(root/'dataset_metadata.json'),
        shards=shards,rows=rows,replays=len(seen),decision_ticks=8,training_ready=False,
        ppo_rollouts_allowed=False))
    summary = dict(unique_games=len(seen),rows=rows,partition_counts=dict(Counter(assignments.values())),
        cross_shard_duplicates=len(decisions),training_ready=False,
        remaining_gate='Model contract acceptance and expert/context quality review')
    atomic_json(root/'catalog_summary.json',summary)
    return _publish_final_validation_rebalance(
        root, state, assignments, summary, prior_split, eval_decks)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(catalog(parser.parse_args().root)))
