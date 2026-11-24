import importlib.util
import json
from pathlib import Path

import pytest

from cr_rl.training.persistence import file_sha256


def module(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/(name+'.py')
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def shard(root, name, partition='train', *, signature='same-game', decks=None):
    folder = root/'datasets'/name
    folder.mkdir(parents=True)
    arrays = folder/'game.npz'
    arrays.write_bytes(b'opaque array fixture; catalog only hashes content')
    metadata = dict(schema={},vocabulary=[],decision_ticks=8,mask_version=2,level=11,
        tower_level=11,import_policy={},replays=[dict(
            canonical_game_signature=signature,
            decks=decks or [['A', 'B'], ['C', 'D']])])
    (folder/'dataset_metadata.json').write_text(json.dumps(metadata))
    manifest = dict(metadata='dataset_metadata.json',
        metadata_sha256=file_sha256(folder/'dataset_metadata.json'),
        shards=[dict(path='game.npz',sha256=file_sha256(arrays),replay_offset=0,rows=12,replays=1)])
    (folder/'dataset.manifest.json').write_text(json.dumps(manifest))
    (folder/'quality_summary.json').write_text(json.dumps({'accepted_games':1}))
    (folder/'split_registry.json').write_text(json.dumps(dict(assignments={signature:partition})))
    return folder


def test_cross_shard_dedup_preserves_split_and_does_not_admit_training(tmp_path):
    shard(tmp_path,'a');shard(tmp_path,'b')
    summary = module('catalog_quality_bc').catalog(tmp_path)
    assert summary['unique_games']==1 and summary['cross_shard_duplicates']==1
    assert summary['partition_counts']=={'train':1}
    assert summary['training_ready'] is False


def test_conflicting_splits_fail(tmp_path):
    shard(tmp_path,'a');shard(tmp_path,'b','validation')
    with pytest.raises(ValueError,match='conflicting split'):
        module('catalog_quality_bc').catalog(tmp_path)


def test_corrupt_array_fails(tmp_path):
    folder = shard(tmp_path,'a')
    (folder/'game.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='corrupt'):
        module('catalog_quality_bc').catalog(tmp_path)


def test_live_catalog_only_reads_committed_producer_shards(tmp_path):
    shard(tmp_path,'a')
    unfinished=tmp_path/'datasets'/'b';unfinished.mkdir()
    (unfinished/'quality_summary.json').write_text('{}')
    (tmp_path/'state.json').write_text(json.dumps({'completed':{
        'source-a':{'output':'/content/work/datasets/a'}}}))
    summary=module('catalog_quality_bc').catalog(tmp_path)
    assert summary['unique_games']==1


def test_launch_plan_uses_prepared_smoke_and_distinct_awbc_output():
    launcher = module('launch_prepared_pipeline')
    root = Path('/content/crrl_training')
    command = launcher.command_for('ppo-smoke',dict(smoke_argv=['oldpython','oldscript','--v3']),root,None)
    assert command[1]==str(root/'scripts/train_selfplay.py')
    command = launcher.command_for('awbc',{},root,Path('/sessions'),True)
    assert '--wandb' in command and str(root/'runs/awbc_v1') in command


def test_full_scan_rebalances_only_v1_validation_and_retains_legacy_registry(tmp_path):
    names = {
        'moved': '00000005' + '0' * 56,     # old hash % 5 == 0; new % 10 == 5
        'kept': '0000000a' + '0' * 56,      # stays validation
        'train': '00000001' + '0' * 56,     # always train
        'heldout': '00000002' + '0' * 56,   # eval-deck heldout
        'legacy_val': '00000007' + '0' * 56,# old known validation
    }
    old = dict(moved='validation', kept='validation', train='train',
               heldout='heldout', legacy_val='validation')
    completed = {}
    for index, (tag, signature) in enumerate(names.items()):
        path = shard(tmp_path, tag, old[tag], signature=signature,
                     decks=[['H', 'J'], ['C', 'D']] if tag == 'heldout' else None)
        completed[f'replays/part-{index}.parquet'] = dict(
            source_sha256='fake-source-sha', output=str(path),
            summary={'accepted_games':1})
    state = dict(stage='reconstructing', completed=completed,
                 total_source_shards=5, revision='source-revision',
                 training_started=False)
    (tmp_path/'state.json').write_text(json.dumps(state))
    prior = tmp_path/'prior.json'
    prior.write_text(json.dumps({'assignments':{names['legacy_val']:'validation'}}))
    eval_decks = tmp_path/'eval.yaml'
    eval_decks.write_text('decks:\n  test: [H, J]\n')
    catalog = module('catalog_quality_bc').catalog

    initial = catalog(tmp_path, prior_split=prior, eval_decks=eval_decks)
    assert initial['partition_counts']=={'train':1,'validation':3,'heldout':1}
    assert not (tmp_path/'split_versions').exists()

    state['stage']='full_scan_completed'
    (tmp_path/'state.json').write_text(json.dumps(state))
    for _ in range(2):  # repeatable finalizer, without overwritten old split
        result = catalog(tmp_path, prior_split=prior, eval_decks=eval_decks)
        assert result['partition_counts']=={'train':2,'validation':2,'heldout':1}
        assert result['validation_to_train']==1
        assert result['training_ready'] is False
        active = json.loads((tmp_path/'split_registry.json').read_text())['assignments']
        assert active[names['moved']]=='train'
        assert active[names['kept']]=='validation'
        assert active[names['heldout']]=='heldout'
        assert active[names['legacy_val']]=='validation'
        version_dir = tmp_path/result['split_versions_dir']
        v1 = json.loads((version_dir/'split_registry_v1.json').read_text())['assignments']
        assert v1=={names[k]:v for k,v in old.items()}
        assert json.loads((version_dir/'report.json').read_text())['training_ready'] is False


def test_partial_scan_never_publishes_rebalanced_registry(tmp_path):
    signature = '00000005' + '0' * 56
    p=shard(tmp_path,'partial','validation',signature=signature)
    (tmp_path/'state.json').write_text(json.dumps(dict(
        stage='full_scan_completed',total_source_shards=2,
        completed={'source':dict(output=str(p),source_sha256='source',
                                 summary={'accepted_games':1})})))
    summary=module('catalog_quality_bc').catalog(tmp_path)
    assert summary['partition_counts']=={'validation':1}
    assert not (tmp_path/'split_versions').exists()
