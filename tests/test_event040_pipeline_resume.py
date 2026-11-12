"""Local recovery invariants; no remote jobs, JVM, CUDA or data mutation."""
import importlib.util
import json
from pathlib import Path

import pytest


def pipeline_module():
    path=Path(__file__).resolve().parents[1]/'scripts/colab_event040_pipeline.py'
    spec=importlib.util.spec_from_file_location('event040_pipeline_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_incomplete_resample_can_restart_without_deleting_committed_data(tmp_path,monkeypatch):
    pipeline=pipeline_module();monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    target=tmp_path/'event040_data'
    (target/'shards').mkdir(parents=True)
    (target/'shards/replay_000001.npz').write_bytes(b'partial')
    (target/'.source_arrays/shard_000001').mkdir(parents=True)
    (target/'.source_arrays/shard_000001/observations.npy').write_bytes(b'scratch')
    pipeline.discard_uncommitted_resample(target)
    assert not target.exists()
    target.mkdir();(target/'dataset.manifest.json').write_text('{}')
    pipeline.discard_uncommitted_resample(target)
    assert (target/'dataset.manifest.json').exists()


@pytest.mark.parametrize('kind',['outside','unexpected_file'])
def test_cleanup_refuses_unowned_output(tmp_path,monkeypatch,kind):
    pipeline=pipeline_module();monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    target=tmp_path/('other_data' if kind=='outside' else 'event040_data')
    target.mkdir();sentinel=target/'user_notes.txt';sentinel.write_text('keep')
    with pytest.raises(RuntimeError,match='owned|unrecognized'):
        pipeline.discard_uncommitted_resample(target)
    assert sentinel.read_text()=='keep'


@pytest.mark.parametrize('saved,epoch,valid',[
    ({'epoch':1,'next_episode':0,'optimizer_steps':606},1,True),
    ({'epoch':0,'next_episode':350,'optimizer_steps':606},1,False),
    ({'epoch':2,'next_episode':0,'optimizer_steps':956},1,False),
    ({'epoch':1,'next_episode':1,'optimizer_steps':607},1,False),
])
def test_cached_progress_cannot_label_mid_or_different_epoch_weights(monkeypatch,saved,epoch,valid):
    pipeline=pipeline_module();monkeypatch.setattr(pipeline,'checkpoint_progress',lambda _:saved)
    if valid:
        assert pipeline.require_completed_epoch(Path('latest.pt'),epoch)==saved
    else:
        with pytest.raises(RuntimeError,match='differs from completed checkpoint'):
            pipeline.require_completed_epoch(Path('latest.pt'),epoch)


def evaluation_fixture(tmp_path,pipeline):
    destination=tmp_path/'epoch2_bc';destination.mkdir()
    policy=tmp_path/'latest.pt';policy.write_bytes(b'completed-epoch2')
    opponent=tmp_path/'old_bc.pt';opponent.write_bytes(b'frozen-bc')
    record={
        'policy_sha256':pipeline.file_sha256(policy),'opponent_sha256':pipeline.file_sha256(opponent),
        'decision_ticks':8,'pair_count':4,'pair_start':4,'candidate_inference_budget_passed':True,
    }
    evidence=destination/'evaluation.json';evidence.write_text(json.dumps(record))
    (destination/'evaluation_summary.json').write_text(json.dumps({'evaluation_sha256':pipeline.file_sha256(evidence)}))
    return destination,policy,opponent


def test_cached_evaluation_is_reused_only_for_same_weights_and_epoch(tmp_path):
    pipeline=pipeline_module();destination,policy,opponent=evaluation_fixture(tmp_path,pipeline)
    pipeline.verify_evaluation(destination,policy,opponent,2)
    policy.write_bytes(b'newer-epoch3')
    with pytest.raises(RuntimeError,match='different frozen weights'):
        pipeline.verify_evaluation(destination,policy,opponent,2)


@pytest.mark.parametrize('changed',['opponent','evidence','epoch'])
def test_cached_evaluation_rejects_stale_opponent_tampered_evidence_or_epoch(tmp_path,changed):
    pipeline=pipeline_module();destination,policy,opponent=evaluation_fixture(tmp_path,pipeline)
    if changed=='opponent':opponent.write_bytes(b'replaced-opponent')
    if changed=='evidence':
        evidence=destination/'evaluation.json';evidence.write_text(evidence.read_text()+' ')
    with pytest.raises(RuntimeError,match='frozen weights|evidence|protocol'):
        pipeline.verify_evaluation(destination,policy,opponent,3 if changed=='epoch' else 2)
