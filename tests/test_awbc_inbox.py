import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from cr_rl.training import awbc_inbox as module


def recording(parent, name='game_one'):
    run = parent/name
    (run/'diagnostics').mkdir(parents=True)
    for filename, data in (
        ('manifest.json', dict(collection_complete=True, outcome_verified=True,
                               stop_reason='battle_end', outcome='win')),
        ('result.json', dict(session_id=name, outcome_verified=True, outcome='win')),
        ('terminal.json', dict(session_id=name, clock_ended=True)),
    ):
        (run/filename).write_text(json.dumps(data))
    (run/'events.jsonl').write_text('{}\n')
    marker=dict(format='crrl-sealed-session-v1',
                manifest=json.loads((run/'manifest.json').read_text()),
                result=json.loads((run/'result.json').read_text()))
    (run/'collection_sealed.json').write_text(json.dumps(marker))
    with gzip.open(run/'diagnostics/frames_000000.jsonl.gz', 'wt') as f:
        f.write('{}\n')
    return run


def batch(tmp_path, runs, name='batch_test'):
    archive = tmp_path/(name+'.zip')
    receipt = module.pack_batch(runs, archive, name)
    inbox = tmp_path/'drive'
    module.publish_batch(archive, inbox, receipt)
    return inbox, receipt


def test_complete_batch_atomic_import_retry_and_new_ssd(tmp_path, monkeypatch):
    runs = [recording(tmp_path/'source', name) for name in ('one', 'two')]
    inbox, receipt = batch(tmp_path, runs)
    sessions, state = tmp_path/'ssd/sessions', tmp_path/'ssd/state.json'
    sessions.mkdir(parents=True)
    original_copy = module.shutil.copyfile

    def check_hidden_before_copy(source, target):
        assert not list(sessions.rglob('result.json'))
        return original_copy(source, target)

    monkeypatch.setattr(module.shutil, 'copyfile', check_hidden_before_copy)
    report = module.ingest_ready_batches(inbox, sessions, state)
    assert report['added_games'] == 2 and report['errors'] == []
    monkeypatch.setattr(module.shutil, 'copyfile', original_copy)
    assert sorted(p.parent.name for p in sessions.glob('*/result.json')) == ['one', 'two']
    assert module.ingest_ready_batches(inbox, sessions, state)['added_games'] == 0
    state.unlink()  # Crash after directory commit but before ledger commit.
    report = module.ingest_ready_batches(inbox, sessions, state)
    assert report['duplicate_games'] == 2 and report['added_games'] == 0
    fresh = tmp_path/'new_ssd'
    assert module.ingest_ready_batches(inbox, fresh/'sessions', fresh/'state.json')['added_games'] == 2


def test_incomplete_recording_and_incomplete_upload_never_commit(tmp_path, monkeypatch):
    run = recording(tmp_path/'source')
    (run/'result.json').unlink()
    with pytest.raises(ValueError, match='Missing'):
        module.pack_batch([run], tmp_path/'bad.zip', 'batch_incomplete')
    run = recording(tmp_path/'other')
    archive = tmp_path/'good.zip'
    receipt = module.pack_batch([run], archive, 'batch_interrupted')

    def fail(*args):
        raise OSError('interrupted upload')

    monkeypatch.setattr(module.shutil, 'copyfile', fail)
    inbox = tmp_path/'drive'
    with pytest.raises(OSError):
        module.publish_batch(archive, inbox, receipt)
    assert not list(inbox.glob('*.ready.json'))
    assert module.ingest_ready_batches(inbox, tmp_path/'sessions', tmp_path/'state.json')['added_games'] == 0


def test_name_collision_never_overwrites_existing_game(tmp_path):
    one, two = [recording(tmp_path/'source', name) for name in ('one', 'two')]
    inbox, receipt = batch(tmp_path, [one, two])
    sessions = tmp_path/'sessions'
    existing = recording(sessions, 'one')
    (existing/'manifest.json').write_text('{"different":true}')
    report = module.ingest_ready_batches(inbox, sessions, tmp_path/'state.json')
    assert report['errors'] and report['added_games'] == 0
    assert not (sessions/'two').exists()
    assert (existing/'manifest.json').read_text() == '{"different":true}'


def test_unsafe_archive_rejected_before_any_game_exposed(tmp_path):
    inbox = tmp_path/'drive'; inbox.mkdir()
    archive = inbox/'batch_unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('one/../../outside.txt', 'bad')
    module.atomic_json(inbox/'batch_unsafe.ready.json', dict(format=module.FORMAT,
        batch_id='batch_unsafe', archive=archive.name, bytes=archive.stat().st_size, games=['one']))
    report = module.ingest_ready_batches(inbox, tmp_path/'sessions', tmp_path/'state.json')
    assert report['errors'] and report['added_games'] == 0
    assert not (tmp_path/'outside.txt').exists()
    assert not list((tmp_path/'sessions').rglob('result.json'))


def test_cli_zero_exit_with_remote_python_error_is_not_success(monkeypatch):
    path = Path(__file__).resolve().parents[1]/'scripts/upload_awbc_batches.py'
    spec = importlib.util.spec_from_file_location('uploader_test', path)
    uploader = importlib.util.module_from_spec(spec); spec.loader.exec_module(uploader)
    monkeypatch.setattr(uploader, 'run_bounded', lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=b'Traceback: failed'))
    monkeypatch.setattr(uploader.time, 'sleep', lambda *a: None)
    with pytest.raises(RuntimeError, match='pending batch retained'):
        uploader.colab(['exec'], expected='BATCH_COMMITTED test')


def test_upload_lock_excludes_second_uploader_and_releases(tmp_path):
    lock = tmp_path/'upload.lock'
    with module.exclusive_lock(lock):
        with pytest.raises(OSError):
            with module.exclusive_lock(lock):
                pytest.fail('Second uploader acquired the same lock')
    with module.exclusive_lock(lock):
        pass


def test_sealed_result_cannot_change_and_legacy_writer_must_finish(tmp_path):
    run=recording(tmp_path/'source')
    module.completed_files(run)
    marker=run/'collection_sealed.json'
    marker.unlink()
    with pytest.raises(ValueError,match='writer has not sealed'):
        module.completed_files(run)

    (run/'events.jsonl').write_text('{"event":"menu_lobby_verified"}\n')
    module.completed_files(run)
    result=json.loads((run/'result.json').read_text())
    result['reward']=1
    marker.write_text(json.dumps(dict(format='crrl-sealed-session-v1',
        manifest=json.loads((run/'manifest.json').read_text()),result=result)))
    with pytest.raises(ValueError,match='changed after sealing'):
        module.completed_files(run)


def test_flush_marker_sends_short_batch_without_stopping_continuous_uploader(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/upload_awbc_batches.py'
    spec=importlib.util.spec_from_file_location('uploader_flush_test',path)
    uploader=importlib.util.module_from_spec(spec);spec.loader.exec_module(uploader)
    source=tmp_path/'source';run=recording(source)
    outbox=tmp_path/'outbox'
    args=SimpleNamespace(source=source,outbox=outbox,batch_games=10,loop=True,session='test')
    state={'known':{}}
    assert not uploader.cycle(args,state)
    (outbox/'FLUSH').touch()
    assert uploader.cycle(args,state)
    calls=[]
    monkeypatch.setattr(uploader,'upload',lambda archive,receipt,session:calls.append(receipt))
    assert not uploader.cycle(args,state)
    assert len(calls)==1 and calls[0]['games']==[run.name]
    assert run.name in state['known'] and not (outbox/'FLUSH').exists()


def test_bad_batches_do_not_starve_later_valid_recordings(tmp_path):
    inbox=tmp_path/'drive';inbox.mkdir()
    for index in range(4):
        name=f'batch_bad{index}'
        archive=inbox/(name+'.zip');archive.write_bytes(b'not zip')
        module.atomic_json(inbox/(name+'.ready.json'),dict(format=module.FORMAT,
            batch_id=name,archive=archive.name,bytes=7,games=['one']))
    run=recording(tmp_path/'source','one')
    archive=tmp_path/'valid.zip';receipt=module.pack_batch([run],archive,'batch_zvalid')
    module.publish_batch(archive,inbox,receipt)
    sessions=tmp_path/'sessions';state=tmp_path/'state.json'
    first=module.ingest_ready_batches(inbox,sessions,state,max_batches=4)
    assert len(first['errors'])==4 and first['added_games']==0
    second=module.ingest_ready_batches(inbox,sessions,state,max_batches=4)
    assert second['added_games']==1 and (sessions/'one/result.json').exists()
