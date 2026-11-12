import json
import threading

import pytest
from test_incremental_backup import add_rollout, setup_run

from cr_rl.training import async_publication as module
from cr_rl.training.persistence import restore_run


def replace_bytes(path, value):
    temporary = path.with_suffix('.next')
    temporary.write_bytes(value)
    temporary.replace(path)


def test_upload_uses_frozen_generation_while_live_files_change(tmp_path, monkeypatch):
    local, backup = tmp_path/'run', tmp_path/'drive'
    setup_run(local)
    (local/'actor_export.pt').write_bytes(b'actor-one')
    (local/'metrics.jsonl').write_text('generation-one\n')
    entered, release = threading.Event(), threading.Event()
    original = module.publish_incremental

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(module, 'publish_incremental', blocked)
    publisher = module.BoundedAsyncPublisher(local, backup)
    try:
        publisher.submit(1)
        assert entered.wait(5)
        assert not (backup/'latest.json').exists()
        replace_bytes(local/'ppo_latest.pt', b'second-state')
        replace_bytes(local/'actor_export.pt', b'actor-two')
        (local/'metrics.jsonl').write_text('generation-two\n')
        release.set()
        manifest = publisher.wait()
        assert manifest['update'] == 1
        assert (backup/manifest['checkpoint']).read_bytes() == b'state'
        assert (backup/'exports/update_000001.actor.pt').read_bytes() == b'actor-one'
        assert (backup/'metrics.jsonl').read_text() == 'generation-one\n'
        restore_run(backup, tmp_path/'restored-one')
        add_rollout(local, 2)
        publisher.submit(2)
        manifest = publisher.wait()
        assert manifest['update'] == 2
        assert (backup/manifest['checkpoint']).read_bytes() == b'second-state'
        restore_run(backup, tmp_path/'restored-two')
        assert not list((local/'.publication_snapshots').iterdir())
    finally:
        release.set()
        publisher.close()


def test_failure_propagates_and_preserves_latest_and_evidence(tmp_path, monkeypatch):
    local, backup = tmp_path/'run', tmp_path/'drive'
    setup_run(local)

    def fail(*args):
        raise OSError('Drive disconnected')

    monkeypatch.setattr(module, 'publish_incremental', fail)
    publisher = module.BoundedAsyncPublisher(local, backup)
    publisher.submit(1)
    with pytest.raises(OSError, match='disconnected'):
        publisher.wait()
    with pytest.raises(OSError, match='disconnected'):
        publisher.submit(2)
    assert not (backup/'latest.json').exists()
    assert (local/'.publication_snapshots/update_000001/ppo_latest.pt').is_file()
    assert not (local/'.publication_snapshots/update_000002').exists()
    with pytest.raises(OSError, match='disconnected'):
        publisher.close()


def test_at_most_one_snapshot_is_queued_and_close_drains_upload(tmp_path, monkeypatch):
    local, backup = tmp_path/'run', tmp_path/'drive'
    setup_run(local)
    entered, release = threading.Event(), threading.Event()
    original = module.publish_incremental

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(module, 'publish_incremental', blocked)
    publisher = module.BoundedAsyncPublisher(local, backup)
    publisher.submit(1)
    assert entered.wait(5)
    replace_bytes(local/'ppo_latest.pt', b'two')
    add_rollout(local, 2)
    errors = []

    def submit_second():
        try:
            publisher.submit(2)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=submit_second)
    worker.start()
    try:
        worker.join(timeout=.05)
        assert worker.is_alive()
        assert not (local/'.publication_snapshots/update_000002').exists()
    finally:
        release.set()
        worker.join(timeout=5)
        publisher.close()
    assert not errors
    assert json.loads((backup/'latest.json').read_text())['update'] == 2
    with pytest.raises(RuntimeError, match='closed'):
        publisher.submit(3)
