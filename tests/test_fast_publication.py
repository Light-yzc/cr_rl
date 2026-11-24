import json
from pathlib import Path

from cr_rl.training.fast_publication import publish_snapshot, restore_latest


def test_repeated_publication_retains_complete_resume_and_bounded_history(tmp_path):
    destination = tmp_path/'drive'
    for update in range(1,5):
        snapshot = tmp_path/f'local{update}'
        (snapshot/'opponents').mkdir(parents=True)
        actor = snapshot/'opponents'/f'actor{update}.pt'
        actor.write_bytes(bytes([update])*17)
        (snapshot/'ppo_latest.pt').write_bytes(bytes([update])*100)
        (snapshot/'opponent_pool.json').write_text(json.dumps(dict(entries=[dict(checkpoint=str(actor))])))
        (snapshot/'wandb_run.json').write_text('{"id":"resumable"}')
        publish_snapshot(snapshot,destination,update)
    assert len(list((destination/'generations').iterdir())) == 2
    assert {p.name for p in (destination/'opponents').iterdir()} == {'actor3.pt','actor4.pt'}
    restored = tmp_path/'new_ssd'
    path = restore_latest(destination,restored)
    assert path.read_bytes() == bytes([4])*100
    row = json.loads((restored/'opponent_pool.json').read_text())['entries'][0]
    assert Path(row['checkpoint']).read_bytes() == bytes([4])*17
    assert json.loads((restored/'wandb_run.json').read_text())['id'] == 'resumable'
    path.write_bytes(b'newer local progress')
    assert restore_latest(destination,restored).read_bytes() == b'newer local progress'


def test_failed_copy_does_not_advance_committed_generation(tmp_path,monkeypatch):
    import cr_rl.training.fast_publication as module
    snapshot=tmp_path/'ssd';snapshot.mkdir()
    (snapshot/'opponent_pool.json').write_text('{"entries":[]}')
    (snapshot/'ppo_latest.pt').write_bytes(b'checkpoint')
    destination=tmp_path/'drive'
    publish_snapshot(snapshot,destination,1)
    def fail(*args): raise OSError('connection interrupted')
    monkeypatch.setattr(module,'copy_once',fail)
    import pytest
    with pytest.raises(OSError): publish_snapshot(snapshot,destination,2)
    assert json.loads((destination/'latest.json').read_text())['update'] == 1


def test_actor_export_published_with_same_committed_update(tmp_path):
    snapshot=tmp_path/'ssd';snapshot.mkdir()
    (snapshot/'opponent_pool.json').write_text('{"entries":[]}')
    (snapshot/'ppo_latest.pt').write_bytes(b'full checkpoint')
    (snapshot/'actor_export.pt').write_bytes(b'actor only')
    destination=tmp_path/'drive'
    publish_snapshot(snapshot,destination,12)
    receipt=json.loads((destination/'latest.json').read_text())
    assert receipt['update']==12
    assert receipt['actor_bytes']==10
    assert (destination/receipt['actor']).read_bytes()==b'actor only'


def test_macro_search_shard_survives_publication_and_ssd_restore(tmp_path):
    from cr_rl.training.async_publication import freeze_publication_files

    local = tmp_path / 'local'
    buffer = local / 'macro_search_buffer'
    buffer.mkdir(parents=True)
    (local / 'rollouts').mkdir()
    (local / 'opponent_pool.json').write_text('{"entries":[]}')
    (local / 'ppo_latest.pt').write_bytes(b'checkpoint')
    shard = buffer / 'combo_000001.npz'
    shard.write_bytes(b'admitted search trajectory')
    snapshot = freeze_publication_files(local, 12)
    shard.unlink()  # Search can evict the underlying local shard during upload.

    destination = tmp_path / 'drive'
    restored = tmp_path / 'new_ssd'
    publish_snapshot(snapshot, destination, 12)
    restore_latest(destination, restored)
    assert (restored / 'macro_search_buffer' / shard.name).read_bytes() == b'admitted search trajectory'
