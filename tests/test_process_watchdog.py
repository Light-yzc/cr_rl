import json

import pytest

from cr_rl.training.process_watchdog import ProgressWatchdog, restore_local_pair


def test_stall_detection_resets_when_existing_progress_changes(tmp_path):
    path=tmp_path/'progress.jsonl'
    watch=ProgressWatchdog([path],30,now=0)
    assert not watch.stalled(now=29)
    path.write_text('one')
    assert not watch.stalled(now=31)
    assert not watch.stalled(now=60)
    assert watch.stalled(now=61)
    path.write_text('two rows')
    assert not watch.stalled(now=62)


def test_stall_detection_resets_on_validation_directory_wave_activity(tmp_path):
    progress = tmp_path / 'progress.jsonl'
    val_dir = tmp_path / 'validation'
    watch = ProgressWatchdog([progress, val_dir], 30, now=0)
    assert not watch.stalled(now=25)
    u_dir = val_dir / 'update_00001590' / 'vs_bc'
    u_dir.mkdir(parents=True)
    (u_dir / 'match_000.json').write_text('{}')
    assert not watch.stalled(now=28)
    assert not watch.stalled(now=55)
    (u_dir / 'match_001.json').write_text('{"win": true}')
    assert not watch.stalled(now=57)
    assert not watch.stalled(now=85)
    assert watch.stalled(now=88)


def test_recovery_pairs_weights_with_original_pool_not_newer_live_pool(tmp_path):
    checkpoints=tmp_path/'checkpoints';checkpoints.mkdir()
    opponent=tmp_path/'frozen.pt';opponent.write_bytes(b'actor')
    pool={'entries':[{'checkpoint':str(opponent)}]}
    (checkpoints/'update_000980.pt').write_bytes(b'complete checkpoint')
    (checkpoints/'update_000980.pool.json').write_text(json.dumps(pool))
    (checkpoints/'update_000981.pt').write_bytes(b'incomplete pair')
    (tmp_path/'ppo_latest.pt').write_bytes(b'older')
    (tmp_path/'opponent_pool.json').write_text('{"entries":[]}')
    assert restore_local_pair(tmp_path)==980
    assert (tmp_path/'ppo_latest.pt').read_bytes()==b'complete checkpoint'
    assert json.loads((tmp_path/'opponent_pool.json').read_text())==pool
    opponent.unlink()
    with pytest.raises(RuntimeError,match='dependency'):
        restore_local_pair(tmp_path)
