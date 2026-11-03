import shutil

from cr_rl.opponents import OpponentEntry, OpponentPool


def test_pool_snapshots_resolve_after_run_directory_moves(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    snapshot = original / "opponent.pt"
    snapshot.write_bytes(b"snapshot")
    pool = OpponentPool([OpponentEntry(0, str(snapshot), 0)])
    pool.save(original / "opponent_pool.json")
    relocated = tmp_path / "relocated"
    shutil.move(original, relocated)
    loaded = OpponentPool.load(relocated / "opponent_pool.json")
    assert loaded.entries[0].checkpoint == str(relocated / "opponent.pt")
    assert (relocated / "opponent.pt").read_bytes() == b"snapshot"
