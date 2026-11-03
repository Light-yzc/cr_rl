import json

import pytest

from cr_rl.opponents import OpponentPool
from cr_rl.training import persistence


def make_run(path):
    path.mkdir()
    (path / "ppo_latest.pt").write_bytes(b"optimizer-and-weights")
    (path / "opponent.pt").write_bytes(b"frozen-policy")
    (path / "rollouts").mkdir()
    (path / "rollouts/game.npz").write_bytes(b"raw-two-seat-data")
    (path / "metrics.jsonl").write_text('{"update":1}\n')
    (path / "opponent_pool.json").write_text(
        json.dumps(
            {
                "uniform_mix": 0.1,
                "entries": [{"policy_id": 0, "checkpoint": "opponent.pt", "generation": 0}],
            }
        )
    )


def test_publish_restore_preserves_checkpoint_pool_and_raw_data(tmp_path):
    local, drive, restored = [tmp_path / x for x in ("local", "drive", "restored")]
    make_run(local)
    (local / "actor_export.pt").write_bytes(b"policy-only-weights")
    manifest = persistence.publish_update(local, drive, 1)
    assert manifest["update"] == 1
    assert any(row["path"] == "exports/update_000001.actor.pt" for row in manifest["files"])
    persistence.restore_run(drive, restored)
    assert (restored / "ppo_latest.pt").read_bytes() == b"optimizer-and-weights"
    assert (restored / "rollouts/game.npz").read_bytes() == b"raw-two-seat-data"
    assert (restored / "exports/update_000001.actor.pt").read_bytes() == b"policy-only-weights"
    assert OpponentPool.load(restored / "opponent_pool.json").entries[0].checkpoint == str(
        restored / "opponent.pt"
    )
    (drive / manifest["checkpoint"]).write_bytes(b"corrupted")
    with pytest.raises(IOError, match="integrity"):
        persistence.restore_run(drive, tmp_path / "bad_restore")


def test_failed_publish_does_not_advance_latest(tmp_path, monkeypatch):
    local, drive = [tmp_path / x for x in ("local", "drive")]
    make_run(local)
    persistence.publish_update(local, drive, 1)
    original = persistence.copy_verified

    def fail_checkpoint(source, destination):
        if destination.name == "update_000002.pt":
            raise OSError("simulated interrupted upload")
        return original(source, destination)

    monkeypatch.setattr(persistence, "copy_verified", fail_checkpoint)
    with pytest.raises(OSError, match="interrupted"):
        persistence.publish_update(local, drive, 2)
    assert json.loads((drive / "latest.json").read_text())["update"] == 1
