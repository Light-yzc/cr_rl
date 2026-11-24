import json

import pytest

from cr_rl.training.persistence import prune_backup, prune_local_opponents


def test_bounded_periodic_and_recent_restore_points(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    for update in (64, 128, 192, 256, 299, 300):
        for suffix in ("pt", "pool.json"):
            (root / f"update_{update:06d}.{suffix}").write_text("{}")
    prune_backup(tmp_path, 300, max_periodic=2)
    assert sorted(p.stem for p in root.glob("*.pt")) == [
        "update_000192",
        "update_000256",
        "update_000299",
        "update_000300",
    ]


def test_gc_preserves_current_rollback_active_and_external_opponents(tmp_path):
    local = tmp_path / "run"
    local.mkdir()
    checkpoints = local / "checkpoints"
    checkpoints.mkdir()
    files = [local / f"opponent_{i:06d}.pt" for i in range(4)]
    for file in files:
        file.write_bytes(b"weights")
    external = tmp_path / "opponent_external.pt"
    external.write_bytes(b"source archive")
    (local / "opponent_pool.json").write_text(
        json.dumps({"entries": [{"checkpoint": files[0].name}, {"checkpoint": str(external)}]})
    )
    (checkpoints / "update_000100.pool.json").write_text(
        json.dumps({"entries": [{"checkpoint": str(files[1])}]})
    )
    result = prune_local_opponents(local, active=[files[2]])
    assert result["removed_opponents"] == 1
    assert all(file.exists() for file in files[:3])
    assert not files[3].exists()
    assert external.exists()
    # Broken dependency metadata must never trigger collection.
    (local / "opponent_pool.json").write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        prune_local_opponents(local)
    assert files[0].exists()


def test_sparse_checkpoint_cadence_keeps_two_actual_restore_points(tmp_path):
    root=tmp_path/"checkpoints"
    root.mkdir()
    for update in (640,697,800,1600):
        (root/f"update_{update:06d}.pt").write_bytes(b"checkpoint")
    prune_backup(tmp_path,1600,recent=2,max_periodic=1)
    assert sorted(p.name for p in root.glob("*.pt")) == ["update_000800.pt","update_001600.pt"]
