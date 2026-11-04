import json

import pytest

from cr_rl.training import incremental_backup as module
from cr_rl.training.persistence import restore_run


def setup_run(root):
    root.mkdir()
    (root / "rollouts").mkdir()
    (root / "ppo_latest.pt").write_bytes(b"state")
    (root / "opponent.pt").write_bytes(b"anchor")
    (root / "opponent_pool.json").write_text(
        json.dumps(dict(entries=[dict(policy_id=0, checkpoint="opponent.pt", generation=0)]))
    )
    for suffix in ("npz", "npz.json"):
        (root / "rollouts" / f"rollout_000001.{suffix}").write_bytes(b"first")


def add_rollout(root, generation):
    for suffix in ("npz", "npz.json"):
        (root / "rollouts" / f"rollout_{generation:06d}.{suffix}").write_bytes(b"next")


def test_incremental_skips_old_file_reads_but_restore_checks_all_hashes(tmp_path, monkeypatch):
    local, backup = tmp_path / "run", tmp_path / "drive"
    setup_run(local)
    module.publish_incremental(local, backup, 1)
    old = module.file_sha256
    reads = []

    def checked(path):
        reads.append(str(path))
        return old(path)

    monkeypatch.setattr(module, "file_sha256", checked)
    add_rollout(local, 2)
    manifest = module.publish_incremental(local, backup, 2)
    assert not any("rollout_000001" in path for path in reads)
    assert len([r for r in manifest["files"] if r["path"].startswith("rollouts/")]) == 4
    restore_run(backup, tmp_path / "restored")
    (backup / "rollouts/rollout_000001.npz").write_bytes(b"corrupt")
    with pytest.raises(OSError, match="integrity"):
        restore_run(backup, tmp_path / "invalid")


def test_periodic_audit_and_immutable_source_guard_prevent_advancing_latest(tmp_path):
    local, backup = tmp_path / "run", tmp_path / "drive"
    setup_run(local)
    module.publish_incremental(local, backup, 1, audit_every=2)
    add_rollout(local, 2)
    (backup / "rollouts/rollout_000001.npz").write_bytes(b"corrupt")
    with pytest.raises(OSError, match="audit failed"):
        module.publish_incremental(local, backup, 2, audit_every=2)
    assert json.loads((backup / "latest.json").read_text())["update"] == 1
    (local / "opponent.pt").write_bytes(b"changed-anchor")
    with pytest.raises(ValueError, match="Immutable"):
        module.publish_incremental(local, backup, 2)


def test_receipt_cannot_be_reused_for_another_destination_or_skipped_generation(tmp_path):
    local, backup = tmp_path / "run", tmp_path / "drive"
    setup_run(local)
    module.publish_incremental(local, backup, 1)
    with pytest.raises(ValueError, match="different backup"):
        module.publish_incremental(local, tmp_path / "other", 2)
    with pytest.raises(ValueError, match="one generation"):
        module.publish_incremental(local, backup, 3)
