import json

import pytest

from cr_rl.training.persistence import file_sha256
from scripts.finish_and_release import verified_results


def setup(tmp_path):
    run, backup = tmp_path / "run", tmp_path / "drive"
    run.mkdir()
    (run / "protocol.json").write_text(
        json.dumps(
            {
                "parent": {"optimizer_steps": 4096},
                "new_updates_per_arm": 1000,
                "arms": {"repaired": {}, "baseline": {}},
            }
        )
    )
    status = {"phase": "completed", "pid": 0, "child_pid": 0}
    (run / "status.json").write_text(json.dumps(status))
    for name in ["repaired", "baseline"]:
        directory = backup / name / "bc"
        directory.mkdir(parents=True)
        checkpoint = directory / "bc_latest.pt"
        checkpoint.write_bytes(b"verified model state")
        (directory / "latest.json").write_text(
            json.dumps(
                {
                    "optimizer_steps": 5096,
                    "path": checkpoint.name,
                    "bytes": checkpoint.stat().st_size,
                    "sha256": file_sha256(checkpoint),
                }
            )
        )
    (run / "comparison.json").write_text('{"completed":true}')
    (backup / "comparison.json").write_bytes((run / "comparison.json").read_bytes())
    return run, backup, status


def test_release_requires_both_final_backups_and_refuses_active_or_corrupt_runs(tmp_path):
    run, backup, status = setup(tmp_path)
    with pytest.raises(ValueError, match="not finished"):
        verified_results(run, backup, {**status, "phase": "training"})
    assert set(verified_results(run, backup, status)) == {"baseline", "repaired"}
    (backup / "baseline/bc/bc_latest.pt").write_bytes(b"corrupt model")
    with pytest.raises(ValueError, match="checksum"):
        verified_results(run, backup, status)


def test_failed_trial_preserves_last_local_checkpoint_before_release(tmp_path):
    run, backup, status = setup(tmp_path)
    (run / "repaired").mkdir()
    checkpoint = run / "repaired/bc.pt"
    checkpoint.write_bytes(b"last stable local checkpoint")
    result = verified_results(run, backup, {**status, "phase": "failed"})
    saved = backup / "repaired/recovery" / result["repaired"]["path"]
    assert saved.read_bytes() == checkpoint.read_bytes()
