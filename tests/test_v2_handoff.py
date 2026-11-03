import copy

import pytest

from scripts.run_v2_handoff import audit_updates


def test_handoff_requires_real_actor_critic_games_and_continuous_episodes():
    rows = [
        {
            "optimizer_steps": actor,
            "critic_optimizer_steps": 2,
            "terminal_games": 1,
            "actor_enabled": actor > 0,
            "faulted_samples": 0,
            "episode_steps_at_update_start": [index * 10],
            "episode_steps_at_update_end": [(index + 1) * 10],
        }
        for index, actor in enumerate([0, 0, 3])
    ]
    assert audit_updates(rows)["actor_steps"] == 3
    assert audit_updates(rows)["continuous_boundaries"] == 2
    broken = copy.deepcopy(rows)
    broken[2]["episode_steps_at_update_start"] = [0]
    with pytest.raises(ValueError, match="continuity"):
        audit_updates(broken)
    with pytest.raises(ValueError, match="actor has not updated"):
        audit_updates(rows[:2])
    broken = copy.deepcopy(rows)
    broken[2]["faulted_samples"] = 1
    with pytest.raises(ValueError, match="Faulted"):
        audit_updates(broken)


def test_completed_staged_checkpoint_is_resolved_inside_run(tmp_path):
    from scripts.run_v2_handoff import completed_bc_checkpoint

    stage = tmp_path / "stage_0003"
    stage.mkdir()
    checkpoint = stage / "bc.pt"
    checkpoint.write_bytes(b"weights")
    assert completed_bc_checkpoint(tmp_path, {"final_checkpoint": str(checkpoint)}) == checkpoint
    with pytest.raises(ValueError, match="outside"):
        completed_bc_checkpoint(tmp_path, {"final_checkpoint": str(tmp_path.parent / "other.pt")})
    checkpoint.unlink()
    with pytest.raises(ValueError, match="missing"):
        completed_bc_checkpoint(tmp_path, {"final_checkpoint": str(checkpoint)})
