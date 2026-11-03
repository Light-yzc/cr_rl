import json
import os

from scripts.train_bc import bc_tracking_options


def test_new_stage_does_not_inherit_previous_wandb_run(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "previous_stage")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    monkeypatch.setenv("WANDB_API_KEY", "test-placeholder")
    assert bc_tracking_options(tmp_path / "bc.wandb.json", True) == {}
    assert "WANDB_RUN_ID" not in os.environ
    assert "WANDB_RESUME" not in os.environ
    assert os.environ["WANDB_API_KEY"] == "test-placeholder"


def test_preupdate_restart_resumes_exact_stage_run(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "wrong_parent_id")
    path = tmp_path / "bc.wandb.json"
    path.write_text(json.dumps(dict(id="current_stage")))
    assert bc_tracking_options(path, True) == dict(id="current_stage", resume="allow")
