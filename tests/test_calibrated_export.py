import json

import pytest
import torch

from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.calibrated_export import export_calibrated_bc
from cr_rl.training.initialization import initialize_from_bc
from cr_rl.training.persistence import file_sha256


def test_calibrated_weights_keep_ppo_initialization_exact_and_prevent_double_bias(tmp_path):
    cfg = ModelConfig.preset("tiny")
    model = ActorCritic(cfg)
    source, report, exported = (tmp_path / name for name in ("bc.pt", "fit.json", "calibrated.pt"))
    metadata = {
        "backend_commit": "pin",
        "data_version": "16",
        "content_sha": "tables",
        "level": 11,
        "decision_ticks": 5,
        "mask_version": 2,
    }
    torch.save(
        {
            "stage": "bc",
            "model_architecture_version": 2,
            "config": cfg.__dict__,
            "model": model.state_dict(),
            "dataset_metadata": metadata,
            "dataset_sha256": "dataset",
            "epochs": 1,
            "optimizer": {"old": True},
        },
        source,
    )
    source_hash = file_sha256(source)
    report.write_text(
        json.dumps({"provenance": {"checkpoint_sha256": source_hash, "dataset_sha256": "dataset"}})
    )
    record = export_calibrated_bc(source, 1.5, report, exported)
    target = ActorCritic(cfg)
    provenance = initialize_from_bc(target, exported, cfg.__dict__, "cpu")
    assert provenance["calibration"]["wait_logit_bias"] == 1.5
    assert provenance["checkpoint_sha256"] == record["checkpoint_sha256"]
    for name, original in model.state_dict().items():
        expected = original + 1.5 if name == "wait_actor.bias" else original
        torch.testing.assert_close(target.state_dict()[name], expected, rtol=0, atol=0)
    state = torch.load(exported, weights_only=False)
    assert "optimizer" not in state and state["weights_only"]
    assert file_sha256(source) == source_hash
    report.write_text(
        json.dumps(
            {
                "provenance": {
                    "checkpoint_sha256": file_sha256(exported),
                    "dataset_sha256": "dataset",
                }
            }
        )
    )
    with pytest.raises(ValueError, match="twice"):
        export_calibrated_bc(exported, 1.5, report, tmp_path / "double.pt")
    with pytest.raises(ValueError, match="finite"):
        export_calibrated_bc(source, float("nan"), report, tmp_path / "invalid.pt")
