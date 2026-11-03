from pathlib import Path

import pytest
import torch

from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.initialization import initialize_from_bc


@pytest.mark.parametrize("upgraded", [False, True])
def test_bc_handoff_preserves_policy_and_starts_fresh_optimizer(tmp_path: Path, upgraded):
    cfg = ModelConfig(
        **{
            **ModelConfig.preset("tiny").__dict__,
            "tile_query_residual": upgraded,
            "tile_head_fp32": upgraded,
        }
    )
    source = ActorCritic(cfg)
    state = {
        "stage": "bc",
        "model_architecture_version": 2,
        "config": cfg.__dict__,
        "model": source.state_dict(),
        "optimizer": {"should_not_be_loaded": True},
        "dataset_sha256": "test",
        "epochs": 2,
        "dataset_metadata": {
            "backend_commit": "pin",
            "data_version": "16",
            "content_sha": "tables",
            "level": 11,
            "decision_ticks": 5,
            "mask_version": 2,
        },
    }
    path = tmp_path / "bc.pt"
    if not upgraded:
        state["config"] = {
            k: v
            for k, v in cfg.__dict__.items()
            if k not in ("tile_query_residual", "tile_head_fp32")
        }
    torch.save(state, path)
    target = ActorCritic(cfg)
    optimizer = torch.optim.AdamW(target.parameters())
    provenance = initialize_from_bc(target, path, cfg.__dict__, "cpu")
    assert not optimizer.state
    assert provenance["optimizer_reset"]
    assert all(
        torch.equal(value, target.state_dict()[key]) for key, value in source.state_dict().items()
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        initialize_from_bc(target, path, {**cfg.__dict__, "context_len": 128}, "cpu")
    state["stage"] = "ppo"
    torch.save(state, path)
    with pytest.raises(ValueError, match="BC checkpoint"):
        initialize_from_bc(target, path, cfg.__dict__, "cpu")
