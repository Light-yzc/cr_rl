"""Explicit weights-only BC to PPO handoff; PPO resume is a separate operation."""

from pathlib import Path

import torch

from cr_rl.models import ModelConfig
from cr_rl.training.persistence import file_sha256


def initialize_from_bc(model, checkpoint: Path, config: dict, device: str) -> dict:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("stage") != "bc" or state.get("model_architecture_version") != 2:
        raise ValueError("PPO initialization requires an architecture-v2 BC checkpoint")
    if ModelConfig(**state["config"]) != ModelConfig(**config):
        raise ValueError("BC/PPO model configuration mismatch (including context and vocabulary)")
    model.load_state_dict(state["model"], strict=True)
    metadata = state["dataset_metadata"]
    return {
        "checkpoint_sha256": file_sha256(checkpoint),
        "dataset_sha256": state["dataset_sha256"],
        "bc_epochs": state["epochs"],
        "backend_commit": metadata["backend_commit"],
        "data_version": metadata["data_version"],
        "content_sha": metadata["content_sha"],
        "level": metadata["level"],
        "decision_ticks": metadata["decision_ticks"],
        "mask_version": metadata["mask_version"],
        "optimizer_reset": True,
        "calibration": state.get("calibration"),
    }
