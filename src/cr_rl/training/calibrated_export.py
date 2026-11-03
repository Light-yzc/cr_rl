"""Bake an evaluated timing correction into weights used for PPO initialization."""

import json
import math
from pathlib import Path

import torch

from cr_rl.training.persistence import atomic_json, file_sha256


def export_calibrated_bc(checkpoint, bias, calibration_report, destination):
    checkpoint, calibration_report, destination = map(
        Path, (checkpoint, calibration_report, destination)
    )
    if not math.isfinite(bias):
        raise ValueError("Calibration bias must be finite")
    if destination.exists() or checkpoint.resolve() == destination.resolve():
        raise ValueError("Choose a new calibration export destination")
    source_sha = file_sha256(checkpoint)
    report = json.loads(calibration_report.read_text(encoding="utf-8"))
    if report.get("provenance", {}).get("checkpoint_sha256") != source_sha:
        raise ValueError("Calibration report belongs to a different checkpoint")
    source = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if source.get("stage") != "bc" or source.get("model_architecture_version") != 2:
        raise ValueError("Expected an architecture-v2 BC checkpoint")
    if source.get("calibration", {}).get("baked_into_weights"):
        raise ValueError("Refusing to apply a timing bias twice")
    if report["provenance"].get("dataset_sha256") != source.get("dataset_sha256"):
        raise ValueError("Calibration dataset differs from its BC checkpoint")
    kept = (
        "stage",
        "model_architecture_version",
        "config",
        "dataset_metadata",
        "dataset_sha256",
        "epochs",
        "optimizer_steps",
        "training_settings",
    )
    exported = {key: source[key] for key in kept if key in source}
    exported["model"] = dict(source["model"])
    exported["model"]["wait_actor.bias"] = source["model"]["wait_actor.bias"].clone() + bias
    exported["weights_only"] = True
    exported["calibration"] = {
        "baked_into_weights": True,
        "wait_logit_bias": float(bias),
        "source_checkpoint_sha256": source_sha,
        "calibration_report_sha256": file_sha256(calibration_report),
        "scope": "Weights prepared for evaluation; promotion requires paired game evidence",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".writing")
    torch.save(exported, temporary)
    temporary.replace(destination)
    result = {"checkpoint_sha256": file_sha256(destination), **exported["calibration"]}
    atomic_json(destination.with_suffix(".calibration.json"), result)
    return result
