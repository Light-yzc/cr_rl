"""Continuous replay/seat BC for the independent architecture-v3 state policy."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.data.state_validation import sample_validation_games
from cr_rl.models import ModelConfig
from cr_rl.models.state_policy import StateActorCritic, state_model_from_checkpoint
from cr_rl.models.state_temporal import StateTemporalConfig
from cr_rl.training.lr_schedule import WarmupCosineLR, cosine_config, remaining_episode_updates
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256
from cr_rl.training.state_bc import (
    ReplaySequenceDataset,
    check_ram_budget,
    fixed_replay_split,
    load_state_checkpoint,
    restore_state_rng,
    run_state_episode,
    save_state_checkpoint,
)
from cr_rl.training.state_bc_parallel import run_state_episode_group
from cr_rl.training.state_bc_metrics import BCMetricWindow
from cr_rl.training.placement_stage import (
    STAGE_FORMAT, initialize_placement_stage, set_placement_training_mode,
)
from cr_rl.training.placement_unfreeze import (
    arm_unfreeze, configure_training_mode, optimizer_parameters, maybe_unfreeze,
)


def prepare_event(stage, message, **details):
    print(
        json.dumps(
            {"phase": "state_bc_prepare", "stage": stage, "message": message, **details},
            ensure_ascii=False,
        ),
        flush=True,
    )


def validation_progress(epoch, completed, total, initial_validation):
    print(
        json.dumps(
            {
                "phase": "state_bc_validation_progress",
                "epoch": epoch,
                "episodes_completed": completed,
                "episodes_total": total,
                "initial_validation": initial_validation,
            }
        ),
        flush=True,
    )


def copy_artifact(source, destination, *, trust_data_files=False):
    """Keep strict backups by default; trusted weight transfers are single pass."""
    if not trust_data_files or Path(source).suffix != ".pt":
        return copy_verified(source, destination)
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".uploading")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": destination.name,
        "bytes": destination.stat().st_size,
        "content_verification": "trusted_storage_transfer",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--trust-data-files",
        action="store_true",
        help="Trust storage transfers; no extra NPZ/checkpoint hash passes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Run directory; resumable latest.pt and inference best.pt",
    )
    parser.add_argument("--cache-dir", type=Path, help="One-replay decompression scratch")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument(
        "--init-from-bc", type=Path, help="Copy compatible entity/action weights only"
    )
    initialization.add_argument(
        "--init-from-state",
        type=Path,
        help="Start a new dataset stage with all architecture-v3 weights",
    )
    initialization.add_argument(
        "--upgrade-placement-from", type=Path,
        help="Start a new frozen-trunk card-prior/CNN stage from the latest same-dataset v3 checkpoint",
    )
    parser.add_argument(
        "--placement-prior", type=Path,
        help="Optional exact-dataset/train-split-bound prior JSON; otherwise build once from train games",
    )
    parser.add_argument(
        "--placement-training-only", action=argparse.BooleanOptionalAction, default=None,
        help="Freeze original trunk/actor and train the learned placement decoder only (upgrade default)",
    )
    parser.add_argument('--unfreeze-after-checkpoints', type=int, default=None)
    parser.add_argument('--unfreeze-learning-rate', type=float, default=1e-5)
    parser.add_argument(
        "--init-state-dataset-manifest",
        type=Path,
        help="Parent manifest+metadata sidecar bound to the state checkpoint",
    )
    parser.add_argument("--model", choices=("tiny", "production"), default="production")
    parser.add_argument("--temporal-width", type=int)
    parser.add_argument("--temporal-heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--pattern", default=None, help="Comma-separated kda/gqa layer pattern")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--frame-microbatch", type=int, default=8)
    parser.add_argument("--episode-batch-size", type=int, default=1)
    parser.add_argument(
        "--allow-episode-batch-transition",
        action="store_true",
        help="Explicitly preserve weights/Adam while changing episode update batch",
    )
    parser.add_argument("--freeze-frame", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame-encoder-fp32", action="store_true")
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--float32-matmul-precision", choices=("highest", "high"), default="highest"
    )
    parser.add_argument("--defer-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--max-episodes", type=int, default=0, help="Bound processed training seats this invocation"
    )
    parser.add_argument("--max-validation-episodes", type=int, default=0)
    parser.add_argument(
        "--max-validation-games",
        type=int,
        default=0,
        help="Randomly sample this many validation games, evaluating both seats",
    )
    parser.add_argument("--validation-every-episodes", type=int, default=16)
    parser.add_argument("--initial-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many complete epochs without validation improvement; 0 disables",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss decrease for the complete-epoch patience counter",
    )
    parser.add_argument("--checkpoint-every-episodes", type=int, default=8)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default=None,
        help="Default constant; resume inherits an existing cosine schedule",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Successful warmup updates, default 350 for a new cosine schedule",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=None,
        help="Cosine floor, default 3e-6 for a new schedule",
    )
    parser.add_argument(
        "--schedule-total-steps",
        type=int,
        default=None,
        help="Successful-update horizon; default all remaining epoch seats",
    )
    parser.add_argument(
        "--start-lr-schedule",
        action="store_true",
        help="Explicitly hand off a constant resume checkpoint to cosine once",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--position-coef", type=float, default=0.5)
    parser.add_argument("--tile-label-sigma", type=float, default=1.0)
    parser.add_argument("--timing-objective", choices=("natural", "balanced"), default="natural")
    parser.add_argument("--timing-target-action-prior", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-registry", type=Path)
    parser.add_argument("--decision-ticks", type=int, default=8)
    parser.add_argument(
        "--allow-smoke-data",
        action="store_true",
        help="Permit explicitly bounded diagnostic runs on incomplete reconstruction",
    )
    parser.add_argument(
        "--ram-limit-gib",
        type=float,
        default=3,
        help="Local default guard; explicitly raise for large Colab RAM",
    )
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument('--metrics-every-updates',type=int,default=1,
                        help='Count-weighted metric window; does not change the training loss or updates')
    args = parser.parse_args(argv)
    if args.metrics_every_updates<1:
        parser.error('metrics-every-updates must be positive')
    if args.unfreeze_after_checkpoints is not None and args.unfreeze_after_checkpoints < 1:
        parser.error('unfreeze-after-checkpoints must be positive')
    if not math.isfinite(args.unfreeze_learning_rate) or args.unfreeze_learning_rate <= 0:
        parser.error('unfreeze-learning-rate must be finite and positive')
    if args.init_from_state and not args.init_state_dataset_manifest:
        parser.error("--init-from-state requires --init-state-dataset-manifest")
    if args.init_state_dataset_manifest and not (args.init_from_state or args.upgrade_placement_from):
        parser.error("--init-state-dataset-manifest requires a state transfer or placement upgrade")
    if args.placement_training_only is False:
        parser.error("This placement upgrade supports frozen-trunk training only")
    if args.placement_prior and not args.upgrade_placement_from:
        parser.error("--placement-prior is only used when starting --upgrade-placement-from")
    if args.placement_training_only and not (args.upgrade_placement_from or args.resume):
        parser.error("--placement-training-only requires a placement upgrade or its own stage resume")
    if (
        min(
            args.chunk_size,
            args.frame_microbatch,
            args.episode_batch_size,
            args.epochs,
            args.decision_ticks,
            args.cpu_threads,
        )
        < 1
    ):
        parser.error("Chunk/frame size, epochs, interval, and threads must be positive")
    if (
        min(
            args.max_episodes,
            args.max_validation_episodes,
            args.max_validation_games,
            args.validation_every_episodes,
            args.checkpoint_every_episodes,
            args.ram_limit_gib,
            args.tile_label_sigma,
            args.early_stopping_patience,
        )
        < 0
    ):
        parser.error("Bounds, intervals, RAM guard, and tile sigma cannot be negative")
    if not math.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
        parser.error("early-stopping-min-delta must be finite and nonnegative")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.position_coef < 0:
        parser.error("Invalid learning rate, gradient clip, or position coefficient")
    if not 0 < args.timing_target_action_prior < 1:
        parser.error("Timing target prior must be in (0,1)")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("warmup steps cannot be negative")
    if args.schedule_total_steps is not None and args.schedule_total_steps <= 0:
        parser.error("schedule total steps must be positive")
    if args.min_learning_rate is not None and not 0 <= args.min_learning_rate <= args.learning_rate:
        parser.error("minimum learning rate must be between zero and peak learning rate")
    if args.start_lr_schedule and (not args.resume or args.lr_schedule == "constant"):
        parser.error("--start-lr-schedule requires --resume and a cosine schedule")
    schedule_options = any(
        value is not None
        for value in (args.warmup_steps, args.min_learning_rate, args.schedule_total_steps)
    )
    if args.lr_schedule == "constant" and schedule_options:
        parser.error("Warmup/cosine options cannot be used with a constant schedule")
    if schedule_options and not args.resume and args.lr_schedule != "cosine":
        parser.error("Warmup/cosine options require --lr-schedule cosine")
    return args


class ValidationControl:
    """Resume-stable model selection; patience advances only at complete epochs.

    Mid-epoch and initial probes can select inference weights, but cannot
    consume patience. The checkpoint, rather than a possibly newer sidecar,
    is authoritative when resuming an older saved cursor.
    """

    def __init__(self, scope, *, patience=0, min_delta=0.0, saved=None):
        self.scope = deepcopy(scope)
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.best_complete_loss = None
        self.bad_complete_epochs = 0
        self.last_complete_epoch = None
        self.last_validation_step = None
        self.last_validation_metrics = None
        if saved is not None:
            if saved.get("format") != "crrl-state-bc-validation-v1" or saved.get("scope") != scope:
                raise ValueError("Resume validation scope differs; use a new dataset stage")
            if saved.get("min_delta") != min_delta:
                raise ValueError("Resume early-stopping minimum delta differs")
            for name in (
                "best",
                "best_complete_loss",
                "bad_complete_epochs",
                "last_complete_epoch",
                "last_validation_step",
                "last_validation_metrics",
            ):
                setattr(self, name, deepcopy(saved[name]))

    def observe(self, metrics, *, epoch, optimizer_steps, complete_epoch=False):
        loss = float(metrics["loss"])
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        improved = self.best is None or loss < self.best["loss"]
        if improved:
            self.best = {
                **metrics,
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "complete_epoch": complete_epoch,
            }
        self.last_validation_step = optimizer_steps
        self.last_validation_metrics = deepcopy(metrics)
        if complete_epoch and epoch != self.last_complete_epoch:
            if self.last_complete_epoch is not None and epoch < self.last_complete_epoch:
                raise ValueError("Complete validation epochs cannot move backwards")
            if self.best_complete_loss is None or loss < self.best_complete_loss - self.min_delta:
                self.best_complete_loss = loss
                self.bad_complete_epochs = 0
            else:
                self.bad_complete_epochs += 1
            self.last_complete_epoch = epoch
        return improved

    @property
    def should_stop(self):
        return bool(self.patience and self.bad_complete_epochs >= self.patience)

    def state_dict(self):
        return {
            "format": "crrl-state-bc-validation-v1",
            "scope": deepcopy(self.scope),
            "patience": self.patience,
            "min_delta": self.min_delta,
            **{
                name: deepcopy(getattr(self, name))
                for name in (
                    "best",
                    "best_complete_loss",
                    "bad_complete_epochs",
                    "last_complete_epoch",
                    "last_validation_step",
                    "last_validation_metrics",
                )
            },
        }


def save_best_weights(
    path, model, *, dataset_fingerprint, split, validation, trust_data_files=False
):
    """Atomic inference-only selection; never replaces resumable latest.pt."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "format": "crrl-state-bc-weights-v1",
        "architecture": "state_actor_critic",
        "resume_supported": False,
        **model.checkpoint_config(),
        "model": model.state_dict(),
        "dataset_fingerprint": dataset_fingerprint,
        "split": {str(key): value for key, value in split.items()},
        "progress": {"optimizer_steps": validation["optimizer_steps"]},
        "validation": deepcopy(validation),
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return None if trust_data_files else file_sha256(path)


def resolve_lr_schedule(args, definition, *, state=None, train_seats, migration=None):
    """Upgrade only the schedule definition; all other resume fields stay exact."""
    saved = state["training_config"] if state else {}
    existing = saved.get("lr_schedule")
    previous_batch = saved.get("episode_batch_size", 1)
    batch_transition = state is not None and previous_batch != args.episode_batch_size
    if batch_transition:
        if not args.allow_episode_batch_transition:
            raise ValueError("Changing episode batch requires --allow-episode-batch-transition")
        original = {
            k: v for k, v in saved.items() if k not in ("lr_schedule", "episode_batch_size")
        }
        requested = {k: v for k, v in definition.items() if k != "episode_batch_size"}
        if original != requested or not existing:
            raise ValueError("Episode batch transition cannot change other training objectives")
        progress = state["progress"]
        remaining = math.ceil(
            (train_seats - progress["next_episode"]) / args.episode_batch_size
        ) + (args.epochs - progress["epoch"] - 1) * math.ceil(train_seats / args.episode_batch_size)
        peak = float(state["optimizer"]["param_groups"][0]["lr"])
        config = cosine_config(
            peak_lr=peak,
            min_lr=min(existing["min_lr"], peak),
            warmup_steps=0,
            total_steps=remaining,
            anchor_step=progress["optimizer_steps"],
            total_epochs=args.epochs,
            start_progress=progress,
            train_seats=train_seats,
            transition={
                "kind": "episode_batch_transition",
                "previous_batch_size": previous_batch,
                "episode_batch_size": args.episode_batch_size,
                "original_warmup_steps": existing["warmup_steps"],
                "model_optimizer_rng_preserved": True,
                "exact_training_equivalence": False,
            },
        )
        state["lr_schedule"] = WarmupCosineLR(config).state_dict()
        definition["lr_schedule"] = config
        return config
    if state:
        original_definition = {key: value for key, value in saved.items() if key != "lr_schedule"}
        if original_definition != definition:
            raise ValueError(
                "Resume training configuration differs; start a distinct experimental run"
            )
    mode = args.lr_schedule or ("cosine" if existing else "constant")
    if args.start_lr_schedule:
        mode = "cosine"
    if existing:
        if args.start_lr_schedule:
            raise ValueError("Learning-rate handoff already occurred; warmup cannot restart")
        if mode != "cosine" or args.epochs != existing["total_epochs"]:
            raise ValueError(
                "Resume must preserve the saved cosine schedule and total epoch horizon"
            )
        for requested, field in (
            (args.warmup_steps, "warmup_steps"),
            (args.min_learning_rate, "min_lr"),
            (args.schedule_total_steps, "total_steps"),
        ):
            allowed_original_warmup = (
                field == "warmup_steps"
                and existing.get("transition", {}).get("kind") == "episode_batch_transition"
                and requested == existing["transition"]["original_warmup_steps"]
            )
            if (
                requested is not None
                and requested != existing[field]
                and not allowed_original_warmup
            ):
                raise ValueError("Resume learning-rate schedule parameters differ")
        if existing["train_seats"] != train_seats or state.get("lr_schedule") is None:
            raise ValueError("Checkpoint lacks a consistent saved learning-rate schedule state")
        definition["lr_schedule"] = deepcopy(existing)
        return definition["lr_schedule"]
    if mode == "constant":
        if any(
            value is not None
            for value in (args.warmup_steps, args.min_learning_rate, args.schedule_total_steps)
        ):
            raise ValueError("Cosine options require an explicit cosine schedule")
        return None  # Keep old constant checkpoint definitions byte-for-byte equivalent.
    if state and not args.start_lr_schedule:
        raise ValueError(
            "A constant checkpoint needs explicit --start-lr-schedule to change LR policy"
        )
    progress = state["progress"] if state else {"epoch": 0, "next_episode": 0}
    anchor = (
        state["progress"]["optimizer_steps"]
        if state
        else migration.get("optimizer_steps_base", 0)
        if migration
        else 0
    )
    remaining = remaining_episode_updates(args.epochs, progress, train_seats)
    if args.episode_batch_size > 1:
        remaining = math.ceil(
            (train_seats - progress["next_episode"]) / args.episode_batch_size
        ) + (args.epochs - progress["epoch"] - 1) * math.ceil(train_seats / args.episode_batch_size)
    transition = None
    if state:
        transition = {
            "kind": "constant_to_warmup_cosine",
            "parent_checkpoint_sha256": None if args.trust_data_files else file_sha256(args.resume),
            "parent_optimizer_steps": anchor,
            "model_optimizer_rng_preserved": True,
            "prior_learning_rate": saved["learning_rate"],
        }
    config = cosine_config(
        peak_lr=args.learning_rate,
        min_lr=3e-6 if args.min_learning_rate is None else args.min_learning_rate,
        warmup_steps=350 if args.warmup_steps is None else args.warmup_steps,
        total_steps=remaining if args.schedule_total_steps is None else args.schedule_total_steps,
        anchor_step=anchor,
        total_epochs=args.epochs,
        start_progress=progress,
        train_seats=train_seats,
        transition=transition,
    )
    definition["lr_schedule"] = config
    return config


def vocabulary_sha256(names):
    return hashlib.sha256(
        json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def initialize_state_stage(args, dataset):
    """Transfer all v3 weights; optimizer, RNG, epoch and memories stay fresh."""
    source = torch.load(args.init_from_state, map_location="cpu", weights_only=False, mmap=True)
    if source.get("model_architecture_version") != 3:
        raise ValueError("--init-from-state requires an architecture-v3 checkpoint")
    if source.get("learned_placement_definition") is not None:
        raise ValueError(
            "A learned placement stage must resume its exact dataset; "
            "new-dataset placement-prior rebinding is not implemented"
        )
    manifest_path = args.init_state_dataset_manifest.resolve()
    parent_fingerprint = file_sha256(manifest_path)
    if parent_fingerprint != source.get("dataset_fingerprint"):
        raise ValueError("Parent dataset manifest fingerprint differs from the source checkpoint")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = (manifest_path.parent / manifest["metadata"]).resolve()
    if not metadata_path.is_relative_to(manifest_path.parent):
        raise ValueError("Parent metadata must stay inside its sidecar directory")
    if file_sha256(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError("Parent dataset metadata checksum differs")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("decision_ticks") != source.get("decision_ticks"):
        raise ValueError("Parent checkpoint and bound metadata decision grid differ")
    cfg = ModelConfig(**source["config"])
    parent_schema, target_schema = metadata["schema"], dataset.metadata["schema"]
    for field in (
        "header_fields",
        "entity_fields",
        "header_ints",
        "entity_ints",
        "seat_ints",
        "max_entities",
    ):
        if parent_schema.get(field) != target_schema.get(field):
            raise ValueError(f"State transfer dataset schema differs: {field}")
    if (
        cfg.header_dim != dataset.schema.header_ints
        or cfg.entity_feature_dim != dataset.schema.entity_ints
        or tuple(cfg.header_fields) != dataset.schema.header_fields
        or tuple(cfg.entity_fields) != dataset.schema.entity_fields
        or cfg.max_entities != dataset.schema.max_entities
        or cfg.action_count != 2307
    ):
        raise ValueError("State source ModelConfig differs from the dataset representation")
    parent_names = metadata["vocabulary"]["names"]
    names = dataset.metadata["vocabulary"]["names"]
    if parent_names != names or cfg.vocab_size != len(names):
        raise ValueError("State source card vocabulary differs from the new dataset")
    temporal = StateTemporalConfig(**source["temporal_config"])
    if (
        cfg.temporal_dim != temporal.width
        or cfg.temporal_heads != temporal.heads
        or cfg.temporal_layers != len(temporal.pattern)
    ):
        raise ValueError("State source ModelConfig and temporal configuration disagree")
    for argument, actual in (
        (args.temporal_width, temporal.width),
        (args.temporal_heads, temporal.heads),
        (args.kv_heads, temporal.kv_heads),
    ):
        if argument is not None and argument != actual:
            raise ValueError(
                "Explicit temporal configuration differs from the transferred state model"
            )
    if args.pattern is not None and tuple(args.pattern.split(",")) != tuple(temporal.pattern):
        raise ValueError("Explicit temporal pattern differs from the transferred state model")
    base_step = source.get("progress", {}).get("optimizer_steps")
    if not isinstance(base_step, int) or isinstance(base_step, bool) or base_step < 0:
        raise ValueError("State source has no valid optimizer step counter")
    model = state_model_from_checkpoint(source, args.device)
    model.decision_ticks = args.decision_ticks
    model.frame_microbatch = args.frame_microbatch
    report = {
        "mode": "architecture_v3_weights_only_new_dataset_stage",
        "exact_resume": False,
        "all_model_tensors_loaded": True,
        "temporal_weights_preserved": True,
        "optimizer_migrated": False,
        "scaler_migrated": False,
        "rng_restored": False,
        "memory_reset": True,
        "stage_epoch_start": 0,
        "optimizer_steps_base": base_step,
        "parent_weight_sha256": None
        if args.trust_data_files
        else file_sha256(args.init_from_state),
        "parent_dataset_fingerprint": parent_fingerprint,
        "parent_metadata_sha256": manifest["metadata_sha256"],
        "new_dataset_fingerprint": dataset.fingerprint,
        "parent_decision_ticks": source["decision_ticks"],
        "decision_ticks": args.decision_ticks,
        "parent_vocabulary_sha256": vocabulary_sha256(parent_names),
        "new_vocabulary_sha256": vocabulary_sha256(names),
    }
    del source
    gc.collect()
    check_ram_budget(args.ram_limit_gib)
    return model, report


def build_model(args, dataset):
    if args.init_from_state:
        return initialize_state_stage(args, dataset)
    source = (
        torch.load(args.init_from_bc, map_location="cpu", weights_only=False, mmap=True)
        if args.init_from_bc
        else None
    )
    if source is not None:
        cfg = ModelConfig(**source["config"])
        source_vocabulary = source.get("dataset_metadata", {}).get("vocabulary", {}).get("names")
        target_vocabulary = dataset.metadata["vocabulary"]["names"]
        if cfg.vocab_size != len(target_vocabulary) or (
            source_vocabulary is not None and source_vocabulary != target_vocabulary
        ):
            raise ValueError("BC source card vocabulary differs from the rebuilt dataset")
        if (
            cfg.header_dim != dataset.schema.header_ints
            or cfg.entity_feature_dim != dataset.schema.entity_ints
            or tuple(cfg.header_fields) != dataset.schema.header_fields
            or tuple(cfg.entity_fields) != dataset.schema.entity_fields
        ):
            raise ValueError("BC source representation differs from the rebuilt dataset schema")
    else:
        cfg = ModelConfig.preset(args.model)
        cfg = replace(
            cfg,
            header_dim=dataset.schema.header_ints,
            entity_feature_dim=dataset.schema.entity_ints,
            max_entities=dataset.schema.max_entities,
            vocab_size=len(dataset.metadata["vocabulary"]["names"]),
        )
        cfg = cfg.for_v2(dataset.schema, "dynamic4")
    tiny = args.model == "tiny" and source is None
    width = args.temporal_width or (32 if tiny else 768)
    heads = args.temporal_heads or (2 if tiny else 12)
    kv_heads = args.kv_heads or (1 if tiny else 3)
    if width % heads:
        raise ValueError("Temporal width must be divisible by heads")
    pattern = (
        tuple(args.pattern.split(","))
        if args.pattern
        else (("kda", "gqa") if tiny else ("kda", "kda", "kda", "gqa") * 2)
    )
    temporal_cfg = StateTemporalConfig(
        width=width,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=width // heads,
        pattern=pattern,
        chunk_size=args.chunk_size,
    )
    cfg = replace(cfg, frame_encoder_fp32=cfg.frame_encoder_fp32 or args.frame_encoder_fp32)
    # Construct directly on the chosen device instead of a full CPU model copy.
    with torch.device(args.device):
        model = StateActorCritic(
            cfg,
            temporal_cfg,
            decision_ticks=args.decision_ticks,
            frame_microbatch=args.frame_microbatch,
        )
    migration = model.migrate_bc_representation(source["model"]) if source is not None else None
    del source
    gc.collect()
    check_ram_budget(args.ram_limit_gib)
    return model, migration


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this Torch/runtime has no CUDA device")
    if device.type != "cuda" and args.precision != "fp32":
        raise ValueError("CPU smoke tests require --precision fp32")
    if args.fused_optimizer and device.type != "cuda":
        raise ValueError("Fused state BC optimizer requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use FP16 on V100")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "latest.pt").exists() and not args.resume:
        raise ValueError("Output already has latest.pt; resume explicitly or choose a new run")
    prepare_event(
        "dataset_open_start", "Opening replay sequence dataset", dataset=str(args.dataset)
    )
    dataset = ReplaySequenceDataset(
        args.dataset,
        args.cache_dir or args.out / ".replay-cache",
        expected_decision_ticks=args.decision_ticks,
        ram_limit_gib=args.ram_limit_gib,
        trust_data_files=args.trust_data_files,
    )
    prepare_event(
        "dataset_open_complete",
        "Replay sequence dataset opened",
        games=len(dataset.replay_ids),
        dataset_fingerprint=dataset.fingerprint,
    )
    if dataset.metadata.get("smoke_only") or not dataset.metadata.get("training_ready", False):
        if not args.allow_smoke_data or not 0 < args.max_episodes <= 8:
            raise ValueError(
                "Reconstruction is smoke-only/not training-ready; "
                "only an explicit <=8-episode smoke is allowed"
            )
    registry = (
        json.loads(args.split_registry.read_text(encoding="utf-8"))["assignments"]
        if args.split_registry
        else None
    )
    split = fixed_replay_split(
        dataset.metadata, dataset.replay_ids, args.validation_fraction, registry
    )
    prepare_event("label_counts_start", "Counting train/validation labels and timing priors")
    train_counts = dataset.partition_counts(split, "train")
    validation_counts = dataset.partition_counts(split, "validation")
    train_seats = 2 * sum(value == "train" for value in split.values())
    validation_seats = 2 * sum(value == "validation" for value in split.values())
    prepare_event(
        "label_counts_complete",
        "Label counts and priors ready",
        train_seats=train_seats,
        validation_seats=validation_seats,
        train_rows=train_counts["rows"],
        validation_rows=validation_counts["rows"],
    )
    if not train_counts["rows"] or not validation_counts["rows"]:
        raise ValueError("Fixed replay split needs nonempty training and validation supervision")
    if args.timing_objective == "natural" and not 0 < train_counts["action_prior"] < 1:
        raise ValueError("Natural timing training needs both action and WAIT labels")
    definition = {
        "chunk_size": args.chunk_size,
        "freeze_frame": args.freeze_frame,
        "precision": args.precision,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "position_coef": args.position_coef,
        "tile_label_sigma": args.tile_label_sigma,
        "timing_objective": args.timing_objective,
        "timing_target_action_prior": args.timing_target_action_prior,
        "sampled_action_prior": train_counts["action_prior"],
        "seed": args.seed,
        "decision_ticks": args.decision_ticks,
        "max_grad_norm": args.max_grad_norm,
        "recurrent_matrix_dtype": "float32",
        "kv_cache_dtype": args.precision,
        "vocabulary_sha256": vocabulary_sha256(dataset.metadata["vocabulary"]["names"]),
    }
    if args.episode_batch_size > 1:
        definition["episode_batch_size"] = args.episode_batch_size
    prepare_event(
        "checkpoint_load_start",
        "Loading checkpoint and constructing policy",
        resume=str(args.resume) if args.resume else None,
    )
    state = (
        load_state_checkpoint(args.resume, dataset_fingerprint=dataset.fingerprint, split=split)
        if args.resume
        else None
    )
    if state:
        # A new-stage transfer's immutable parent audit remains part of that
        # stage when subsequently resuming its own dataset and optimizer.
        if "initialization" in state["training_config"]:
            definition["initialization"] = state["training_config"]["initialization"]
        saved_placement = state["training_config"].get("placement_stage")
        if saved_placement is not None:
            if saved_placement.get("format") != STAGE_FORMAT or saved_placement.get("mode") not in ("placement_only", "full_finetune"):
                raise ValueError("Unsupported placement training-stage format or freeze mode")
            definition["placement_stage"] = deepcopy(saved_placement)
        elif args.placement_training_only:
            raise ValueError("Original joint checkpoint must use --upgrade-placement-from, not placement-only resume")
        if args.placement_training_only and saved_placement and saved_placement.get('mode') == 'full_finetune':
            raise ValueError('Cannot force placement-only mode on a fully unfrozen checkpoint')
        arm_unfreeze(args, definition, state)
        schedule_config = resolve_lr_schedule(
            args,
            definition,
            state=state,
            train_seats=2 * sum(value == "train" for value in split.values()),
        )
        model = state_model_from_checkpoint(state, args.device)
        model.frame_microbatch = args.frame_microbatch
        if saved_placement is not None:
            configure_training_mode(model, definition['placement_stage'])
        migration = None
    else:
        if args.upgrade_placement_from:
            model, migration = initialize_placement_stage(args, dataset, split)
            definition["placement_stage"] = deepcopy(migration)
        else:
            model, migration = build_model(args, dataset)
        if args.init_from_state:
            definition["initialization"] = migration
        arm_unfreeze(args, definition)
        schedule_config = resolve_lr_schedule(
            args,
            definition,
            train_seats=2 * sum(value == "train" for value in split.values()),
            migration=migration,
        )
    prepare_event("checkpoint_load_complete", "Checkpoint loaded and policy constructed")
    if args.freeze_frame:
        model.frame.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        optimizer_parameters(model, definition.get('placement_stage')),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=args.fused_optimizer,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.precision == "fp16", init_scale=32
    )
    epoch_start, next_episode, optimizer_steps = 0, 0, 0
    if migration and migration.get("mode") == "architecture_v3_weights_only_new_dataset_stage":
        optimizer_steps = migration["optimizer_steps_base"]
    if state:
        # Torch's optimizer loader consults the saved group flags when deciding
        # where scalar step tensors belong. Change only these small runtime
        # fields before loading; do not clone the large Adam moment tensors.
        for group in state["optimizer"]["param_groups"]:
            group["fused"] = args.fused_optimizer
            group["foreach"] = False if args.fused_optimizer else None
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["fused"] = args.fused_optimizer
            group["foreach"] = False if args.fused_optimizer else None
        # CPU step scalars (and all CPU-smoke moments) must not keep latest.pt
        # memory-mapped while the atomic saver replaces that file on Windows.
        for parameter, values in optimizer.state.items():
            for key, value in values.items():
                if key == "step" and args.fused_optimizer and isinstance(value, torch.Tensor):
                    values[key] = value.to(device=parameter.device, dtype=torch.float32)
                    continue
                if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                    values[key] = value.clone()
        if state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        restore_state_rng(state)
        epoch_start, next_episode, optimizer_steps = (
            state["progress"][key] for key in ("epoch", "next_episode", "optimizer_steps")
        )
    schedule = None
    if schedule_config is not None:
        if state and state.get("lr_schedule") is not None:
            schedule = WarmupCosineLR.from_state_dict(
                state["lr_schedule"],
                expected_config=schedule_config,
                optimizer_step=optimizer_steps,
                optimizer=optimizer,
            )
        else:
            schedule = WarmupCosineLR(schedule_config)
            if optimizer_steps != schedule_config["anchor_step"]:
                raise ValueError(
                    "New learning-rate schedule anchor differs from successful optimizer step"
                )
        schedule.apply(optimizer)
    validation_scope = {
        "dataset_fingerprint": dataset.fingerprint,
        "split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
        "max_validation_episodes": args.max_validation_episodes,
        "objective": {
            key: definition[key]
            for key in (
                "position_coef",
                "tile_label_sigma",
                "timing_objective",
                "timing_target_action_prior",
            )
        },
        "aggregation": "equal_weight_episode_seat",
    }
    if args.max_validation_games:
        validation_scope["sampling"] = {
            "games_per_validation": args.max_validation_games,
            "seed": args.seed,
            "policy": "whole_games_without_replacement_per_epoch_and_optimizer_step",
        }
    control = ValidationControl(
        validation_scope,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        saved=state.get("training_control") if state else None,
    )
    if control.last_validation_step is not None and control.last_validation_step > optimizer_steps:
        raise ValueError("Saved validation control is ahead of the resumable checkpoint")
    del state
    gc.collect()
    check_ram_budget(args.ram_limit_gib)
    adapter = CrForgeObservationAdapter(dataset.schema, model.cfg.vocab_size, entity_bucket=16)
    run_config = {
        "dataset_fingerprint": dataset.fingerprint,
        "model": model.checkpoint_config(),
        "training": definition,
        "train_counts": train_counts,
        "validation_counts": validation_counts,
        "migration": migration,
        "gradient_semantics": (
            "chunk_tbptt_independent_stream_memory_mean_episode_gradient_step_per_group"
            if args.episode_batch_size > 1
            else "chunk_tbptt_forward_memory_from_tick_zero_step_per_seat"
        ),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "ram_limit_gib": args.ram_limit_gib,
        "validation_monitor": {
            "scope": validation_scope,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "best_weights": "best.pt",
            "best_weights_inference_only": True,
        },
        "runtime_performance": {
            "fused_optimizer": args.fused_optimizer,
            "float32_matmul_precision": args.float32_matmul_precision,
            "defer_metrics": args.defer_metrics,
            "frame_microbatch": args.frame_microbatch,
            "bitwise_equivalence_claimed": False,
        },
    }
    if schedule is not None:
        run_config["lr_schedule"] = schedule.state_dict()
    atomic_json(args.out / "run_config.json", run_config)
    atomic_json(args.out / "split.json", {str(key): value for key, value in split.items()})
    started = time.perf_counter()
    processed = 0
    last_saved_progress = None
    metric_window=BCMetricWindow(args.metrics_every_updates,optimizer_steps)

    def log(record):
        record = {**record, "elapsed_seconds": time.perf_counter() - started}
        with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def flush_training_metrics():
        record=metric_window.flush()
        if record is not None:
            log(record)

    def save(epoch, next_index):
        nonlocal last_saved_progress
        flush_training_metrics()
        progress_key = (epoch, next_index, optimizer_steps)
        if progress_key == last_saved_progress:
            return
        save_state_checkpoint(
            args.out / "latest.pt",
            model,
            optimizer,
            scaler,
            dataset_fingerprint=dataset.fingerprint,
            split=split,
            epoch=epoch,
            next_episode=next_index,
            optimizer_steps=optimizer_steps,
            training_config=definition,
            lr_schedule=schedule.state_dict() if schedule else None,
            training_control=control.state_dict(),
        )
        if args.backup_dir:
            for name in (
                "latest.pt",
                "run_config.json",
                "split.json",
                "metrics.jsonl",
                "best_validation.json",
            ):
                if not (args.out / name).exists():
                    continue
                copy_artifact(
                    args.out / name, args.backup_dir / name, trust_data_files=args.trust_data_files
                )
        last_saved_progress = progress_key

    def validate(epoch, *, complete_epoch=False, initial_validation=False):
        flush_training_metrics()
        if control.last_validation_step == optimizer_steps and not complete_epoch:
            return
        if control.last_validation_step == optimizer_steps:
            metrics = deepcopy(control.last_validation_metrics)
        else:
            reports = []
            sampled_games = sample_validation_games(
                split,
                args.max_validation_games,
                seed=args.seed,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
            )
            validation_assignments = {game: "validation" for game in sampled_games}
            sampled_seats = 2 * len(sampled_games)
            total_validation_seats = (
                min(sampled_seats, args.max_validation_episodes)
                if args.max_validation_episodes
                else sampled_seats
            )
            validation_progress(epoch + 1, 0, total_validation_seats, initial_validation)
            episodes = dataset.episodes(validation_assignments, "validation")
            try:
                for episode in episodes:
                    reports.append(
                        run_state_episode(
                            model,
                            episode,
                            adapter,
                            chunk_size=args.chunk_size,
                            device=args.device,
                            precision=args.precision,
                            position_coef=args.position_coef,
                            tile_label_sigma=args.tile_label_sigma,
                            timing_objective=args.timing_objective,
                            timing_target_action_prior=args.timing_target_action_prior,
                            sampled_action_prior=validation_counts["action_prior"],
                            ram_limit_gib=args.ram_limit_gib,
                            defer_metrics=args.defer_metrics,
                        )
                    )
                    progress_every = 2 if args.max_validation_games else 32
                    if len(reports) % progress_every == 0 or len(reports) == total_validation_seats:
                        validation_progress(
                            epoch + 1, len(reports), total_validation_seats, initial_validation
                        )
                    if (
                        args.max_validation_episodes
                        and len(reports) >= args.max_validation_episodes
                    ):
                        break
            finally:
                episodes.close()
            if not reports:
                raise ValueError("No validation episodes")
            fields = (
                "loss",
                "timing_loss",
                "conditional_slot_loss",
                "position_loss",
                "wait_accuracy",
                "act_recall",
                "slot_accuracy",
                "ability_accuracy",
                "tile_mean_distance",
            )
            metrics = {key: sum(report[key] for report in reports) / len(reports) for key in fields}
            metrics.update(
                episodes=len(reports),
                rows=sum(report["rows"] for report in reports),
                aggregation="equal_weight_episode_seat",
            )
            if args.max_validation_games:
                metrics.update(
                    validation_games=len(sampled_games),
                    validation_game_ids=sampled_games,
                    validation_pool_games=validation_seats // 2,
                )
        improved = control.observe(
            metrics, epoch=epoch + 1, optimizer_steps=optimizer_steps, complete_epoch=complete_epoch
        )
        if improved:
            control.best["weights_sha256"] = save_best_weights(
                args.out / "best.pt",
                model,
                dataset_fingerprint=dataset.fingerprint,
                split=split,
                validation=control.best,
                trust_data_files=args.trust_data_files,
            )
            if args.backup_dir:
                copy_artifact(
                    args.out / "best.pt",
                    args.backup_dir / "best.pt",
                    trust_data_files=args.trust_data_files,
                )
        atomic_json(args.out / "best_validation.json", control.state_dict())
        if args.backup_dir:
            copy_verified(
                args.out / "best_validation.json", args.backup_dir / "best_validation.json"
            )
        log(
            {
                "phase": "state_bc_validation",
                "epoch": epoch + 1,
                "optimizer_steps": optimizer_steps,
                "complete_epoch": complete_epoch,
                "best_weights_updated": improved,
                "bad_complete_epochs": control.bad_complete_epochs,
                **metrics,
            }
        )

    log(
        {
            "phase": "state_bc_start",
            "train_seats": train_seats,
            "validation_seats": validation_seats,
            "episode_batch_size": args.episode_batch_size,
            **run_config,
        }
    )
    if args.initial_validation:
        prepare_event(
            "initial_validation_start",
            "Starting initial full validation",
            validation_seats=validation_seats,
        )
        validate(epoch_start, initial_validation=True)
    if control.should_stop:
        save(epoch_start, next_episode)
        summary = {
            "status": "early_stopped",
            "optimizer_steps": optimizer_steps,
            "episodes_this_run": 0,
            "epoch": epoch_start,
            "next_episode": next_episode,
            "best_validation": control.best,
            "bad_complete_epochs": control.bad_complete_epochs,
        }
        atomic_json(args.out / "summary.json", summary)
        return summary
    for epoch in range(epoch_start, args.epochs):
        if args.episode_batch_size > 1:
            episodes = dataset.episode_groups(
                split,
                "train",
                batch_size=args.episode_batch_size,
                epoch=epoch,
                seed=args.seed,
                start_episode=next_episode if epoch == epoch_start else 0,
                max_episodes=args.max_episodes - processed if args.max_episodes else 0,
            )
            training_iterator = episodes
        else:
            episodes = dataset.episodes(split, "train", epoch=epoch, seed=args.seed)
            training_iterator = enumerate(episodes)
        count = 0
        try:
            for index, item in training_iterator:
                group_size = len(item) if args.episode_batch_size > 1 else 1
                end_index = index + group_size
                count = end_index
                if epoch == epoch_start and index < next_episode:
                    continue
                transition = maybe_unfreeze(model, optimizer, definition, epoch=epoch,
                    next_episode=index, train_seats=train_seats, optimizer_steps=optimizer_steps)
                if transition is not None:
                    last_saved_progress = None
                    run_config['trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    atomic_json(args.out / 'run_config.json', run_config)
                    save(epoch, index)
                    log({'phase': 'state_bc_unfreeze', **transition,
                         'trainable_parameter_count': run_config['trainable_parameter_count']})
                used_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
                runner = (
                    run_state_episode_group if args.episode_batch_size > 1 else run_state_episode
                )
                report = runner(
                    model,
                    item,
                    adapter,
                    chunk_size=args.chunk_size,
                    device=args.device,
                    precision=args.precision,
                    optimizer=optimizer,
                    scaler=scaler,
                    position_coef=args.position_coef,
                    tile_label_sigma=args.tile_label_sigma,
                    timing_objective=args.timing_objective,
                    timing_target_action_prior=args.timing_target_action_prior,
                    sampled_action_prior=train_counts["action_prior"],
                    max_grad_norm=args.max_grad_norm,
                    ram_limit_gib=args.ram_limit_gib,
                    defer_metrics=args.defer_metrics,
                )
                optimizer_steps += int(report["optimizer_stepped"])
                if schedule is not None:
                    schedule.after_attempt(optimizer, successful=bool(report["optimizer_stepped"]))
                    if (
                        schedule_config["anchor_step"] + schedule.successful_updates
                        != optimizer_steps
                    ):
                        raise ValueError(
                            "Learning-rate schedule and successful optimizer steps diverged"
                        )
                previous_processed = processed
                processed += group_size
                metric_window.add(report, epoch=epoch+1, episode_index=end_index-1,
                    optimizer_steps=optimizer_steps, amp_scale=scaler.get_scale(),
                    full_finetune=definition.get('placement_stage',{}).get('mode')=='full_finetune',
                    trainable_parameter_count=run_config['trainable_parameter_count'],
                    learning_rate=used_learning_rates[0],learning_rates=used_learning_rates,
                    next_learning_rate=float(optimizer.param_groups[0]['lr']),
                    lr_schedule_step=schedule.successful_updates if schedule else None,
                    episode_batch_size=group_size,rss_mib=report['rss_mib'])
                if metric_window.ready():
                    flush_training_metrics()
                elif args.metrics_every_updates>1:
                    # Lightweight progress only; do not plot a per-seat score.
                    log({'phase':'state_bc_progress','epoch':epoch+1,
                         'episode_index':end_index-1,'optimizer_steps':optimizer_steps})
                if report["amp_overflow_skipped"] and scaler.get_scale() < 1:
                    raise FloatingPointError("Persistent FP16 overflow exhausted the loss scale")
                if (
                    args.validation_every_episodes
                    and processed // args.validation_every_episodes
                    > previous_processed // args.validation_every_episodes
                ):
                    validate(epoch)
                if (
                    args.checkpoint_every_episodes
                    and processed // args.checkpoint_every_episodes
                    > previous_processed // args.checkpoint_every_episodes
                ):
                    save(epoch, end_index)
                if args.max_episodes and processed >= args.max_episodes:
                    complete_epoch = end_index == train_seats
                    validate(epoch, complete_epoch=complete_epoch)
                    saved_epoch = epoch + 1 if complete_epoch else epoch
                    saved_index = 0 if complete_epoch else end_index
                    save(saved_epoch, saved_index)
                    summary = {
                        "status": "early_stopped"
                        if control.should_stop
                        else "bounded_run_complete",
                        "optimizer_steps": optimizer_steps,
                        "episodes_this_run": processed,
                        "epoch": saved_epoch,
                        "next_episode": saved_index,
                        "best_validation": control.best,
                        "bad_complete_epochs": control.bad_complete_epochs,
                    }
                    atomic_json(args.out / "summary.json", summary)
                    return summary
        finally:
            episodes.close()
        if epoch == epoch_start and next_episode > count:
            raise ValueError("Resume episode index exceeds the fixed epoch sequence")
        next_episode = 0
        validate(epoch, complete_epoch=True)
        save(epoch + 1, 0)
        if control.should_stop:
            summary = {
                "status": "early_stopped",
                "optimizer_steps": optimizer_steps,
                "episodes_this_run": processed,
                "epoch": epoch + 1,
                "next_episode": 0,
                "best_validation": control.best,
                "bad_complete_epochs": control.bad_complete_epochs,
            }
            atomic_json(args.out / "summary.json", summary)
            log({"phase": "state_bc_early_stop", **summary})
            return summary
    summary = {
        "status": "epochs_complete",
        "optimizer_steps": optimizer_steps,
        "episodes_this_run": processed,
        "best_validation": control.best,
    }
    atomic_json(args.out / "summary.json", summary)
    return summary


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
