from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.data.bc_sampling import class_prior_weights, epoch_rows
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.data.sharded_bc import load_sharded_bc
from cr_rl.data.validation_cache import prepare_validation_arrays
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.bc import hierarchical_action_predictions, hierarchical_bc_loss
from cr_rl.training.bc_batch import attention_context, build_bc_batch
from cr_rl.training.bc_optimizer import (
    carry_bc_optimizer,
    make_bc_optimizer,
    optimizer_parameter_names,
)
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def bc_tracking_options(path, resume):
    if resume and path.exists():
        return {"id": json.loads(path.read_text())["id"], "resume": "allow"}
    # A fresh stage must not inherit an earlier stage's run from its supervisor.
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    return {}


def make_schema(metadata: dict):
    raw = metadata["schema"]
    header = {}
    for i, name in enumerate(raw["header_fields"]):
        if name != "reserved":
            header.setdefault(name, []).append(i)
    return SimpleNamespace(
        seat_ints=int(raw["seat_ints"]),
        header_ints=int(raw["header_ints"]),
        entity_ints=int(raw["entity_ints"]),
        max_entities=int(raw["max_entities"]),
        header={key: tuple(value) for key, value in header.items()},
        header_fields=tuple(raw["header_fields"]),
        entity_fields=tuple(raw["entity_fields"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--model",
        choices=("tiny", "small", "compact", "general", "large", "production", "general_300m"),
        default="tiny",
    )
    parser.add_argument("--context", type=int)
    parser.add_argument("--v2", action="store_true", help="Clean-slate v2 representation")
    parser.add_argument("--entity-pooling", choices=("mean", "static1", "static4", "dynamic4"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--entity-bucket", type=int, default=16, help="Trim masked entity padding; 0 disables"
    )
    parser.add_argument("--fused-optimizer", action="store_true")
    parser.add_argument("--attention-backend", choices=("auto", "flash"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument(
        "--all-plays",
        action="store_true",
        help="Cover every training play/ability once per epoch; balance WAIT samples",
    )
    parser.add_argument(
        "--stream-cache-gib",
        type=float,
        default=0,
        help="Bound decompressed observation/mask storage; 0 uses eager memmaps",
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--validation-every-steps", type=int, default=0)
    parser.add_argument("--quick-validation-samples", type=int, default=256)
    parser.add_argument("--quick-natural-validation-samples", type=int, default=512)
    parser.add_argument("--carry-optimizer", action="store_true")
    parser.add_argument(
        "--upgrade-tile-head",
        action="store_true",
        help="Explicitly migrate BC weights to query residual + FP32 placement",
    )
    parser.add_argument("--tile-learning-rate", type=float)
    parser.add_argument("--upgrade-spatial", action="store_true")
    parser.add_argument("--stable-split", action="store_true")
    parser.add_argument("--split-registry", type=Path)
    parser.add_argument("--play-fraction", type=float, default=0.5)
    parser.add_argument("--ability-fraction", type=float)
    parser.add_argument(
        "--card-prior-correction", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--tile-label-sigma", type=float)
    parser.add_argument("--bc-objective", choices=("joint", "conditional"), default="joint")
    parser.add_argument("--wait-margin-before", type=int, default=0, help="Decision frames")
    parser.add_argument("--wait-margin-after", type=int, default=0, help="Decision frames")
    parser.add_argument("--natural-validation-samples", type=int, default=0)
    parser.add_argument("--position-coef", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-from-bc", type=Path, help="Weights-only initialization for a new, expanded dataset"
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name")
    parser.add_argument("--holdout-decks", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/bc_tiny.pt"))
    args = parser.parse_args()
    if args.cache_dir is None:
        args.cache_dir = args.out.parent / ".bc_arrays"
    if min(args.wait_margin_before, args.wait_margin_after) < 0:
        parser.error("WAIT margins cannot be negative")
    if args.stream_cache_gib < 0 or args.checkpoint_every_steps < 0:
        parser.error("cache and checkpoint interval cannot be negative")
    if (
        args.validation_every_steps < 0
        or min(args.quick_validation_samples, args.quick_natural_validation_samples) < 1
    ):
        parser.error("validation interval must be nonnegative and quick sample counts positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation fraction must be between zero and one")
    if min(args.epochs, args.batch_size, args.samples_per_epoch, args.grad_accumulation) < 1:
        parser.error("training counts must be positive")
    if not 0.0 < args.play_fraction < 1.0:
        parser.error("--play-fraction must be strictly between 0 and 1")

    dataset_path = args.dataset
    if args.resume and args.init_from_bc:
        parser.error("Use either --resume or --init-from-bc")
    if args.carry_optimizer and not args.init_from_bc:
        parser.error("--carry-optimizer requires --init-from-bc")
    if args.upgrade_tile_head and not args.init_from_bc:
        parser.error("--upgrade-tile-head requires --init-from-bc; resume retains saved behavior")
    if args.upgrade_spatial and not args.init_from_bc:
        parser.error("--upgrade-spatial requires --init-from-bc")
    if args.tile_learning_rate is not None and args.tile_learning_rate <= 0:
        parser.error("tile learning rate must be positive")
    if (dataset_path / "dataset.manifest.json").is_file():
        dataset_path = dataset_path / "dataset.manifest.json"
    if dataset_path.name == "dataset.manifest.json":
        data, metadata = load_sharded_bc(dataset_path, args.cache_dir, args.stream_cache_gib)
        obs, masks = data["observations"], data["masks"]
        actions = data["actions"].astype(np.int64)
        replay_ids = data["replay_ids"].astype(np.int32)
        seats = data["seats"].astype(np.int32)
        flags = data["flags"].astype(np.int32)
        ticks = data["ticks"].astype(np.int32)
    else:
        if dataset_path.suffix == ".npz":
            metadata_path = dataset_path.with_suffix(".json")
        else:
            metadata_path = dataset_path / "human_bc_reconstructed.json"
            dataset_path = dataset_path / "human_bc_reconstructed.npz"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(dataset_path) as data:
            obs = data["observations"]
            masks = data["masks"]
            actions = data["actions"].astype(np.int64)
            replay_ids = data["replay_ids"].astype(np.int32)
            seats = data["seats"].astype(np.int32)
            flags = data["flags"].astype(np.int32)
            ticks = data["ticks"].astype(np.int32)

    valid = (flags == 0) & (masks[np.arange(len(actions)), actions] == 1)
    valid &= ~np.isin(replay_ids, metadata.get("excluded_replay_ids", []))
    heldout_replays = []
    if args.holdout_decks:
        import yaml

        heldout = yaml.safe_load(args.holdout_decks.read_text())["decks"]
        vocabulary = {name: i for i, name in enumerate(metadata["vocabulary"]["names"])}
        schema = make_schema(metadata)
        if dataset_path.name == "dataset.manifest.json" and metadata.get("replays"):
            wanted = {tuple(sorted(deck)) for deck in heldout.values()}
            heldout_replays = [
                i
                for i, replay in enumerate(metadata["replays"])
                if any(tuple(sorted(deck)) in wanted for deck in replay["decks"])
            ]
        else:
            observed_decks = np.sort(obs[:, list(schema.header["deck"])], axis=1)
            excluded = np.zeros(len(actions), dtype=bool)
            for deck in heldout.values():
                deck_ids = np.sort([vocabulary[name] for name in deck])
                excluded |= (observed_decks == deck_ids).all(axis=1)
            heldout_replays = np.unique(replay_ids[excluded]).tolist()
        valid &= ~np.isin(replay_ids, heldout_replays)
    rows = np.flatnonzero(valid)
    plays = rows[actions[rows] != 0]
    waits = rows[actions[rows] == 0]
    if not len(plays) or not len(waits):
        raise ValueError("BC dataset needs both legal play and WAIT labels")

    # Hold out entire replays, never individual rows from a match.
    replay_values = np.unique(replay_ids[rows])
    if len(replay_values) < 2:
        raise ValueError("Need at least two independent replays for BC validation")
    split_rng = np.random.default_rng(args.seed)
    shuffled_replays = split_rng.permutation(replay_values)
    n_val = max(
        1, min(len(replay_values) - 1, int(np.ceil(len(replay_values) * args.validation_fraction)))
    )
    validation_replays = shuffled_replays[:n_val]
    if args.stable_split:
        scores = {}
        for i in replay_values:
            replay = metadata["replays"][int(i)]
            identity = [replay.get("source"), replay.get("source_revision"), replay["replay_tag"]]
            score = int.from_bytes(
                hashlib.sha256(json.dumps(identity).encode()).digest()[:8], "big"
            )
            scores[int(i)] = score / 2**64
        validation_replays = np.array(
            [i for i in replay_values if scores[int(i)] < args.validation_fraction]
        )
        if not 0 < len(validation_replays) < len(replay_values):
            raise ValueError("Stable split needs more replay records to populate both partitions")
    if args.split_registry:
        registry = json.loads(args.split_registry.read_text())["assignments"]
        validation_set = set(validation_replays.tolist())
        registry_heldout = []
        for index in replay_values:
            replay = metadata["replays"][int(index)]
            identity = replay.get("canonical_game_signature") or json.dumps(
                [replay.get("source"), replay.get("source_revision"), replay["replay_tag"]]
            )
            partition = registry.get(identity)
            if partition == "validation":
                validation_set.add(int(index))
            elif partition in ("train", "heldout"):
                validation_set.discard(int(index))
                if partition == "heldout":
                    registry_heldout.append(int(index))
        if registry_heldout:
            heldout_replays = sorted(set(heldout_replays) | set(registry_heldout))
            rows = rows[~np.isin(replay_ids[rows], registry_heldout)]
        validation_replays = np.asarray(sorted(validation_set), dtype=replay_values.dtype)
        if not 0 < len(validation_replays) < len(replay_values):
            raise ValueError("Split registry leaves an empty training or validation partition")
    train_rows = rows[~np.isin(replay_ids[rows], validation_replays)]
    val_rows = rows[np.isin(replay_ids[rows], validation_replays)]
    natural_rng = np.random.default_rng(args.seed + 1)
    natural_rows = natural_rng.choice(
        val_rows, min(len(val_rows), args.natural_validation_samples), replace=False
    )
    natural_rows.sort()
    if len(val_rows) > args.validation_samples:
        vp = val_rows[(actions[val_rows] > 0) & (actions[val_rows] < 2305)]
        vw = val_rows[actions[val_rows] == 0]
        va = val_rows[actions[val_rows] >= 2305]
        n_ability = min(len(va), args.validation_samples // 16)
        remaining = args.validation_samples - n_ability
        val_rows = np.concatenate(
            (
                split_rng.choice(vp, min(len(vp), remaining // 2), replace=False),
                split_rng.choice(vw, min(len(vw), remaining - remaining // 2), replace=False),
                split_rng.choice(va, n_ability, replace=False),
            )
        )
    if args.stream_cache_gib:
        val_rows = np.sort(val_rows)

    train_plays = train_rows[actions[train_rows] != 0]
    train_waits = train_rows[actions[train_rows] == 0]
    waits_before_filter = len(train_waits)
    if args.wait_margin_before or args.wait_margin_after:
        from cr_rl.data.replay_supervision import isolated_wait_rows

        train_waits = isolated_wait_rows(
            train_waits,
            actions,
            replay_ids,
            seats,
            ticks,
            before_ticks=args.wait_margin_before * metadata["decision_ticks"],
            after_ticks=args.wait_margin_after * metadata["decision_ticks"],
        )
    groups: dict[tuple[int, int], list[int]] = {}
    if not len(train_plays) or not len(train_waits):
        raise ValueError("Training partition needs both play and WAIT labels")
    for index in range(len(actions)):
        groups.setdefault((int(replay_ids[index]), int(seats[index])), []).append(int(index))
    for key in groups:
        groups[key].sort(key=lambda i: int(ticks[i]))
    positions = {
        key: {row: pos for pos, row in enumerate(indices)} for key, indices in groups.items()
    }

    schema = make_schema(metadata)
    adapter = CrForgeObservationAdapter(
        schema,
        len(metadata["vocabulary"]["names"]),
        entity_bucket=args.entity_bucket,
    )
    cfg = ModelConfig.preset(args.model)
    source_state = None
    reference = args.resume or args.init_from_bc
    if reference:
        source_state = torch.load(reference, map_location="cpu", weights_only=False, mmap=True)
    source_config = ModelConfig(**source_state["config"]) if source_state else cfg
    if args.v2 and args.init_from_bc:
        parser.error("Start v2 from scratch; later v2 dataset stages inherit without --v2")
    if args.v2 and source_state and source_config.representation_version != 2:
        parser.error("A legacy checkpoint cannot resume into the clean-slate v2 representation")
    cfg = cfg.inherit_representation(source_config)
    if args.v2 and not source_state:
        cfg = cfg.for_v2(schema, args.entity_pooling or "dynamic4")
        source_config = cfg
    elif args.entity_pooling and args.entity_pooling != cfg.entity_pooling:
        parser.error("Pooling ablations require a new --v2 run")
    v2 = cfg.representation_version == 2
    previous_settings = source_state.get("training_settings", {}) if source_state else {}
    for name, default in (
        ("ability_fraction", 0.1 if v2 else 0.0),
        ("tile_label_sigma", 1.0 if v2 else 0.0),
        ("card_prior_correction", v2),
    ):
        if getattr(args, name) is None:
            setattr(args, name, previous_settings.get(name, default))
    if not 0 <= args.ability_fraction < 1 or args.tile_label_sigma < 0:
        parser.error("Invalid ability fraction or tile label sigma")
    if args.bc_objective == "conditional" and args.card_prior_correction:
        parser.error("Conditional BC uses balanced timing; pass --no-card-prior-correction")
    if args.tile_learning_rate is None and source_state:
        args.tile_learning_rate = source_state.get("training_settings", {}).get(
            "tile_learning_rate"
        )
    if args.tile_learning_rate is None and v2:
        args.tile_learning_rate = 3e-4
    args.context = args.context or cfg.context_len
    cfg = ModelConfig(
        **{
            **cfg.__dict__,
            "context_len": args.context,
            "vocab_size": len(metadata["vocabulary"]["names"]),
            "tile_query_residual": args.upgrade_tile_head or source_config.tile_query_residual,
            "tile_head_fp32": args.upgrade_tile_head or source_config.tile_head_fp32,
            "spatial_fourier": args.upgrade_spatial or source_config.spatial_fourier,
        }
    )
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    precision = "highest" if cfg.tile_head_fp32 else "high"
    torch.set_float32_matmul_precision(precision)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ActorCritic(cfg).to(device)
    parent_bc_sha256 = None
    initial_optimizer_state = None
    parent_optimizer_steps = 0
    initial_optimizer_names = None
    resume_baseline = None
    resume_natural_baseline = None
    if args.init_from_bc:
        state = source_state
        expected_config = dict(cfg.__dict__)
        if args.upgrade_tile_head:
            for name in ("tile_query_residual", "tile_head_fp32"):
                expected_config[name] = getattr(source_config, name)
        if args.upgrade_spatial:
            expected_config["spatial_fourier"] = source_config.spatial_fourier
        if state.get("stage") != "bc" or source_config.__dict__ != expected_config:
            raise ValueError("BC initialization stage/model configuration mismatch")
        model.load_state_dict(state["model"], strict=True)
        parent_bc_sha256 = file_sha256(args.init_from_bc)
        if args.carry_optimizer:
            initial_optimizer_state = state["optimizer"]
            initial_optimizer_names = state.get("optimizer_param_names")
            parent_optimizer_steps = state["optimizer_steps"]
        del state
    optimizer = make_bc_optimizer(
        model,
        args.learning_rate,
        args.tile_learning_rate,
        fused=args.fused_optimizer and device.startswith("cuda"),
    )
    if initial_optimizer_state is not None:
        carry_bc_optimizer(model, optimizer, initial_optimizer_state, initial_optimizer_names)
        del initial_optimizer_state
    dataset_hash = file_sha256(dataset_path)
    data_report = {
        "dataset_sha256": dataset_hash,
        "total_rows": len(actions),
        "train_rows": len(train_rows),
        "train_replays": len(np.unique(replay_ids[train_rows])),
        "training_card_plays": int(
            ((actions[train_rows] > 0) & (actions[train_rows] < 2305)).sum()
        ),
        "training_abilities": int((actions[train_rows] >= 2305).sum()),
        "validation_replays": len(validation_replays),
        "validation_rows": len(val_rows),
        "validation_rows_sha256": hashlib.sha256(val_rows.tobytes()).hexdigest(),
        "heldout_deck_replays_excluded": len(heldout_replays),
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accumulation,
        "entity_bucket": args.entity_bucket,
        "attention_backend": args.attention_backend,
        "fused_optimizer": args.fused_optimizer,
        "epoch_mode": "all_plays" if args.all_plays else "sampled",
        "stream_cache_gib": args.stream_cache_gib,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "validation_every_steps": args.validation_every_steps,
        "stable_split": args.stable_split,
        "split_registry_sha256": file_sha256(args.split_registry) if args.split_registry else None,
        "parent_optimizer_steps": parent_optimizer_steps,
        "tile_query_residual": cfg.tile_query_residual,
        "representation_version": cfg.representation_version,
        "entity_pooling": cfg.entity_pooling,
        "tile_head_fp32": cfg.tile_head_fp32,
        "spatial_fourier": cfg.spatial_fourier,
        "tile_learning_rate": args.tile_learning_rate,
        "learning_rate": args.learning_rate,
        "explicit_tile_head_upgrade": args.upgrade_tile_head,
        "float32_matmul_precision": precision,
        "ability_fraction": args.ability_fraction,
        "card_prior_correction": args.card_prior_correction,
        "bc_objective": args.bc_objective,
        "wait_margin_before": args.wait_margin_before,
        "wait_margin_after": args.wait_margin_after,
        "training_wait_pool": len(train_waits),
        "near_action_waits_excluded": waits_before_filter - len(train_waits),
        "tile_label_sigma": args.tile_label_sigma,
        "natural_validation_rows": len(natural_rows),
        "duplicate_replays_excluded": len(metadata.get("excluded_replay_ids", [])),
    }
    atomic_json(args.out.with_suffix(".data.json"), data_report)
    print(json.dumps({"phase": "dataset_loaded", **data_report}), flush=True)
    start_epoch, history, optimizer_steps = 0, [], parent_optimizer_steps
    resume_progress = None
    if args.resume:
        state = source_state
        if (
            state.get("stage") != "bc"
            or ModelConfig(**state["config"]).__dict__ != cfg.__dict__
            or state.get("dataset_sha256") != dataset_hash
            or state.get("validation_replays") != validation_replays.tolist()
            or state.get("heldout_deck_replays_excluded", []) != heldout_replays
        ):
            raise ValueError("BC resume stage/config/dataset mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        parent_bc_sha256 = state.get("parent_bc_sha256")
        start_epoch, history = state["epochs"], state["history"]
        optimizer_steps = state["optimizer_steps"]
        data_report["parent_optimizer_steps"] = state.get("training_settings", {}).get(
            "parent_optimizer_steps", 0
        )
        resume_progress = state.get("epoch_progress")
        resume_baseline = state.get("baseline")
        resume_natural_baseline = state.get("natural_baseline")
        if resume_progress:
            settings = state["training_settings"]
            for name in ("batch_size", "gradient_accumulation", "epoch_mode"):
                if settings[name] != data_report[name]:
                    raise ValueError(f"Mid-epoch resume setting differs: {name}")
            if settings.get("tile_learning_rate") != args.tile_learning_rate:
                raise ValueError("Mid-epoch resume cannot change the placement learning rate")
            for name, default in (
                ("ability_fraction", 0.0),
                ("tile_label_sigma", 0.0),
                ("card_prior_correction", False),
                ("bc_objective", "joint"),
                ("wait_margin_before", 0),
                ("wait_margin_after", 0),
            ):
                if settings.get(name, default) != getattr(args, name):
                    raise ValueError(f"Mid-epoch resume cannot change {name}")
        rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_state"]])
        del state
    source_state = None
    tracking = None
    if args.wandb:
        import wandb

        tracking_path = args.out.with_suffix(".wandb.json")
        tracking_options = bc_tracking_options(
            tracking_path, bool(args.resume or args.init_from_bc)
        )
        tracking = wandb.init(
            project="cr-rl",
            name=args.wandb_name or args.out.parent.name + "_bc",
            job_type="bc",
            config={
                "model": args.model,
                "dataset_sha256": dataset_hash,
                "context": args.context,
                "stage": "bc",
                **data_report,
            },
            **tracking_options,
        )
        atomic_json(args.out.with_suffix(".wandb.json"), {"url": tracking.url, "id": tracking.id})
        tracking.define_metric("bc/optimizer_steps", hidden=True)
        for name in (
            "train_loss",
            "grad_norm",
            "optimizer_batch_loss",
            "microbatch_loss",
            "card_loss",
            "position_loss",
            "timing_loss",
            "conditional_slot_loss",
            "clip_scale",
        ):
            tracking.define_metric("bc/" + name, step_metric="bc/optimizer_steps")
        tracking.define_metric("bc_v3/epoch", hidden=True)
        tracking.define_metric("bc_v3/*", step_metric="bc/optimizer_steps")
        tracking.define_metric("bc_natural/*", step_metric="bc/optimizer_steps")
        tracking.define_metric("bc_quick/*", step_metric="bc/optimizer_steps")
        tracking.define_metric("bc_quick_natural/*", step_metric="bc/optimizer_steps")
        for name in (
            "metric_version",
            "rows",
            "validation_play_fraction",
            "validation_play_count",
            "validation_wait_count",
        ):
            for group in ("bc_v3", "bc_natural", "bc_quick", "bc_quick_natural"):
                tracking.define_metric(group + "/" + name, hidden=True)

    evaluation_obs, evaluation_masks = obs, masks
    if args.stream_cache_gib and hasattr(obs, "arrays"):
        print(json.dumps({"phase": "bc_validation_cache", "status": "preparing"}), flush=True)
        evaluation_obs, evaluation_masks = prepare_validation_arrays(
            obs,
            masks,
            np.concatenate((val_rows, natural_rows)),
            replay_ids,
            seats,
            groups,
            positions,
            args.context,
            args.out.parent / ".validation_cache",
            dataset_hash,
            progress=lambda field, done, total: print(
                json.dumps(
                    {
                        "phase": "bc_validation_cache",
                        "field": field,
                        "shards_done": done,
                        "shards_total": total,
                    }
                ),
                flush=True,
            ),
        )
        print(json.dumps({"phase": "bc_validation_cache", "status": "ready"}), flush=True)

    def batch(rows_to_build: np.ndarray, evaluation=False):
        return build_bc_batch(
            rows_to_build,
            evaluation_obs if evaluation else obs,
            evaluation_masks if evaluation else masks,
            actions,
            replay_ids,
            seats,
            groups,
            positions,
            adapter,
            args.context,
            device,
        )

    def evaluate(rows_to_eval: np.ndarray, prepared=None) -> dict:
        if not len(rows_to_eval):
            return {}
        model.eval()
        losses, exact_correct, slot_correct, tile_correct = [], 0, 0, 0
        play_count, wait_count, wait_correct, play_recalled = 0, 0, 0, 0
        card_loss_sum = tile_loss_sum = 0.0
        timing_wait_sum = timing_act_sum = conditional_card_sum = 0.0
        wait_false_probability = nonwait_probability = 0.0
        wait_false_decisions = 0
        tile_distance_sum = tile_near = ability_count = ability_correct = 0
        with torch.no_grad():
            for start in range(0, len(rows_to_eval), args.batch_size):
                ids = rows_to_eval[start : start + args.batch_size]
                tensors = (
                    batch(ids, evaluation=True)
                    if prepared is None
                    else tuple(t.to(device) for t in prepared[start // args.batch_size])
                )
                h, e, em, legal, tm, hand_ids, y, deck_ids, seen_ids = tensors
                with (
                    attention_context(args.attention_backend),
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=device == "cuda" and torch.cuda.is_bf16_supported(),
                    ),
                ):
                    logp, _ = model(
                        h,
                        e,
                        em,
                        legal,
                        tm,
                        hand_card_ids=hand_ids,
                        deck_card_ids=deck_ids,
                        enemy_seen_card_ids=seen_ids,
                    )
                    loss, components = hierarchical_bc_loss(
                        logp, y, legal, position_coef=args.position_coef
                    )
                batch_plays = int(((actions[ids] > 0) & (actions[ids] < 2305)).sum())
                card_loss_sum += float(components["card_loss"].cpu()) * len(ids)
                tile_loss_sum += float(components["position_loss"].cpu()) * batch_plays
                pred = logp.argmax(-1)
                losses.append(float(loss.cpu()))
                exact_correct += int((pred == y).sum().cpu())
                is_play = (y > 0) & (y < 2305)
                is_wait = y == 0
                is_ability = y >= 2305
                p_nonwait = (1 - logp[..., 0].float().exp()).clamp(0, 1)
                nonwait_probability += float(p_nonwait.sum().cpu())
                wait_false_probability += float(p_nonwait[is_wait].sum().cpu())
                wait_false_decisions += int(((p_nonwait > 0.5) & is_wait).sum().cpu())
                ability_count += int(is_ability.sum().cpu())
                ability_correct += int(((pred == y) & is_ability).sum().cpu())
                play_count += int(is_play.sum().cpu())
                wait_count += int(is_wait.sum().cpu())
                card_logp, predicts_play, pred_slot = hierarchical_action_predictions(logp)
                target_card = torch.where(is_play, 1 + (y - 1) // 576, y.clamp(0, 0))
                target_card = torch.where(y >= 2305, 5 + y - 2305, target_card)
                act_logp = torch.logsumexp(card_logp[:, 1:], dim=-1)
                timing_wait_sum += float((-card_logp[is_wait, 0]).sum().cpu())
                timing_act_sum += float((-act_logp[~is_wait]).sum().cpu())
                chosen_card = card_logp.gather(-1, target_card[:, None]).squeeze(-1)
                conditional_card_sum += float(
                    (-chosen_card[~is_wait] + act_logp[~is_wait]).sum().cpu()
                )
                wait_correct += int((~predicts_play & is_wait).sum().cpu())
                play_recalled += int((predicts_play & is_play).sum().cpu())
                slot_correct += int(((pred_slot + 1 == target_card) & is_play).sum().cpu())
                if is_play.any():
                    true_slot = ((y[is_play] - 1) // 576).long()
                    true_tile = ((y[is_play] - 1) % 576).long()
                    conditional = []
                    for slot in range(4):
                        start = 1 + slot * 576
                        conditional.append(
                            logp[..., start : start + 576] - card_logp[..., slot + 1 : slot + 2]
                        )
                    tile_logp = torch.stack(conditional, dim=-2)
                    row = torch.arange(len(y), device=device)[is_play]
                    tile_pred = tile_logp[row, true_slot].argmax(-1)
                    tile_correct += int((tile_pred == true_tile).sum().cpu())
                    distance = (tile_pred % 18 - true_tile % 18).abs() + (
                        tile_pred // 18 - true_tile // 18
                    ).abs()
                    tile_distance_sum += int(distance.sum().cpu())
                    tile_near += int((distance <= 2).sum().cpu())
        model.train()
        wait_accuracy = wait_correct / max(1, wait_count)
        play_slot_accuracy = slot_correct / max(1, play_count)
        act_count = play_count + ability_count
        balanced_timing_nll = (
            timing_wait_sum / max(1, wait_count) + timing_act_sum / max(1, act_count)
        ) / max(1, int(wait_count > 0) + int(act_count > 0))
        conditional_card_nll = conditional_card_sum / max(1, act_count)
        return {
            "loss": float(np.mean(losses)),
            "card_loss": card_loss_sum / len(rows_to_eval),
            "position_loss": tile_loss_sum / max(1, play_count),
            "global_loss": card_loss_sum / len(rows_to_eval)
            + args.position_coef * tile_loss_sum / max(1, play_count),
            "balanced_timing_nll": balanced_timing_nll,
            "conditional_card_nll": conditional_card_nll,
            "conditional_validation_score": balanced_timing_nll
            + conditional_card_nll
            + args.position_coef * tile_loss_sum / max(1, play_count),
            "exact_accuracy": exact_correct / len(rows_to_eval),
            "wait_accuracy": wait_accuracy,
            "balanced_wait_play_accuracy": 0.5
            * (wait_accuracy + play_recalled / max(1, play_count)),
            "play_recall": play_recalled / max(1, play_count),
            "label_nonwait_fraction": (play_count + ability_count) / len(rows_to_eval),
            "policy_nonwait_probability": nonwait_probability / len(rows_to_eval),
            "nonwait_probability_on_action_rows": (nonwait_probability - wait_false_probability)
            / max(1, play_count + ability_count),
            "wait_false_action_probability": wait_false_probability / max(1, wait_count),
            "wait_false_action_decision_rate": wait_false_decisions / max(1, wait_count),
            "expected_false_actions_per_wait_second": (
                wait_false_probability / max(1, wait_count) / (metadata["decision_ticks"] * 0.05)
            ),
            "metric_version": 4,
            "play_slot_accuracy": play_slot_accuracy,
            "play_tile_accuracy": tile_correct / max(1, play_count),
            "play_tile_mean_l1_distance": tile_distance_sum / max(1, play_count),
            "play_tile_within_2": tile_near / max(1, play_count),
            "ability_accuracy": ability_correct / ability_count if ability_count else None,
            "validation_play_fraction": play_count / len(rows_to_eval),
            "validation_play_count": play_count,
            "validation_wait_count": wait_count,
            "rows": len(rows_to_eval),
        }

    started = time.perf_counter()
    baseline_path = args.out.with_suffix(".baseline.json")
    if args.resume and resume_baseline is not None:
        baseline = resume_baseline
    elif args.resume and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
    else:
        baseline = evaluate(val_rows) if not history else {}
    atomic_json(args.out.with_suffix(".baseline.json"), baseline or {})
    if tracking and baseline:
        tracking.log({f"bc/baseline_{k}": v for k, v in baseline.items()})
    natural_baseline = {}
    if len(natural_rows):
        natural_path = args.out.with_suffix(".natural_baseline.json")
        if args.resume and resume_natural_baseline is not None:
            natural_baseline = resume_natural_baseline
        elif args.resume and natural_path.exists():
            natural_baseline = json.loads(natural_path.read_text())
        else:
            natural_baseline = evaluate(natural_rows)
        atomic_json(natural_path, natural_baseline)

    quick_sets = []
    if args.validation_every_steps:
        # Separate RNG: enabling monitoring must not change training sample order.
        probe_rng = np.random.default_rng(args.seed + 71237)
        for name, available, count in (
            ("bc_quick", val_rows, args.quick_validation_samples),
            ("bc_quick_natural", natural_rows, args.quick_natural_validation_samples),
        ):
            if not len(available):
                continue
            rows = np.sort(probe_rng.choice(available, min(count, len(available)), replace=False))
            # Keep only this small fixed cohort in CPU RAM. Re-reading random
            # shards each minute would churn the bounded decompression cache.
            prepared = [
                build_bc_batch(
                    rows[start : start + args.batch_size],
                    evaluation_obs,
                    evaluation_masks,
                    actions,
                    replay_ids,
                    seats,
                    groups,
                    positions,
                    adapter,
                    args.context,
                    "cpu",
                )
                for start in range(0, len(rows), args.batch_size)
            ]
            quick_sets.append((name, rows, prepared))
        atomic_json(
            args.out.with_suffix(".quick_cohort.json"),
            {
                "interval_steps": args.validation_every_steps,
                "cohorts": {
                    name: {"rows": len(rows), "sha256": hashlib.sha256(rows.tobytes()).hexdigest()}
                    for name, rows, _ in quick_sets
                },
                "scope": "Fixed small monitoring subsets; full validation still runs each epoch",
            },
        )

    def quick_validation(epoch):
        began = time.perf_counter()
        result = {name: evaluate(rows, prepared) for name, rows, prepared in quick_sets}
        record = {
            "phase": "bc_quick_validation",
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "metrics": result,
            "seconds": time.perf_counter() - began,
        }
        with args.out.with_suffix(".quick.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if tracking:
            tracking.log(
                {
                    "bc/optimizer_steps": optimizer_steps,
                    **{
                        f"{name}/{key}": value
                        for name, metrics in result.items()
                        for key, value in metrics.items()
                    },
                }
            )
        if args.backup_dir:
            for suffix in (".quick.jsonl", ".quick_cohort.json"):
                path = args.out.with_suffix(suffix)
                copy_verified(path, args.backup_dir / path.name)

    def save_checkpoint(completed_epochs, progress=None):
        state_to_save = {
            "stage": "bc",
            "model_architecture_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_param_names": optimizer_parameter_names(model, optimizer),
            "config": cfg.__dict__,
            "dataset_metadata": metadata,
            "dataset_sha256": dataset_hash,
            "parent_bc_sha256": parent_bc_sha256,
            "training_settings": data_report,
            "epochs": completed_epochs,
            "optimizer_steps": optimizer_steps,
            "history": history,
            "baseline": baseline,
            "natural_baseline": natural_baseline,
            "validation_replays": validation_replays.tolist(),
            "heldout_deck_replays_excluded": heldout_replays,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
            "epoch_progress": progress,
        }
        temporary = args.out.with_suffix(".tmp.pt")
        torch.save(state_to_save, temporary)
        temporary.replace(args.out)
        atomic_json(
            args.out.with_suffix(".json"),
            {
                "parameters": model.parameter_count(),
                "representation_version": cfg.representation_version,
                "bc_objective": args.bc_objective,
                "history": history,
                "baseline": baseline,
                "natural_baseline": natural_baseline,
                "train_rows": len(train_rows),
                "validation_rows": len(val_rows),
                "validation_replays": validation_replays.tolist(),
                "optimizer_steps": optimizer_steps,
                "dataset_sha256": dataset_hash,
                "seconds": time.perf_counter() - started,
                "completed_epochs": completed_epochs,
                "samples_in_current_epoch": progress["next_sample"] if progress else 0,
            },
        )
        if args.backup_dir:
            name = "bc_latest.pt" if progress else f"bc_epoch_{completed_epochs:04d}.pt"
            record = copy_verified(args.out, args.backup_dir / name)
            copy_verified(args.out.with_suffix(".json"), args.backup_dir / "summary.json")
            if args.out.with_suffix(".wandb.json").exists():
                copy_verified(
                    args.out.with_suffix(".wandb.json"), args.backup_dir / "bc.wandb.json"
                )
            atomic_json(
                args.backup_dir / "latest.json",
                {
                    "epoch": completed_epochs,
                    "optimizer_steps": optimizer_steps,
                    "mid_epoch": progress is not None,
                    **record,
                },
            )

    for epoch in range(start_epoch, args.epochs):
        if resume_progress:
            selected = resume_progress["selected"]
            first_sample = resume_progress["next_sample"]
            epoch_losses = resume_progress["epoch_losses"]
            card_losses = resume_progress["card_losses"]
            position_losses = resume_progress["position_losses"]
            resume_progress = None
        else:
            selected = epoch_rows(
                train_plays,
                train_waits,
                rng,
                args.play_fraction,
                sample_count=None
                if args.all_plays
                else min(args.samples_per_epoch, len(train_rows)),
                offsets=obs.offsets if args.stream_cache_gib and hasattr(obs, "offsets") else None,
                ability_rows=train_rows[actions[train_rows] >= 2305],
                ability_fraction=args.ability_fraction,
            )
            first_sample = 0
            epoch_losses, card_losses, position_losses = [], [], []
        model.train()
        prior_weights = (
            torch.as_tensor(
                class_prior_weights(actions[train_rows], actions[selected]),
                device=device,
                dtype=torch.float32,
            )
            if args.card_prior_correction
            else None
        )
        optimizer.zero_grad(set_to_none=True)
        group_loss = group_card = group_position = 0.0
        group_timing = group_slot = 0.0
        print(
            json.dumps(
                {
                    "phase": "epoch_start",
                    "epoch": epoch + 1,
                    "samples": len(selected),
                    "next_sample": first_sample,
                    "epoch_mode": data_report["epoch_mode"],
                }
            ),
            flush=True,
        )
        for start in range(first_sample, len(selected), args.batch_size):
            batch_index = start // args.batch_size
            updated = False
            ids = selected[start : start + args.batch_size]
            h, e, em, legal, tm, hand_ids, y, deck_ids, seen_ids = batch(ids)
            with (
                attention_context(args.attention_backend),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device == "cuda" and torch.cuda.is_bf16_supported(),
                ),
            ):
                logp, _ = model(
                    h,
                    e,
                    em,
                    legal,
                    tm,
                    hand_card_ids=hand_ids,
                    deck_card_ids=deck_ids,
                    enemy_seen_card_ids=seen_ids,
                )
                loss, loss_parts = hierarchical_bc_loss(
                    logp,
                    y,
                    legal,
                    position_coef=args.position_coef,
                    card_class_weights=prior_weights,
                    tile_label_sigma=args.tile_label_sigma,
                    conditional_card_loss=args.bc_objective == "conditional",
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BC loss")
            loss.backward()
            loss_value = float(loss.detach().cpu())
            card_value = float(loss_parts["card_loss"].cpu())
            position_value = float(loss_parts["position_loss"].cpu())
            group_loss += loss_value
            group_card += card_value
            group_position += position_value
            group_timing += float(loss_parts["timing_loss"].cpu())
            group_slot += float(loss_parts["conditional_slot_loss"].cpu())
            accumulated = batch_index % args.grad_accumulation + 1
            if accumulated == args.grad_accumulation or start + args.batch_size >= len(selected):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                updated = True
                if optimizer_steps % 16 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "bc_train",
                                "epoch": epoch + 1,
                                "optimizer_steps": optimizer_steps,
                                "samples_in_epoch": min(start + args.batch_size, len(selected)),
                                "loss": loss_value,
                                "optimizer_batch_loss": group_loss / accumulated,
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
                if tracking:
                    tracking.log(
                        {
                            "bc/optimizer_steps": optimizer_steps,
                            "bc/train_loss": group_loss / accumulated,
                            "bc/microbatch_loss": loss_value,
                            "bc/optimizer_batch_loss": group_loss / accumulated,
                            "bc/card_loss": group_card / accumulated,
                            "bc/position_loss": group_position / accumulated,
                            "bc/timing_loss": group_timing / accumulated,
                            "bc/conditional_slot_loss": group_slot / accumulated,
                            "bc/grad_norm": float(norm.cpu()),
                            "bc/clip_scale": min(1.0, 1.0 / (float(norm.cpu()) + 1e-6)),
                        }
                    )
                group_loss = group_card = group_position = 0.0
                group_timing = group_slot = 0.0
                if (
                    args.validation_every_steps
                    and optimizer_steps % args.validation_every_steps == 0
                ):
                    quick_validation(epoch + 1)
            epoch_losses.append(loss_value)
            card_losses.append(card_value)
            position_losses.append(position_value)
            if (
                updated
                and args.checkpoint_every_steps
                and optimizer_steps % args.checkpoint_every_steps == 0
                and start + args.batch_size < len(selected)
            ):
                save_checkpoint(
                    epoch,
                    {
                        "selected": selected,
                        "next_sample": start + args.batch_size,
                        "epoch_losses": epoch_losses,
                        "card_losses": card_losses,
                        "position_losses": position_losses,
                    },
                )
        report = {
            "epoch": epoch + 1,
            "action_labels_seen": int((actions[selected] != 0).sum()),
            "unique_action_labels_seen": len(np.unique(selected[actions[selected] != 0])),
            "available_training_action_labels": len(train_plays),
            "training_draws": len(selected),
            "sample_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
            "train_loss": float(np.mean(epoch_losses)),
            "card_loss": float(np.mean(card_losses)),
            "position_loss": float(np.mean(position_losses)),
            "validation": evaluate(val_rows),
        }
        if len(natural_rows):
            report["natural_validation"] = evaluate(natural_rows)
        history.append(report)
        print(json.dumps(report), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(epoch + 1)
        if tracking:
            tracking.log(
                {
                    "bc/epoch": epoch + 1,
                    "bc/optimizer_steps": optimizer_steps,
                    "bc_v3/epoch": epoch + 1,
                    **{f"bc_v3/{k}": v for k, v in report["validation"].items()},
                    **{f"bc/validation_{k}": v for k, v in report["validation"].items()},
                    **{
                        f"bc_natural/{k}": v
                        for k, v in report.get("natural_validation", {}).items()
                    },
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if tracking:
        tracking.finish()
    summary = {
        "checkpoint": str(args.out),
        "device": device,
        "parameters": model.parameter_count(),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "play_rows": len(train_plays),
        "wait_rows": len(train_waits),
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
