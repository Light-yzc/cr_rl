"""Stream full immutable state-BC snapshots, refreshing only at epoch boundaries.

One trainer invocation finishes at most one epoch. A persisted transaction and
checkpoint cursor make controller restarts safe; manifest growth starts an
explicit weights-only data stage. Old validation/heldout identities stay fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare_event(stage, message, **details):
    print(
        json.dumps(
            {"phase": "state_bc_prepare", "stage": stage, "message": message, **details},
            ensure_ascii=False,
        ),
        flush=True,
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_trusted(source, target, **_ignored):
    """Atomic single-pass transfer without an extra content hash/readback."""
    source, target = Path(source), Path(target)
    stat = source.stat()
    identity = {
        "source": str(source),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "content_verification": "trusted_storage_transfer",
    }
    receipt = target.with_name(target.name + ".trusted.json")
    if target.exists() and receipt.exists() and read_json(receipt) == identity:
        return identity
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".uploading")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        atomic_json(receipt, identity)
    finally:
        temporary.unlink(missing_ok=True)
    return identity


def identity(replay):
    return str(
        replay.get("canonical_game_signature")
        or replay.get("game_signature")
        or json.dumps([replay.get("source"), replay.get("source_revision"), replay["replay_tag"]])
    )


def identity_keys(replay):
    keys = {identity(replay)}
    for field in ("canonical_game_signature", "game_signature"):
        if replay.get(field):
            keys.add(str(replay[field]))
    if replay.get("replay_tag"):
        values = [replay.get("source"), replay.get("source_revision"), replay["replay_tag"]]
        keys.update((json.dumps(values), json.dumps(values, separators=(",", ":"))))
    return keys


def assignments(path):
    value = read_json(path)
    return value.get("assignments", value)


def frozen_registry(metadata, initial, existing=None):
    registry = dict(initial)
    for key, value in (existing or {}).items():
        if key in registry and registry[key] != value:
            raise ValueError("Existing and seed split registries disagree")
        registry[key] = value
    split = {}
    excluded = set(metadata.get("excluded_replay_ids", []))
    for replay_id, replay in enumerate(metadata["replays"]):
        key = identity(replay)
        aliases = identity_keys(replay)
        values = {registry[k] for k in aliases if k in registry}
        if len(values) > 1:
            raise ValueError("Replay identity has contradictory split assignments")
        partition = next(iter(values)) if values else "train"
        if replay_id in excluded:
            if values and partition != "heldout":
                raise ValueError("Dataset exclusion contradicts frozen split")
            partition = "heldout"
        if partition not in ("train", "validation", "heldout"):
            raise ValueError("Unknown split assignment")
        for alias in aliases:
            registry[alias] = partition
        split[replay_id] = partition
    return registry, split


def discover_manifests(explicit, roots):
    result = list(map(Path, explicit))
    for root in map(Path, roots):
        catalog = root / "collection.manifest.json"
        if not catalog.is_file():
            continue
        value = read_json(catalog)
        if value.get("format") != "crrl-state-bc-collection-v1":
            raise ValueError("Unsupported collection catalog format")
        for entry in value.get("batches", []):
            # collect_state_bc_full.py publishes batches[].manifest and sha256.
            name = entry.get("manifest")
            if not name:
                raise ValueError("Committed batch lacks a dataset manifest path")
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Committed manifest escapes collection root")
            if entry.get("sha256") and sha(path) != entry["sha256"]:
                raise ValueError("Committed collection manifest checksum differs")
            result.append(path)
    return list(
        dict.fromkeys(
            (path / "dataset.manifest.json" if path.is_dir() else path).resolve() for path in result
        )
    )


def collection_completeness(roots, *, expected_total_games=0, expected_source_shards=0):
    """Validate producer migration receipts before any dataset/checkpoint read."""
    if not roots:
        return False, {"phase": "waiting_for_complete_collection", "reason": "no_collection_roots"}
    reports = []
    for root in map(Path, roots):
        catalog = root / "collection.manifest.json"
        if not catalog.is_file():
            return False, {
                "phase": "waiting_for_complete_collection",
                "root": str(root),
                "reason": "missing_collection_manifest",
            }
        value = read_json(catalog)
        if value.get("format") != "crrl-state-bc-collection-v1":
            return False, {
                "phase": "waiting_for_complete_collection",
                "root": str(root),
                "reason": "unsupported_collection_format",
            }
        required = (
            "migration_complete",
            "source_shards_total",
            "source_shards_migrated",
            "source_replays_total",
            "accepted_replays",
        )
        if value.get("migration_complete") is not True or any(key not in value for key in required):
            return False, {
                "phase": "waiting_for_complete_collection",
                "root": str(root),
                "reason": "migration_receipt_incomplete",
                "required": required,
            }
        counts = (
            "source_shards_total",
            "source_shards_migrated",
            "source_replays_total",
            "accepted_replays",
        )
        if any(
            not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] <= 0
            for key in counts
        ):
            return False, {
                "phase": "waiting_for_complete_collection",
                "root": str(root),
                "reason": "invalid_collection_counts",
            }
        if value["source_shards_migrated"] != value["source_shards_total"]:
            return False, {
                "phase": "waiting_for_complete_collection",
                "root": str(root),
                "reason": "source_shards_incomplete",
                "migrated": value["source_shards_migrated"],
                "total": value["source_shards_total"],
            }
        if value["accepted_replays"] != value["source_replays_total"]:
            return False, {
                "phase": "waiting_for_complete_collection",
                "root": str(root),
                "reason": "source_replays_incomplete",
                "accepted": value["accepted_replays"],
                "total": value["source_replays_total"],
            }
        reports.append(value)
    if expected_total_games:
        total = sum(int(value["source_replays_total"]) for value in reports)
        if total != expected_total_games:
            return False, {
                "phase": "waiting_for_complete_collection",
                "reason": "source_game_count_mismatch",
                "expected_total_games": expected_total_games,
                "source_total_games": total,
            }
    if expected_source_shards:
        total = sum(value["source_shards_total"] for value in reports)
        if total != expected_source_shards:
            return False, {
                "phase": "waiting_for_complete_collection",
                "reason": "source_shard_count_mismatch",
                "expected_source_shards": expected_source_shards,
                "source_total_shards": total,
            }
    return True, {"collections": reports}


def controller_lock_path(local_work, out):
    """Use local POSIX locking; Drive/FUSE does not reliably support flock."""
    digest = hashlib.sha256(str(Path(out).resolve()).encode()).hexdigest()
    directory = Path(local_work) / ".controller-locks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (digest + ".lock")


def registry_for_sources(manifests, initial, existing, *, initialize_validation=False):
    registry = dict(existing)
    original_keys = set(initial) | set(existing)
    games = []
    for path in manifests:
        manifest = read_json(path)
        metadata_path = (path.parent / manifest["metadata"]).resolve()
        if not metadata_path.is_relative_to(path.parent.resolve()):
            raise ValueError("Metadata escapes source dataset")
        if sha(metadata_path) != manifest["metadata_sha256"]:
            raise ValueError("Source metadata checksum differs")
        metadata = read_json(metadata_path)
        games.extend(metadata["replays"])
        registry, _ = frozen_registry(metadata, initial, registry)
    if initialize_validation and not any(
        registry[identity(game)] == "validation" for game in games
    ):
        eligible = {
            identity(game)
            for game in games
            if not (identity_keys(game) & original_keys) and registry[identity(game)] == "train"
        }
        chosen = {
            key
            for key in eligible
            if int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64 < 0.1
        }
        if not chosen and len(eligible) >= 2:
            chosen.add(min(eligible, key=lambda key: hashlib.sha256(key.encode()).digest()))
        if chosen:
            for game in games:
                if identity(game) in chosen:
                    # All canonical/legacy aliases must receive the same frozen assignment.
                    game_registry, _ = frozen_registry({"replays": [game]}, {}, {})
                    for key in game_registry:
                        registry[key] = "validation"
    return registry


def local_progress(checkpoint, fingerprint, split):
    """Inspect already-cached checkpoint metadata; never build a policy model."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    try:
        if (
            state.get("format") != "crrl-state-bc-v1"
            or state.get("model_architecture_version") != 3
        ):
            raise ValueError("Controller requires resumable architecture-v3 latest.pt")
        if state.get("dataset_fingerprint") != fingerprint:
            raise ValueError("Stage checkpoint fingerprint differs")
        if state.get("split") != {str(k): v for k, v in split.items()}:
            raise ValueError("Stage checkpoint fixed split differs")
        p = dict(state["progress"])
        if any(
            not isinstance(p.get(k), int) or isinstance(p[k], bool) or p[k] < 0
            for k in ("epoch", "next_episode", "optimizer_steps")
        ):
            raise ValueError("Checkpoint progress is invalid")
        control = state.get("training_control", {})
        return p, {
            "bad_complete_epochs": control.get("bad_complete_epochs", 0),
            "best_validation": control.get("best"),
            "training_config": state.get("training_config", {}),
        }
    finally:
        del state


def restore_training(local_stage, stage, copy_verified):
    local = local_stage / "training"
    local.mkdir(parents=True, exist_ok=True)
    for name in (
        "latest.pt",
        "best.pt",
        "best_validation.json",
        "metrics.jsonl",
        "run_config.json",
        "split.json",
        "summary.json",
    ):
        source, target = stage / "training" / name, local / name
        if source.is_file() and not target.exists():
            copy_verified(source, target)
    return local


def child_is_alive(pending):
    if pending.get("child_host") != socket.gethostname() or not pending.get("child_pid"):
        return False
    try:
        os.kill(int(pending["child_pid"]), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def stage_parent(args, state, copy_verified, *, copy_weights=True):
    local = args.local_work / "parent"
    local.mkdir(parents=True, exist_ok=True)
    source = Path(state.get("backup_checkpoint") or args.parent_checkpoint)
    manifest = Path(state.get("backup_manifest") or args.parent_manifest)
    data = read_json(manifest)
    metadata = (manifest.parent / data["metadata"]).resolve()
    if not metadata.is_relative_to(manifest.parent.resolve()):
        raise ValueError("Parent metadata escapes its dataset")
    copy_verified(
        metadata, local / "dataset_metadata.json", expected_sha256=data["metadata_sha256"]
    )
    # Preserve manifest bytes/fingerprint, including its original metadata name.
    target_meta = (local / data["metadata"]).resolve()
    if not target_meta.is_relative_to(local.resolve()):
        raise ValueError("Parent metadata path escapes local cache")
    if target_meta != local / "dataset_metadata.json":
        copy_verified(
            local / "dataset_metadata.json", target_meta, expected_sha256=data["metadata_sha256"]
        )
    copy_verified(manifest, local / "dataset.manifest.json")
    if copy_weights:
        copy_verified(source, local / "latest.pt")
    return local / "latest.pt", local / "dataset.manifest.json"


def training_argv(args, snapshot, local_stage, registry, *, resume, parent, parent_manifest, seats):
    argv = [
        sys.executable,
        str(ROOT / "scripts/train_state_bc.py"),
        str(snapshot),
        "--out",
        str(local_stage / "training"),
        "--backup-dir",
        str(local_stage / "backup"),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--max-episodes",
        str(seats),
        "--max-validation-episodes",
        "0",
        "--validation-every-episodes",
        "0",
        "--max-validation-games",
        str(args.validation_games),
        "--validation-fraction",
        str(args.validation_fraction if args.validation_fraction is not None else 0.1),
        "--no-initial-validation",
        "--checkpoint-every-episodes",
        str(args.checkpoint_every_seats),
        "--chunk-size",
        "32",
        "--frame-microbatch",
        str(args.frame_microbatch),
        "--episode-batch-size",
        str(args.episode_batch_size),
        "--precision",
        "fp16" if args.device.startswith("cuda") else "fp32",
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        "0.01",
        "--position-coef",
        "1.0",
        "--tile-label-sigma",
        "1.0",
        "--timing-objective",
        "natural",
        "--timing-target-action-prior",
        "0.20",
        "--seed",
        "20261110",
        "--decision-ticks",
        "8",
        "--max-grad-norm",
        "1.0",
        "--no-freeze-frame",
        "--frame-encoder-fp32",
        "--split-registry",
        str(registry),
        "--lr-schedule",
        "cosine" if resume else ("constant" if args.upgrade_placement else "cosine"),
        "--early-stopping-patience",
        str(args.patience),
        "--early-stopping-min-delta",
        str(args.min_delta),
        "--ram-limit-gib",
        str(args.ram_limit_gib),
        "--defer-metrics",
    ]
    if not args.upgrade_placement:
        argv += ["--warmup-steps", str(args.warmup_steps),
                 "--min-learning-rate", str(args.min_learning_rate)]
    if resume:
        argv += ["--resume", str(local_stage / "training/latest.pt")]
    else:
        argv += [
            "--upgrade-placement-from" if args.upgrade_placement else "--init-from-state",
            str(parent),
            "--init-state-dataset-manifest",
            str(parent_manifest),
        ]
    if args.device.startswith("cuda"):
        argv += ["--fused-optimizer", "--float32-matmul-precision", "high"]
    if args.trust_data_files:
        argv += ["--trust-data-files"]
    if args.allow_episode_batch_transition:
        argv += ["--allow-episode-batch-transition"]
    if args.unfreeze_after_checkpoints is not None and not resume:
        argv += ['--unfreeze-after-checkpoints', str(args.unfreeze_after_checkpoints),
                 '--unfreeze-learning-rate', str(args.unfreeze_learning_rate)]
    return argv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, action="append", default=[])
    p.add_argument("--collection-root", type=Path, action="append", default=[])
    p.add_argument(
        "--trust-data-files",
        action="store_true",
        help="Skip extra data/checkpoint content hash passes; trust storage transfers",
    )
    p.add_argument("--require-complete-collection", action="store_true")
    p.add_argument("--expected-total-games", type=int, default=0)
    p.add_argument("--expected-source-shards", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--local-work", type=Path, required=True)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--parent-manifest", type=Path, required=True)
    p.add_argument("--upgrade-placement", action="store_true",
                   help="Start a separate frozen-trunk card-prior/CNN placement stage")
    p.add_argument("--placement-cnn-channels", type=int, choices=(64,), default=64)
    p.add_argument('--unfreeze-after-checkpoints', type=int, default=None)
    p.add_argument('--unfreeze-learning-rate', type=float, default=1e-5)
    p.add_argument("--split-registry", type=Path, required=True)
    p.add_argument("--min-train-games", type=int, default=1000)
    p.add_argument("--episode-batch-size", type=int, default=1)
    p.add_argument("--frame-microbatch", type=int, default=32)
    p.add_argument("--allow-episode-batch-transition", action="store_true")
    p.add_argument("--validation-games", type=int, default=16)
    p.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Resize the local collection's validation pool before the new stage",
    )
    p.add_argument("--validation-seed", type=int, default=20261111)
    p.add_argument(
        "--epochs", type=int, default=8, help="Maximum completed epochs across all data snapshots"
    )
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Complete epochs without improvement before stopping; 0 disables",
    )
    p.add_argument("--min-delta", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=350)
    p.add_argument(
        "--checkpoint-every-seats",
        type=int,
        default=512,
        help="Verified full latest checkpoint interval; complete epochs always save",
    )
    p.add_argument("--ram-limit-gib", type=float, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument('--allow-epoch-extension', action='store_true',
                   help='Explicitly extend the same constant-LR placement/joint run without resetting state')
    p.add_argument('--metrics-every-updates',type=int,default=32)
    a = p.parse_args(argv)
    if a.metrics_every_updates<1:
        p.error('metrics-every-updates must be positive')
    if a.unfreeze_after_checkpoints is not None and (a.unfreeze_after_checkpoints < 1 or not a.upgrade_placement):
        p.error('Scheduled unfreeze requires --upgrade-placement and a positive checkpoint count')
    if not math.isfinite(a.unfreeze_learning_rate) or a.unfreeze_learning_rate <= 0:
        p.error('Full-finetune learning rate must be finite and positive')
    if (
        a.min_train_games < 1
        or a.episode_batch_size < 1
        or a.frame_microbatch < 1
        or a.epochs < 1
        or a.patience < 0
        or a.warmup_steps < 0
        or a.checkpoint_every_seats < 1
        or a.expected_total_games < 0
        or a.expected_source_shards < 0
        or a.validation_games < 1
        or (a.validation_fraction is not None and not 0 < a.validation_fraction < 1)
        or not math.isfinite(a.min_delta)
        or a.min_delta < 0
        or not 0 <= a.min_learning_rate <= a.learning_rate
        or a.learning_rate <= 0
    ):
        p.error("Invalid dataset, epoch, patience or learning-rate bounds")
    if not a.manifest and not a.collection_root:
        p.error("At least one manifest or collection root is required")
    if a.validation_fraction is not None and (len(a.collection_root) != 1 or a.manifest):
        p.error("Validation repartition requires one local collection root")
    if a.upgrade_placement and a.warmup_steps:
        p.error("Placement stage uses constant LR; set --warmup-steps 0")
    return a


def commit_epoch(state, pending, progress, info, args):
    """Derive completion from committed cursor; never blindly increment twice."""
    target = pending["start_epoch"] + 1
    epoch, index = progress["epoch"], progress["next_episode"]
    if index == pending["train_seats"]:
        epoch, index = epoch + 1, 0  # Compatible old end-of-epoch cursor.
    if epoch != target or index != 0:
        raise ValueError("Trainer did not commit exactly one complete epoch")
    if state["completed_epochs"] != pending["base_completed_epochs"]:
        raise ValueError("Controller epoch transaction was already committed or is inconsistent")
    state.update(
        completed_epochs=pending["base_completed_epochs"] + 1,
        stage=pending["stage"],
        fingerprint=pending["fingerprint"],
        dataset_manifest=pending["snapshot"],
        backup_manifest=pending["backup_manifest"],
        latest_checkpoint=pending["latest_checkpoint"],
        backup_checkpoint=pending["backup_checkpoint"],
        local_stage=pending["local_stage"],
        stage_completed_epochs=epoch,
        updated_unix=time.time(),
    )
    state.pop("pending", None)
    if args.patience and info["bad_complete_epochs"] >= args.patience:
        state["early_stopped"] = True


def allow_unfreeze_contract_extension(state, contract, requested):
    old_contract = state.get('contract')
    if (requested is not None and old_contract is not None
            and 'unfreeze_schedule' not in old_contract
            and old_contract == {k:v for k,v in contract.items() if k != 'unfreeze_schedule'}):
        state['contract'] = contract
        state.setdefault('unfreeze_changes', []).append({'config': contract['unfreeze_schedule'], 'unix': time.time()})


def allow_epoch_contract_extension(state, contract, requested):
    previous=state.get('contract')
    if not requested or previous is None or previous==contract:
        return
    before={k:v for k,v in previous.items() if k!='epochs'}
    after={k:v for k,v in contract.items() if k!='epochs'}
    if (before!=after or contract['epochs']<=previous['epochs']
            or contract.get('placement_stage',{}).get('lr_schedule')!='constant'):
        raise ValueError('Epoch extension may only increase the horizon of the same constant-LR run')
    state.setdefault('epoch_extensions',[]).append({'previous_epochs':previous['epochs'],
        'epochs':contract['epochs'],'completed_epochs':state.get('completed_epochs',0),
        'optimizer_and_cursor_preserved':True,'unix':time.time()})
    state['contract']=contract


def run(args):
    import fcntl

    from notebook_state_bc_setup import copy_verified

    from cr_rl.data.state_dataset_catalog import build_catalog_snapshot

    request_path = args.out / 'unfreeze_request.json'
    if args.unfreeze_after_checkpoints is None and request_path.is_file():
        request = read_json(request_path)
        if request.get('format') != 'crrl-user-unfreeze-request-v1' or not args.upgrade_placement:
            raise ValueError('Unsupported unfreeze handoff request')
        count, lr = request['checkpoints'], request['learning_rate']
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('Invalid unfreeze handoff request')
        args.unfreeze_after_checkpoints, args.unfreeze_learning_rate = count, lr

    if args.trust_data_files:
        copy_verified = copy_trusted

    prepare_event(
        "directories",
        "Preparing controller and local SSD work directories",
        output=str(args.out),
        local_work=str(args.local_work),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    args.local_work.mkdir(parents=True, exist_ok=True)
    if args.validation_fraction is not None:
        from cr_rl.data.state_validation import prepare_validation_registry

        args.split_registry = prepare_validation_registry(
            args.collection_root[0],
            args.local_work / "validation_budget_registry.json",
            fraction=args.validation_fraction,
            seed=args.validation_seed,
        )
        budget = read_json(args.split_registry)
        prepare_event(
            "validation_budget",
            "Using reduced validation pool and random game budget",
            split_counts=budget["split_counts"],
            games_per_validation=args.validation_games,
            initial_validation=False,
        )
    with controller_lock_path(args.local_work, args.out).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = args.out / "controller.json"
        state = read_json(state_path) if state_path.exists() else {"completed_epochs": 0}
        contract = {
            key: getattr(args, key)
            for key in (
                "epochs",
                "patience",
                "min_delta",
                "learning_rate",
                "min_learning_rate",
                "warmup_steps",
                "require_complete_collection",
                "expected_total_games",
                "expected_source_shards",
                "validation_games",
                "validation_fraction",
                "validation_seed",
            )
        }
        contract["seed_registry_sha256"] = sha(args.split_registry)
        if args.upgrade_placement:
            contract["placement_stage"] = {
                "variant": "card_prior_cnn_001", "cnn_channels": args.placement_cnn_channels,
                "frozen_trunk": True, "lr_schedule": "constant", "temperature": 1.,
            }
        if args.unfreeze_after_checkpoints is not None:
            contract['unfreeze_schedule'] = {'checkpoints': args.unfreeze_after_checkpoints,
                'checkpoint_interval': args.checkpoint_every_seats, 'learning_rate': args.unfreeze_learning_rate}
        if args.episode_batch_size > 1:
            contract['episode_batch_size'] = args.episode_batch_size
        # Explicitly enabling the user-requested handoff may extend an existing
        # placement controller; every other part of its contract must match.
        allow_unfreeze_contract_extension(state, contract, args.unfreeze_after_checkpoints)
        allow_epoch_contract_extension(state, contract, args.allow_epoch_extension)
        if args.episode_batch_size > 1:
            contract["episode_batch_size"] = args.episode_batch_size
        previous_contract = state.get("contract", contract)
        if (
            previous_contract != contract
            and args.patience == 0
            and {k: v for k, v in previous_contract.items() if k != "patience"}
            == {k: v for k, v in contract.items() if k != "patience"}
        ):
            state.setdefault("early_stopping_changes", []).append(
                {
                    "previous_patience": previous_contract.get("patience"),
                    "patience": 0,
                    "unix": time.time(),
                }
            )
            state["contract"] = contract
        if args.patience == 0:
            state.pop("early_stopped", None)
        if (
            previous_contract != contract
            and args.allow_episode_batch_transition
            and {k: v for k, v in previous_contract.items() if k != "episode_batch_size"}
            == {k: v for k, v in contract.items() if k != "episode_batch_size"}
        ):
            state.setdefault("batch_transitions", []).append(
                {
                    "previous_batch_size": previous_contract.get("episode_batch_size", 1),
                    "episode_batch_size": args.episode_batch_size,
                    "unix": time.time(),
                    "optimizer_preserved": True,
                }
            )
            state["contract"] = contract
        if state.get("contract", contract) != contract:
            raise ValueError("Controller configuration changed; use a distinct output directory")
        state["contract"] = contract
        seed_registry = assignments(args.split_registry)
        registry_path = args.out / "split_registry.json"
        existing = assignments(registry_path) if registry_path.exists() else {}
        while state["completed_epochs"] < args.epochs and not state.get("early_stopped"):
            pending = state.get("pending")
            if pending:
                if child_is_alive(pending):
                    raise RuntimeError(
                        "The pending trainer may still be running; refusing a duplicate child"
                    )
                manifests = list(map(Path, pending["source_manifests"]))
                registry = assignments(Path(pending["stage_path"]) / "split_registry.json")
            else:
                if args.require_complete_collection:
                    complete, report = collection_completeness(
                        args.collection_root,
                        expected_total_games=args.expected_total_games,
                        expected_source_shards=args.expected_source_shards,
                    )
                    if not complete:
                        atomic_json(args.out / "data_status.json", report)
                        print(json.dumps(report, ensure_ascii=False), flush=True)
                        return report
                prepare_event("manifest_discovery_start", "Discovering committed dataset manifests")
                manifests = discover_manifests(args.manifest, args.collection_root)
                prepare_event(
                    "manifest_discovery_complete",
                    "Dataset manifest discovery complete",
                    manifests=len(manifests),
                )
                if not manifests:
                    report = {
                        "phase": "waiting_for_more_data",
                        "games": 0,
                        "train": 0,
                        "validation": 0,
                        "heldout": 0,
                        "required_train_games": args.min_train_games,
                        "message": "No producer-committed compatible batches have been published.",
                    }
                    atomic_json(args.out / "data_status.json", report)
                    return report
                registry = registry_for_sources(
                    manifests,
                    seed_registry,
                    existing,
                    initialize_validation=not state.get("split_initialized", False),
                )
            prepare_event(
                "catalog_build_start",
                "Verifying datasets and building local catalog",
                manifests=len(manifests),
            )
            result = build_catalog_snapshot(
                manifests,
                args.local_work / "snapshots",
                cache_root=args.local_work / "compressed-cache",
                split_registry=registry,
                trust_data_files=args.trust_data_files,
            )
            prepare_event(
                "catalog_build_complete",
                "Local catalog ready",
                games=result.get("games"),
                manifest=result["manifest_path"],
            )
            snapshot = Path(result["manifest_path"])
            manifest = read_json(snapshot)
            metadata = read_json(snapshot.parent / manifest["metadata"])
            registry, split = frozen_registry(metadata, seed_registry, registry)
            counts = {
                key: sum(value == key for value in split.values())
                for key in ("train", "validation", "heldout")
            }
            if args.expected_total_games and result.get("games") != args.expected_total_games:
                report = {
                    "phase": "waiting_for_complete_collection",
                    "reason": "catalog_game_count_mismatch",
                    "expected_total_games": args.expected_total_games,
                    "catalog_games": result.get("games"),
                }
                atomic_json(args.out / "data_status.json", report)
                return report
            for partition in ("validation", "heldout"):
                identities = sorted(
                    identity(metadata["replays"][i])
                    for i, value in split.items()
                    if value == partition
                )
                key = "fixed_" + partition + "_identities"
                # Source-wide seed registries may name validation games before
                # their converted shards are published. Existing available
                # validation games may never disappear; newly published seeded
                # validation games are admitted with their frozen assignment.
                changed = not set(state[key]).issubset(identities) if key in state else False
                if changed:
                    raise ValueError(f"The frozen {partition} identity set changed")
                state[key] = identities
            state["split_initialized"] = True
            atomic_json(registry_path, {"assignments": registry})
            atomic_json(state_path, state)
            existing = registry
            if counts["train"] < args.min_train_games:
                report = {
                    "phase": "waiting_for_more_data",
                    "games": len(split),
                    **counts,
                    "required_train_games": args.min_train_games,
                    "message": "More compatible episodes are required before model loading.",
                }
                atomic_json(args.out / "data_status.json", report)
                print(json.dumps(report, ensure_ascii=False), flush=True)
                return report
            if not counts["validation"]:
                raise ValueError("A fixed nonempty validation partition is required")
            digest = sha(snapshot)
            if pending and digest != pending["fingerprint"]:
                raise ValueError(
                    "Pending epoch snapshot changed; refusing a mixed-epoch continuation"
                )
            stage_number = int(state.get("stage", 0))
            if not pending and state.get("fingerprint") and state["fingerprint"] != digest:
                if args.upgrade_placement:
                    raise ValueError(
                        "Placement stage dataset changed; use an explicit new stage with a rebuilt train-only prior"
                    )
                stage_number += 1
            if pending:
                stage_number = pending["stage"]
            stage = args.out / f"stage_{stage_number:03d}_{digest[:12]}"
            local_stage = args.local_work / stage.name
            stage.mkdir(parents=True, exist_ok=True)
            local_stage.mkdir(parents=True, exist_ok=True)
            copy_verified(snapshot, stage / "dataset.manifest.json")
            copy_verified(snapshot.parent / manifest["metadata"], stage / manifest["metadata"])
            copy_verified(registry_path, stage / "split_registry.json")
            training = (
                restore_training(local_stage, stage, copy_verified)
                if not args.plan_only
                else local_stage / "training"
            )
            latest = training / "latest.pt"
            progress, info = (
                {"epoch": 0, "next_episode": 0, "optimizer_steps": 0},
                {"bad_complete_epochs": 0, "best_validation": None},
            )
            if latest.exists():
                progress, info = local_progress(latest, digest, split)
            seats = 2 * counts["train"]
            if progress["next_episode"] > seats:
                raise ValueError("Checkpoint cursor exceeds current training seats")
            # Recover a finished child even if the controller died before its JSON commit.
            normalized_epoch = progress["epoch"] + int(progress["next_episode"] == seats)
            if pending and normalized_epoch == pending["start_epoch"] + 1:
                commit_epoch(state, pending, progress, info, args)
                atomic_json(state_path, state)
                continue
            if args.patience and info["bad_complete_epochs"] >= args.patience:
                state["early_stopped"] = True
                atomic_json(state_path, state)
                break
            if pending and progress["epoch"] != pending["start_epoch"]:
                raise ValueError("Checkpoint epoch contradicts pending transaction")
            if not pending and state.get("fingerprint") == digest and latest.exists():
                expected = int(state.get("stage_completed_epochs", progress["epoch"]))
                if progress["epoch"] != expected or progress["next_episode"]:
                    raise ValueError("Stage checkpoint differs from committed controller progress")
            remaining = seats - progress["next_episode"]
            if remaining <= 0:
                raise ValueError("Noncanonical completed cursor lacks a recovery transaction")
            parent = parent_manifest = None
            if not latest.exists():
                prepare_event("parent_checkpoint_start", "Preparing parent checkpoint on local SSD")
                parent, parent_manifest = stage_parent(
                    args, state, copy_verified, copy_weights=not args.plan_only
                )
                prepare_event(
                    "parent_checkpoint_complete",
                    "Parent checkpoint preparation complete",
                    checkpoint=str(parent),
                    manifest=str(parent_manifest),
                )
            argv = training_argv(
                args,
                snapshot,
                local_stage,
                registry_path,
                resume=latest.exists(),
                parent=parent,
                parent_manifest=parent_manifest,
                seats=remaining,
            )
            argv[argv.index("--backup-dir") + 1] = str(stage / "training")
            plan = {
                "argv": argv,
                "snapshot": result,
                "split_counts": counts,
                "global_completed_epochs": state["completed_epochs"],
                "remaining_seats": remaining,
            }
            atomic_json(args.out / "next_command.json", plan)
            if args.plan_only:
                return plan
            pending = pending or {
                "base_completed_epochs": state["completed_epochs"],
                "start_epoch": progress["epoch"],
                "stage": stage_number,
                "fingerprint": digest,
                "snapshot": str(snapshot),
                "backup_manifest": str(stage / "dataset.manifest.json"),
                "stage_path": str(stage),
                "local_stage": str(local_stage),
                "latest_checkpoint": str(latest),
                "backup_checkpoint": str(stage / "training/latest.pt"),
                "train_seats": seats,
                "source_manifests": list(map(str, manifests)),
            }
            state["pending"] = pending
            atomic_json(state_path, state)
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
            summary_path = training / "summary.json"
            summary_path.unlink(
                missing_ok=True
            )  # A stale prior invocation cannot signal completion.
            with (args.out / "metrics.jsonl").open("a", buffering=1) as metrics:
                process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                pending.update(child_pid=process.pid, child_host=socket.gethostname())
                atomic_json(state_path, state)
                for line in process.stdout:
                    print(line.rstrip(), flush=True)
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict) and "phase" in record:
                            metrics.write(
                                json.dumps(
                                    {
                                        **record,
                                        "data_stage": stage_number,
                                        "snapshot_sha256": digest,
                                    }
                                )
                                + "\n"
                            )
                    except json.JSONDecodeError:
                        pass
                if process.wait():
                    raise RuntimeError(
                        "Trainer failed; committed checkpoint can resume; see trainer output"
                    )
            if not summary_path.is_file():
                raise ValueError("Trainer exited without its summary receipt")
            summary = read_json(summary_path)
            if summary.get("status") not in (
                "bounded_run_complete",
                "epochs_complete",
                "early_stopped",
            ):
                raise ValueError("Unknown trainer completion status")
            progress, info = local_progress(latest, digest, split)
            commit_epoch(state, pending, progress, info, args)
            copy_verified(summary_path, stage / "training/summary.json")
            atomic_json(state_path, state)
        result = {
            "status": "early_stopped"
            if state.get("early_stopped")
            else "total_epoch_budget_complete",
            **state,
        }
        atomic_json(args.out / "summary.json", result)
        return result


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, default=str))
