"""Explicit, dataset-bound upgrade into a frozen-trunk learned-placement stage.

Only the new placement decoder is optimized. The source checkpoint remains an
immutable parent; its epoch, Adam state and successful-update clock are not the
new stage's resume state.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from cr_rl.models.learned_placement import PlacementAdapter, definition_for
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.persistence import atomic_json, file_sha256
from cr_rl.training.state_bc import (CHECKPOINT_FORMAT, check_ram_budget, load_state_checkpoint,
                                    replay_identity)


STAGE_FORMAT = "crrl-placement-training-stage-v1"
PRIOR_FORMAT = "crrl-training-card-location-prior-v1"


def split_sha256(split):
    encoded = json.dumps({str(key): value for key, value in sorted(split.items())},
                         separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prior_binding(dataset, split, sigma):
    return {"format": PRIOR_FORMAT, "fit_partition": "train",
            "dataset_fingerprint": dataset.fingerprint,
            "metadata_sha256": dataset.manifest["metadata_sha256"],
            "split_sha256": split_sha256(split),
            "fit_replay_ids": sorted(key for key, value in split.items() if value == "train"),
            "fit_games": sum(value == "train" for value in split.values()),
            "sigma": float(sigma), "training_play_rows": dataset.partition_counts(split, "train")["play"]}


def validate_training_prior(prior, dataset, split, sigma):
    expected = _prior_binding(dataset, split, sigma)
    if any(prior.get(key) != value for key, value in expected.items()):
        raise ValueError("Placement prior is not bound to this exact train partition, metadata and target")
    masses, counts = prior.get("mass"), prior.get("card_counts")
    if not isinstance(masses, dict) or not isinstance(counts, dict) or masses.keys() != counts.keys():
        raise ValueError("Placement prior needs matching card mass and count tables")
    total = 0
    vocab_size = len(dataset.metadata["vocabulary"]["names"])
    for card, mass in masses.items():
        if str(int(card)) != card or not 0 <= int(card) < vocab_size:
            raise ValueError("Placement prior card identity is outside this vocabulary")
        count, values = counts[card], np.asarray(mass, dtype=np.float64)
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1 or values.shape != (576,)
                or not np.isfinite(values).all() or (values < 0).any()
                or not np.isclose(values.sum(), count, rtol=1e-7, atol=1e-6)):
            raise ValueError("Placement prior contains invalid count or conditional target mass")
        total += count
    if total != expected["training_play_rows"] or not total:
        raise ValueError("Placement prior does not cover exactly the valid training play labels")
    return prior


def build_training_prior(dataset, split, sigma, path):
    """Read each training game once; retain only small per-card 576-grid totals."""
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Placement prior sigma must be finite and nonnegative")
    path = Path(path)
    if path.exists():
        return validate_training_prior(json.loads(path.read_text()), dataset, split, sigma)
    started, completed_seats, play_rows = time.perf_counter(), 0, 0
    total_seats = 2 * sum(value == "train" for value in split.values())

    def progress(stage, message):
        print(json.dumps({"phase": "state_bc_prepare", "stage": stage, "message": message,
                          "completed_training_seats": completed_seats, "total_training_seats": total_seats,
                          "training_play_rows": play_rows, "seconds": time.perf_counter() - started}), flush=True)

    progress("placement_prior_start", "Building the train-only card-location prior once")
    hand = np.asarray(dataset.schema.header["hand_card"])
    grid = np.arange(576)
    masses, counts = {}, {}
    episodes = dataset.episodes(split, "train", epoch=0, seed=0)
    try:
        for episode in episodes:
            completed_seats += 1
            chosen = np.flatnonzero(episode.supervised & (episode.labels > 0) & (episode.labels < 2305))
            rows, labels = episode.rows[chosen], episode.labels[chosen]
            slots, tiles = (labels - 1) // 576, (labels - 1) % 576
            play_rows += len(rows)
            if not len(rows):
                continue
            cards = episode.arrays["observations"][rows, hand[slots]]
            legal = np.asarray(episode.arrays["masks"][rows, 1:2305]).reshape(-1, 4, 576)[
                np.arange(len(rows)), slots].astype(bool)
            if sigma:
                distance = (grid[None] % 18 - tiles[:, None] % 18)**2
                distance += (grid[None] // 18 - tiles[:, None] // 18)**2
                targets = np.exp(-distance / (2 * sigma**2)) * legal
            else:
                targets = (grid[None] == tiles[:, None]).astype(np.float64) * legal
            targets /= targets.sum(-1, keepdims=True)
            for card in np.unique(cards):
                key, selected = str(int(card)), cards == card
                masses.setdefault(key, np.zeros(576, dtype=np.float64))
                masses[key] += targets[selected].sum(0)
                counts[key] = counts.get(key, 0) + int(selected.sum())
            check_ram_budget(dataset.ram_limit_gib)
            if completed_seats % 128 == 0:
                progress("placement_prior_progress", "Scanning training games for the card-location prior")
    finally:
        episodes.close()
    prior = {**_prior_binding(dataset, split, sigma), "card_counts": counts,
             "mass": {key: value.tolist() for key, value in masses.items()},
             "producer": "cr_rl.training.placement_stage.build_training_prior"}
    validate_training_prior(prior, dataset, split, sigma)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, prior)
    progress("placement_prior_complete", "Train-only card-location prior is ready")
    return prior


def set_placement_training_mode(model):
    if not hasattr(model, "learned_placement"):
        raise ValueError("Placement-only training requires an explicit learned placement decoder")
    model.requires_grad_(False)
    model.learned_placement.requires_grad_(True)
    # The spatial wrapper keeps a scalar compatibility anchor with no function.
    for name, parameter in model.learned_placement.named_parameters():
        if name.endswith(".anchor"):
            parameter.requires_grad_(False)


def _load_upgrade_source(args, dataset, split):
    """Allow file repacking only when an explicit parent sidecar proves identity."""
    parent_manifest = getattr(args, "init_state_dataset_manifest", None)
    if parent_manifest is None:
        return load_state_checkpoint(args.upgrade_placement_from,
                                     dataset_fingerprint=dataset.fingerprint, split=split), None
    source = torch.load(args.upgrade_placement_from, map_location="cpu", weights_only=False, mmap=True)
    if source.get("format") != CHECKPOINT_FORMAT or source.get("architecture") != "state_actor_critic":
        raise ValueError("Checkpoint is not an independent state BC architecture")
    progress = source.get("progress", {})
    if any(isinstance(progress.get(key), bool) or not isinstance(progress.get(key), int)
           or progress[key] < 0 for key in ("epoch", "next_episode", "optimizer_steps")):
        raise ValueError("Invalid state BC episode-boundary progress")
    parent_manifest = Path(parent_manifest).resolve()
    if file_sha256(parent_manifest) != source.get("dataset_fingerprint"):
        raise ValueError("Parent dataset manifest fingerprint differs from the source checkpoint")
    manifest = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "crrl-bc-shards-v1":
        raise ValueError("Placement upgrade parent manifest has an unsupported format")
    metadata_path = (parent_manifest.parent / manifest["metadata"]).resolve()
    if not metadata_path.is_relative_to(parent_manifest.parent):
        raise ValueError("Parent metadata must stay inside its sidecar directory")
    if file_sha256(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError("Parent dataset metadata checksum differs")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("schema") != dataset.metadata.get("schema")
            or metadata.get("vocabulary") != dataset.metadata.get("vocabulary")
            or metadata.get("decision_ticks") != dataset.metadata.get("decision_ticks")):
        raise ValueError("Placement upgrade parent schema, vocabulary or decision grid differs")
    try:
        source_split = {int(key): value for key, value in source["split"].items()}
        previous = {}
        for replay_id, partition in source_split.items():
            if not 0 <= replay_id < len(metadata["replays"]):
                raise ValueError("Invalid source replay identity")
            identity = replay_identity(metadata["replays"][replay_id])
            if partition not in ("train", "validation", "heldout"):
                raise ValueError("Invalid source partition")
            if previous.setdefault(identity, partition) != partition:
                raise ValueError("Conflicting source identity partitions")
        current = {}
        for replay_id, partition in split.items():
            identity = replay_identity(dataset.metadata["replays"][replay_id])
            if current.setdefault(identity, partition) != partition:
                raise ValueError("Conflicting current identity partitions")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Placement upgrade parent has invalid replay/split identities") from error
    if previous != current or len(source_split) != len(split):
        raise ValueError("Placement upgrade requires the same replay identities and train/validation split")
    binding = {"parent_manifest_sha256": source["dataset_fingerprint"],
               "parent_metadata_sha256": manifest["metadata_sha256"],
               "new_dataset_fingerprint": dataset.fingerprint,
               "same_replay_identities_and_partitions": True}
    return source, binding


def initialize_placement_stage(args, dataset, split):
    source_path = args.upgrade_placement_from
    source, source_binding = _load_upgrade_source(args, dataset, split)
    if source.get("model_architecture_version") != 3 or source.get("decision_ticks") != args.decision_ticks:
        raise ValueError("Placement upgrade requires the same architecture-v3 decision grid")
    if source.get("learned_placement_definition") is not None:
        raise ValueError("Source already has learned placement; resume its own stage instead of upgrading again")
    model = state_model_from_checkpoint(source, args.device)
    cfg = model.cfg
    if (cfg.header_dim != dataset.schema.header_ints or cfg.entity_feature_dim != dataset.schema.entity_ints
            or tuple(cfg.header_fields) != dataset.schema.header_fields
            or tuple(cfg.entity_fields) != dataset.schema.entity_fields
            or cfg.vocab_size != len(dataset.metadata["vocabulary"]["names"])):
        raise ValueError("Placement source configuration differs from the exact dataset representation")
    model.frame_microbatch = args.frame_microbatch
    prior_path = args.placement_prior or args.out / "placement_prior.json"
    if args.placement_prior:
        prior = validate_training_prior(json.loads(prior_path.read_text()), dataset, split, args.tile_label_sigma)
    else:
        prior = build_training_prior(dataset, split, args.tile_label_sigma, prior_path)
    definition = definition_for(model, "card_prior_cnn_001", cnn_channels=64, alpha=.01, temperature=1.)
    adapter = PlacementAdapter(model, definition)
    table = adapter.decoder.original.card_location_bias.weight
    final = adapter.decoder.original.original.tile_actor[-1]
    with torch.no_grad():
        table.fill_(.01)
        for card, mass in prior["mass"].items():
            table[int(card) + 1].copy_(torch.as_tensor(mass, device=table.device, dtype=table.dtype) + .01)
        table.log_()
        final.weight.zero_()
        if final.bias is not None:
            final.bias.zero_()
    model.add_module("learned_placement", adapter)
    model.learned_placement_definition = dict(definition)
    set_placement_training_mode(model)
    report = {"format": STAGE_FORMAT, "mode": "placement_only",
              "variant": "card_prior_cnn_001", "definition": definition,
              "source_checkpoint_sha256": file_sha256(source_path),
              "source_progress": dict(source["progress"]),
              "source_dataset_fingerprint": source["dataset_fingerprint"],
              "prior_sha256": file_sha256(prior_path),
              "train_split_sha256": split_sha256(split),
              "temperature": 1., "parent_optimizer_restored": False,
              "parent_epoch_restored": False, "stage_epoch_start": 0,
              "stage_optimizer_steps_start": 0,
              "original_trunk_and_actor_weights_preserved": True}
    if source_binding is not None:
        report["source_binding"] = source_binding
    del source
    gc.collect()
    check_ram_budget(args.ram_limit_gib)
    return model, report
