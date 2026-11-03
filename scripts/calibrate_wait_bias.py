"""Fit WAIT timing on unbalanced held-out rows; never modifies training checkpoints.

Probability-only mode uses small NPZ arrays exported by a trusted producer. Direct
checkpoint mode owns a separate decompression cache and can reuse an existing
training cache read-only. Pause the trainer externally before sharing GPU/IO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cr_rl.training.calibration import audit_validation_provenance, calibrate_timing
from cr_rl.training.persistence import atomic_json, file_sha256


class ReadOnlyReuseCache:
    """Read immutable digest-named arrays without touching another cache's LRU."""

    def __init__(self, own, readonly):
        self.own, self.readonly = own, readonly

    def read(self, source, digest, field, key):
        path = self.readonly / digest / f"{field}.npy"
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except FileNotFoundError:
            return self.own.read(source, digest, field, key)
        try:
            return np.array(array[key], copy=True)
        finally:
            array._mmap.close()


def extract_validation(args):
    # Imported lazily so probability-only calibration needs no Torch or JVM.
    import torch
    from train_bc import make_schema

    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.data.sharded_bc import load_sharded_bc
    from cr_rl.data.validation_cache import prepare_validation_arrays
    from cr_rl.models import ActorCritic, ActorPolicy, ModelConfig
    from cr_rl.training.bc_batch import build_bc_batch
    from cr_rl.training.separate import autocast_for

    checkpoint = args.checkpoint.resolve()
    manifest = args.dataset / "dataset.manifest.json" if args.dataset.is_dir() else args.dataset
    cache_dir = args.cache_dir or args.out.parent / ".calibration_arrays"
    if cache_dir.name in ("array_cache", ".bc_arrays"):
        raise ValueError("Use a dedicated calibration cache, not the trainer's cache directory")
    if args.reuse_cache_dir and cache_dir.resolve() == args.reuse_cache_dir.resolve():
        raise ValueError("Read-only reuse and private calibration caches must differ")
    checkpoint_digest = file_sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    digest = file_sha256(manifest)
    if state.get("dataset_sha256") != digest:
        raise ValueError("Checkpoint and frozen calibration dataset hashes differ")
    if state.get("model_architecture_version") != 2:
        raise ValueError("Calibration requires architecture-v2 weights")
    cfg = ModelConfig(**state["config"])
    validation = np.asarray(state.get("validation_replays", []), dtype=np.int64)
    heldout = np.asarray(state.get("heldout_deck_replays_excluded", []), dtype=np.int64)
    if not len(validation):
        raise ValueError("Checkpoint contains no validation partition")
    data, metadata = load_sharded_bc(manifest, cache_dir, args.cache_gib)
    if args.reuse_cache_dir:
        for name in ("observations", "masks"):
            for array in data[name].arrays:
                array.cache = ReadOnlyReuseCache(array.cache, args.reuse_cache_dir)
    obs, masks = data["observations"], data["masks"]
    actions, ids, seats = data["actions"], data["replay_ids"], data["seats"]
    excluded = np.asarray(metadata.get("excluded_replay_ids", []), dtype=np.int64)
    candidates = np.flatnonzero(
        (data["flags"] == 0)
        & np.isin(ids, validation)
        & ~np.isin(ids, heldout)
        & ~np.isin(ids, excluded)
    )
    permutation = np.random.default_rng(args.seed).permutation(candidates)
    selected = []
    # Rejection sampling is uniform among valid rows, without ACT oversampling.
    for left in range(0, len(permutation), max(512, args.samples * 2)):
        draws = permutation[left : left + max(512, args.samples * 2)]
        valid = masks[draws, actions[draws]].astype(bool)
        selected.extend(draws[valid].tolist())
        if len(selected) >= args.samples:
            break
    rows = np.sort(np.asarray(selected[: args.samples], dtype=np.int64))
    if not len(rows):
        raise ValueError("Frozen validation partition contains no legal supervised rows")
    selected_games = np.unique(ids[rows])
    groups = {}
    for index in np.flatnonzero(np.isin(ids, selected_games)):
        groups.setdefault((int(ids[index]), int(seats[index])), []).append(int(index))
    for group in groups.values():
        group.sort(key=lambda i: int(data["ticks"][i]))
    positions = {key: {row: i for i, row in enumerate(group)} for key, group in groups.items()}
    schema = make_schema(metadata)
    obs, masks = prepare_validation_arrays(
        obs,
        masks,
        rows,
        ids,
        seats,
        groups,
        positions,
        cfg.context_len,
        cache_dir.parent / ".timing_validation_cache",
        digest,
        progress=lambda field, done, total: print(
            json.dumps(
                {
                    "phase": "calibration_cache",
                    "field": field,
                    "shards_done": done,
                    "shards_total": total,
                }
            ),
            flush=True,
        ),
    )
    adapter = CrForgeObservationAdapter(
        schema, len(metadata["vocabulary"]["names"]), entity_bucket=16
    )
    model = (ActorPolicy if state.get("policy_only") else ActorCritic)(cfg)
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    if file_sha256(checkpoint) != checkpoint_digest or file_sha256(manifest) != digest:
        raise ValueError("Checkpoint or frozen manifest changed during calibration extraction")
    provenance = {
        "partition": "validation",
        "natural_sampling": True,
        "sampling": "uniform without replacement among flag-zero, legal validation rows",
        "checkpoint_sha256": checkpoint_digest,
        "dataset_sha256": digest,
        "checkpoint": str(checkpoint),
        "dataset": str(manifest.resolve()),
        "optimizer_steps": state.get("optimizer_steps"),
        "validation_replays": validation.tolist(),
        "heldout_deck_replays_excluded": heldout.tolist(),
        "excluded_replay_ids": excluded.tolist(),
        "seed": args.seed,
        "requested_rows": args.samples,
        "decision_seconds": float(metadata["decision_ticks"]) * 0.05,
        "private_cache": str(cache_dir.resolve()),
        "read_only_reuse_cache": str(args.reuse_cache_dir) if args.reuse_cache_dir else None,
    }
    del state
    odds, legal_act = [], []
    with torch.inference_mode():
        for left in range(0, len(rows), args.batch_size):
            selected_rows = rows[left : left + args.batch_size]
            tensors = build_bc_batch(
                selected_rows,
                obs,
                masks,
                actions,
                ids,
                seats,
                groups,
                positions,
                adapter,
                cfg.context_len,
                args.device,
            )
            h, e, em, legal, tm, hand, _, deck, seen = tensors
            if not legal[:, 0].all():
                raise ValueError("WAIT must be legal for scalar WAIT calibration")
            with autocast_for(args.device):
                logp, _ = model(
                    h,
                    e,
                    em,
                    legal,
                    tm,
                    hand_card_ids=hand,
                    deck_card_ids=deck,
                    enemy_seen_card_ids=seen,
                )
            logp = logp.float()
            odds.extend((torch.logsumexp(logp[:, 1:], -1) - logp[:, 0]).cpu().tolist())
            legal_act.extend(legal[:, 1:].any(-1).cpu().tolist())
            print(
                json.dumps(
                    {
                        "phase": "calibration_extract",
                        "rows": min(left + args.batch_size, len(rows)),
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    arrays = {
        "act_log_odds": np.asarray(odds),
        "actions": actions[rows],
        "legal_act": np.asarray(legal_act),
        "replay_ids": ids[rows],
        "rows": rows,
    }
    capture = args.capture_out or args.out.with_suffix(".samples.npz")
    capture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(capture, **arrays, provenance_json=json.dumps(provenance))
    provenance["sample_artifact_sha256"] = file_sha256(capture)
    return arrays, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input", type=Path, help="NPZ with act_log_odds/actions/legal_act/replay_ids"
    )
    mode.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--provenance", type=Path, help="JSON provenance for input-mode NPZ")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-out", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--reuse-cache-dir", type=Path, help="Existing .npy cache, read-only reuse")
    parser.add_argument("--cache-gib", type=float, default=8.0)
    parser.add_argument("--decision-seconds", type=float)
    parser.add_argument("--training-act-fraction", type=float, default=0.5)
    parser.add_argument("--bias-bounds", type=float, nargs=2, default=(-12.0, 12.0))
    parser.add_argument("--grid", type=float, nargs="+", default=[0, 1, 1.5, 2, 2.5])
    parser.add_argument("--target-rate", type=float)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.cache_gib <= 0:
        parser.error("Sample count, batch size and private cache budget must be positive")
    if args.decision_seconds is not None and args.decision_seconds <= 0:
        parser.error("--decision-seconds must be positive")
    if args.checkpoint:
        if not args.dataset:
            parser.error("--checkpoint requires --dataset")
        arrays, provenance = extract_validation(args)
    else:
        with np.load(args.input, allow_pickle=False) as data:
            arrays = {key: data[key] for key in ("act_log_odds", "actions", "replay_ids")}
            if "legal_act" in data:
                arrays["legal_act"] = data["legal_act"]
            if "rows" in data:
                arrays["rows"] = data["rows"]
            provenance = (
                json.loads(args.provenance.read_text())
                if args.provenance
                else json.loads(str(data["provenance_json"]))
            )
        provenance["sample_artifact_sha256"] = file_sha256(args.input)
    if len(arrays["replay_ids"]) != len(arrays["actions"]):
        raise ValueError("Row counts differ between replay provenance and action labels")
    if "rows" in arrays and (
        len(arrays["rows"]) != len(arrays["actions"])
        or len(np.unique(arrays["rows"])) != len(arrays["actions"])
    ):
        raise ValueError("Natural calibration cohort contains duplicate or missing row ids")
    partition = audit_validation_provenance(provenance, arrays["replay_ids"])
    report = calibrate_timing(
        arrays["act_log_odds"],
        arrays["actions"],
        legal_act=arrays.get("legal_act"),
        decision_seconds=args.decision_seconds or provenance.get("decision_seconds", 0.25),
        training_act_fraction=args.training_act_fraction,
        bias_bounds=args.bias_bounds,
        grid=args.grid,
        target_rate=args.target_rate,
    )
    report.update(provenance=provenance, partition_audit=partition)
    atomic_json(args.out, report)
    print(
        json.dumps(
            {
                "report": str(args.out),
                "nll_fitted_bias": report["nll_fitted"]["bias"],
                "theoretical_prior_bias": report["theoretical_prior_bias"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
