"""Audit timing across fixed expert-trajectory windows, separately from closed-loop recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from calibrate_wait_bias import ReadOnlyReuseCache
from train_bc import make_schema

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.data.sharded_bc import load_sharded_bc
from cr_rl.data.validation_cache import prepare_validation_arrays
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.bc_batch import build_bc_batch
from cr_rl.training.calibration import adjusted_act_probability, audit_validation_provenance
from cr_rl.training.persistence import atomic_json, file_sha256
from cr_rl.training.separate import autocast_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--reuse-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    d = args.calibration_dir
    report = json.loads((d / "report.json").read_text())
    provenance = report["provenance"]
    checkpoint = d / "candidate.pt"
    if file_sha256(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("Window audit checkpoint differs from natural calibration")
    manifest = Path(provenance["dataset"])
    if file_sha256(manifest) != provenance["dataset_sha256"]:
        raise ValueError("Frozen window dataset changed")
    capture = d / "report.samples.npz"
    if file_sha256(capture) != provenance["sample_artifact_sha256"]:
        raise ValueError("Natural calibration sample capture changed")
    with np.load(capture, allow_pickle=False) as sample:
        centers = sample["rows"][sample["actions"] != 0]
    if not len(centers):
        raise ValueError("No held-out action rows are available for window auditing")
    data, metadata = load_sharded_bc(manifest, d / "window_private_arrays", 8)
    for field in ["observations", "masks"]:
        for array in data[field].arrays:
            array.cache = ReadOnlyReuseCache(array.cache, args.reuse_cache_dir)
    ids, seats, ticks = data["replay_ids"], data["seats"], data["ticks"]
    audit_validation_provenance(provenance, ids[centers])
    if int(metadata["decision_ticks"]) != 5:
        raise ValueError("Window offsets require the pinned 250 ms decision grid")
    games = np.unique(ids[centers])
    groups = {}
    for index in np.flatnonzero(np.isin(ids, games)):
        groups.setdefault((int(ids[index]), int(seats[index])), []).append(int(index))
    for group in groups.values():
        group.sort(key=lambda i: int(ticks[i]))
    positions = {key: {row: i for i, row in enumerate(group)} for key, group in groups.items()}
    windows = []
    for center in centers:
        group = groups[(int(ids[center]), int(seats[center]))]
        window = [
            i
            for i in group
            if ticks[center] - 20 <= ticks[i] <= ticks[center] + 10 and data["flags"][i] == 0
        ]
        windows.append(window)
    rows = np.unique(np.concatenate(windows))
    cfg_state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = ModelConfig(**cfg_state["config"])
    window_obs, window_masks = prepare_validation_arrays(
        data["observations"],
        data["masks"],
        rows,
        ids,
        seats,
        groups,
        positions,
        cfg.context_len,
        d / ".window_validation_cache",
        provenance["dataset_sha256"],
        progress=lambda field, done, total: print(
            json.dumps(
                {
                    "phase": "window_cache",
                    "field": field,
                    "shards_done": done,
                    "shards_total": total,
                }
            ),
            flush=True,
        ),
    )
    model = ActorCritic(cfg)
    model.load_state_dict(cfg_state["model"], strict=True)
    del cfg_state
    model.to(args.device).eval()
    adapter = CrForgeObservationAdapter(
        make_schema(metadata), len(metadata["vocabulary"]["names"]), 16
    )
    odds = []
    legal_acts = []
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            values = build_bc_batch(
                rows[start : start + 8],
                window_obs,
                window_masks,
                data["actions"],
                ids,
                seats,
                groups,
                positions,
                adapter,
                cfg.context_len,
                args.device,
            )
            h, e, em, legal, tm, hand, _, deck, seen = values
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
            odds.extend(
                (torch.logsumexp(logp.float()[:, 1:], -1) - logp.float()[:, 0]).cpu().tolist()
            )
            legal_acts.extend(legal[:, 1:].any(-1).cpu().tolist())
            print(json.dumps(dict(rows=min(start + 8, len(rows)), total=len(rows))), flush=True)
    indices = {int(row): i for i, row in enumerate(rows)}
    metrics = []
    for bias in [0, 1, 1.5, 2, 2.5, report["nll_fitted"]["bias"]]:
        probability = adjusted_act_probability(odds, bias, legal_acts)
        recalls = [
            1 - float(np.prod(1 - probability[[indices[i] for i in window]])) for window in windows
        ]
        metrics.append(
            dict(
                bias=bias,
                mean_fixed_expert_window_any_act_probability=float(np.mean(recalls)),
                greedy_binary_timing_any_act_fraction=float(
                    np.mean(
                        [
                            (probability[[indices[i] for i in window]] > 0.5).any()
                            for window in windows
                        ]
                    )
                ),
            )
        )
    atomic_json(
        d / "action_windows.json",
        dict(
            checkpoint_sha256=provenance["checkpoint_sha256"],
            dataset_sha256=provenance["dataset_sha256"],
            center_action_rows=len(centers),
            unique_window_rows=len(rows),
            window_rows_min=min(map(len, windows)),
            window_rows_max=max(map(len, windows)),
            before_seconds=1,
            after_seconds=0.5,
            metrics=metrics,
            limitation="Independent action draws along fixed expert histories, not closed-loop "
            "recall after state changes; tests timing only, not correct card or placement.",
        ),
    )


if __name__ == "__main__":
    main()
