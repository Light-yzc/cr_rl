"""Real-replay v2 CUDA optimization and incremental-frame equivalence probe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from train_bc import make_schema

from cr_rl.data.bc_snapshot import read_published
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.models.frame_cache import FrameFeatureCache
from cr_rl.training.bc import hierarchical_bc_loss
from cr_rl.training.bc_batch import build_bc_batch
from cr_rl.training.bc_optimizer import make_bc_optimizer
from cr_rl.training.persistence import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="production")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20261103)
    manifest, metadata, dataset_hash = read_published(args.dataset)
    schema = make_schema(metadata)
    cfg = ModelConfig.preset(args.model).for_v2(schema)
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        raise RuntimeError("Need 8 GiB free for the v2 capacity probe")
    with np.load(args.dataset / manifest["shards"][0]["path"]) as data:
        obs, masks, actions = (data[k] for k in ("observations", "masks", "actions"))
        replays, seats, flags, ticks = (data[k] for k in ("replay_ids", "seats", "flags", "ticks"))
    groups = {}
    for row in range(len(actions)):
        groups.setdefault((int(replays[row]), int(seats[row])), []).append(row)
    for key in groups:
        groups[key].sort(key=lambda i: int(ticks[i]))
    positions = {key: {row: pos for pos, row in enumerate(rows)} for key, rows in groups.items()}
    valid = (flags == 0) & masks[np.arange(len(actions)), actions].astype(bool)
    plays = np.flatnonzero(valid & (actions > 0) & (actions < 2305))
    assert len(plays) >= args.batch_size
    rows = plays[: args.batch_size]
    adapter = CrForgeObservationAdapter(schema, cfg.vocab_size, entity_bucket=16)
    batch = build_bc_batch(
        rows,
        obs,
        masks,
        actions,
        replays,
        seats,
        groups,
        positions,
        adapter,
        cfg.context_len,
        "cuda",
    )
    h, e, em, legal, tm, hand, y, deck, seen = batch
    model = ActorCritic(cfg).cuda().train()
    optimizer = make_bc_optimizer(model, 3e-5, 3e-4, fused=True)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
            loss, _ = hierarchical_bc_loss(logp, y, legal, tile_label_sigma=1.0)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.frame.category_embeddings["row"].weight.grad.norm() > 0
        assert model.frame.dynamic_queries[-1].weight.grad.norm() > 0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "passed": False,
        "model": args.model,
        "config": cfg.__dict__,
        "parameters": model.parameter_count(),
        "dataset_manifest_sha256": dataset_hash,
        "gpu": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "history_len": cfg.context_len,
        "losses": losses,
        "last_preclip_grad_norm": float(norm),
        "two_optimizer_steps_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "limitation": "Capacity/equivalence probe; timing includes warmup, not a strength result",
    }
    model.eval()
    sequence = max(groups.values(), key=len)
    # Prefix-safe shards can contain only short opening fragments. Find a real
    # full-length history rather than padding/repeating a short trajectory.
    cache_shard = manifest["shards"][0]["path"]
    if len(sequence) < cfg.context_len + 2:
        for shard in manifest["shards"][1:]:
            with np.load(args.dataset / shard["path"]) as data:
                labels = data["replay_ids"].astype(np.int64) * 2 + data["seats"]
                ids, counts = np.unique(labels, return_counts=True)
                if counts.max() < cfg.context_len + 2:
                    continue
                selected = ids[counts.argmax()]
                sequence = np.flatnonzero(labels == selected)
                sequence = sequence[np.argsort(data["ticks"][sequence])]
                obs, masks = data["observations"], data["masks"]
                cache_shard = shard["path"]
                break
        assert len(sequence) >= cfg.context_len + 2, "No real 130-frame prefix is published yet"
    report["cache_source_shard"] = cache_shard
    cache = FrameFeatureCache(model, adapter, "cuda", capacity=cfg.context_len + 4)
    errors, times = [], {"full": [], "cached": []}
    for offset in range(3):
        window = obs[sequence[offset : offset + cfg.context_len]][None]
        time_mask = np.ones((1, cfg.context_len), dtype=bool)
        hh, ee, mm = (torch.as_tensor(v, device="cuda") for v in adapter(window))
        current = window[:, -1]
        kwargs = {
            name: torch.as_tensor(current[:, list(schema.header[field])], device="cuda")
            for name, field in (
                ("hand_card_ids", "hand_card"),
                ("deck_card_ids", "deck"),
                ("enemy_seen_card_ids", "enemy_seen"),
            )
        }
        lm = torch.as_tensor(
            masks[sequence[offset + cfg.context_len - 1]][None], device="cuda", dtype=torch.bool
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name in ("full", "cached"):
                torch.cuda.synchronize()
                start = time.perf_counter()
                encoded = cache.encode(window, time_mask) if name == "cached" else None
                result = model(
                    hh,
                    ee,
                    mm,
                    lm,
                    torch.as_tensor(time_mask, device="cuda"),
                    encoded_history=encoded,
                    **kwargs,
                )[0]
                torch.cuda.synchronize()
                times[name].append(time.perf_counter() - start)
                if name == "full":
                    expected = result
                else:
                    legal_mask = lm.bool()
                    error = (result[legal_mask] - expected[legal_mask]).abs().max().item()
                    errors.append(error)
                    torch.testing.assert_close(
                        result[legal_mask], expected[legal_mask], atol=0.05, rtol=0.005
                    )
    assert cache.encoded_frames == cfg.context_len + 2
    report.update(
        passed=True,
        cache_encoded_frames=cache.encoded_frames,
        cache_legal_logp_max_errors=errors,
        inference_seconds=times,
    )
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
