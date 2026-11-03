"""Tune BC kernels/batch size on real history windows, then stress 200 entities."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from train_bc import make_schema

from cr_rl.data.bc_snapshot import read_published
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.bc import hierarchical_bc_loss
from cr_rl.training.bc_batch import attention_context, build_bc_batch
from cr_rl.training.bc_optimizer import make_bc_optimizer
from cr_rl.training.persistence import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--v2", action="store_true")
    args = parser.parse_args()
    if args.dataset.is_dir():
        manifest, metadata, _ = read_published(args.dataset)
        dataset_path = args.dataset / manifest["shards"][0]["path"]
    else:
        dataset_path = args.dataset
        metadata = json.loads(args.dataset.with_suffix(".json").read_text(encoding="utf-8"))
    schema = make_schema(metadata)
    with np.load(dataset_path) as data:
        obs, masks, actions = data["observations"], data["masks"], data["actions"].astype(np.int64)
        replays, seats, flags, ticks = (data[k] for k in ("replay_ids", "seats", "flags", "ticks"))
    groups = {}
    for row in range(len(actions)):
        groups.setdefault((int(replays[row]), int(seats[row])), []).append(row)
    for key in groups:
        groups[key].sort(key=lambda i: int(ticks[i]))
    positions = {key: {row: pos for pos, row in enumerate(rows)} for key, rows in groups.items()}
    valid = (flags == 0) & masks[np.arange(len(actions)), actions].astype(bool)
    plays, waits = np.flatnonzero(valid & (actions != 0)), np.flatnonzero(valid & (actions == 0))
    cfg = ModelConfig.preset("production")
    if args.v2:
        cfg = cfg.for_v2(schema)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest" if args.v2 else "high")
    torch.manual_seed(20261103)
    model = ActorCritic(cfg).cuda().train()
    results = []

    def measure(bs, bucket, fused, backend, stress=False):
        optimizer = make_bc_optimizer(model, 3e-5, 3e-4 if args.v2 else None, fused=fused)
        adapter = CrForgeObservationAdapter(schema, cfg.vocab_size, entity_bucket=bucket)
        rng = np.random.default_rng(20261103)
        entity_sizes = []
        accumulation = 96 // bs
        microbatch = 0
        optimizer.zero_grad(set_to_none=True)

        def step():
            nonlocal microbatch
            rows = np.concatenate(
                (rng.choice(plays, max(1, bs // 2)), rng.choice(waits, bs - bs // 2))
            )
            h, e, em, legal, tm, hand, y, deck, seen = build_bc_batch(
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
            if stress:
                padding = cfg.max_entities - e.shape[-2]
                e = torch.nn.functional.pad(e, (0, 0, 0, padding))
                em = torch.ones((*em.shape[:-1], cfg.max_entities), dtype=torch.bool, device="cuda")
            entity_sizes.append(e.shape[-2])
            with attention_context(backend), torch.autocast("cuda", dtype=torch.bfloat16):
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
                loss, _ = hierarchical_bc_loss(
                    logp, y, legal, position_coef=0.2, tile_label_sigma=1.0 if args.v2 else 0.0
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite benchmark loss")
            (loss / accumulation).backward()
            microbatch += 1
            if microbatch % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        row = {
            "batch_size": bs,
            "entity_bucket": bucket,
            "fused": fused,
            "attention_backend": backend,
            "stress_200_entities": stress,
            "gradient_accumulation": accumulation,
            "effective_batch_size": 96,
        }
        try:
            for _ in range(accumulation):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            iterations = accumulation * (1 if stress else args.iterations)
            for _ in range(iterations):
                step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row.update(
                success=True,
                samples_per_second=bs * iterations / elapsed,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                entity_sizes=entity_sizes,
            )
        except (torch.OutOfMemoryError, RuntimeError) as error:
            row.update(success=False, error=str(error)[:250])
        finally:
            optimizer.zero_grad(set_to_none=True)
            optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps(row), flush=True)
        return row

    results.append(measure(4, 0, False, "auto"))
    for backend in ("auto", "flash"):
        for bs in (4, 8, 16, 24, 32):
            results.append(measure(bs, 16, True, backend))
    candidates = sorted(
        (r for r in results if r["success"] and r["entity_bucket"]),
        key=lambda r: r["samples_per_second"],
        reverse=True,
    )
    chosen = None
    stress_results = []
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    for candidate in candidates:
        stress = measure(candidate["batch_size"], 16, True, candidate["attention_backend"], True)
        stress_results.append(stress)
        if stress["success"] and stress["peak_gib"] < 0.88 * total_gib:
            chosen = candidate
            break
    if chosen is None:
        raise RuntimeError("No batch passed the full-entity memory-headroom check")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "context_len": cfg.context_len,
        "model": "production",
        "representation_version": cfg.representation_version,
        "measurement": "CPU history preparation, transfer, forward/backward, AdamW step",
        "results": results,
        "stress_results": stress_results,
        "chosen": chosen,
    }
    atomic_json(args.out, report)
    print(json.dumps({"chosen": chosen}), flush=True)


if __name__ == "__main__":
    main()
