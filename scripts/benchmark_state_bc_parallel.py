"""Bounded, same-data complete-sequence throughput benchmark on an existing GPU."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.state_bc import (
    ReplaySequenceDataset,
    StateEpisode,
    make_state_schema,
    run_state_episode,
)
from cr_rl.training.state_bc_parallel import run_state_episode_group


def load_benchmark_data(manifest_path, source_state, scratch, games=8):
    """Only open selected local shards; do not scan the full dataset or hash contents."""
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((manifest_path.parent / manifest["metadata"]).read_text())
    schema = make_state_schema(metadata)
    split = {int(k): v for k, v in source_state["split"].items()}
    pool = [i for i, part in split.items() if part == "train"]
    chosen = random.Random(20261111).sample(pool, games)
    records = {int(record["replay_offset"]): record for record in manifest["shards"]}
    dataset = ReplaySequenceDataset.__new__(ReplaySequenceDataset)
    dataset.path, dataset.manifest, dataset.metadata, dataset.schema = (
        manifest_path,
        manifest,
        metadata,
        schema,
    )
    dataset.cache_root = scratch
    dataset.ram_limit_gib = 35
    scratch.mkdir(parents=True, exist_ok=True)
    return dataset, chosen, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", default="1:32,2:64,4:128,8:256")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--repeats", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    ds, chosen, records = load_benchmark_data(a.dataset, state, a.out / "scratch", a.games)
    adapter = CrForgeObservationAdapter(ds.schema, state["config"]["vocab_size"], entity_bucket=16)
    settings = state["training_config"]
    options = {
        k: settings[k]
        for k in [
            "chunk_size",
            "position_coef",
            "tile_label_sigma",
            "timing_objective",
            "timing_target_action_prior",
            "sampled_action_prior",
            "max_grad_norm",
        ]
    }
    options.update(device="cuda", precision="fp16", ram_limit_gib=35, defer_metrics=True)
    results = []
    with ExitStack() as stack:
        episodes = []
        for replay_id in chosen:
            record = records[replay_id]
            arrays = stack.enter_context(ds._open_shard(record))
            ids = arrays["replay_ids"] + int(record.get("replay_offset", 0))
            for seat in [0, 1]:
                rows = np.flatnonzero((ids == replay_id) & (arrays["seats"] == seat))
                episodes.append(StateEpisode(replay_id, seat, rows, arrays, 8))
        frames = sum(len(e.rows) for e in episodes)
        print(
            json.dumps(
                {
                    "phase": "benchmark_start",
                    "games": chosen,
                    "seats": len(episodes),
                    "frames": frames,
                    "lengths": [len(e.rows) for e in episodes],
                    "gpu": torch.cuda.get_device_name(0),
                }
            ),
            flush=True,
        )
        for case in a.cases.split(","):
            batch, frame = map(int, case.split(":"))
            model = state_model_from_checkpoint(state, "cuda")
            model.frame_microbatch = frame
            optimizer = torch.optim.AdamW(model.parameters(), lr=0, fused=True)
            saved = {
                **state["optimizer"],
                "param_groups": [
                    {**g, "lr": 0.0, "fused": True, "foreach": False}
                    for g in state["optimizer"]["param_groups"]
                ],
            }
            optimizer.load_state_dict(saved)
            for param, values in optimizer.state.items():
                if isinstance(values.get("step"), torch.Tensor):
                    values["step"] = values["step"].to(param.device).clone()
            scaler = torch.amp.GradScaler("cuda", init_scale=32)
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            timings = []
            losses = []
            overflows = 0
            try:
                # Real complete sequences, including recurrent state growth.
                for repeat in range(a.repeats + 1):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    current = []
                    for offset in range(0, len(episodes), batch):
                        group = episodes[offset : offset + batch]
                        if batch == 1:
                            report = run_state_episode(
                                model,
                                group[0],
                                adapter,
                                optimizer=optimizer,
                                scaler=scaler,
                                **options,
                            )
                            current.append(report["loss"])
                            overflows += int(report["amp_overflow_skipped"])
                        else:
                            report = run_state_episode_group(
                                model, group, adapter, optimizer=optimizer, scaler=scaler, **options
                            )
                            current.extend(r["loss"] for r in report["reports"])
                            overflows += int(report["amp_overflow_skipped"])
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    if repeat:
                        timings.append(seconds)
                        losses = current
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                median = float(np.median(timings))
                result = {
                    "batch_size": batch,
                    "frame_microbatch": frame,
                    "seconds": timings,
                    "median_seconds": median,
                    "history_frames_per_second": frames / median,
                    "seats_per_second": len(episodes) / median,
                    "peak_allocated_gib": peak,
                    "loss_mean": float(np.mean(losses)),
                    "amp_overflow_skipped": overflows,
                    "optimizer_lr": 0.0,
                    "full_sequences": True,
                    "games": chosen,
                }
            except torch.cuda.OutOfMemoryError:
                result = {"batch_size": batch, "frame_microbatch": frame, "status": "oom"}
            results.append(result)
            print(json.dumps({"phase": "benchmark_case", **result}), flush=True)
            del model, optimizer, scaler, saved
            gc.collect()
            torch.cuda.empty_cache()
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "results.json").write_text(
                json.dumps({"results": results, "frames": frames, "seats": len(episodes)}, indent=2)
                + "\n"
            )
    baseline = results[0].get("history_frames_per_second")
    if baseline:
        for r in results:
            if "history_frames_per_second" in r:
                r["speedup"] = r["history_frames_per_second"] / baseline
    (a.out / "results.json").write_text(
        json.dumps({"results": results, "frames": frames, "seats": len(episodes)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
