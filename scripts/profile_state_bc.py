"""Bounded, frozen-weight BC training-stage profile on real continuous seats.

The optimizer uses lr=0 so forward/backward/Adam work is measured without
changing model weights. Every case reconstructs the same model and saved Adam
state. Production throughput comes from a separate pass with timing disabled;
CUDA-event diagnostics are a second, explicitly instrumented pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.state_bc import (
    EpisodeProfiler, ReplaySequenceDataset, prepare_state_chunk, run_state_episode,
)


CASES = {
    "baseline": {"defer_metrics": False, "frame_microbatch": 8, "matmul": "highest", "fused": False},
    "deferred": {"defer_metrics": True, "frame_microbatch": 8, "matmul": "highest", "fused": False},
    "frame16": {"defer_metrics": True, "frame_microbatch": 16, "matmul": "highest", "fused": False},
    "frame32": {"defer_metrics": True, "frame_microbatch": 32, "matmul": "highest", "fused": False},
    "tf32_high": {"defer_metrics": True, "frame_microbatch": 16, "matmul": "high", "fused": False},
    "fused": {"defer_metrics": True, "frame_microbatch": 16, "matmul": "highest", "fused": True},
    "combined": {"defer_metrics": True, "frame_microbatch": 32, "matmul": "high", "fused": True},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cases", default="baseline,deferred,frame16,frame32")
    parser.add_argument("--episodes", type=int, default=2, help="Real replay/seat sequences per measured pass")
    parser.add_argument("--warmup", type=int, default=1, help="Repeat first real seat before measurement")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--ram-limit-gib", type=float, default=40)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--max-probability-delta", type=float, default=1e-3)
    parser.add_argument("--verify-weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.cases = args.cases.split(",")
    if not args.cases or args.cases[0] != "baseline" or any(case not in CASES for case in args.cases):
        parser.error("Cases must start with baseline and use: " + ",".join(CASES))
    if not 1 <= args.episodes <= 4 or not 0 <= args.warmup <= 2 or args.cpu_threads < 1:
        parser.error("Use 1..4 measured seats, 0..2 warmups and positive CPU threads")
    return args


def weights_digest(model):
    """Bounded CPU copies; no duplicate CPU model or full optimizer copy."""
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        result.update(name.encode())
        flat = parameter.detach().reshape(-1)
        for block in flat.split(1024**2):
            result.update(block.cpu().numpy().tobytes())
    return result.hexdigest()


@contextmanager
def nested_forward_profile(model, profiler):
    old_frame = model.encode_frames
    old_temporal = model.temporal.prefill
    def frame(*args, **kwargs):
        with profiler.measure("frame_encoder_inclusive"):
            return old_frame(*args, **kwargs)
    def temporal(*args, **kwargs):
        with profiler.measure("temporal_inclusive"):
            return old_temporal(*args, **kwargs)
    model.encode_frames, model.temporal.prefill = frame, temporal
    try:
        yield
    finally:
        model.encode_frames, model.temporal.prefill = old_frame, old_temporal


def build_case(checkpoint, settings, case, args):
    torch.set_float32_matmul_precision(case["matmul"])
    torch.manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = state_model_from_checkpoint(checkpoint, args.device)
    model.frame_microbatch = case["frame_microbatch"]
    if settings.get("freeze_frame"):
        model.frame.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0,
                                 fused=case["fused"])
    saved_optimizer = {**checkpoint["optimizer"], "param_groups": [
        {**group, "lr": 0., "fused": case["fused"], "foreach": False if case["fused"] else None}
        for group in checkpoint["optimizer"]["param_groups"]]}
    optimizer.load_state_dict(saved_optimizer)
    # Non-fused Adam's CPU step scalar may otherwise alias the mmap-backed
    # source object and increment the next candidate's supposedly fresh state.
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                values[key] = value.clone()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(args.device).type == "cuda" and args.precision == "fp16",
                                 init_scale=32)
    if checkpoint.get("scaler") and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])
    return model, optimizer, scaler


def episode_options(settings, counts, args, case):
    return dict(chunk_size=settings["chunk_size"], device=args.device, precision=args.precision,
                position_coef=settings["position_coef"], tile_label_sigma=settings["tile_label_sigma"],
                timing_objective=settings["timing_objective"],
                timing_target_action_prior=settings["timing_target_action_prior"],
                sampled_action_prior=counts["action_prior"], max_grad_norm=settings["max_grad_norm"],
                ram_limit_gib=args.ram_limit_gib, defer_metrics=case["defer_metrics"])


def probability_probe(model, episode, adapter, args, chunk_size):
    """The same real first chunk, complete tile distribution, no optimizer work."""
    model.train()
    inputs = prepare_state_chunk(episode, episode.rows[:chunk_size], adapter, args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    state = model.initial_state(1, device=args.device, dtype=dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
            enabled=torch.device(args.device).type == "cuda" and args.precision == "fp16"):
        logp, _, _ = model.forward_state(**inputs, state=state, compute_value=False)
    if not torch.isfinite(logp).all():
        raise FloatingPointError("Probe policy is nonfinite")
    return logp.float().exp().cpu()


def measured_pass(model, optimizer, scaler, dataset, split, adapter, options, args, *, instrumented):
    reports, timings, load_seconds = [], [], []
    iterator = dataset.episodes(split, "train", epoch=0, seed=args.seed)
    started = time.perf_counter()
    try:
        for _ in range(args.episodes):
            loading = time.perf_counter()
            try:
                episode = next(iterator)
            except StopIteration:
                break
            load_seconds.append(time.perf_counter() - loading)
            if instrumented:
                profile = EpisodeProfiler(args.device)
                with nested_forward_profile(model, profile):
                    report = run_state_episode(model, episode, adapter, optimizer=optimizer,
                                               scaler=scaler, profile=profile, **options)
                timings.append(profile.resolve())
            else:
                report = run_state_episode(model, episode, adapter, optimizer=optimizer, scaler=scaler, **options)
            reports.append(report)
    finally:
        iterator.close()
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    if len(reports) != args.episodes:
        raise ValueError("Not enough real training seats for the requested bounded profile")
    return {"seconds": elapsed, "load_seconds": sum(load_seconds), "load_per_seat_seconds": load_seconds,
            "reports": reports, "component_timings": timings,
            "history_frames_per_second": sum(row["history_frames"] for row in reports) / elapsed,
            "supervised_rows_per_second": sum(row["rows"] for row in reports) / elapsed}


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime; no Torch installation is attempted")
    if device.type != "cuda" and args.precision != "fp32":
        raise ValueError("CPU diagnostic requires --precision fp32")
    torch.set_num_threads(args.cpu_threads)
    # Trusted private NumPy2 checkpoints also contain RNG state; keep compatibility
    # with the existing local NumPy1 environment without installing anything.
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("model_architecture_version") != 3:
        raise ValueError("Requires the current architecture-v3 full checkpoint")
    settings = checkpoint["training_config"]
    dataset = ReplaySequenceDataset(args.dataset, args.cache_dir or args.out.parent / ".profile-replay-cache",
        expected_decision_ticks=checkpoint["decision_ticks"], ram_limit_gib=args.ram_limit_gib)
    if dataset.fingerprint != checkpoint["dataset_fingerprint"]:
        raise ValueError("Profile data differs from the saved training stage")
    split = {int(key): value for key, value in checkpoint["split"].items()}
    counts = dataset.partition_counts(split, "train")
    original_precision = torch.get_float32_matmul_precision()
    result = {"torch": torch.__version__, "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "dataset_fingerprint": dataset.fingerprint, "checkpoint_progress": checkpoint["progress"],
              "parameter_count": sum(value.numel() for value in checkpoint["model"].values()),
              "episodes_per_pass": args.episodes, "optimizer_lr": 0.,
              "scope": "Unprofiled real forward/backward/Adam; independent second pass for inclusive stage timing. "
                       "All cases reload saved model/Adam; lr0 and a streaming weight hash protect checkpoint weights.",
              "limitation": "A bounded sample of 1..4 real seats is a speed/admission check, not convergence evidence. "
                            "Frame/temporal timings overlap forward_loss; CUDA event intervals can include device idle.",
              "cases": []}
    baseline_probability = None
    try:
        for name in args.cases:
            case = CASES[name]
            if case["fused"] and device.type != "cuda":
                result["cases"].append({"name": name, "status": "skipped", "reason": "Fused candidate requires CUDA"})
                continue
            model = optimizer = scaler = None
            try:
                model, optimizer, scaler = build_case(checkpoint, settings, case, args)
                adapter = CrForgeObservationAdapter(dataset.schema, model.cfg.vocab_size, entity_bucket=16)
                options = episode_options(settings, counts, args, case)
                iterator = dataset.episodes(split, "train", epoch=0, seed=args.seed)
                try:
                    first = next(iterator)
                    before = weights_digest(model) if args.verify_weights else None
                    for _ in range(args.warmup):
                        run_state_episode(model, first, adapter, optimizer=optimizer, scaler=scaler, **options)
                    probability = probability_probe(model, first, adapter, args, settings["chunk_size"])
                finally:
                    iterator.close()
                if baseline_probability is None:
                    baseline_probability = probability
                probability_delta = float((probability - baseline_probability).abs().max())
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                production = measured_pass(model, optimizer, scaler, dataset, split, adapter, options, args, instrumented=False)
                instrumented = measured_pass(model, optimizer, scaler, dataset, split, adapter, options, args, instrumented=True)
                after = weights_digest(model) if args.verify_weights else None
                unchanged = before == after if args.verify_weights else None
                reports = production["reports"] + instrumented["reports"]
                accepted = (unchanged is not False and probability_delta <= args.max_probability_delta
                            and all(row["optimizer_stepped"] and not row["amp_overflow_skipped"]
                                    and math.isfinite(row["grad_norm"]) for row in reports))
                record = {"name": name, "settings": case, "status": "ok", "admission_passed": accepted,
                          "probability_max_abs_delta": probability_delta, "weights_unchanged": unchanged,
                          "amp_scale": scaler.get_scale(), "production": production, "instrumented": instrumented,
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
                baseline = next((row for row in result["cases"] if row.get("name") == "baseline" and row.get("status") == "ok"), None)
                if baseline:
                    record["unprofiled_speedup"] = baseline["production"]["seconds"] / production["seconds"]
            except Exception as error:
                record = {"name": name, "settings": case, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
            finally:
                del model, optimizer, scaler
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            result["cases"].append(record)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"case": name, "status": record["status"], "admission_passed": record.get("admission_passed"),
                              "unprofiled_speedup": record.get("unprofiled_speedup"), "error": record.get("error")}), flush=True)
    finally:
        torch.set_float32_matmul_precision(original_precision)
    if not result["cases"] or result["cases"][0].get("status") != "ok":
        raise RuntimeError("Baseline profile failed; see saved report")


if __name__ == "__main__":
    main()
