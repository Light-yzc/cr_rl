"""Measure complete episode-state decisions on a streamed real-frame fixture.

No training or cloud session is started. A repeated real frame constructs cache
lengths for a performance fixture, not an estimate of tactical policy quality.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ModelConfig
from cr_rl.models.state_policy import PolicyMemory, StateActorCritic, state_model_from_checkpoint
from cr_rl.models.state_temporal import GQAState, KDAState, StateTemporalConfig


def read_frame_fixture(path: Path):
    """Read only the last frame of the first history, plus its legal mask.

    ZipExtFile.seek moves forward through the first history's compressed stream;
    it never materializes the full windows array or writes an extracted archive.
    """
    consumed = 0

    def read_array_slice(archive, name, last_frame=False):
        nonlocal consumed
        with archive.open(name + ".npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported fixture NPY version {version}")
            if fortran or dtype.hasobject:
                raise ValueError("Fixture must contain C-order numeric arrays")
            if last_frame:
                if len(shape) != 3 or min(shape) <= 0:
                    raise ValueError("Expected windows [sample, time, seat_ints]")
                offset = (shape[1] - 1) * shape[2] * dtype.itemsize
                size = shape[2] * dtype.itemsize
            else:
                if len(shape) != 2 or min(shape) <= 0:
                    raise ValueError("Expected masks [sample, action]")
                offset = 0
                size = shape[1] * dtype.itemsize
            stream.seek(offset, os.SEEK_CUR)
            raw = stream.read(size)
            consumed += offset + len(raw)
            if len(raw) != size:
                raise ValueError("Incomplete fixture array")
            return np.frombuffer(raw, dtype=dtype).copy(), shape

    with zipfile.ZipFile(path) as archive:
        observation, shape = read_array_slice(archive, "windows", last_frame=True)
        legal, _ = read_array_slice(archive, "masks")
    return observation, legal.astype(bool), {
        "archive": str(path.resolve()), "history_shape": list(shape),
        "decompressed_array_bytes_read_or_skipped": consumed,
        "retained_raw_bytes": observation.nbytes + legal.nbytes,
        "history_fixture": "first sample, final frame repeated; performance only",
    }


def schema_from_config(cfg):
    header = {}
    for i, name in enumerate(cfg.header_fields):
        if name != "reserved":
            header.setdefault(name, []).append(i)
    return SimpleNamespace(
        seat_ints=cfg.header_dim + cfg.max_entities * cfg.entity_feature_dim,
        header_ints=cfg.header_dim, entity_ints=cfg.entity_feature_dim,
        max_entities=cfg.max_entities, header=header,
        header_fields=cfg.header_fields, entity_fields=cfg.entity_fields,
    )


def cache_accounting(memory):
    rows = []
    for i, state in enumerate(memory.temporal.layers):
        tensors = [state.matrix] if isinstance(state, KDAState) else [state.key, state.value]
        rows.append({"layer": i, "kind": "kda" if isinstance(state, KDAState) else "gqa",
                     "bytes": sum(t.numel() * t.element_size() for t in tensors),
                     "dtypes": [str(t.dtype) for t in tensors],
                     "shapes": [list(t.shape) for t in tensors]})
    return {"total_bytes": sum(r["bytes"] for r in rows) + memory.temporal.positions.numel() * 8,
            "layers": rows}


def peak_ram_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except ImportError:
        try:
            import psutil
            memory = psutil.Process().memory_info()
            return int(getattr(memory, "peak_wset", memory.rss))
        except ImportError:
            return None


def statistics(values):
    data = np.asarray(values, dtype=np.float64)
    return {"p50_ms": float(np.percentile(data, 50)),
            "p95_ms": float(np.percentile(data, 95)),
            "p99_ms": float(np.percentile(data, 99)), "max_ms": float(data.max()),
            "mean_ms": float(data.mean()), "samples": len(values)}


def parse_ints(value):
    numbers = [int(p) for p in value.split(",")]
    if not numbers or min(numbers) <= 0:
        raise argparse.ArgumentTypeError("Provide comma-separated positive integers")
    return numbers


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("artifacts/bc_diagnostic_20261110")
    parser.add_argument("--probe", type=Path, default=base / "prepared/probe.npz")
    parser.add_argument("--metadata", type=Path, default=base / "probe_parent_metadata.pt")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path, help="Optional original BC tensors to migrate with mmap")
    weights.add_argument("--state-checkpoint", type=Path,
                         help="Architecture-v3 checkpoint to load strictly, including trained KDA/GQA tensors")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(512, 768, 1024, 1280), default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lengths", type=parse_ints, default=parse_ints("1,375,750,1200"))
    parser.add_argument("--streams", type=parse_ints, default=parse_ints("1,8"))
    parser.add_argument("--entity-count", type=int, default=200,
                        help="Synthetic entity stress by copying valid fixture tokens; 0 uses observed count")
    parser.add_argument("--precision", choices=("amp_fp16", "fp16", "fp32"), default="amp_fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")
    return args


def validate_fixture_config(fixture_cfg, model_cfg):
    """Fixture metadata describes raw input semantics, never model weights."""
    fields = ("header_dim", "entity_feature_dim", "max_entities", "action_count",
              "vocab_size", "header_fields", "entity_fields")
    mismatched = [name for name in fields if getattr(fixture_cfg, name) != getattr(model_cfg, name)]
    if mismatched:
        raise ValueError("Fixture metadata/checkpoint schema mismatch: " + ", ".join(mismatched))


def load_benchmark_model(args, fixture_cfg, device):
    """Construct on the target device; mmap checkpoint storages only on CPU.

    Architecture-v3 checkpoints use the strict factory. The optional legacy BC
    path remains a representation-only migration and is labeled separately.
    """
    if args.state_checkpoint:
        checkpoint = torch.load(args.state_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if checkpoint.get("model_architecture_version") != 3:
            raise ValueError("--state-checkpoint requires architecture-v3 weights")
        trained_cfg = StateTemporalConfig(**checkpoint["temporal_config"])
        if args.width != trained_cfg.width:
            raise ValueError(f"--width {args.width} differs from state checkpoint width {trained_cfg.width}")
        if checkpoint.get("decision_ticks", 8) != 8:
            raise ValueError("State latency benchmark requires decision_ticks=8 (0.4 seconds)")
        validate_fixture_config(fixture_cfg, ModelConfig(**checkpoint["config"]))
        model = state_model_from_checkpoint(checkpoint, device=device)
        model.frame_microbatch = max(args.streams)
        loaded = {"source": str(args.state_checkpoint.resolve()), "strict": True,
                  "loaded_tensors": len(checkpoint["model"]),
                  "loaded_temporal_tensors": sum(name.startswith("temporal.") for name in checkpoint["model"]),
                  "optimizer_migrated": False}
        del checkpoint
        return model, None, loaded
    temporal_cfg = StateTemporalConfig(width=args.width, heads=args.width // 64,
                                       kv_heads=args.width // 256, head_dim=64)
    with torch.device(device):
        model = StateActorCritic(fixture_cfg, temporal_cfg, frame_microbatch=max(args.streams))
    migration = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        source = checkpoint.get("model", checkpoint.get("state_dict"))
        if source is None:
            raise ValueError("Checkpoint has no model/state_dict tensors")
        migration = model.migrate_bc_representation(source)
        del source, checkpoint
    return model, migration, None


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently benchmark CPU")
    if device.type == "cpu" and args.precision == "amp_fp16":
        raise ValueError("Use --precision fp32 or fp16 for an explicit CPU diagnostic")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    metadata = torch.load(args.metadata, map_location="cpu", weights_only=False)
    fixture_cfg = ModelConfig(**metadata["config"])
    model, migration, state_load = load_benchmark_model(args, fixture_cfg, device)
    cfg = model.cfg
    temporal_cfg = model.temporal_cfg
    schema = schema_from_config(cfg)
    observation, legal, fixture = read_frame_fixture(args.probe)
    if len(observation) != schema.seat_ints:
        raise ValueError("Fixture/schema seat dimensions disagree")
    count_index = schema.header["entity_count"][0]
    observed_entities = int(observation[count_index])
    if args.entity_count:
        if not 1 <= args.entity_count <= cfg.max_entities:
            raise ValueError("entity-count must be 0 or within model max_entities")
        cells = observation[cfg.header_dim:].reshape(cfg.max_entities, cfg.entity_feature_dim)
        source = cells[:max(1, min(observed_entities, cfg.max_entities))].copy()
        cells[:args.entity_count] = source[np.arange(args.entity_count) % len(source)]
        observation[count_index] = args.entity_count
    fixture.update(observed_entity_count=observed_entities,
                   measured_entity_count=int(observation[count_index]),
                   synthetic_entity_token_copies=bool(args.entity_count),
                   unchanged_input_scaling=True)
    if args.precision == "fp16":
        model.half()
    model.eval()
    adapter = CrForgeObservationAdapter(schema, cfg.vocab_size, entity_bucket=16)
    activation_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    kv_dtype = torch.float16 if args.precision != "fp32" else torch.float32

    def precision_context():
        return torch.autocast("cuda", dtype=torch.float16) if args.precision == "amp_fp16" else nullcontext()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def prepare(raw, legal_cpu):
        header, entities, mask = adapter(raw)
        hands = raw[:, schema.header["hand_card"]].astype(np.int64, copy=True)
        arrays = (header, entities, mask, legal_cpu, hands)
        return tuple(torch.from_numpy(np.ascontiguousarray(array)).to(
            device=device, dtype=activation_dtype if index < 2 else
            (torch.long if index == 4 else torch.bool)) for index, array in enumerate(arrays))

    report = {"torch": torch.__version__, "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "precision": args.precision, "fixture": fixture,
              "width": args.width, "pattern": list(temporal_cfg.pattern),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "weights": ("strict architecture-v3 trained checkpoint" if state_load else
                          "migrated matching BC representation" if migration else "random fixture weights"),
              "state_checkpoint_load": state_load,
              "migration": migration, "sampling": "multinomial of legal joint distribution, copied to CPU",
              "latency_gate": "all measured warm maxima <400ms AND p99 <300ms",
              "results": []}

    with torch.inference_mode():
        for streams in args.streams:
            raw = np.repeat(observation[None], streams, axis=0)
            legal_cpu = np.repeat(legal[None], streams, axis=0)
            inputs = prepare(raw, legal_cpu)
            with precision_context():
                summary, _ = model.encode_frames(*inputs[:3])
                temporal_input = model.to_temporal(summary)[:, None]
            for length in args.lengths:
                memory = model.initial_state(streams, device=device, dtype=kv_dtype)
                # Prime the temporal-only prefix. Entity encoding is independent
                # of history, and unused prefix placement heads need not run.
                with precision_context():
                    for start in range(0, length - 1, 32):
                        amount = min(32, length - 1 - start)
                        _, temporal = model.temporal.prefill(
                            temporal_input.expand(-1, amount, -1), memory.temporal)
                        memory = PolicyMemory(temporal, id(model), model.memory_generation)

                def decision(state):
                    synchronize()
                    start = time.perf_counter()
                    header, entities, mask = adapter(raw)
                    hands = raw[:, schema.header["hand_card"]].astype(np.int64, copy=True)
                    adapted = time.perf_counter()
                    arrays = (header, entities, mask, legal_cpu, hands)
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if device.type == "cuda" else None
                    if events:
                        events[0].record()
                    moved = tuple(torch.from_numpy(np.ascontiguousarray(array)).to(
                        device=device, dtype=activation_dtype if index < 2 else
                        (torch.long if index == 4 else torch.bool)) for index, array in enumerate(arrays))
                    transferred = time.perf_counter()
                    if events:
                        events[1].record()
                    with precision_context():
                        logp, _, next_state = model.step(*moved[:4], state=state,
                                                       hand_card_ids=moved[4], compute_value=False)
                    evaluated = time.perf_counter()
                    if events:
                        events[2].record()
                    sampled = torch.multinomial(logp.exp(), 1).cpu().numpy()
                    if events:
                        events[3].record()
                    synchronize()
                    end = time.perf_counter()
                    if not np.isfinite(sampled).all():
                        raise FloatingPointError("Nonfinite sampled action")
                    parts = {"adapter_ms": (adapted - start) * 1000}
                    if events:
                        parts.update(h2d_ms=events[0].elapsed_time(events[1]),
                                     model_ms=events[1].elapsed_time(events[2]),
                                     sample_d2h_ms=events[2].elapsed_time(events[3]))
                    else:
                        parts.update(h2d_ms=(transferred - adapted) * 1000,
                                     model_ms=(evaluated - transferred) * 1000,
                                     sample_d2h_ms=(end - evaluated) * 1000)
                    return (end - start) * 1000, parts, next_state

                warm = memory
                for _ in range(args.warmup):
                    _, _, warm = decision(warm)
                del warm
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                timings, stages = [], []
                for _ in range(args.steps):
                    elapsed, parts, memory = decision(memory)
                    timings.append(elapsed); stages.append(parts)
                row = {"streams": streams, "initial_decision_length": length,
                       "final_decision_length": length + args.steps - 1,
                       "warm": statistics(timings),
                       "stages": {name: statistics([s[name] for s in stages]) for name in stages[0]},
                       "peak_ram_bytes": peak_ram_bytes(),
                       "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                       "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                       "cache": cache_accounting(memory)}
                row["gate_passed"] = row["warm"]["max_ms"] < 400 and row["warm"]["p99_ms"] < 300
                report["results"].append(row)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(row), flush=True)
    report["all_measured_cases_passed"] = all(r["gate_passed"] for r in report["results"])
    report["l4_gate_verified"] = bool(device.type == "cuda" and "L4" in report["gpu"]
                                      and report["all_measured_cases_passed"]
                                      and 200 == report["fixture"]["measured_entity_count"]
                                      and {1, 8}.issubset(args.streams)
                                      and {1, 375, 750, 1200}.issubset(args.lengths))
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
