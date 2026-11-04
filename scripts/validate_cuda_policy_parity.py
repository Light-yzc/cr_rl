"""Audit real frozen behavior probabilities without changing stored old log-probs."""

import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import ActorPolicy, ModelConfig
from cr_rl.training.episode_trace import EpisodeTrace
from cr_rl.training.inference import FrozenInference, history_tensors
from cr_rl.training.persistence import atomic_json, file_sha256
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.separate import forward_inputs


def select_histories(data, context_len, count_index, per_group=8):
    """Stratify lane/seat/action and density; reject unavailable prefix history."""
    traces, candidates = {}, {}
    for step in range(len(data["actions"])):
        for lane in range(data["actions"].shape[1]):
            key = lane, int(data["episode_ids"][step, lane])
            if key not in traces:
                traces[key] = (EpisodeTrace(key[1], 0), bool(data["episode_start"][step, lane]))
            trace, starts_here = traces[key]
            pos = trace.append(data["obs"][step, lane], data["masks"][step, lane], [0, 0])
            if not starts_here and pos + 1 < context_len:
                continue
            for seat in range(2):
                if (
                    not data["learner_mask"][step, lane, seat]
                    or not data["valid"][step, lane, seat]
                ):
                    continue
                action = int(data["actions"][step, lane, seat])
                row = dict(
                    step=step,
                    lane=lane,
                    seat=seat,
                    action=action,
                    history_length=min(pos + 1, context_len),
                    entities=int(data["obs"][step, lane, seat, count_index]),
                )
                candidates.setdefault((lane, seat, action != 0), []).append((row, trace, pos))
    selected = []
    for group in sorted(candidates):
        rows = sorted(candidates[group], key=lambda r: (r[0]["entities"], r[0]["history_length"]))
        indices = np.unique(np.linspace(0, len(rows) - 1, min(per_group, len(rows))).astype(int))
        selected.extend(rows[i] for i in indices)
    if not selected or not any(row[0]["action"] for row in selected):
        raise ValueError("No valid real non-WAIT histories")
    return selected


def differences(reference, candidate, actions):
    reference, candidate = reference.float(), candidate.float()
    old, new = reference.gather(1, actions[:, None]), candidate.gather(1, actions[:, None])
    return dict(
        selected_logp_max_abs=float((old - new).abs().max()),
        selected_ratio_max_deviation=float(((new - old).exp() - 1).abs().max()),
        probability_max_abs=float((reference.exp() - candidate.exp()).abs().max()),
        total_variation_max=float(((reference.exp() - candidate.exp()).abs().sum(-1) / 2).max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--actor-sha256", required=True)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--rollout-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frame-encoder-fp32", action="store_true")
    args = parser.parse_args()
    if (
        file_sha256(args.actor) != args.actor_sha256
        or file_sha256(args.rollout) != args.rollout_sha256
    ):
        raise ValueError("Frozen source SHA mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This audit requires a BF16-capable CUDA GPU")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    state = torch.load(args.actor, map_location="cpu", weights_only=False, mmap=True)
    config = ModelConfig(**state["config"])
    if args.frame_encoder_fp32:
        config = replace(config, frame_encoder_fp32=True)
    model = ActorPolicy(config).cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    with np.load(args.rollout) as archive:
        data = {
            key: archive[key]
            for key in (
                "obs",
                "masks",
                "actions",
                "episode_ids",
                "episode_start",
                "learner_mask",
                "valid",
                "old_log_probs",
                "policy_versions",
            )
        }
    if not np.all(data["policy_versions"][data["learner_mask"].astype(bool)] == state["updates"]):
        raise ValueError("Actor is not the behavior generation used by this rollout")
    env = CrForgeVector(1, 1, independent=True)
    try:
        adapter = CrForgeObservationAdapter(
            env.vector.schema, env.vector.vocabulary_size, entity_bucket=16
        )
        selected = select_histories(
            data, config.context_len, adapter.schema.header["entity_count"][0]
        )
        refs = [(trace, pos, row["seat"]) for row, trace, pos in selected]
        prepared = PreparedHistoryBatches(refs, config.context_len, adapter, "cuda")
        actions = torch.tensor([row["action"] for row, _, _ in selected])

        def windows(indices):
            raw, times, legal = [], [], []
            for i in indices:
                trace, pos, seat = refs[i]
                w, t = trace.window(pos, seat, config.context_len)
                raw.append(w)
                times.append(t)
                legal.append(trace.masks[pos][seat])
            return np.stack(raw), np.stack(times), np.stack(legal)

        results, outputs = [], {}
        with torch.inference_mode():
            for precision in ("fp32", "bf16"):
                for batch in (1, 4, 8):
                    for path in ("full", "prepared", "cached"):
                        cache = (
                            FrozenInference(model, adapter, "cuda") if path == "cached" else None
                        )
                        values = []
                        for left in range(0, len(refs), batch):
                            indices = np.arange(left, min(left + batch, len(refs)))
                            w, t, legal = windows(indices)
                            context = (
                                torch.autocast("cuda", dtype=torch.bfloat16)
                                if precision == "bf16"
                                else nullcontext()
                            )
                            with context:
                                if cache is None:
                                    inputs = (
                                        prepared(indices)
                                        if path == "prepared"
                                        else history_tensors(w, t, legal, adapter, "cuda")
                                    )
                                    value = forward_inputs(model, inputs)[0]
                                else:
                                    # Bypass the BF16 wrapper for the explicit FP32 audit.
                                    encoded = cache.cache.encode(w, t)
                                    h, e, mask = adapter(w[:, -1])
                                    value = model(
                                        torch.as_tensor(h, device="cuda"),
                                        torch.as_tensor(e, device="cuda"),
                                        torch.as_tensor(mask, device="cuda"),
                                        torch.as_tensor(legal, dtype=torch.bool, device="cuda"),
                                        hand_card_ids=torch.as_tensor(
                                            w[:, -1, list(adapter.schema.header["hand_card"])],
                                            device="cuda",
                                        ),
                                        encoded_history=encoded,
                                    )[0]
                            values.append(value.float().log_softmax(-1).cpu())
                        output = torch.cat(values)
                        if not torch.isfinite(output).all():
                            raise FloatingPointError("Nonfinite audited action log-probabilities")
                        outputs[precision, batch, path] = output
                        baseline = outputs[precision, 1, "full"]
                        row = dict(
                            precision=precision,
                            batch=batch,
                            path=path,
                            **differences(baseline, output, actions),
                        )
                        results.append(row)
                        print(json.dumps(row), flush=True)
        original = torch.tensor(
            [float(data["old_log_probs"][r["step"], r["lane"], r["seat"]]) for r, _, _ in selected]
        )
        original_checks = {}
        for precision in ("fp32", "bf16"):
            predicted = outputs[precision, 4, "cached"].gather(1, actions[:, None]).squeeze(1)
            original_checks[precision] = float((predicted - original).abs().max())
        passed = (
            all(
                r["selected_logp_max_abs"] <= (0.0005 if r["precision"] == "fp32" else 0.03)
                and r["total_variation_max"] <= (0.0001 if r["precision"] == "fp32" else 0.01)
                for r in results
            )
            and original_checks["bf16"] <= 0.05
        )
        report = dict(
            passed=passed,
            device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            frame_encoder_fp32=config.frame_encoder_fp32,
            actor_sha256=args.actor_sha256,
            rollout_sha256=args.rollout_sha256,
            samples=[row for row, _, _ in selected],
            results=results,
            stored_old_logp_max_abs=original_checks,
            thresholds=dict(
                fp32_logp=0.0005,
                fp32_tv=0.0001,
                bf16_logp=0.03,
                bf16_tv=0.01,
                stored_bf16_logp=0.05,
            ),
            note=(
                "Bounded numeric acceptance, not bitwise equivalence; "
                "original old probabilities unchanged"
            ),
        )
        atomic_json(args.out, report)
        if not passed:
            raise RuntimeError(
                "CUDA probability audit exceeded registered bounds; inspect before training"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
