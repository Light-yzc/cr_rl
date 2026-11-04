"""Real-weight CUDA window/cache comparison; no simulator or optimizer timing."""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import ActorPolicy, ModelConfig
from cr_rl.models.incremental_frames import IncrementalFrameHistory
from cr_rl.training.inference import FrozenInference
from cr_rl.training.persistence import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--measure-steps", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    state = torch.load(args.actor, map_location="cpu", mmap=True, weights_only=False)
    cfg = ModelConfig(**state["config"])
    model = ActorPolicy(cfg).cuda().eval()
    model.load_state_dict(state["model"])
    with np.load(args.rollout) as archive:
        data = {name: archive[name] for name in ("obs", "masks", "episode_ids", "actions")}
    env = CrForgeVector(1, 1, independent=True)
    try:
        adapter = CrForgeObservationAdapter(
            env.vector.schema, env.vector.vocabulary_size, entity_bucket=16
        )
        results, outputs = [], {}
        for repeat in range(3):
            for mode in ["legacy", "incremental"] if repeat % 2 == 0 else ["incremental", "legacy"]:
                inference = FrozenInference(model, adapter, "cuda")
                history = IncrementalFrameHistory(
                    args.streams, cfg.context_len, adapter.schema.seat_ints
                )
                queues = [deque(maxlen=cfg.context_len) for _ in range(args.streams)]
                previous = np.full(args.streams, -1)
                timings, values, actions = [], [], []
                with torch.inference_mode():
                    for step in range(cfg.context_len + args.measure_steps):
                        offsets = np.arange(args.streams) // 4 * 256 + step
                        lanes = np.arange(args.streams) % 4 // 2
                        seats = np.arange(args.streams) % 2
                        raw, masks = (
                            data["obs"][offsets, lanes, seats],
                            data["masks"][offsets, lanes, seats],
                        )
                        episodes = (
                            data["episode_ids"][offsets, lanes]
                            + np.arange(args.streams) // 4 * 1000000
                        )
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        if mode == "incremental":
                            history.append(raw, episodes)
                            result = inference.evaluate_incremental(history.request(), masks)
                        else:
                            window = np.zeros(
                                (args.streams, cfg.context_len, raw.shape[-1]), dtype=np.int32
                            )
                            valid = np.zeros(window.shape[:2], dtype=bool)
                            for i in range(args.streams):
                                if episodes[i] != previous[i]:
                                    queues[i].clear()
                                queues[i].append(raw[i].copy())
                                window[i, -len(queues[i]) :] = np.stack(queues[i])
                                valid[i, -len(queues[i]) :] = True
                            result = inference.evaluate(window, valid, masks)
                        previous = episodes
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - began
                        if step >= cfg.context_len:
                            timings.append(elapsed)
                            values.append(result.cpu())
                            actions.append(
                                torch.from_numpy(
                                    data["actions"][offsets, lanes, seats].astype(np.int64)
                                )
                            )
                output = torch.cat(values)
                chosen = torch.cat(actions)
                outputs[repeat, mode] = output
                results.append(
                    dict(
                        repeat=repeat,
                        mode=mode,
                        seconds=float(sum(timings)),
                        milliseconds_per_step=float(np.mean(timings) * 1000),
                    )
                )
                print(json.dumps(results[-1]), flush=True)
        differences = []
        for repeat in range(3):
            a, b = outputs[repeat, "legacy"], outputs[repeat, "incremental"]
            difference = (a.gather(1, chosen[:, None]) - b.gather(1, chosen[:, None])).abs().max()
            tv = (a.exp() - b.exp()).abs().sum(-1).max() / 2
            differences.append(
                dict(selected_logp_max_abs=float(difference), total_variation_max=float(tv))
            )
        medians = {
            mode: float(np.median([r["seconds"] for r in results if r["mode"] == mode]))
            for mode in ("legacy", "incremental")
        }
        report = dict(
            actor_sha256=file_sha256(args.actor),
            rollout_sha256=file_sha256(args.rollout),
            streams=args.streams,
            context_len=cfg.context_len,
            measure_steps=args.measure_steps,
            results=results,
            differences=differences,
            median_seconds=medians,
            speedup=medians["legacy"] / medians["incremental"],
            numerical_pass=all(
                r["selected_logp_max_abs"] <= 0.05 and r["total_variation_max"] <= 0.01
                for r in differences
            ),
            limitation=(
                "Synchronized inference microbenchmark only; "
                "excludes environment, optimization and backups"
            ),
        )
        atomic_json(args.out, report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
