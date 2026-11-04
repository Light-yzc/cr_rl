"""Real Critic zero-LR optimization grid with fixed effective batch and samples."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from validate_cuda_policy_parity import select_histories

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import HistoryValueNetwork, ModelConfig
from cr_rl.training.persistence import atomic_json, file_sha256
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.separate import optimize_separate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic", required=True, type=Path)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    state = torch.load(args.critic, map_location="cpu", mmap=True, weights_only=False)
    cfg = ModelConfig(**state["critic_config"])
    critic = HistoryValueNetwork(cfg).cuda()
    critic.load_state_dict(state["critic"])
    with np.load(args.rollout) as archive:
        data = {
            k: archive[k]
            for k in (
                "obs",
                "masks",
                "actions",
                "episode_ids",
                "episode_start",
                "learner_mask",
                "valid",
            )
        }
    env = CrForgeVector(1, 1, independent=True)
    try:
        adapter = CrForgeObservationAdapter(
            env.vector.schema, env.vector.vocabulary_size, entity_bucket=16
        )
        rows = select_histories(data, cfg.context_len, adapter.schema.header["entity_count"][0])
        refs = [(trace, position, row["seat"]) for row, trace, position in rows]
        if len(refs) != 64:
            raise ValueError("This fixed grid requires the registered 64-row cohort")
        prepared = PreparedHistoryBatches(refs, cfg.context_len, adapter, "cuda")
        dummy = torch.nn.Linear(1, 1).cuda()
        dummy_optimizer = torch.optim.SGD(dummy.parameters(), lr=0)
        indices = np.arange(64)
        values = torch.ones(64, device="cuda")
        results = []
        for batch, deferred in [
            (4, False),
            (8, False),
            (16, False),
            (4, True),
            (8, True),
            (16, True),
        ]:
            optimizer = torch.optim.AdamW(critic.parameters(), lr=0)
            timings = []
            for _repeat in range(3):
                torch.cuda.synchronize()
                began = time.perf_counter()
                result = optimize_separate(
                    dummy,
                    critic,
                    dummy,
                    dummy_optimizer,
                    optimizer,
                    prepared,
                    np.zeros(0, dtype=np.int64),
                    indices,
                    torch.zeros_like(values),
                    torch.zeros(64, device="cuda", dtype=torch.long),
                    values,
                    values,
                    rng=np.random.default_rng(29),
                    device="cuda",
                    batch_size=batch,
                    accumulation=64 // batch,
                    actor_epochs=0,
                    critic_epochs=1,
                    clip_coef=0.15,
                    entropy_coef=0.005,
                    reference_kl_coef=0.01,
                    target_kl=0.02,
                    max_grad_norm=0.5,
                    actor_enabled=False,
                    defer_scalar_metrics=deferred,
                )
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - began)
            row = dict(
                batch=batch,
                effective_batch=64,
                samples=64,
                defer_scalar_metrics=deferred,
                timings=timings,
                seconds_median=float(np.median(timings[1:])),
                critic_steps=result["critic_steps"],
            )
            results.append(row)
            print(json.dumps(row), flush=True)
        atomic_json(
            args.out,
            dict(
                results=results,
                critic_sha256=file_sha256(args.critic),
                limitation=("Fixed real histories and weights, synthetic fixed targets, "
                            "zero LR; no rollout or backups"),
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
