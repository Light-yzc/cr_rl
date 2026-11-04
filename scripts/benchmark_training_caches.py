"""Real Actor forward/backward timings with fixed inputs and zero learning rate."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from validate_cuda_policy_parity import select_histories

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import ActorPolicy, ModelConfig
from cr_rl.training.persistence import atomic_json, file_sha256
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.separate import autocast_for, forward_inputs, optimize_separate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-grid-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    state = torch.load(args.actor, map_location="cpu", mmap=True, weights_only=False)
    cfg = ModelConfig(**state["config"])
    actor = ActorPolicy(cfg).cuda().eval()
    actor.load_state_dict(state["model"])
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    with np.load(args.rollout) as archive:
        data = {
            name: archive[name]
            for name in (
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
        refs = [(trace, pos, row["seat"]) for row, trace, pos in rows]
        prepared = PreparedHistoryBatches(refs, cfg.context_len, adapter, "cuda")
        indices = np.arange(len(refs))
        old, actions = [], []
        with torch.no_grad(), autocast_for("cuda"):
            for left in range(0, len(refs), 8):
                logp, _ = forward_inputs(actor, prepared(indices[left : left + 8]))
                distribution = torch.distributions.Categorical(logits=logp.float())
                chosen = distribution.probs.argmax(-1)
                old.append(distribution.log_prob(chosen))
                actions.append(chosen)
        old, actions = torch.cat(old), torch.cat(actions)
        results = []
        configurations = [
            (8, False, False),
            (8, True, False),
            (8, False, True),
            (8, True, True),
            (16, True, True),
        ]
        if args.batch_grid_only:
            configurations = [(batch, True, True) for batch in (4, 8, 16)]
        for batch, cache, deferred in configurations:
            optimizer = torch.optim.AdamW(actor.parameters(), lr=0)
            dummy = torch.nn.Linear(1, 1).cuda()
            dummy_optimizer = torch.optim.SGD(dummy.parameters(), lr=0)
            timings = []
            for _repeat in range(3):
                torch.cuda.synchronize()
                began = time.perf_counter()
                result = optimize_separate(
                    actor,
                    dummy,
                    behavior,
                    optimizer,
                    dummy_optimizer,
                    prepared,
                    indices,
                    np.zeros(0, dtype=np.int64),
                    old,
                    actions,
                    torch.linspace(-1, 1, len(refs), device="cuda"),
                    torch.zeros_like(old),
                    rng=np.random.default_rng(29),
                    device="cuda",
                    batch_size=batch,
                    accumulation=64 // batch,
                    actor_epochs=2,
                    critic_epochs=0,
                    clip_coef=0.15,
                    entropy_coef=0.005,
                    reference_kl_coef=0.01,
                    target_kl=1,
                    max_grad_norm=0.5,
                    cache_reference=cache,
                    defer_scalar_metrics=deferred,
                )
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - began)
            row = dict(
                batch=batch,
                effective_batch=64,
                cache_reference=cache,
                defer_scalar_metrics=deferred,
                seconds_median=float(np.median(timings[1:])),
                timings=timings,
                reference_cache=result["reference_cache"],
                actor_steps=result["actor_steps"],
                samples_per_epoch=len(refs),
            )
            results.append(row)
            print(json.dumps(row), flush=True)
        atomic_json(
            args.out,
            dict(
                results=results,
                actor_sha256=file_sha256(args.actor),
                limitation=(
                    "Two epochs of zero-LR Actor optimization on fixed real inputs; "
                    "excludes rollout, Critic and backups"
                ),
                note=(
                    "Lazy reference cache has no expected forward saving "
                    "when KL stops before sample reuse"
                ),
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
