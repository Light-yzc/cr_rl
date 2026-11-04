"""Bound BF16 frozen-reference differences across shuffled same-size microbatches."""

import argparse
from pathlib import Path

import numpy as np
import torch
from validate_cuda_policy_parity import select_histories

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import ActorPolicy, ModelConfig
from cr_rl.training.persistence import atomic_json, file_sha256
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.reference_cache import ReferenceDistributionCache
from cr_rl.training.separate import autocast_for, forward_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    state = torch.load(args.actor, map_location="cpu", mmap=True, weights_only=False)
    cfg = ModelConfig(**state["config"])
    model = ActorPolicy(cfg).cuda().eval()
    model.load_state_dict(state["model"])
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
        selected = select_histories(data, cfg.context_len, adapter.schema.header["entity_count"][0])
        refs = [(trace, position, row["seat"]) for row, trace, position in selected]
        actions = torch.tensor([r[0]["action"] for r in selected], device="cuda")
        prepared = PreparedHistoryBatches(refs, cfg.context_len, adapter, "cuda")
        cache = ReferenceDistributionCache(len(refs))
        worst_logp = worst_tv = 0.0
        rng = np.random.default_rng(29)
        with torch.no_grad():
            for _epoch in range(2):
                permutation = rng.permutation(len(refs))
                for left in range(0, len(refs), 8):
                    rows = permutation[left : left + 8]
                    with autocast_for("cuda"):
                        full, _ = forward_inputs(model, prepared(rows))
                    full = full.float()
                    cached = cache.evaluate(rows, lambda full=full: full)
                    chosen = actions[rows, None]
                    worst_logp = max(
                        worst_logp,
                        float((full.gather(1, chosen) - cached.gather(1, chosen)).abs().max()),
                    )
                    worst_tv = max(
                        worst_tv, float((full.exp() - cached.exp()).abs().sum(-1).max() / 2)
                    )
        passed = worst_logp <= 0.05 and worst_tv <= 0.01
        atomic_json(
            args.out,
            dict(
                passed=passed,
                selected_logp_max_abs=worst_logp,
                total_variation_max=worst_tv,
                samples=len(refs),
                microbatch=8,
                forward_calls=cache.forward_calls,
                hit_rows=cache.hit_rows,
                actor_sha256=file_sha256(args.actor),
                rollout_sha256=file_sha256(args.rollout),
            ),
        )
        if not passed:
            raise RuntimeError("Reference cache drift exceeds registered bounds")
    finally:
        env.close()


if __name__ == "__main__":
    main()
