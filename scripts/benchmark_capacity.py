"""CUDA capacity probe using saved real observations, not an online learning evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.ppo import ppo_loss_with_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="production")
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261103)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = ModelConfig.preset(args.model)
    metadata = json.loads(args.rollout.with_suffix(".npz.json").read_text())
    schema = SimpleNamespace(**metadata["schema"])
    schema.header = {}
    for index, field in enumerate(schema.header_fields):
        schema.header.setdefault(field, []).append(index)
    adapter = CrForgeObservationAdapter(schema, cfg.vocab_size)
    with np.load(args.rollout) as data:
        seat = int(data["learner_mask"][0, 0].argmax())
        candidates = np.arange(cfg.context_len - 1, len(data["obs"]))
        candidates = [
            i
            for i in candidates
            if data["episode_ids"][i, 0] == data["episode_ids"][i - cfg.context_len + 1, 0]
        ]
        assert len(candidates), "No complete within-episode history window"
        end = int(max(candidates, key=lambda i: data["masks"][i, 0, seat].sum()))
        raw = data["obs"][end - cfg.context_len + 1 : end + 1, 0, seat].copy()
        legal = data["masks"][end, 0, seat].astype(bool)
        assert legal.sum() > 1, "Probe needs a real card-placement decision"
    header, entities, mask = adapter(raw[None])

    def inputs(batch):
        arrays = [header, entities, mask, legal[None], np.ones((1, cfg.context_len), dtype=bool)]
        tensors = [torch.as_tensor(np.repeat(a, batch, axis=0), device="cuda") for a in arrays]
        card_ids = {
            name: torch.as_tensor(
                np.repeat(raw[-1, schema.header[field]][None], batch, axis=0), device="cuda"
            )
            for name, field in [
                ("hand_card_ids", "hand_card"),
                ("deck_card_ids", "deck"),
                ("enemy_seen_card_ids", "enemy_seen"),
            ]
        }
        return tensors, card_ids

    one, one_cards = inputs(1)
    batch, cards = inputs(args.batch_size)
    model = ActorCritic(cfg).cuda()
    parameters = model.parameter_count()
    behavior = copy.deepcopy(model).eval().requires_grad_(False)
    opponent = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)
    before = model.critic[-1].weight.detach().clone()

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16)

    with torch.no_grad(), amp():
        old, values = behavior(*batch, **cards)
        old_dist = torch.distributions.Categorical(logits=old.float())
        actions = old_dist.sample()
        old_logp = old_dist.log_prob(actions).clone()
        targets = values.float().clone() + 0.5

    def rollout_pair():
        with torch.inference_mode(), amp():
            behavior(*one, **one_cards)
            opponent(*one, **one_cards)

    losses = []

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), amp():
            reference, _ = behavior(*batch, **cards)
        with amp():
            logits, value = model(*batch, **cards)
            distribution = torch.distributions.Categorical(logits=logits.float())
            loss, _ = ppo_loss_with_reference(
                logits,
                reference,
                distribution.log_prob(actions),
                old_logp,
                torch.ones(args.batch_size, device="cuda"),
                value.float(),
                targets,
                distribution.entropy(),
            )
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))

    def measure(function):
        for _ in range(2):
            function()
        times = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            function()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
        return {"mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95))}

    rollout = measure(rollout_pair)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    training = measure(train_step)
    changed = not torch.equal(before, model.critic[-1].weight.detach())
    assert changed
    result = {
        "model": args.model,
        "config": cfg.__dict__,
        "parameters": parameters,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": "fp32_weights_bf16_autocast",
        "context": cfg.context_len,
        "batch_size": args.batch_size,
        "resident_models": 3,
        "iterations": args.iterations,
        "input": "saved_real_simulator_observations",
        "legal_actions_at_probe": int(legal.sum()),
        "objective": "PPO_with_reference_KL_synthetic_advantages_and_value_targets",
        "rollout_pair": rollout,
        "train_step": training,
        "train_samples_per_second": args.batch_size * 1000 / training["mean_ms"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "finite_losses": bool(np.isfinite(losses).all()),
        "weights_changed": changed,
        "boundary": "network_capacity_and_optimizer_probe_not_online_selfplay_or_policy_strength",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
