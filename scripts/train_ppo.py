from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.advantages import compute_gae
from cr_rl.training.ppo import ppo_loss


def train(args) -> None:
    from cr_rl.envs.crforge import CrForgeVector

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = ModelConfig.preset(args.model)
    checkpoint = (
        torch.load(args.resume, map_location=device, weights_only=False)
        if args.resume is not None
        else None
    )
    if checkpoint is not None:
        saved_cfg = ModelConfig(**checkpoint["config"])
        cfg = ModelConfig(
            **{
                **cfg.__dict__,
                "tile_query_residual": saved_cfg.tile_query_residual,
                "tile_head_fp32": saved_cfg.tile_head_fp32,
                "spatial_fourier": saved_cfg.spatial_fourier,
            }
        )
    model = ActorCritic(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    update_number = 0
    resume_state = None
    if args.resume is not None:
        if ModelConfig(**checkpoint["config"]) != cfg:
            raise ValueError("Resume checkpoint model config does not match --model")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        update_number = int(checkpoint.get("updates", 0))
        resume_state = checkpoint
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.startswith("cuda") and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with CrForgeVector(count=args.envs, threads=min(args.envs, 8)) as env:
        raw_obs, raw_masks = env.reset(list(range(args.seed, args.seed + args.envs)))
        adapter = CrForgeObservationAdapter(env.vector.schema, env.vector.vocabulary_size)
        schema = env.vector.schema
        n_agents = args.envs * 2
        histories = [deque(maxlen=cfg.context_len) for _ in range(n_agents)]
        observations = np.zeros((args.steps, args.envs, 2, schema.seat_ints), dtype=np.int32)
        masks = np.zeros((args.steps, args.envs, 2, schema.actions), dtype=np.uint8)
        actions = np.zeros((args.steps, args.envs, 2), dtype=np.int32)
        old_logp = np.zeros((args.steps, args.envs, 2), dtype=np.float32)
        values = np.zeros_like(old_logp)
        rewards = np.zeros_like(old_logp)
        dones = np.zeros((args.steps, args.envs, 2), dtype=np.uint8)
        episode_start = np.zeros((args.steps, args.envs), dtype=np.int32)
        current_episode_start = np.zeros(args.envs, dtype=np.int32)
        rng = np.random.default_rng(args.seed)
        if resume_state is not None and "numpy_rng_state" in resume_state:
            rng.bit_generator.state = resume_state["numpy_rng_state"]

        for step in range(args.steps):
            observations[step] = raw_obs
            masks[step] = raw_masks
            episode_start[step] = current_episode_start
            current = raw_obs.reshape(n_agents, schema.seat_ints)
            for agent in range(n_agents):
                histories[agent].append(current[agent].copy())
            window = np.stack(
                [
                    np.pad(
                        np.stack(list(h))[-cfg.context_len :],
                        ((max(0, cfg.context_len - len(h)), 0), (0, 0)),
                    )
                    for h in histories
                ]
            ).astype(np.int32, copy=False)
            time_mask = np.stack(
                [np.arange(cfg.context_len) >= max(0, cfg.context_len - len(h)) for h in histories]
            )
            header, entities, entity_mask = adapter(window)
            legal = raw_masks.reshape(n_agents, schema.actions).astype(bool)
            tensors = [
                torch.as_tensor(header, device=device),
                torch.as_tensor(entities, device=device),
                torch.as_tensor(entity_mask, device=device),
                torch.as_tensor(legal, device=device),
                torch.as_tensor(time_mask, device=device),
            ]
            hand_ids = torch.as_tensor(current[:, list(schema.header["hand_card"])], device=device)
            with torch.no_grad():
                joint_logp, value = model(
                    tensors[0],
                    tensors[1],
                    tensors[2],
                    tensors[3],
                    tensors[4],
                    hand_card_ids=hand_ids,
                )
                distribution = torch.distributions.Categorical(logits=joint_logp)
                action = distribution.sample()
                logp = distribution.log_prob(action)
            action_np = action.cpu().numpy().reshape(args.envs, 2)
            actions[step] = action_np
            old_logp[step] = logp.cpu().numpy().reshape(args.envs, 2)
            values[step] = value.cpu().numpy().reshape(args.envs, 2)
            env.actions[:] = action_np
            env.step()
            next_obs = env.observations.copy()
            rewards[step] = env.rewards
            dones[step] = np.repeat(env.dones[:, None], 2, axis=1)
            raw_obs, raw_masks = next_obs, env.masks.copy()
            for env_index, done in enumerate(env.dones):
                if done:
                    histories[env_index * 2].clear()
                    histories[env_index * 2 + 1].clear()
                    current_episode_start[env_index] = step + 1

        # Bootstrap the final state from its observation history.
        final_current = raw_obs.reshape(n_agents, schema.seat_ints)
        final_windows = []
        final_time_masks = []
        for agent in range(n_agents):
            h = list(histories[agent]) + [final_current[agent]]
            h = h[-cfg.context_len :]
            pad = cfg.context_len - len(h)
            w = np.zeros((cfg.context_len, schema.seat_ints), dtype=np.int32)
            w[pad:] = np.stack(h)
            tm = np.zeros(cfg.context_len, dtype=np.bool_)
            tm[pad:] = True
            final_windows.append(w)
            final_time_masks.append(tm)
        fh, fe, fm = adapter(np.stack(final_windows))
        final_hand_ids = torch.as_tensor(
            final_current[:, list(schema.header["hand_card"])], device=device
        )
        with torch.no_grad():
            _, last_value = model(
                torch.as_tensor(fh, device=device),
                torch.as_tensor(fe, device=device),
                torch.as_tensor(fm, device=device),
                time_mask=torch.as_tensor(np.asarray(final_time_masks), device=device),
                hand_card_ids=final_hand_ids,
            )

        reward_t = torch.as_tensor(rewards.reshape(args.steps, n_agents), device=device)
        value_t = torch.as_tensor(values.reshape(args.steps, n_agents), device=device)
        done_t = torch.as_tensor(dones.reshape(args.steps, n_agents), device=device)
        advantages, returns = compute_gae(
            reward_t, value_t, done_t, last_value, gamma=args.gamma, gae_lambda=args.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        flat_adv = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_old_logp = torch.as_tensor(old_logp.reshape(-1), device=device)
        flat_actions = torch.as_tensor(actions.reshape(-1), device=device)
        flat_obs = observations.reshape(args.steps, n_agents, schema.seat_ints)
        flat_masks = masks.reshape(args.steps, n_agents, schema.actions)
        metric_rows = []

        for _epoch in range(args.epochs):
            permutation = rng.permutation(args.steps * n_agents)
            for start in range(0, len(permutation), args.batch_size):
                indices = permutation[start : start + args.batch_size]
                batch_windows, batch_time, batch_legal, batch_hand_ids = [], [], [], []
                for index in indices:
                    t, agent = divmod(int(index), n_agents)
                    # Rebuild this exact action's causal context without crossing an episode.
                    env_index = agent // 2
                    left = max(int(episode_start[t, env_index]), t - cfg.context_len + 1)
                    seq = flat_obs[left : t + 1, agent]
                    pad = cfg.context_len - len(seq)
                    w = np.zeros((cfg.context_len, schema.seat_ints), dtype=np.int32)
                    w[pad:] = seq
                    tm = np.zeros(cfg.context_len, dtype=np.bool_)
                    tm[pad:] = True
                    batch_windows.append(w)
                    batch_time.append(tm)
                    batch_legal.append(flat_masks[t, agent].astype(bool))
                    batch_hand_ids.append(flat_obs[t, agent, list(schema.header["hand_card"])])
                bh, be, bm = adapter(np.stack(batch_windows))
                legal_t = torch.as_tensor(np.stack(batch_legal), device=device)
                logp_all, value_new = model(
                    torch.as_tensor(bh, device=device),
                    torch.as_tensor(be, device=device),
                    torch.as_tensor(bm, device=device),
                    legal_t,
                    torch.as_tensor(np.stack(batch_time), device=device),
                    hand_card_ids=torch.as_tensor(np.stack(batch_hand_ids), device=device),
                )
                dist = torch.distributions.Categorical(logits=logp_all)
                new_logp = dist.log_prob(flat_actions[indices])
                loss, metrics = ppo_loss(
                    new_logp,
                    flat_old_logp[indices],
                    flat_adv[indices],
                    value_new,
                    flat_returns[indices],
                    dist.entropy(),
                    args.clip_coef,
                    args.value_coef,
                    args.entropy_coef,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                metric_rows.append({k: float(v.cpu()) for k, v in metrics.items()})

        checkpoint = output / "ppo_latest.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.__dict__,
                "backend_commit": "7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6",
                "data_version": env.vector.data_version,
                "content_sha": env.vector.content_sha,
                "seed": args.seed,
                "updates": update_number + 1,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if device.startswith("cuda")
                else None,
                "numpy_rng_state": rng.bit_generator.state,
            },
            checkpoint,
        )
        if args.keep_history:
            history_checkpoint = output / f"ppo_update_{update_number + 1:06d}.pt"
            torch.save(
                {"model": model.state_dict(), "config": cfg.__dict__, "updates": update_number + 1},
                history_checkpoint,
            )
        summary = {
            "update": update_number + 1,
            "steps": args.steps,
            "envs": args.envs,
            "agents": n_agents,
            "samples": args.steps * n_agents,
            "device": device,
            "params": model.parameter_count(),
            "mean_reward": float(rewards.mean()),
            "mean_value": float(values.mean()),
            "metrics": {k: float(np.mean([row[k] for row in metric_rows])) for k in metric_rows[0]},
            "seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary) + "\n")
        print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy shared-seat PPO smoke. Use train_selfplay.py for all current training; "
            "this script is retained only for historical comparisons."
        )
    )
    parser.add_argument(
        "--model",
        choices=("tiny", "small", "compact", "general", "large", "production", "general_300m"),
        default="tiny",
    )
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="artifacts/ppo_smoke")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--keep-history", action="store_true")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--allow-legacy-shared-policy", action="store_true")
    args = parser.parse_args()
    if not args.allow_legacy_shared_policy:
        parser.error(
            "legacy shared-seat PPO is disabled by default; use scripts/train_selfplay.py "
            "or pass --allow-legacy-shared-policy for a historical comparison"
        )
    if args.updates < 1:
        parser.error("--updates must be >= 1")
    initial_resume = args.resume
    base_seed = args.seed
    for update in range(args.updates):
        if update:
            args.resume = Path(args.out) / "ppo_latest.pt"
        else:
            args.resume = initial_resume
        args.seed = base_seed + update
        train(args)


if __name__ == "__main__":
    main()
