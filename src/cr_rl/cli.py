from __future__ import annotations

import argparse
import platform
import sys

import numpy as np
import torch

from cr_rl.config import SimulatorPaths
from cr_rl.data.trajectory import TrajectoryBuffer
from cr_rl.models import ActorCritic, ModelConfig


def doctor() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.machine()})")
    print(
        f"PyTorch: {torch.__version__}; MPS={torch.backends.mps.is_available()}; "
        f"CUDA={torch.cuda.is_available()}"
    )
    paths = SimulatorPaths.discover()
    print(f"crforge install: {paths.install} [{'OK' if paths.install.exists() else 'MISSING'}]")
    print(f"game tables: {paths.tables} [{'OK' if paths.tables.exists() else 'MISSING'}]")
    if paths.install.exists() and paths.tables.exists():
        from cr_rl.envs.crforge import CrForgeVector

        with CrForgeVector(count=1, threads=1) as env:
            obs, masks = env.reset([20261102])
            print(f"observation shape={obs.shape}; action mask shape={masks.shape}")
            print(f"metadata keys={list(env.metadata())[:12]}")
    return 0


def smoke(episodes: int, random_legal: bool = False, steps: int = 40) -> int:
    from cr_rl.envs.crforge import CrForgeVector

    with CrForgeVector(count=episodes, threads=min(4, episodes)) as env:
        env.set_autoreset(False)
        obs, masks = env.reset(list(range(20261102, 20261102 + episodes)))
        print(f"reset: obs={obs.shape}, masks={masks.shape}, legal={masks.sum(axis=-1).mean():.1f}")
        rng = np.random.default_rng(20261102)
        action_histogram: dict[int, int] = {}
        played_cards: dict[str, int] = {}
        hand_fields = env.vector.schema.header["hand_card"]
        for _ in range(steps):
            if random_legal:
                for i in range(episodes):
                    for seat in range(2):
                        legal = np.flatnonzero(env.masks[i, seat])
                        action = int(rng.choice(legal))
                        env.actions[i, seat] = action
                        action_histogram[action] = action_histogram.get(action, 0) + 1
                        if 1 <= action < 2305:
                            slot = (action - 1) // 576
                            card_id = int(env.observations[i, seat, hand_fields[slot]])
                            if card_id >= 0:
                                name = env.vector.vocabulary[card_id]
                                played_cards[name] = played_cards.get(name, 0) + 1
            else:
                env.actions[:] = 0
            env.step()
            for index, done in enumerate(env.dones):
                if done:
                    env.reset_env(index, int(rng.integers(1, 2**31 - 1)))
        print(
            f"{steps} decisions completed; random_legal={random_legal}; "
            f"unique_actions={len(action_histogram)}; played_cards={played_cards}; "
            f"fault kinds={env.vector.fault_kind_counts()}"
        )
    return 0


def model_check(size: str) -> int:
    cfg = ModelConfig.preset(size)
    model = ActorCritic(cfg)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    b, t, n, f = 2, min(8, cfg.context_len), cfg.max_entities, cfg.entity_feature_dim
    header = torch.zeros(b, t, cfg.header_dim, device=device)
    entities = torch.zeros(b, t, n, f, device=device)
    entity_mask = torch.zeros(b, t, n, dtype=torch.bool, device=device)
    entity_mask[:, :, 0] = True
    logits, values = model(header, entities, entity_mask)
    print(f"preset={size}; params={model.parameter_count():,}; device={device}")
    print(f"logits={tuple(logits.shape)}; values={tuple(values.shape)}")
    return 0


def collect(steps: int, envs: int, output: str) -> int:
    from cr_rl.envs.crforge import CrForgeVector

    rng = np.random.default_rng(20261102)
    with CrForgeVector(count=envs, threads=min(envs, 8)) as env:
        env.set_autoreset(False)
        obs, masks = env.reset(list(range(20261102, 20261102 + envs)))
        metadata = env.metadata()
        metadata.update(
            {
                "backend": "crforge",
                "backend_commit": "7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6",
                "deck": env.deck,
                "card_level": env.level,
                "policy": "uniform_random_legal",
                "autoreset": False,
                "trajectory_purpose": "random_legal_baseline_only",
                "seed_start": 20261102,
                "requested_steps": steps,
            }
        )
        buffer = TrajectoryBuffer(metadata)
        for _ in range(steps):
            current_obs = obs.copy()
            current_masks = masks.copy()
            actions = np.zeros((envs, 2), dtype=np.int32)
            for i in range(envs):
                for seat in range(2):
                    actions[i, seat] = int(rng.choice(np.flatnonzero(current_masks[i, seat])))
            env.actions[:] = actions
            env.step()
            next_obs, next_masks = env.observations.copy(), env.masks.copy()
            done_flags = env.dones.copy().astype(bool)
            info_rows = [env.vector.info(i) for i in range(envs)]
            terminated = np.asarray(
                [done_flags[i] and not info_rows[i]["truncated"] for i in range(envs)],
                dtype=np.uint8,
            )
            truncated = np.asarray(
                [done_flags[i] and info_rows[i]["truncated"] for i in range(envs)],
                dtype=np.uint8,
            )
            buffer.append(
                current_obs,
                current_masks,
                actions,
                env.rewards.copy(),
                next_obs,
                next_masks,
                done_flags.astype(np.uint8),
                terminated=terminated,
                truncated=truncated,
            )
            for index, done in enumerate(done_flags):
                if done:
                    env.reset_env(index, int(rng.integers(1, 2**31 - 1)))
            obs, masks = env.observations.copy(), env.masks.copy()
        path = buffer.save(output)
        print(f"saved {steps * envs} two-seat transitions to {path}")
        print(f"fault kinds={env.vector.fault_kind_counts()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="cr-rl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--episodes", type=int, default=4)
    smoke_parser.add_argument("--steps", type=int, default=40)
    smoke_parser.add_argument("--random-legal", action="store_true")
    model_parser = sub.add_parser("model-check")
    model_parser.add_argument(
        "--size",
        choices=("tiny", "small", "compact", "general", "large", "production", "general_300m"),
        default="tiny",
    )
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--steps", type=int, default=256)
    collect_parser.add_argument("--envs", type=int, default=4)
    collect_parser.add_argument("--out", default="artifacts/random_rollout.npz")
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "smoke":
        return smoke(args.episodes, args.random_legal, args.steps)
    if args.command == "collect":
        return collect(args.steps, args.envs, args.out)
    return model_check(args.size)


if __name__ == "__main__":
    raise SystemExit(main())
