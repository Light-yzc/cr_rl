"""Validate shaping against a complete pinned-simulator game and a PPO rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cr_rl.envs.crforge import CrForgeVector
from cr_rl.training.persistence import atomic_json
from cr_rl.training.rewards import RewardConfig, shaped_rewards, tower_potentials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ppo-run", type=Path)
    args = parser.parse_args()
    gamma = 0.9995
    cfg = RewardConfig(enabled=True)
    rng = np.random.default_rng(20261103)
    discounted = np.zeros((1, 2), dtype=np.float64)
    report = {"passed": False, "gamma": gamma, "nonzero_hp_steps": 0, "nonzero_crown_steps": 0}
    with CrForgeVector(1, 1) as env:
        schema = env.vector.schema
        env.set_autoreset(False)
        env.reset([20261103])
        initial = sum(tower_potentials(env.observations, env.vector.schema, cfg))
        for step in range(2000):
            current = env.observations.copy()
            actions = np.array(
                [[rng.choice(np.flatnonzero(env.masks[0, seat])) for seat in range(2)]],
                dtype=np.int32,
            )
            env.step(actions)
            info = env.vector.info(0)
            assert not info["fault"], info
            terminal = bool(env.dones[0]) and not info["truncated"]
            reward, components = shaped_rewards(
                current,
                env.observations,
                actions,
                env.rewards,
                np.array([terminal]),
                env.vector.schema,
                gamma,
                cfg,
            )
            assert np.isfinite(reward).all()
            np.testing.assert_allclose(reward.sum(-1), 0, atol=2e-6)
            discounted += gamma**step * (reward - env.rewards)
            for name in ("hp", "crown"):
                report[f"nonzero_{name}_steps"] += int(np.any(components[name] != 0))
            if env.dones[0]:
                assert terminal, "Need a true terminal for the telescoping acceptance"
                report.update(
                    steps=step + 1,
                    raw_result=env.rewards.tolist(),
                    discounted_shaping=discounted.tolist(),
                )
                np.testing.assert_allclose(discounted, -initial, atol=2e-6)
                break
        else:
            raise AssertionError("No complete game within the acceptance budget")
    assert report["nonzero_hp_steps"] > 0
    if args.ppo_run:
        summary = json.loads((args.ppo_run / "summary.json").read_text())
        assert summary["training_reward_config"] == cfg.metadata()
        with np.load(summary["rollout"]) as data:
            expected, _ = shaped_rewards(
                data["obs"],
                data["next_obs"],
                data["actions"],
                data["rewards"],
                data["terminated"],
                schema,
                gamma,
                cfg,
                faulted=data["fault_codes"] != 0,
            )
            np.testing.assert_allclose(data["training_rewards"], expected, atol=1e-7)
        report["ppo_optimizer_steps"] = summary["optimizer_steps"]
        report["critic_optimizer_steps"] = summary["critic_optimizer_steps"]
        assert report["critic_optimizer_steps"] > 0
    report["passed"] = True
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
