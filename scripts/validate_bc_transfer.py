"""Actual Linux simulator isolation and BC -> PPO -> PPO-resume acceptance check."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from cr_rl.envs.crforge import CrForgeVector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("league", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    league = yaml.safe_load(args.league.read_text())
    names = list(league["decks"])
    decks = [league["decks"][name] for name in names[:2]]
    forms = [league["slots"][name] for name in names[:2]]
    with CrForgeVector(2, 2, independent=True) as env:
        env.reset([101, 202], decks, decks, forms, forms)
        env.step(np.zeros((2, 2), dtype=np.int32))
        before = env.observations[1].copy()
        env.vector.reset_env(0, 303, decks[1], decks[0], forms[1], forms[0])
        assert np.array_equal(before, env.observations[1])
        assert env.vector.episode_seed(1) == 202
        assert env.vector.episode_seed(0) == 303
        assert not env.vector.info(0)["fault"]
    bc = args.out / "tiny_bc.pt"
    commands = [
        [
            sys.executable,
            "scripts/train_bc.py",
            str(args.dataset),
            "--model",
            "tiny",
            "--epochs",
            "1",
            "--samples-per-epoch",
            "128",
            "--validation-samples",
            "64",
            "--batch-size",
            "4",
            "--device",
            "cuda",
            "--out",
            str(bc),
        ],
        [
            sys.executable,
            "scripts/train_bc.py",
            str(args.dataset),
            "--model",
            "tiny",
            "--epochs",
            "2",
            "--samples-per-epoch",
            "128",
            "--validation-samples",
            "64",
            "--batch-size",
            "4",
            "--device",
            "cuda",
            "--out",
            str(bc),
            "--resume",
            str(bc),
        ],
        [
            sys.executable,
            "scripts/train_selfplay.py",
            "--critic-mode",
            "shared",
            "--model",
            "tiny",
            "--envs",
            "2",
            "--steps",
            "1300",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cuda",
            "--out",
            str(args.out / "ppo"),
            "--decks",
            str(args.league),
            "--init-from-bc",
            str(bc),
        ],
        [
            sys.executable,
            "scripts/train_selfplay.py",
            "--critic-mode",
            "shared",
            "--model",
            "tiny",
            "--envs",
            "2",
            "--steps",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cuda",
            "--out",
            str(args.out / "ppo"),
            "--decks",
            str(args.league),
            "--resume",
            str(args.out / "ppo/ppo_latest.pt"),
        ],
    ]
    summaries = []
    for command in commands:
        subprocess.run(command, check=True)
        if "scripts/train_selfplay.py" in command:
            summaries.append(json.loads((args.out / "ppo/summary.json").read_text()))
    state = torch.load(args.out / "ppo/ppo_latest.pt", map_location="cpu", weights_only=False)
    assert state["updates"] == 2
    assert state["initialization"]["optimizer_reset"]
    assert state["initialization"]["bc_epochs"] == 2
    assert state["optimizer"]["state"]
    assert len(summaries[0]["episode_matchups"]) > 2, "Must exercise real end-game resampling"
    assert all(summary["faulted_samples"] == 0 for summary in summaries)
    (args.out / "verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "lane_reset_isolated": True,
                "bc_resume": True,
                "bc_to_ppo": True,
                "ppo_resume": True,
                "summaries": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
