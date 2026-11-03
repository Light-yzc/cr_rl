"""Restore a real production checkpoint into a new directory and optimize on GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from cr_rl.training.persistence import atomic_json, restore_run


def optimizer_steps(state):
    return max(int(row["step"].item()) for row in state["optimizer"]["state"].values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "ppo_latest.pt").exists():
        raise ValueError("Use a new restore-probe directory")
    manifest = restore_run(args.source, args.out)
    checkpoint = args.out / "ppo_latest.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    before_steps = optimizer_steps(state)
    initialization = state["initialization"]
    del state
    # No --decks: this specifically verifies restoration of the frozen deck pool,
    # even if the workspace default has subsequently changed.
    command = [
        sys.executable,
        "scripts/train_selfplay.py",
        "--model",
        "production",
        "--device",
        "cuda",
        "--envs",
        "2",
        "--steps",
        "16",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--out",
        str(args.out),
        "--resume",
        str(checkpoint),
    ]
    with (args.out / "resume.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["updates"] == manifest["update"] + 1
    after_steps = optimizer_steps(state)
    assert after_steps > before_steps
    assert state["initialization"] == initialization
    saved_pool = json.loads((args.out / "run_config.json").read_text())["decks"]
    assert state["league_config"]["decks"] == saved_pool["decks"]
    report = {
        "passed": True,
        "restored_update": manifest["update"],
        "resumed_update": state["updates"],
        "optimizer_steps_before": before_steps,
        "optimizer_steps_after": after_steps,
        "restored_deck_entries": len(saved_pool["decks"]),
        "device": "cuda",
        "semantics": "weights/optimizer/RNG restored; simulator starts fresh matches",
    }
    atomic_json(args.out / "verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
