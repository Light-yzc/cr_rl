"""Verify a v3 PPO checkpoint and its progress record without running a game."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--progress-jsonl", type=Path)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    required = {"format", "training_driver", "model_architecture_version", "config",
                "temporal_config", "decision_ticks", "model", "optimizer", "progress"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError("v3 PPO checkpoint is missing: " + ", ".join(missing))
    if state["format"] != "crrl-state-ppo-v1" or state["training_driver"] != "persistent-state-v3":
        raise ValueError("Checkpoint is not from the v3 PPO driver")
    if state["model_architecture_version"] != 3 or state["decision_ticks"] != 8:
        raise ValueError("Checkpoint does not use the v3 0.4s contract")
    if not state["optimizer"]["state"]:
        raise ValueError("v3 PPO checkpoint has no optimizer state")
    events = []
    if args.progress_jsonl:
        for line in args.progress_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                if event.get("phase") == "state_ppo":
                    events.append(event)
        if events and events[-1]["update"] != state["progress"]["update"]:
            raise ValueError("Progress JSONL and checkpoint update differ")
    result = {"status": "pass", "format": state["format"], "architecture": 3,
              "decision_ticks": 8, "update": state["progress"]["update"],
              "optimizer_parameters": len(state["optimizer"]["state"]),
              "progress_events": len(events)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
