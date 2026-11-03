"""Real-simulator acceptance: BC-compatible actor, value warmup, PPO, restore, export."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from evaluate_policy import load_policy
from validate_training import audit_rollout

from cr_rl.training.persistence import atomic_json, publish_update, restore_run, sync_logs
from cr_rl.training.telemetry import TrainingTelemetry

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drive", type=Path)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--init-from-bc", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "model": args.model,
        "python": sys.version,
        "torch": torch.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
    }
    initial = args.out / "initial"
    resumed = args.out / "restored"
    destination = args.drive or args.out / "backup"
    tracking = telemetry = None
    if args.wandb:
        import wandb

        tracking = wandb.init(
            project="cr-rl",
            name=args.out.name,
            tags=["independent-critic", "acceptance"],
            config={
                "model": args.model,
                "warmup_rounds": 1,
                "warmup_games": 1,
                "critic_epochs": 1,
                "purpose": "pipeline acceptance, not strength evaluation",
            },
        )
        telemetry = TrainingTelemetry(tracking, args.steps * 2)
        atomic_json(args.out / "wandb_run.json", {"url": tracking.url, "id": tracking.id})
        report["wandb_url"] = tracking.url

    def run(output, extra, steps):
        command = [
            sys.executable,
            "scripts/train_selfplay.py",
            "--model",
            args.model,
            "--device",
            args.device,
            "--envs",
            "2",
            "--steps",
            str(steps),
            "--epochs",
            "1",
            "--critic-epochs",
            "1",
            "--batch-size",
            "4",
            "--grad-accumulation",
            "4",
            "--critic-mode",
            "independent",
            "--out",
            str(output),
            "--progress-jsonl",
            str(output / "progress.jsonl"),
            *extra,
        ]
        output.mkdir(parents=True, exist_ok=True)
        progress = output / "progress.jsonl"
        seen = len(progress.read_text().splitlines()) if progress.exists() else 0

        def drain():
            nonlocal seen
            rows = progress.read_text().splitlines() if progress.exists() else []
            for row in rows[seen:]:
                try:
                    event = json.loads(row)
                except json.JSONDecodeError:
                    break
                if telemetry:
                    telemetry.log(event)
                seen += 1

        with (output / "train.log").open("a") as stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2"},
            )
            last_sync = time.monotonic()
            while child.poll() is None:
                drain()
                if args.drive and time.monotonic() - last_sync >= 20:
                    sync_logs(output, destination)
                    last_sync = time.monotonic()
                time.sleep(2)
        drain()
        if child.returncode:
            raise RuntimeError((output / "train.log").read_text()[-5000:])
        if telemetry:
            telemetry.log_summary(json.loads((output / "summary.json").read_text()))

    try:
        extra = ["--critic-warmup-updates", "1", "--critic-warmup-games", "1"]
        if args.init_from_bc:
            extra += ["--init-from-bc", str(args.init_from_bc)]
        run(initial, extra, args.steps)
        state = torch.load(initial / "ppo_latest.pt", map_location="cpu", weights_only=False)
        snapshot = next(initial.glob("opponent_000000_*.pt"))
        before = torch.load(snapshot, map_location="cpu", weights_only=False)["model"]
        assert all(torch.equal(before[k], v) for k, v in state["model"].items())
        assert state["critic_warmup"]["games"] >= 1, "Need a complete game for real warmup"
        assert not state["optimizer"]["state"]
        assert state["critic_optimizer"]["state"]
        value_steps = max(int(s["step"]) for s in state["critic_optimizer"]["state"].values())
        report["warmup"] = {
            "actor_unchanged": True,
            "critic_steps": value_steps,
            "progress": state["critic_warmup"],
            "initialization": state["critic_initialization"],
        }
        report["rollout"] = audit_rollout(initial / "rollouts/rollout_000001.npz")
        del state, before
        report["backup_1"] = publish_update(initial, destination, 1)
        restore_run(destination, resumed)
        # A new rollout after restore proves warmup state and both optimizers are usable.
        run(resumed, ["--resume", str(resumed / "ppo_latest.pt")], 32)
        state = torch.load(resumed / "ppo_latest.pt", map_location="cpu", weights_only=False)
        before = torch.load(snapshot, map_location="cpu", weights_only=False)["model"]
        assert state["updates"] == 2
        assert state["optimizer"]["state"]
        assert (
            max(int(s["step"]) for s in state["critic_optimizer"]["state"].values()) > value_steps
        )
        assert any(not torch.equal(before[k], v) for k, v in state["model"].items())
        assert not any(k.startswith("critic.") for k in state["model"])
        report["resume"] = {
            "actor_changed": True,
            "critic_steps_increased": True,
            "actor_steps": max(int(s["step"]) for s in state["optimizer"]["state"].values()),
        }
        exported = torch.load(resumed / "actor_export.pt", map_location="cpu", weights_only=False)
        assert set(exported["model"]) == set(state["model"])
        assert all(torch.equal(v, exported["model"][k]) for k, v in state["model"].items())
        assert "critic" not in exported and "optimizer" not in exported
        del state, before, exported
        exported_model, _ = load_policy(resumed / "actor_export.pt", "cpu")
        report["export"] = {
            "loadable": True,
            "contains_critic": False,
            "parameters": exported_model.parameter_count(),
        }
        del exported_model
        report["backup_2"] = publish_update(resumed, destination, 2)
        report["passed"] = True
    finally:
        atomic_json(args.out / "validation.json", report)
        if args.drive:
            atomic_json(destination / "validation.json", report)
        if tracking:
            tracking.summary["acceptance_passed"] = report["passed"]
            tracking.save(str(args.out / "validation.json"), base_path=str(args.out), policy="now")
            for directory in (initial, resumed):
                for name in ("train.log", "summary.json", "progress.jsonl", "metrics.jsonl"):
                    path = directory / name
                    if path.exists():
                        tracking.save(str(path), base_path=str(args.out), policy="now")
            tracking.finish(exit_code=0 if report["passed"] else 1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
