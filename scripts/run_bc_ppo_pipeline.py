"""Bounded human BC, weights-only transfer and multi-deck PPO on mounted Colab.

Start over SSH with --start and a stdin JSON containing WANDB_API_KEY. No secret files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    local = ROOT / "artifacts" / args.run_name
    destination = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    data = ROOT / "artifacts/multideck"
    local.mkdir(parents=True, exist_ok=True)
    status = {"phase": "starting", "run_name": args.run_name, "drive_path": str(destination)}
    child = None

    def save_status():
        status["heartbeat_unix"] = time.time()
        atomic_json(local / "pipeline_status.json", status)
        copy_verified(local / "pipeline_status.json", destination / "pipeline_status.json")

    def run(command, phase):
        nonlocal child
        status["phase"] = phase
        save_status()
        print("COMMAND", command, flush=True)
        with (local / f"{phase}.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                save_status()
                copy_verified(local / f"{phase}.log", destination / f"{phase}.log")
                time.sleep(30)
        copy_verified(local / f"{phase}.log", destination / f"{phase}.log")
        if child.returncode:
            raise RuntimeError(f"{phase} exited {child.returncode}")

    try:
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Drive must be mounted")
        ready = data / "data_status.json"
        if not ready.exists() or not json.loads(ready.read_text()).get("ready"):
            data.mkdir(parents=True, exist_ok=True)
            run(
                [
                    sys.executable,
                    "scripts/collect_training_sources.py",
                    "--out",
                    str(data / "sources"),
                    "--shards",
                    "2",
                ],
                "collect_sources",
            )
            run(
                [
                    sys.executable,
                    "scripts/validate_league.py",
                    str(data / "sources/league_candidates.yaml"),
                    "--out",
                    str(data / "league.yaml"),
                ],
                "validate_league",
            )
            run(
                [
                    sys.executable,
                    "scripts/import_il_replay.py",
                    str(data / "sources/replays.parquet"),
                    "--limit",
                    "40",
                    "--source-revision",
                    "059d43a02138a34b1b3009cc2acc7630fb99a638",
                    "--out",
                    str(data / "human_bc"),
                ],
                "import_human_replays",
            )
            atomic_json(ready, {"ready": True})
        if (local / "bc.pt").exists():
            raise ValueError("Existing run: use a new name rather than overwrite")
        for folder in (data / "human_bc", data / "sources"):
            for source in folder.rglob("*"):
                if source.is_file():
                    copy_verified(source, destination / "data" / source.relative_to(data))
        for source in data.glob("league*"):
            copy_verified(source, destination / "data" / source.name)
        copy_verified(Path("/content/crrl-live-src.tar.gz"), destination / "source.tar.gz")
        run([sys.executable, "-m", "pytest", "-q"], "tests")
        run(
            [
                sys.executable,
                "scripts/validate_bc_transfer.py",
                str(data / "human_bc"),
                str(data / "league_train.yaml"),
                "--out",
                str(local / "acceptance"),
            ],
            "acceptance",
        )
        verification = json.loads((local / "acceptance/verification.json").read_text())
        if not verification["passed"]:
            raise RuntimeError("BC transfer acceptance failed")
        copy_verified(local / "acceptance/verification.json", destination / "acceptance.json")
        run(
            [
                sys.executable,
                "scripts/train_bc.py",
                str(data / "human_bc"),
                "--model",
                "production",
                "--epochs",
                str(args.bc_epochs),
                "--samples-per-epoch",
                str(args.bc_samples),
                "--batch-size",
                "4",
                "--grad-accumulation",
                "4",
                "--learning-rate",
                "3e-5",
                "--validation-samples",
                "1024",
                "--device",
                "cuda",
                "--wandb",
                "--holdout-decks",
                str(data / "league_eval.yaml"),
                "--backup-dir",
                str(destination / "bc"),
                "--out",
                str(local / "bc.pt"),
            ],
            "bc",
        )
        run(
            [
                sys.executable,
                "scripts/evaluate_bc_checkpoints.py",
                "--dataset",
                str(data / "human_bc"),
                "--run-dir",
                str(local),
                "--backup-dir",
                str(destination / "bc"),
            ],
            "bc_full_validation",
        )
        from cr_rl.training.selection import select_bc_checkpoint

        selected_bc, selection = select_bc_checkpoint(local / "bc.pt")
        baseline = selection["baseline"]
        final = selection["validation"]
        # A process exit alone must not promote a collapsed or worse BC policy.
        passed = (
            final["play_slot_accuracy"] > 0
            and final["balanced_wait_play_accuracy"] > baseline["balanced_wait_play_accuracy"]
        )
        status["bc_gate"] = {
            "passed": passed,
            "baseline": baseline,
            "final": final,
            "meaning": "held-out imitation improvement, not game-strength proof",
        }
        save_status()
        if not passed:
            raise RuntimeError("BC held-out imitation gate failed; PPO was not started")
        run(
            [
                sys.executable,
                "scripts/run_colab_training.py",
                "--model",
                "production",
                "--updates",
                str(args.ppo_updates),
                "--run-name",
                args.run_name + "_ppo",
                "--init-from-bc",
                str(selected_bc),
                "--decks",
                str(data / "league_train.yaml"),
            ],
            "ppo",
        )
        ppo_status = json.loads(
            (ROOT / "artifacts" / (args.run_name + "_ppo") / "job_status.json").read_text()
        )
        if ppo_status["phase"] != "completed":
            raise RuntimeError(f"PPO did not complete: {ppo_status['phase']}")
        run(
            [
                sys.executable,
                "scripts/evaluate_policy.py",
                "--policy",
                str(ROOT / "artifacts" / (args.run_name + "_ppo") / "ppo_latest.pt"),
                "--opponent",
                str(selected_bc),
                "--decks",
                str(data / "league_eval.yaml"),
                "--out",
                str(local / "evaluation"),
            ],
            "evaluation",
        )
        for source in (local / "evaluation").iterdir():
            copy_verified(source, destination / "evaluation" / source.name)
        status.update(phase="completed", ppo=ppo_status)
    except Exception:
        status.update(
            phase="failed",
            error=traceback.format_exc().replace(
                os.environ.get("WANDB_API_KEY", "<absent>"), "[REDACTED]"
            ),
        )
        print(status["error"], flush=True)
    finally:
        save_status()
        print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--run-name", default="119m_multideck_bc_20261103")
    parser.add_argument("--bc-epochs", type=int, default=4)
    parser.add_argument("--bc-samples", type=int, default=2048)
    parser.add_argument("--ppo-updates", type=int, default=3)
    args = parser.parse_args()
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    if args.start:
        secrets = json.load(sys.stdin)
        if set(secrets) != {"WANDB_API_KEY"}:
            raise ValueError("Only WANDB_API_KEY can enter through launcher stdin")
        local = ROOT / "artifacts" / args.run_name
        local.mkdir(parents=True, exist_ok=True)
        with (local / "pipeline.log").open("a") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--run-name",
                    args.run_name,
                    "--bc-epochs",
                    str(args.bc_epochs),
                    "--bc-samples",
                    str(args.bc_samples),
                    "--ppo-updates",
                    str(args.ppo_updates),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **secrets},
                start_new_session=True,
            )
        print(json.dumps({"pid": child.pid, "run_name": args.run_name}))
    else:
        worker(args)


if __name__ == "__main__":
    main()
