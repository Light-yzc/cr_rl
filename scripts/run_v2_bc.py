"""Fresh v2 BC on every published prefix-safe replay, with verified Drive backups."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cr_rl.data.bc_snapshot import freeze_datasets, read_published
from cr_rl.training.persistence import atomic_json, copy_verified

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--resume-acceptance", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--run-name", default="121m_v2_bc_20261103")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    if not acceptance.get("passed"):
        raise ValueError("Complete v2 acceptance must pass before formal BC")
    resumed = json.loads(args.resume_acceptance.read_text())
    if not all(
        resumed.get(key)
        for key in (
            "passed",
            "bc_model_bitwise_equal",
            "bc_optimizer_bitwise_equal",
            "initial_baselines_preserved",
            "bc_to_ppo_actor_exact",
            "candidate_complete_game_gate",
        )
    ):
        raise ValueError("v2 resume and BC-to-PPO handoff acceptance must pass")
    tuning = json.loads(args.benchmark.read_text())
    if tuning.get("representation_version") != 2:
        raise ValueError("Batch tuning must use the v2 model")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Drive is not mounted")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Supply W&B credentials through launcher stdin")
    output = ROOT / "artifacts" / args.run_name
    drive = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    if (output / "bc.pt").exists() or (drive / "bc/latest.json").exists():
        raise ValueError("Clean-slate v2 requires a new run; resume through train_bc explicitly")
    output.mkdir(parents=True, exist_ok=True)
    status = {
        "phase": "starting",
        "success": False,
        "run_name": args.run_name,
        "drive": str(drive),
        "cold_start": True,
        "epochs": args.epochs,
        "worker_pid": os.getpid(),
    }

    def sync():
        status["heartbeat_unix"] = time.time()
        atomic_json(output / "status.json", status)
        for path in [*output.glob("*.json"), *output.glob("*.log")]:
            snapshot = output / ".snapshots" / path.name
            snapshot.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, snapshot)
            copy_verified(snapshot, drive / path.name)

    try:
        sync()
        manifest, metadata, source_hash = read_published(args.source)
        if any(
            row.get("reconstruction_policy") != "first_omission_prefix"
            for row in metadata["replays"]
        ):
            raise ValueError("Refusing non-prefix-safe BC data")
        dataset = freeze_datasets([(args.source, args.source)], output / "data", deduplicate=True)
        atomic_json(output / "dataset.json", dataset)
        for p in (output / "data").glob("*.json"):
            copy_verified(p, drive / "data" / p.name)
        copy_verified(args.source_archive, drive / "source.tar.gz")
        copy_verified(args.acceptance, drive / "acceptance.json")
        copy_verified(args.resume_acceptance, drive / "resume_acceptance.json")
        copy_verified(args.benchmark, drive / "benchmark.json")
        chosen = tuning["chosen"]
        batch = int(chosen["batch_size"])
        accumulation = int(chosen["gradient_accumulation"])
        command = [
            sys.executable,
            "scripts/train_bc.py",
            str(output / "data"),
            "--v2",
            "--model",
            "production",
            "--entity-pooling",
            "dynamic4",
            "--device",
            "cuda",
            "--epochs",
            str(args.epochs),
            "--all-plays",
            "--stable-split",
            "--card-prior-correction",
            "--ability-fraction",
            "0.1",
            "--tile-label-sigma",
            "1.0",
            "--natural-validation-samples",
            "4096",
            "--stream-cache-gib",
            "8",
            "--entity-bucket",
            "16",
            "--fused-optimizer",
            "--attention-backend",
            chosen["attention_backend"],
            "--batch-size",
            str(batch),
            "--grad-accumulation",
            str(accumulation),
            "--learning-rate",
            "1e-4",
            "--tile-learning-rate",
            "3e-4",
            "--checkpoint-every-steps",
            "128",
            "--validation-every-steps",
            "32",
            "--holdout-decks",
            "configs/decks_eval.yaml",
            "--seed",
            "20261103",
            "--wandb",
            "--wandb-name",
            args.run_name,
            "--out",
            str(output / "bc.pt"),
            "--backup-dir",
            str(drive / "bc"),
        ]
        atomic_json(
            output / "training_config.json",
            {
                "command": command,
                "dataset": dataset,
                "source_manifest_sha256": source_hash,
                "cold_start": True,
            },
        )
        with (output / "train.log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            )
            status.update(
                phase="bc_training", training_pid=child.pid, dataset_replays=dataset["replays"]
            )
            while child.poll() is None:
                sync()
                time.sleep(20)
        if child.returncode:
            raise RuntimeError(f"v2 BC exited with code {child.returncode}; inspect train.log")
        report = json.loads((output / "bc.json").read_text())
        if report["representation_version"] != 2 or report["completed_epochs"] != args.epochs:
            raise RuntimeError("BC completion/configuration audit failed")
        history = report["history"]
        finite = all(math.isfinite(row["natural_validation"]["global_loss"]) for row in history)
        if not finite:
            raise RuntimeError("Non-finite held-out BC metrics")
        status.update(
            phase="completed",
            success=True,
            optimizer_steps=report["optimizer_steps"],
            final_validation=history[-1]["validation"],
            final_natural_validation=history[-1]["natural_validation"],
        )
        atomic_json(
            output / "ppo_readiness.json",
            {
                "bc_completed": True,
                "representation_version": 2,
                "next": "Select the held-out checkpoint, then verify BC-to-PPO warmup",
                "limitation": "BC imitation metrics do not establish battle strength",
            },
        )
        sync()
    except Exception:
        status.update(phase="failed", error=traceback.format_exc())
        sync()
        raise


if __name__ == "__main__":
    main()
