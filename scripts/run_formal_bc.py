"""Tune BC throughput, build cloud-backed shards, then run a bounded formal BC round."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--il-parquet", type=Path, required=True)
    parser.add_argument("--cochon-parquet", type=Path, required=True)
    parser.add_argument("--benchmark-data", type=Path, required=True)
    parser.add_argument("--init-from-bc", type=Path, required=True)
    parser.add_argument("--run-name", default="119m_formal_bc_expanded_20261103")
    parser.add_argument("--target-games", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=16384)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    output = ROOT / "artifacts" / args.run_name
    destination = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    data = ROOT / "artifacts" / (args.run_name + "_data")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "bc.pt").exists():
        raise ValueError(
            "Formal launcher requires a new run name; use train_bc --resume to continue"
        )
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Drive before launching a formal run")
    deadline = time.monotonic() + 3 * 3600
    status = {
        "phase": "starting",
        "run_name": args.run_name,
        "success": False,
        "target_games": args.target_games,
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "drive": str(destination),
    }

    def sync():
        status["heartbeat_unix"] = time.time()
        atomic_json(output / "status.json", status)
        for path in [*output.glob("*.json"), *output.glob("*.log")]:
            snapshot = output / ".log_snapshots" / path.name
            snapshot.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, snapshot)
            copy_verified(snapshot, destination / path.name)

    def run(command, phase):
        status["phase"] = phase
        sync()
        log_path = output / (phase + ".log")
        with log_path.open("a") as stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2"},
            )
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    child.terminate()
                    child.wait(timeout=30)
                    raise TimeoutError("Formal round reached its three-hour budget")
                sync()
                time.sleep(20)
        sync()
        if child.returncode:
            raise RuntimeError(log_path.read_text()[-5000:])

    try:
        if args.source_archive:
            copy_verified(args.source_archive, destination / "source.tar.gz")
        run([sys.executable, "-m", "pytest", "-q"], "tests")
        tuning = output / "bc_tuning.json"
        run(
            [
                sys.executable,
                "scripts/benchmark_bc.py",
                str(args.benchmark_data),
                "--out",
                str(tuning),
                "--iterations",
                "2",
            ],
            "tuning",
        )
        chosen = json.loads(tuning.read_text())["chosen"]
        status["configuration"] = chosen
        run(
            [
                sys.executable,
                "scripts/build_bc_shards.py",
                "--il-parquet",
                str(args.il_parquet),
                "--cochon-parquet",
                str(args.cochon_parquet),
                "--out",
                str(data),
                "--drive",
                str(destination / "data"),
                "--target",
                str(args.target_games),
            ],
            "data",
        )
        built = json.loads((data / "build_status.json").read_text())
        if built["replays"] < 128:
            raise RuntimeError("Expanded formal round needs at least 128 reconstructed games")
        status["dataset"] = {
            key: built[key] for key in ("replays", "valid_plays", "rows", "source_counts")
        }
        command = [
            sys.executable,
            "scripts/train_bc.py",
            str(data),
            "--model",
            "production",
            "--device",
            "cuda",
            "--init-from-bc",
            str(args.init_from_bc),
            "--epochs",
            str(args.epochs),
            "--samples-per-epoch",
            str(args.samples_per_epoch),
            "--batch-size",
            str(chosen["batch_size"]),
            "--grad-accumulation",
            str(chosen["gradient_accumulation"]),
            "--learning-rate",
            "3e-5",
            "--entity-bucket",
            str(chosen["entity_bucket"]),
            "--attention-backend",
            chosen["attention_backend"],
            "--fused-optimizer",
            "--wandb",
            "--holdout-decks",
            "configs/decks_eval.yaml",
            "--out",
            str(output / "bc.pt"),
            "--backup-dir",
            str(destination / "bc"),
        ]
        atomic_json(
            output / "training_config.json",
            {
                "command": command,
                "epochs": args.epochs,
                "samples_per_epoch": args.samples_per_epoch,
                "configuration": chosen,
                "dataset": status["dataset"],
            },
        )
        run(command, "bc_training")
        status.update(phase="completed", success=True)
    except Exception:
        key = os.environ.get("WANDB_API_KEY", "<absent>")
        status.update(phase="failed", error=traceback.format_exc().replace(key, "[REDACTED]"))
    finally:
        sync()
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
