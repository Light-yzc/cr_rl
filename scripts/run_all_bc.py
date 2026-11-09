"""Switch at a verified checkpoint and cover all available BC labels in frozen stages."""

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

from cr_rl.data.bc_snapshot import freeze_datasets, read_published
from cr_rl.training.persistence import atomic_json, copy_verified

ROOT = Path(__file__).resolve().parents[1]


def resolve_shard_mix_window(stage, requested=64):
    """Use the saved setting for an old stage; default new stages to mixing.

    This prevents a resumed historical checkpoint from silently changing its
    within-epoch order, while new BC snapshots avoid shard-local batches.
    """
    stage = Path(stage)
    config_path = stage / "training_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        command = config.get("command", [])
        if "--shard-mix-window" in command:
            index = command.index("--shard-mix-window") + 1
            try:
                saved = int(command[index])
            except (IndexError, TypeError, ValueError):
                saved = None
            if saved is not None and saved >= 1:
                return saved
        # Old stages without an explicit flag used train_bc's historical
        # default of one shard. Preserve that order when resuming them.
        return 1
    return requested


def stage_training_plan(stage, source_complete, final_epochs):
    """Keep a resumed stage's epoch target independent of producer progress."""
    path = stage / "training_plan.json"
    if path.exists():
        return json.loads(path.read_text())
    previous = stage / "training_config.json"
    if previous.exists():
        command = json.loads(previous.read_text())["command"]
        epochs = int(command[command.index("--epochs") + 1])
        # Old in-progress stages have no proof that their snapshot was final.
        source_complete = False
    else:
        epochs = final_epochs if source_complete else 1
    plan = {"epochs": epochs, "source_complete_at_start": source_complete}
    atomic_json(path, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=Path("/content/broad_bc_worker.json"))
    parser.add_argument("--run-name", default="119m_all_bc_20261103")
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--include-legacy-data", action="store_true")
    parser.add_argument("--min-replays", type=int, default=512)
    parser.add_argument("--conditional-v2", action="store_true")
    parser.add_argument("--fallback-source", type=Path)
    parser.add_argument("--split-registry", type=Path)
    parser.add_argument("--final-epochs", type=int, default=1)
    parser.add_argument("--carry-initial-optimizer", action="store_true")
    parser.add_argument("--stream-cache-gib", type=float, default=8.0)
    parser.add_argument("--persistent-validation-cache", type=Path)
    parser.add_argument(
        "--shard-mix-window",
        type=int,
        default=64,
        help="Number of randomized shards in one working set (default: 64)",
    )
    parser.add_argument("--timing-objective", choices=("balanced", "natural"), default="balanced")
    args = parser.parse_args()
    if args.final_epochs < 1:
        parser.error("--final-epochs must be positive")
    if args.stream_cache_gib <= 0:
        parser.error("--stream-cache-gib must be positive")
    jobs = json.loads(args.jobs.read_text())
    output = ROOT / "artifacts" / args.run_name
    drive = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "pipeline.json"
    pipeline = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "stage": 0,
            "stage_completed": True,
            "last_source_hash": None,
        }
    )
    status = {"phase": "waiting_acceptance", "run_name": args.run_name, "success": False}

    def sync(stage=None):
        status["heartbeat_unix"] = time.time()
        atomic_json(output / "status.json", status)
        atomic_json(state_path, pipeline)
        roots = [(output, drive)]
        if stage:
            roots.append((stage, drive / stage.name))
        for local, remote in roots:
            for path in [*local.glob("*.json"), *local.glob("*.log")]:
                snapshot = local / ".snapshots" / path.name
                snapshot.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, snapshot)
                copy_verified(snapshot, remote / path.name)

    try:
        while args.acceptance and not args.acceptance.exists():
            sync()
            time.sleep(10)
        if args.acceptance and not json.loads(args.acceptance.read_text()).get("passed"):
            raise RuntimeError("Full-data BC integration acceptance failed")
        status["phase"] = "waiting_checkpoint"
        while not args.initial_checkpoint:
            path = Path("/content/all_bc_switch.json")
            switch = json.loads(path.read_text()) if path.exists() else {}
            if switch.get("phase") == "ready":
                break
            sync()
            time.sleep(5)
        initial = args.initial_checkpoint or Path(switch["checkpoint"])
        broad = Path(jobs["data_status"]).parent
        old = Path(jobs["bc_status"]).parent.with_name(
            Path(jobs["bc_status"]).parent.name + "_data"
        )
        old_drive = (
            Path("/content/drive/MyDrive/cr_rl/runs") / Path(jobs["bc_status"]).parent.name / "data"
        )
        sources = [(broad, Path(jobs["drive"]))]
        if args.fallback_source:
            sources.append((args.fallback_source, args.fallback_source))
        if args.include_legacy_data:
            sources.insert(0, (old, old_drive))
        if pipeline.get("checkpoint"):
            initial = Path(pipeline["checkpoint"])
        while True:
            try:
                _, source_metadata, source_hash = read_published(broad)
            except (FileNotFoundError, ValueError):
                source_status = Path(jobs["data_status"])
                if source_status.exists() and json.loads(source_status.read_text())["phase"] in (
                    "completed",
                    "failed",
                ):
                    raise RuntimeError(
                        "Prefix producer finished without a readable complete dataset"
                    ) from None
                status["phase"] = "waiting_data"
                sync()
                time.sleep(5)
                continue
            built = json.loads(Path(jobs["data_status"]).read_text())
            if built["phase"] == "failed":
                raise RuntimeError(
                    "BC data producer failed; published shards and checkpoints retained"
                )
            if not args.include_legacy_data and any(
                r.get("reconstruction_policy")
                not in (
                    ("first_omission_prefix", "bounded_omission_prefix")
                    if args.conditional_v2
                    else ("first_omission_prefix",)
                )
                or r.get("max_omissions", 0) > (2 if args.conditional_v2 else 0)
                or (r.get("max_omissions", 0) > 0 and not r.get("omission_budget_enforced_in_loop"))
                for r in source_metadata["replays"]
            ):
                raise ValueError(
                    "Full BC requires prefix-safe reconstruction; rebuild into a new dataset"
                )
            if built["replays"] < args.min_replays and built["phase"] != "completed":
                status["phase"] = "waiting_data"
                sync()
                time.sleep(20)
                continue
            if pipeline["stage_completed"] and source_hash == pipeline["last_source_hash"]:
                if built["phase"] == "completed":
                    if pipeline.get("final_snapshot_epochs_completed", 0) >= args.final_epochs:
                        break
                else:
                    status["phase"] = "waiting_new_data"
                    sync()
                    time.sleep(20)
                    continue
            if pipeline["stage_completed"]:
                pipeline["stage"] += 1
                pipeline["stage_completed"] = False
            stage = output / f"stage_{pipeline['stage']:04d}"
            stage.mkdir(exist_ok=True)
            frozen = stage / "data"
            if not (frozen / "dataset.manifest.json").exists():
                try:
                    dataset = freeze_datasets(sources, frozen, deduplicate=True)
                except ValueError as error:
                    if "Producer is publishing" not in str(error):
                        raise
                    time.sleep(5)
                    continue
                atomic_json(stage / "dataset.json", dataset)
                for path in frozen.glob("*.json"):
                    copy_verified(path, drive / stage.name / "data" / path.name)
            dataset = json.loads((stage / "dataset.json").read_text())
            training_plan = stage_training_plan(
                stage, built["phase"] == "completed", args.final_epochs
            )
            status.update(
                phase="bc_training",
                stage=pipeline["stage"],
                dataset_replays=dataset["replays"],
                dataset_rows=dataset["rows"],
            )
            command = [
                sys.executable,
                "scripts/train_bc.py",
                str(frozen),
                "--model",
                "production",
                "--device",
                "cuda",
                "--epochs",
                str(training_plan["epochs"]),
                "--all-plays",
                "--stable-split",
                "--card-prior-correction",
                "--ability-fraction",
                "0.1",
                "--tile-label-sigma",
                "0.75",
                "--natural-validation-samples",
                "4096",
                "--stream-cache-gib",
                str(args.stream_cache_gib),
                "--cache-dir",
                str(output / "array_cache"),
                "--batch-size",
                "8",
                "--grad-accumulation",
                "12",
                "--entity-bucket",
                "16",
                "--attention-backend",
                "auto",
                "--fused-optimizer",
                "--learning-rate",
                "3e-5",
                "--checkpoint-every-steps",
                "128",
                "--wandb",
                "--holdout-decks",
                "configs/decks_eval.yaml",
                "--out",
                str(stage / "bc.pt"),
                "--backup-dir",
                str(drive / stage.name / "bc"),
            ]
            if args.conditional_v2:
                command[command.index("--card-prior-correction")] = "--no-card-prior-correction"
                for flag, value in (
                    ("--batch-size", "16"),
                    ("--grad-accumulation", "6"),
                    ("--tile-label-sigma", "1.0"),
                ):
                    command[command.index(flag) + 1] = value
                command += [
                    "--tile-learning-rate",
                    "3e-4",
                    "--timing-objective",
                    args.timing_objective,
                    "--bc-objective",
                    "conditional",
                    "--wait-margin-before",
                    "4",
                    "--wait-margin-after",
                    "2",
                    "--position-coef",
                    "1.0",
                    "--validation-every-steps",
                    "32",
                    "--wandb-name",
                    args.run_name + "/" + stage.name,
                ]
            # Apply the safe default to both joint and conditional objectives.
            # Existing stages retain their recorded order for reproducibility.
            command += [
                "--shard-mix-window",
                str(resolve_shard_mix_window(stage, args.shard_mix_window)),
            ]
            if args.split_registry:
                command += ["--split-registry", str(args.split_registry)]
            if args.persistent_validation_cache:
                command += ["--persistent-validation-cache", str(args.persistent_validation_cache)]
            if (stage / "bc.pt").exists():
                command += ["--resume", str(stage / "bc.pt")]
            else:
                command += ["--init-from-bc", str(initial)]
                if not args.conditional_v2 or pipeline["stage"] > 1 or args.carry_initial_optimizer:
                    command += ["--carry-optimizer"]
                if not args.conditional_v2:
                    command += ["--upgrade-spatial"]
            atomic_json(stage / "training_config.json", {"command": command, **dataset})
            sync(stage)
            with (stage / "train.log").open("a") as stream:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
                )
                status["training_pid"] = process.pid
                while process.poll() is None:
                    tracking = stage / "bc.wandb.json"
                    if tracking.exists():
                        status["wandb"] = json.loads(tracking.read_text())["url"]
                    sync(stage)
                    time.sleep(20)
            if process.returncode:
                raise RuntimeError((stage / "train.log").read_text()[-3000:])
            initial = stage / "bc.pt"
            pipeline.update(
                stage_completed=True,
                checkpoint=str(initial),
                last_source_hash=dataset["source_manifest_sha256"][
                    0 if args.fallback_source else -1
                ],
                final_snapshot_epochs_completed=(
                    training_plan["epochs"] if training_plan["source_complete_at_start"] else 0
                ),
            )
            sync(stage)
        # Full source processing and one full pass on its final snapshot are prerequisites.
        atomic_json(
            output / "ppo_readiness.json",
            {
                "full_data_bc_pass_completed": True,
                "final_snapshot_epochs_completed": pipeline["final_snapshot_epochs_completed"],
                "checkpoint": str(initial),
                "phase": "needs_heldout_and_match_evaluation",
                "next": "Compare heldout loss, play/position metrics and paired multi-deck games; "
                "then warm up the independent critic before PPO actor updates.",
            },
        )
        status.update(phase="completed", success=True, final_checkpoint=str(initial))
    except Exception:
        message = traceback.format_exc()
        key = os.environ.get("WANDB_API_KEY")
        status.update(phase="failed", error=message.replace(key, "[REDACTED]") if key else message)
    finally:
        sync()


if __name__ == "__main__":
    main()
