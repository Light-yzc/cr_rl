"""One-GPU, bounded BC comparison; never promotes weights or starts PPO.

Restore the exact Stage 2 dataset via its source receipts, keep the original
checkpoint immutable, and train the repaired and baseline arms sequentially.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def resume_command(command, checkpoint, current_step, target_step):
    """Keep the registered protocol and advance only the remaining updates."""
    if not 0 <= current_step < target_step:
        raise ValueError("Checkpoint is outside the unfinished trial interval")
    result = list(command)
    i = result.index("--init-from-bc")
    result[i : i + 2] = ["--resume", str(checkpoint)]
    for flag in ("--carry-optimizer", "--upgrade-readouts"):
        if flag in result:
            result.remove(flag)
    result[result.index("--max-updates") + 1] = str(target_step - current_step)
    return result


def trim_uncheckpointed_log(path, step):
    """Drop observations of lost updates from the resumed, canonical history."""
    kept = []
    for line in path.read_text().splitlines(keepends=True):
        try:
            row = json.loads(line)
        except ValueError:
            kept.append(line)
            continue
        if row.get("optimizer_steps", 0) <= step:
            kept.append(line)
    path.write_text("".join(kept))


def latest_train(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 100_000))
        lines = stream.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("phase") in ("bc_train", "bounded_trial_completed"):
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="121m_bc_repair_20261104")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--cache-gib", type=float, default=128)
    parser.add_argument("--source-run", default="121m_cycle_bc_20261103")
    parser.add_argument("--retry-preparation", action="store_true")
    parser.add_argument("--resume-trial", action="store_true")
    parser.add_argument("--evaluate-results", action="store_true")
    parser.add_argument("--evaluation-pairs", type=int, default=8)
    args = parser.parse_args()
    if min(args.updates, args.window, args.cache_gib) <= 0:
        parser.error("Trial limits must be positive")
    if args.evaluation_pairs < 1:
        parser.error("At least one paired evaluation is required")
    drive_root = Path("/content/drive/MyDrive/cr_rl/runs")
    source = drive_root / args.source_run
    out, backup = ROOT / "artifacts" / args.run_name, drive_root / args.run_name
    existing = out / "status.json"
    if existing.exists() or (backup / "status.json").exists():
        previous = json.loads(
            (existing if existing.exists() else backup / "status.json").read_text()
        )
        proc = Path(f"/proc/{previous['pid']}/stat")
        alive = proc.exists() and proc.read_text().split()[2] != "Z"
        if args.resume_trial:
            command_line = Path(f"/proc/{previous['pid']}/cmdline")
            same_controller = alive and command_line.exists() and (
                b"run_bc_repair_trial.py" in command_line.read_bytes()
            )
            if same_controller or previous["phase"] == "completed":
                raise ValueError("Cannot resume an active or completed trial")
        elif (
            not args.retry_preparation
            or previous["phase"] != "failed"
            or alive
            or any((out / arm).exists() for arm in ("repaired", "baseline"))
        ):
            raise ValueError(
                "Trial exists; only a failed, inactive pre-training setup can be retried"
            )
        saved = out / f"preparation_failure_{time.time_ns()}.json"
        saved.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(saved, previous)
        copy_verified(saved, backup / saved.name)
    elif args.resume_trial:
        raise ValueError("Resume requires an existing registered trial")
    out.mkdir(parents=True, exist_ok=True)
    state = {"pid": os.getpid(), "ppo_started": False}
    child = None

    def status(phase, **details):
        state.update(phase=phase, observed_unix=time.time(), **details)
        atomic_json(out / "status.json", state)
        copy_verified(out / "status.json", backup / "status.json")
        print(json.dumps(state), flush=True)

    def sync_arm(arm):
        for path in arm.glob("*"):
            if path.suffix in (".json", ".jsonl", ".log"):
                # Snapshot an append-only log before hashing/uploading it.
                snap = out / ".snapshots" / arm.name / path.name
                snap.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, snap)
                copy_verified(snap, backup / arm.name / path.name)

    def wait_evaluation(folder, phase, **details):
        began = time.monotonic()
        while child.poll() is None:
            sync_arm(folder)
            status(phase, child_pid=child.pid, **details)
            if time.monotonic() - began > 3600:
                raise TimeoutError("Evaluation exceeded its one-hour bound")
            time.sleep(20)
        sync_arm(folder)
        if child.returncode:
            raise RuntimeError(f"Evaluation failed: {details}")

    try:
        status("source_and_tests")
        if args.resume_trial and (backup / "source.tar.gz").exists():
            copy_verified(
                backup / "source.tar.gz", backup / f"source_before_resume_{time.time_ns()}.tar.gz"
            )
        with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
            for name in ("src", "scripts", "tests", "configs", "pyproject.toml"):
                archive.add(
                    ROOT / name,
                    arcname=name,
                    filter=lambda m: None if "__pycache__" in m.name else m,
                )
        copy_verified(out / "source.tar.gz", backup / "source.tar.gz")
        with (out / "tests.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        copy_verified(out / "tests.log", backup / "tests.log")
        data = out / "data"
        data.mkdir(exist_ok=True)
        for name in ("dataset.manifest.json", "dataset_metadata.json", "source_locations.json"):
            copy_verified(source / "stage_0002/data" / name, data / name)
        copy_verified(source / "split_registry.json", out / "split_registry.json")
        manifest = json.loads((data / "dataset.manifest.json").read_text())
        locations = json.loads((data / "source_locations.json").read_text())["files"]
        status("restoring_shards", completed_shards=0, total_shards=len(locations))

        def restore(item):
            record = copy_verified(Path(item["source"]), data / item["path"])
            if record["sha256"] != item["sha256"]:
                raise ValueError("Frozen source shard differs from its receipt")

        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(restore, item) for item in locations]
            for done, future in enumerate(as_completed(jobs), 1):
                future.result()
                if done % 32 == 0 or done == len(jobs):
                    status("restoring_shards", completed_shards=done)
        record = json.loads((source / "stage_0002/bc/latest.json").read_text())
        if record["optimizer_steps"] != 4096:
            raise ValueError("This registered comparison requires the verified step4096 parent")
        checkpoint = out / "parent4096.pt"
        copied = copy_verified(source / "stage_0002/bc" / record["path"], checkpoint)
        if copied["sha256"] != record["sha256"]:
            raise ValueError("Parent checkpoint SHA mismatch")
        # Reserve headroom for history spillover and masks outside the working set.
        sizes = []
        for shard in manifest["shards"]:
            with zipfile.ZipFile(data / shard["path"]) as archive:
                sizes.append(
                    sum(
                        archive.getinfo(f"{field}.npy").file_size
                        for field in ("observations", "masks")
                    )
                )
        largest = sorted(sizes, reverse=True)
        window = min(args.window, len(largest))
        while window > 1 and sum(largest[:window]) > args.cache_gib * 1024**3 * 0.6:
            window -= 1
        protocol = {
            "parent": record,
            "dataset_sha256": file_sha256(data / "dataset.manifest.json"),
            "new_updates_per_arm": args.updates,
            "seed": 20261102,
            "batch_size": 16,
            "gradient_accumulation": 6,
            "context": 128,
            "learning_rate": 3e-5,
            "tile_learning_rate": 3e-4,
            "shard_window": window,
            "cache_gib": args.cache_gib,
            "arms": {
                "repaired": {"readout_version": 2, "window": window, "timing": "natural"},
                "baseline": {"readout_version": 1, "window": 1, "timing": "balanced"},
            },
            "interpretation": (
                "Combined-treatment comparison; cannot attribute effects to one change"
            ),
            "ppo_started": False,
            "auto_promote": False,
            "time_budget_note": "No 15-20 minute completion or strength guarantee",
        }
        if args.resume_trial:
            registered = json.loads((backup / "protocol.json").read_text())
            if registered != protocol:
                raise ValueError("Resumed trial differs from its registered protocol")
        atomic_json(out / "protocol.json", protocol)
        copy_verified(out / "protocol.json", backup / "protocol.json")
        if args.evaluate_results:
            atomic_json(out / "evaluation_request.json", {
                "full_validation": True,
                "paired_games_per_candidate": 2 * args.evaluation_pairs,
            })
            copy_verified(out / "evaluation_request.json", backup / "evaluation_request.json")
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
        for name, arm_settings in protocol["arms"].items():
            arm = out / name
            arm.mkdir(exist_ok=args.resume_trial)
            # Sequential arms use identical immutable validation histories. Reuse
            # their signature- and checksum-verified compact cache, not two 11GiB copies.
            shared_validation = out / "validation_cache"
            shared_validation.mkdir(exist_ok=True)
            if not (arm / ".validation_cache").exists():
                (arm / ".validation_cache").symlink_to(shared_validation, target_is_directory=True)
            command = [
                sys.executable,
                "scripts/train_bc.py",
                str(data),
                "--model",
                "production",
                "--device",
                "cuda",
                "--epochs",
                "1",
                "--all-plays",
                "--stable-split",
                "--split-registry",
                str(out / "split_registry.json"),
                "--holdout-decks",
                "configs/decks_eval.yaml",
                "--init-from-bc",
                str(checkpoint),
                "--carry-optimizer",
                "--bc-objective",
                "conditional",
                "--no-card-prior-correction",
                "--ability-fraction",
                "0.1",
                "--tile-label-sigma",
                "1.0",
                "--wait-margin-before",
                "4",
                "--wait-margin-after",
                "2",
                "--position-coef",
                "1.0",
                "--learning-rate",
                "3e-5",
                "--tile-learning-rate",
                "3e-4",
                "--batch-size",
                "16",
                "--grad-accumulation",
                "6",
                "--entity-bucket",
                "16",
                "--fused-optimizer",
                "--stream-cache-gib",
                str(args.cache_gib),
                "--cache-dir",
                str(out / "arrays"),
                "--persistent-validation-cache",
                "/content/drive/MyDrive/cr_rl/caches/validation-v1",
                "--natural-validation-samples",
                "4096",
                "--validation-every-steps",
                "32",
                "--checkpoint-every-steps",
                "128",
                "--max-updates",
                str(args.updates),
                "--shard-mix-window",
                str(arm_settings["window"]),
                "--timing-objective",
                arm_settings["timing"],
                "--out",
                str(arm / "bc.pt"),
                "--backup-dir",
                str(backup / name / "bc"),
            ]
            if name == "repaired":
                command.append("--upgrade-readouts")
            registered_command = command.copy()
            atomic_json(arm / "registered_command.json", {"command": registered_command})
            if args.resume_trial:
                arm_backup = backup / name
                for path in arm_backup.glob("*"):
                    if path.suffix in (".json", ".jsonl", ".log"):
                        copy_verified(path, arm / path.name)
                latest_path = arm_backup / "bc/latest.json"
                if latest_path.exists():
                    latest = json.loads(latest_path.read_text())
                    if Path(latest["path"]).name != latest["path"]:
                        raise ValueError("Invalid checkpoint backup filename")
                    restored = copy_verified(
                        arm_backup / "bc" / latest["path"], arm / "bc.pt"
                    )
                    if (restored["sha256"], restored["bytes"]) != (
                        latest["sha256"], latest["bytes"]
                    ):
                        raise ValueError("Resumed checkpoint checksum mismatch")
                    import torch

                    saved_state = torch.load(
                        arm / "bc.pt", map_location="cpu", weights_only=False, mmap=True
                    )
                    if (
                        saved_state["optimizer_steps"] != latest["optimizer_steps"]
                        or saved_state["dataset_sha256"] != protocol["dataset_sha256"]
                        or saved_state["config"]["readout_version"]
                        != arm_settings["readout_version"]
                    ):
                        raise ValueError("Resumed checkpoint metadata mismatch")
                    del saved_state
                    target = 4096 + args.updates
                    if latest["optimizer_steps"] == target:
                        continue
                    archive = arm_backup / f"interrupted_{time.time_ns()}"
                    for filename in ("train.log", "bc.quick.jsonl", "bc.json", "command.json"):
                        path = arm / filename
                        if path.exists():
                            copy_verified(path, archive / filename)
                            if filename in ("train.log", "bc.quick.jsonl"):
                                trim_uncheckpointed_log(path, latest["optimizer_steps"])
                    command = resume_command(
                        command, arm / "bc.pt", latest["optimizer_steps"], target
                    )
                    atomic_json(arm / "resume_receipt.json", latest)
            atomic_json(arm / "command.json", {"command": command})
            with (arm / "train.log").open("a") as log:
                child = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                status("training", arm=name, child_pid=child.pid, latest_update=None)
                began = time.monotonic()
                while child.poll() is None:
                    sync_arm(arm)
                    status("training", arm=name, latest_update=latest_train(arm / "train.log"))
                    if time.monotonic() - began > 6 * 3600:
                        child.terminate()
                        child.wait(timeout=60)
                        raise TimeoutError("Bounded arm exceeded six hours; inspect saved progress")
                    time.sleep(20)
            sync_arm(arm)
            if child.returncode:
                raise RuntimeError(f"{name} failed; see its saved train.log")
            latest = json.loads((arm / "bc.json").read_text())
            if latest["optimizer_steps"] != 4096 + args.updates:
                raise RuntimeError("Arm did not execute the registered number of updates")
            status("arm_completed", arm=name, latest_update=latest_train(arm / "train.log"))
        if args.evaluate_results:
            status("evaluating_results", child_pid=None)
            evaluation = out / "evaluation"
            evaluation.mkdir(exist_ok=True)
            if not (evaluation / ".validation_cache").exists():
                (evaluation / ".validation_cache").symlink_to(
                    out / "validation_cache", target_is_directory=True
                )
            for label, weights, command_arm in (
                ("parent4096", checkpoint, "baseline"),
                ("repaired5096", out / "repaired/bc.pt", "repaired"),
                ("baseline5096", out / "baseline/bc.pt", "baseline"),
            ):
                command = json.loads(
                    (out / command_arm / "registered_command.json").read_text()
                )["command"]
                for flag in ("--carry-optimizer", "--upgrade-readouts"):
                    if flag in command:
                        command.remove(flag)
                command[command.index("--init-from-bc") + 1] = str(weights)
                command[command.index("--out") + 1] = str(evaluation / f"{label}.pt")
                command[command.index("--backup-dir") + 1] = str(backup / "evaluation")
                command[command.index("--validation-every-steps") + 1] = "0"
                command.append("--evaluate-only")
                with (evaluation / f"{label}.log").open("w") as log:
                    child = subprocess.Popen(
                        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    status("evaluating_results", candidate=label, child_pid=child.pid)
                    wait_evaluation(evaluation, "evaluating_results", candidate=label)
            battle_results = {"repaired": [], "baseline": []}
            for label in ("repaired", "baseline"):
                for start in range(0, args.evaluation_pairs, 4):
                    pairs = min(4, args.evaluation_pairs - start)
                    battle = evaluation / f"{label}_pairs_{start:03d}"
                    command = [
                        sys.executable, "scripts/evaluate_policy.py",
                        "--policy", str(out / label / "bc.pt"),
                        "--opponent", str(checkpoint),
                        "--decks", "configs/decks_eval.yaml",
                        "--out", str(battle), "--pairs", str(pairs),
                        "--pair-start", str(start), "--frame-cache",
                    ]
                    with (evaluation / f"{label}_battle_{start:03d}.log").open("w") as log:
                        child = subprocess.Popen(
                            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                        )
                        status("evaluating_battles", candidate=label, child_pid=child.pid)
                        wait_evaluation(evaluation, "evaluating_battles", candidate=label)
                    for path in battle.glob("*.json"):
                        copy_verified(path, backup / "evaluation" / battle.name / path.name)
                    battle_results[label].append(
                        json.loads((battle / "evaluation.json").read_text())
                    )
            summary = {
                "validation": {
                    label: json.loads((evaluation / f"{label}.evaluation.json").read_text())
                    for label in ("parent4096", "repaired5096", "baseline5096")
                },
                "battles_vs_parent4096": {
                    label: {
                        key: sum(result[key] for result in results)
                        for key in ("games", "invalid_games", "wins", "losses", "draws")
                    }
                    for label, results in battle_results.items()
                },
                "interpretation": (
                    "Matched fixed validation sets and paired simulator games; "
                    "small battle cohort, no official-client strength claim or automatic promotion"
                ),
            }
            atomic_json(evaluation / "summary.json", summary)
            atomic_json(evaluation / "completed.json", {
                "full_validation_candidates": ["parent4096", "repaired5096", "baseline5096"],
                "paired_games_per_candidate": 2 * args.evaluation_pairs,
                "wait_logit_bias": 0.0,
                "limitation": "Small simulator cohort; not proof of official-client strength",
            })
            sync_arm(evaluation)
        comparison = {}
        for name in protocol["arms"]:
            quick = [
                json.loads(line)
                for line in (out / name / "bc.quick.jsonl").read_text().splitlines()
            ]
            comparison[name] = {"first": quick[0], "last": quick[-1], "points": len(quick)}
        atomic_json(out / "comparison.json", comparison)
        copy_verified(out / "comparison.json", backup / "comparison.json")
        status("completed", auto_promote=False)
    except Exception:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=60)
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
