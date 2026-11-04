"""Resume a verified generation for a fixed budget after a real CUDA parity audit.

Run in an already mounted Colab kernel. This does not allocate a runtime or reuse
the old pilot's hardcoded controller. Deck curricula require an explicit override.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cr_rl.training.persistence import (
    atomic_json,
    copy_verified,
    file_sha256,
    restore_run,
    sync_logs,
)


def continuation_command(
    previous, run, backup, additional_updates, *, actor_effective_batch_size=None
):
    if additional_updates < 1:
        raise ValueError("Positive fixed continuation budget required")
    command = list(previous)
    # A restored pool already contains verified seeds. This flag only belongs to
    # the first launch of a fresh history stage, not its recovery command.
    if "--history-seed-manifest" in command:
        index = command.index("--history-seed-manifest")
        if index + 1 >= len(command):
            raise ValueError("Missing historical seed manifest argument")
        del command[index : index + 2]
    if "--expand-decks" in command:
        index = command.index("--expand-decks")
        if index + 1 >= len(command):
            raise ValueError("Missing expanded deck pool argument")
        expanded_path = command[index + 1]
        del command[index : index + 2]
        if "--decks" in command:
            command[command.index("--decks") + 1] = expanded_path
        else:
            command.extend(["--decks", expanded_path])
    for option, value in {
        "--resume": str(run / "ppo_latest.pt"),
        "--updates": str(additional_updates),
        "--out": str(run),
        "--backup-dir": str(backup),
        "--progress-jsonl": str(run / "progress.jsonl"),
    }.items():
        if option not in command:
            raise ValueError(f"Missing required original resume option: {option}")
        command[command.index(option) + 1] = value
    command[0] = sys.executable
    if actor_effective_batch_size is not None:
        if actor_effective_batch_size < 64 or "--v2" not in command:
            raise ValueError("Actor effective batch transitions require v2 and at least 64")
        microbatch_option = (
            "--actor-batch-size" if "--actor-batch-size" in command else "--batch-size"
        )
        microbatch = int(command[command.index(microbatch_option) + 1])
        if microbatch < 1 or actor_effective_batch_size % microbatch:
            raise ValueError("Actor microbatch must divide its effective batch")
        option = "--actor-effective-batch-size"
        if option in command:
            command[command.index(option) + 1] = str(actor_effective_batch_size)
        else:
            command.extend([option, str(actor_effective_batch_size)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--additional-updates", type=int, default=8)
    parser.add_argument("--source-generation", type=int, default=6)
    parser.add_argument("--history-seed-manifest", type=Path)
    parser.add_argument("--frame-encoder-fp32", action="store_true")
    parser.add_argument("--resume-actor-learning-rate", type=float)
    parser.add_argument("--resume-critic-learning-rate", type=float)
    parser.add_argument("--actor-effective-batch-size", type=int)
    parser.add_argument("--expand-decks", type=Path)
    parser.add_argument("--new-wandb-run", action="store_true")
    parser.add_argument("--budget-seconds", type=int, default=3 * 3600)
    args = parser.parse_args()
    if any(
        v is not None and (not math.isfinite(v) or v <= 0)
        for v in (args.resume_actor_learning_rate, args.resume_critic_learning_rate)
    ):
        parser.error("Resume learning rates must be finite and positive")
    if not os.environ.get("TBE_RUNTIME_ADDR") or not args.original.is_dir():
        raise RuntimeError("Use an existing mounted Colab kernel")
    if args.out.exists() or args.backup.exists():
        raise ValueError(
            "Use fresh continuation directories; inspect existing work before recovery"
        )
    manifest = json.loads((args.original / "latest.json").read_text())
    if args.source_generation < 1 or manifest["update"] != args.source_generation:
        raise ValueError("Source differs from the explicitly requested frozen generation")
    root = Path(__file__).resolve().parents[1]
    args.backup.mkdir(parents=True)
    args.out.mkdir(parents=True)
    state_dir = args.out / "state"
    args.out = state_dir
    previous = json.loads((args.original / "run_config.json").read_text())
    protocol = dict(
        source_generation=args.source_generation,
        additional_updates=args.additional_updates,
        target_generation=args.source_generation + args.additional_updates,
        budget_seconds=args.budget_seconds,
        source_manifest_sha256=file_sha256(args.original / "latest.json"),
        simulator_restart=(
            "Unfinished source games reset; models, optimizers, RNG and warmup retained"
        ),
        reference="Existing per-update frozen behavior reference, not a permanent BC anchor",
        evaluation="Frozen generation exports; comparison cohort supplied by parent protocol",
        numeric_thresholds=dict(
            fp32_logp=0.0005, fp32_tv=0.0001, bf16_logp=0.03, bf16_tv=0.01, stored_bf16_logp=0.05
        ),
        role_mixture=dict(current=0.3, historical=0.5, anchor=0.2)
        if args.history_seed_manifest
        else None,
        frame_encoder_fp32=args.frame_encoder_fp32,
        resume_actor_learning_rate=args.resume_actor_learning_rate,
        resume_critic_learning_rate=args.resume_critic_learning_rate,
        actor_effective_batch_size=args.actor_effective_batch_size,
        expanded_decks_sha256=file_sha256(args.expand_decks) if args.expand_decks else None,
        history_seed_manifest_sha256=file_sha256(args.history_seed_manifest)
        if args.history_seed_manifest
        else None,
    )
    atomic_json(args.backup / "continuation_protocol.json", protocol)
    began, child = time.monotonic(), None

    def status(phase, **details):
        record = dict(
            phase=phase,
            pid=os.getpid(),
            child_pid=child.pid if child else None,
            observed_unix=time.time(),
            **details,
        )
        atomic_json(args.backup / "continuation_status.json", record)
        print(json.dumps(record), flush=True)

    def wait(command, log_path, phase):
        nonlocal child
        with log_path.open("a") as log:
            child = subprocess.Popen(
                command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        while child.poll() is None:
            if time.monotonic() - began > args.budget_seconds:
                raise TimeoutError("Fixed continuation wall-clock budget reached")
            latest = args.backup / "latest.json"
            status(
                phase,
                published_update=json.loads(latest.read_text())["update"]
                if latest.exists()
                else args.source_generation,
            )
            if args.out.exists():
                sync_logs(args.out, args.backup)
            time.sleep(30)
        if child.returncode:
            raise RuntimeError(f"{phase} exited {child.returncode}; inspect {log_path.name}")

    try:
        status("restoring")
        restore_run(args.original, args.out)
        if (args.history_seed_manifest or args.new_wandb_run) and (
            args.out / "wandb_run.json"
        ).exists():
            (args.out / "wandb_run.json").rename(args.out / "source_wandb_run.json")
        five = json.loads(
            (
                args.original
                / (f"checkpoints/update_{args.source_generation - 1:06d}.manifest.json")
            ).read_text()
        )
        actor = next(r for r in five["files"] if r["path"].endswith(".actor.pt"))
        rollout = next(
            r
            for r in manifest["files"]
            if r["path"] == (f"rollouts/rollout_{args.source_generation:06d}.npz")
        )
        audit = args.backup / "cuda_policy_parity.json"
        wait(
            [
                sys.executable,
                "scripts/validate_cuda_policy_parity.py",
                "--actor",
                str(args.original / actor["path"]),
                "--actor-sha256",
                actor["sha256"],
                "--rollout",
                str(args.out / rollout["path"]),
                "--rollout-sha256",
                rollout["sha256"],
                "--out",
                str(audit),
                *(["--frame-encoder-fp32"] if args.frame_encoder_fp32 else []),
            ],
            args.backup / "cuda_policy_parity.log",
            "numeric_audit",
        )
        if not json.loads(audit.read_text())["passed"]:
            raise ValueError("Numeric audit did not pass")
        command = continuation_command(
            previous["command"],
            args.out,
            args.backup,
            args.additional_updates,
            actor_effective_batch_size=args.actor_effective_batch_size,
        )
        if args.history_seed_manifest:
            command.extend(
                [
                    "--role-league",
                    "--history-seed-manifest",
                    str(args.history_seed_manifest),
                    "--history-snapshot-interval",
                    "20000",
                ]
            )
        if args.frame_encoder_fp32:
            command.append("--frame-encoder-fp32")
        if args.expand_decks:
            command.extend(["--expand-decks", str(args.expand_decks)])
        for option, value in (
            ("--resume-actor-learning-rate", args.resume_actor_learning_rate),
            ("--resume-critic-learning-rate", args.resume_critic_learning_rate),
        ):
            if value is not None:
                if option in command:
                    command[command.index(option) + 1] = str(value)
                else:
                    command.extend([option, str(value)])
        config = dict(
            previous,
            command=command,
            continuation=protocol,
            created_unix=time.time(),
            budget_seconds=args.budget_seconds,
        )
        atomic_json(args.out / "run_config.json", config)
        copy_verified(args.out / "run_config.json", args.backup / "run_config.json")
        # Capture exactly the source used; data/weights stay exclusively in Colab/Drive.
        for relative in (
            "scripts/continue_ppo_training.py",
            "scripts/validate_cuda_policy_parity.py",
        ):
            copy_verified(root / relative, args.backup / "source" / relative)
        wait(command, args.out / "train.log", "training")
        published = json.loads((args.backup / "latest.json").read_text())
        if published["update"] != protocol["target_generation"]:
            raise ValueError("Requested final generation is not published")
        row = next(r for r in published["files"] if r["path"] == published["checkpoint"])
        if file_sha256(args.backup / row["path"]) != row["sha256"]:
            raise ValueError("Final full-state checkpoint verification failed")
        status(
            "training_completed",
            published_update=published["update"],
            note="Training complete; frozen strength evaluation is a separate required step",
        )
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        status("failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
