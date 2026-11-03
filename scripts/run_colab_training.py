"""Bounded Colab self-play with W&B and verified per-update Google Drive backups.

Secrets enter the detached launcher over stdin, never through command arguments/files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cr_rl.config import load_yaml
from cr_rl.training.persistence import (
    atomic_json,
    copy_verified,
    publish_update,
    restore_run,
    sync_logs,
)
from cr_rl.training.telemetry import (
    TrainingTelemetry,
    create_training_workspace,
    replay_rollout_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def critic_arguments(args):
    command = []
    for name in (
        "critic_mode",
        "critic_model",
        "critic_learning_rate",
        "critic_epochs",
        "critic_warmup_updates",
        "critic_warmup_games",
    ):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def reward_arguments(args):
    command = []
    for name in ("gamma", "gae_lambda", "overflow_penalty"):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.tower_shaping is not None:
        command.append("--tower-shaping" if args.tower_shaping else "--no-tower-shaping")
    return command


def run_v2_training(args):
    """One worker owns all updates and live matches; it publishes each checkpoint."""
    import torch

    local = ROOT / "artifacts" / args.run_name
    destination = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    if " /content/drive " not in Path("/proc/mounts").read_text():
        raise RuntimeError("Google Drive must be mounted for formal v2 PPO")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied through launcher stdin")
    if args.resume_drive and not (local / "ppo_latest.pt").exists():
        restore_run(args.resume_drive, local)
    local.mkdir(parents=True, exist_ok=True)
    checkpoint = local / "ppo_latest.pt"
    completed = 0
    if checkpoint.exists():
        header = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        completed = int(header["updates"])
        del header
    elif not args.init_from_bc:
        raise ValueError("Formal v2 PPO starts from BC; random policies are acceptance-only")
    if completed >= args.updates:
        return
    command = [
        sys.executable,
        "scripts/train_selfplay.py",
        "--v2",
        "--model",
        args.model,
        "--device",
        "cuda",
        "--envs",
        "2",
        "--steps",
        "1536",
        "--epochs",
        "2",
        "--batch-size",
        "4",
        "--grad-accumulation",
        "16",
        "--updates",
        str(args.updates - completed),
        "--out",
        str(local),
        "--backup-dir",
        str(destination),
        "--wandb",
        *critic_arguments(args),
        *reward_arguments(args),
    ]
    if checkpoint.exists():
        command.extend(["--resume", str(checkpoint)])
    else:
        command.extend(["--init-from-bc", str(args.init_from_bc)])
    if args.decks:
        command.extend(["--decks", str(args.decks)])
    atomic_json(
        local / "run_config.json",
        {"driver": "persistent-v2", "command": command, "planned_updates": args.updates},
    )
    status = {"phase": "starting", "completed_updates": completed, "planned_updates": args.updates}
    with (local / "train.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        last_sync = 0
        while child.poll() is None:
            summary = local / "summary.json"
            if summary.exists():
                status["completed_updates"] = json.loads(summary.read_text())["update"]
            status.update(phase="running", training_pid=child.pid, heartbeat_unix=time.time())
            atomic_json(local / "job_status.json", status)
            if time.monotonic() - last_sync > 60:
                sync_logs(local, destination)
                last_sync = time.monotonic()
            time.sleep(5)
    status["phase"] = "completed" if child.returncode == 0 else "failed"
    atomic_json(local / "job_status.json", status)
    sync_logs(local, destination)
    if child.returncode:
        raise RuntimeError(f"Persistent v2 worker exited with code {child.returncode}")


def run_training(args):
    if args.init_from_bc and not args.v2:
        import torch

        header = torch.load(args.init_from_bc, map_location="cpu", weights_only=False, mmap=True)
        args.v2 = header.get("config", {}).get("representation_version", 1) == 2
        del header
    if args.v2:
        return run_v2_training(args)
    for name, default in (
        ("gamma", 0.999),
        ("gae_lambda", 0.95),
        ("tower_shaping", False),
        ("overflow_penalty", 0.0),
    ):
        if getattr(args, name) is None:
            setattr(args, name, default)
    import wandb
    import yaml

    if args.init_from_bc:
        from cr_rl.training.selection import select_bc_checkpoint

        dataset = ROOT / "artifacts/multideck/human_bc"
        if dataset.exists() and not (args.init_from_bc.parent / "bc_selection_full.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_bc_checkpoints.py",
                    "--dataset",
                    str(dataset),
                    "--run-dir",
                    str(args.init_from_bc.parent),
                    "--backup-dir",
                    str(
                        Path("/content/drive/MyDrive/cr_rl/runs")
                        / args.init_from_bc.parent.name
                        / "bc"
                    ),
                ],
                cwd=ROOT,
                check=True,
            )

        args.init_from_bc, _ = select_bc_checkpoint(args.init_from_bc)

    local = ROOT / "artifacts" / args.run_name
    destination = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    local.mkdir(parents=True, exist_ok=True)
    child = None
    tracking = None
    status = {
        "phase": "starting",
        "completed_updates": 0,
        "backed_up_update": 0,
        "planned_updates": args.updates,
        "model": args.model,
        "drive_path": str(destination),
    }
    config = {
        "model": args.model,
        "envs": 2,
        "steps": 1536,
        "epochs": 2,
        "batch_size": 4,
        "grad_accumulation": 16,
        "learning_rate": 2e-5,
        "critic_mode": args.critic_mode,
        "critic_model": args.critic_model,
        "critic_learning_rate": args.critic_learning_rate,
        "critic_epochs": args.critic_epochs,
        "tower_shaping": args.tower_shaping,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "overflow_penalty": args.overflow_penalty,
        "seed": 20261103,
        "planned_updates": args.updates,
        "decks": yaml.safe_load(args.decks.read_text()) if args.decks else load_yaml("decks.yaml"),
        "init_from_bc": str(args.init_from_bc) if args.init_from_bc else None,
        "backend": load_yaml("backends.yaml")["primary"],
        "model_config": load_yaml("model.yaml")["presets"][args.model],
    }
    try:
        if " /content/drive " not in Path("/proc/mounts").read_text():
            raise RuntimeError("Google Drive is not mounted")
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is required; pass through launcher stdin")
        if not json.loads((ROOT / "setup_status.json").read_text()).get("ready"):
            raise RuntimeError("Simulator setup did not pass")
        if args.resume_drive:
            # The launcher log lives outside this new restore target until startup.
            if (local / "ppo_latest.pt").exists():
                raise ValueError("Use a new run name for restoration")
            restored = ROOT / ".cache" / "restore_start" / args.run_name
            manifest = restore_run(args.resume_drive, restored)
            for source in restored.iterdir():
                source.rename(local / source.name)
            status["completed_updates"] = manifest["update"]
            status["backed_up_update"] = manifest["update"]
        elif args.resume_local:
            latest = destination / "latest.json"
            status["backed_up_update"] = (
                json.loads(latest.read_text())["update"] if latest.exists() else 0
            )
            status["completed_updates"] = status["backed_up_update"]
        elif (destination / "latest.json").exists() or (local / "ppo_latest.pt").exists():
            raise ValueError("Existing training run requires explicit --resume-drive")
        if args.decks:
            config["decks"] = yaml.safe_load(args.decks.read_text())
        elif (local / "run_config.json").exists():
            config["decks"] = json.loads((local / "run_config.json").read_text())["decks"]
        args.decks = local / "league.yaml"
        args.decks.write_text(yaml.safe_dump(config["decks"], sort_keys=False))
        destination.mkdir(parents=True, exist_ok=True)
        copy_verified(args.decks, destination / "league.yaml")
        atomic_json(local / "run_config.json", config)
        for name in ("run_config.json",):
            copy_verified(local / name, destination / name)
        source = Path("/content/crrl-live-src.tar.gz")
        from cr_rl.training.persistence import file_sha256

        copy_verified(source, destination / "sources" / f"{file_sha256(source)[:16]}.tar.gz")
        if not (destination / "source.tar.gz").exists():
            copy_verified(source, destination / "source.tar.gz")
        copy_verified(ROOT / "setup.log", destination / "setup.log")
        os.environ["WANDB_DISABLE_GIT"] = "true"
        kwargs = {}
        if (local / "wandb_run.json").exists():
            old = json.loads((local / "wandb_run.json").read_text())
            kwargs.update(id=old["id"], entity=old["entity"], resume="allow")
        tracking = wandb.init(
            project="cr-rl",
            name=args.run_name,
            config=config,
            dir=str(local),
            mode="online",
            **kwargs,
        )
        atomic_json(
            local / "wandb_run.json",
            {
                "id": tracking.id,
                "url": tracking.url,
                "entity": tracking.entity,
                "project": tracking.project,
            },
        )
        status["wandb_url"] = tracking.url
        telemetry = TrainingTelemetry(tracking, config["envs"] * config["steps"])
        workspace_path = local / "wandb_workspace.json"
        if workspace_path.exists():
            status["workspace_url"] = json.loads(workspace_path.read_text())["url"]
        else:
            try:
                url = create_training_workspace(tracking.entity, tracking.project, args.run_name)
                atomic_json(workspace_path, {"url": url})
                status["workspace_url"] = url
            except Exception as error:
                print(
                    "Workspace layout configuration unavailable:", type(error).__name__, flush=True
                )
        replayed = set()

        def replay_legacy():
            for path in sorted((local / "rollouts").glob("*.npz")):
                if path.name in replayed:
                    continue
                metadata = json.loads(path.with_suffix(".npz.json").read_text())
                if metadata.get("telemetry_version", 0) < 2:
                    replay_rollout_metrics(path, telemetry)
                replayed.add(path.name)

        replay_legacy()
        atomic_json(local / "job_status.json", status)
        sync_logs(local, destination)
        seen = 0

        def drain_progress():
            nonlocal seen
            path = local / "progress.jsonl"
            if not path.exists():
                return
            lines = path.read_text().splitlines(keepends=True)
            for line in lines[seen:]:
                if not line.endswith("\n"):
                    break
                event = json.loads(line)
                if event["phase"] == "optimization" and "optimizer_steps_total" not in event:
                    history = local / "metrics.jsonl"
                    summaries = (
                        [json.loads(row) for row in history.read_text().splitlines()]
                        if history.exists()
                        else []
                    )
                    event["optimizer_steps_total"] = event["optimizer_steps"] + sum(
                        row["optimizer_steps"]
                        for row in summaries
                        if row["update"] < event["update"]
                    )
                status.update(phase=event["phase"], active_update=event["update"], last_event=event)
                telemetry.log(event)
                seen += 1

        for update in range(int(status["completed_updates"]) + 1, args.updates + 1):
            command = [
                sys.executable,
                "scripts/train_selfplay.py",
                "--model",
                args.model,
                "--device",
                "cuda",
                "--envs",
                "2",
                "--steps",
                "1536",
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--grad-accumulation",
                "16",
                "--gamma",
                str(args.gamma),
                "--gae-lambda",
                str(args.gae_lambda),
                "--overflow-penalty",
                str(args.overflow_penalty),
                "--updates",
                "1",
                "--seed",
                str(20261103 + update - 1),
                "--out",
                str(local),
                "--progress-jsonl",
                str(local / "progress.jsonl"),
            ]
            if (local / "ppo_latest.pt").exists():
                command.extend(["--resume", str(local / "ppo_latest.pt")])
            elif args.init_from_bc:
                command.extend(["--init-from-bc", str(args.init_from_bc)])
            if args.decks:
                command.extend(["--decks", str(args.decks)])
            command.extend(critic_arguments(args))
            if args.tower_shaping:
                command.append("--tower-shaping")
            status.update(phase="launching_update", active_update=update)
            atomic_json(local / "job_status.json", status)
            prior_summary = (
                json.loads((local / "summary.json").read_text())
                if (local / "summary.json").exists()
                else {}
            )
            adopted = args.adopt_pid if update == int(status["backed_up_update"]) + 1 else None

            def adopted_running(adopted=adopted):
                if not adopted:
                    return False
                path = Path(f"/proc/{adopted}/stat")
                return path.exists() and path.read_text().split()[2] != "Z"

            already_complete = (
                not adopted_running()
                and prior_summary.get("update") == update
                and (local / "ppo_latest.pt").exists()
            )
            with (local / "train.log").open("a") as log:
                child = (
                    None
                    if (adopted_running() or already_complete)
                    else subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                )
                last_sync = 0.0
                while child.poll() is None if child is not None else adopted_running():
                    drain_progress()
                    status["heartbeat_unix"] = time.time()
                    atomic_json(local / "job_status.json", status)
                    if time.monotonic() - last_sync >= 60:
                        sync_logs(local, destination)
                        last_sync = time.monotonic()
                    time.sleep(5)
            drain_progress()
            if child is not None and child.returncode:
                raise RuntimeError(f"Trainer exited {child.returncode}; see train.log")
            summary = json.loads((local / "summary.json").read_text())
            if summary["update"] != update:
                raise RuntimeError("Checkpoint/summary update mismatch")
            replay_legacy()
            telemetry.log_summary(summary)
            status.update(phase="backing_up", completed_updates=update)
            atomic_json(local / "job_status.json", status)
            started = time.monotonic()
            manifest = publish_update(local, destination, update)
            if not status.get("drive_resume_probe_verified"):
                probe = (
                    ROOT
                    / ".cache"
                    / "drive_restore_probe"
                    / args.run_name
                    / f"update_{update:06d}_{time.time_ns()}"
                )
                restored = restore_run(destination, probe)
                import torch

                checkpoint = torch.load(
                    probe / "ppo_latest.pt", map_location="cpu", weights_only=False
                )
                assert checkpoint["updates"] == restored["update"] == update
                if "critic_optimizer" in checkpoint:
                    # During value warmup the policy optimizer intentionally has no state.
                    assert checkpoint["critic_optimizer"]["state"]
                else:
                    assert checkpoint["optimizer"]["state"]
                del checkpoint
                status["drive_resume_probe_verified"] = True
            status.update(backed_up_update=manifest["update"], phase="update_complete")
            atomic_json(local / "job_status.json", status)
            sync_logs(local, destination)
            tracking.summary["backup"] = {
                "update": update,
                "seconds": time.monotonic() - started,
                "verified": True,
            }
            print(f"UPDATE {update} CHECKPOINT AND RAW DATA VERIFIED ON DRIVE", flush=True)
        status["phase"] = "completed"
        tracking.summary["completed_updates"] = status["completed_updates"]
        tracking.summary["drive_backed_up_update"] = status["backed_up_update"]
    except Exception:
        status["phase"] = "failed"
        status["error"] = traceback.format_exc().replace(
            os.environ.get("WANDB_API_KEY", "<absent>"), "[REDACTED]"
        )
        print(status["error"], flush=True)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    finally:
        if tracking is not None:
            tracking.finish(exit_code=0 if status["phase"] == "completed" else 1)
        atomic_json(local / "job_status.json", status)
        if destination.exists():
            sync_logs(local, destination)
        print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--model", default="production")
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument(
        "--run-name",
        default="119m_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--resume-drive", type=Path)
    parser.add_argument("--resume-local", action="store_true")
    parser.add_argument("--adopt-pid", type=int)
    parser.add_argument("--init-from-bc", type=Path)
    parser.add_argument("--decks", type=Path)
    parser.add_argument("--critic-mode", choices=("auto", "independent", "shared"), default="auto")
    parser.add_argument("--critic-model", default="same")
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-epochs", type=int, default=4)
    parser.add_argument("--tower-shaping", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--overflow-penalty", type=float)
    parser.add_argument("--critic-warmup-updates", type=int)
    parser.add_argument("--critic-warmup-games", type=int)
    args = parser.parse_args()
    if args.start:
        secrets = json.load(sys.stdin)
        if set(secrets) != {"WANDB_API_KEY"}:
            raise ValueError("Only WANDB_API_KEY may be passed through stdin")
        output = ROOT / "artifacts" / args.run_name
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            args.model,
            "--updates",
            str(args.updates),
            "--run-name",
            args.run_name,
        ]
        if args.resume_drive:
            command.extend(["--resume-drive", str(args.resume_drive)])
        if args.resume_local:
            command.append("--resume-local")
        if args.adopt_pid:
            command.extend(["--adopt-pid", str(args.adopt_pid)])
        if args.init_from_bc:
            command.extend(["--init-from-bc", str(args.init_from_bc)])
        if args.decks:
            command.extend(["--decks", str(args.decks)])
        command.extend(critic_arguments(args))
        command.extend(reward_arguments(args))
        if args.v2:
            command.append("--v2")
        with (output / "supervisor.log").open("a") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **secrets},
                start_new_session=True,
            )
        print(json.dumps({"pid": child.pid, "run_name": args.run_name, "local_path": str(output)}))
    else:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
        os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
        run_training(args)


if __name__ == "__main__":
    main()
