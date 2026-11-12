"""One owned Colab coordinator: four-update handoff pilot, then total 100 BC epochs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

# These are the helpers from this exact source bundle, not an installed project.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook_state_bc import configure_wandb, log_tail, monitor_process, resume_completed
from notebook_state_bc_setup import build_trainer_argv, copy_verified, inspect_checkpoint, preflight


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def profile_settings(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    value = value.get("settings", value)
    frame = value.get("frame_microbatch")
    matmul = value.get("float32_matmul_precision", value.get("matmul"))
    fused = value.get("fused_optimizer", value.get("fused"))
    deferred = value.get("defer_metrics")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
        raise ValueError("Selected profile needs a positive frame_microbatch")
    if matmul not in ("highest", "high") or not isinstance(fused, bool) or not isinstance(deferred, bool):
        raise ValueError("Selected profile has invalid matmul/fused/defer settings")
    return {"frame_microbatch": frame, "float32_matmul_precision": matmul,
            "fused_optimizer": fused, "defer_metrics": deferred}


def open_wandb(root, drive, settings, sdk=None):
    if sdk is None:
        import wandb as sdk
    local, remote = root / "wandb_run.json", drive / "wandb_run.json"
    if not local.exists() and remote.exists():
        copy_verified(remote, local)
    old = json.loads(local.read_text()) if local.exists() else {}
    mode = "online" if bool(os.environ.get("WANDB_API_KEY")) else "offline"
    identity = old.get("id") or sdk.util.generate_id()
    if mode == "online":
        # Authentication must finish before starting a trainer. A bad supplied
        # key is a reported failure, never an implicit fallback to offline.
        if not sdk.login(key=os.environ["WANDB_API_KEY"], relogin=False):
            raise RuntimeError("W&B online authentication failed")
    directory = root / "wandb"
    directory.mkdir(parents=True, exist_ok=True)
    options = {"project": "cr-rl", "id": identity, "mode": mode,
               "name": "state_event040_bc100_cosine", "job_type": "continuous-bc",
               "dir": str(directory), "config": {"target_total_epochs": 100, "decision_ticks": 8,
                                                    "selected_runtime_profile": settings},
               "settings": sdk.Settings(init_timeout=60, disable_git=True)}
    requested_entity = os.environ.get("WANDB_ENTITY") or os.environ.get("REQUESTED_ENTITY") or old.get("requested_entity")
    entity = requested_entity or old.get("entity")
    if entity:
        options["entity"] = entity
    if mode == "online":
        options["resume"] = "allow"
    run = sdk.init(**options)
    if run is None:
        raise RuntimeError("W&B did not create a logging run")
    configure_wandb(run)
    record = {"id": run.id, "project": "cr-rl", "name": options["name"],
              "requested_entity": requested_entity, "entity": getattr(run, "entity", entity),
              "mode": mode, "url": getattr(run, "url", None) if mode == "online" else None,
              "online_logging": mode == "online", "updated_unix": time.time()}
    atomic_json(local, record)
    copy_verified(local, remote)
    return run, record


def trainer_command(info, *, source_root, root, drive, settings, pilot_updates=0):
    argv = build_trainer_argv(
        info, source_root=source_root, dataset_manifest=root / "dataset/dataset.manifest.json",
        checkpoint=root / "training/latest.pt", out=root / "training", backup_dir=drive / "training",
        epochs=100, ram_limit_gib=40, split_registry=source_root / "fixtures/split_registry.json",
        lr_schedule="cosine", warmup_steps=350, min_learning_rate=3e-6,
        checkpoint_every_seats=128, validation_every_seats=350, **settings,
    )
    if pilot_updates:
        argv.extend(("--max-episodes", str(pilot_updates)))
    return argv


def pilot_remaining(info):
    config = info["training_config"].get("lr_schedule")
    if config is None:
        return 4
    if config.get("kind") != "cosine" or config.get("total_epochs") != 100:
        raise ValueError("Existing checkpoint uses a different schedule horizon")
    return max(0, 4 - (info["progress"]["optimizer_steps"] - config["anchor_step"]))


def verify_pilot(before, after, attempted):
    config = after["training_config"].get("lr_schedule")
    if (not config or config.get("kind") != "cosine" or config.get("total_epochs") != 100
            or config.get("warmup_steps") != 350 or config.get("min_lr") != 3e-6):
        raise RuntimeError("Pilot checkpoint did not preserve the requested warmup/cosine schedule")
    successful = after["progress"]["optimizer_steps"] - before["progress"]["optimizer_steps"]
    if successful != attempted:
        raise RuntimeError("Pilot did not complete every requested successful update; inspect overflow/empty-seat logs")
    if after["dataset_fingerprint"] != before["dataset_fingerprint"]:
        raise RuntimeError("Pilot changed the admitted dataset")
    if before["training_config"].get("lr_schedule") is None and config["anchor_step"] != before["progress"]["optimizer_steps"]:
        raise RuntimeError("Pilot schedule anchor does not match the handoff checkpoint")


def coordinate(root, drive, settings, source_root=None):
    root, drive = Path(root).resolve(), Path(drive).resolve()
    source_root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    drive.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    run = None
    mode = "online" if bool(os.environ.get("WANDB_API_KEY")) else "offline"
    log = root / "training/training.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    manifest, checkpoint = root / "dataset/dataset.manifest.json", root / "training/latest.pt"

    def publish(phase, **details):
        record = {"phase": phase, "pid": os.getpid(),
                  "child_pid": child.pid if child is not None and child.poll() is None else None,
                  "time_unix": time.time(), "wandb_mode": mode, **details}
        atomic_json(root / "status.json", record)
        atomic_json(drive / "status.json", record)
        print(json.dumps(record), flush=True)

    last_sync = 0.0

    def sync_small():
        nonlocal last_sync
        if time.monotonic() - last_sync < 120:
            return
        for path in (root / "wandb_run.json", root / ".notebook" / f"wandb_cursor_{mode}.json"):
            if path.exists():
                copy_verified(path, drive / path.relative_to(root))
        last_sync = time.monotonic()

    def train(argv, info, phase):
        nonlocal child
        environment = {**os.environ, "PYTHONPATH": str(source_root / "src"),
                       "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        preflight(argv, source_root, require_cuda=True, env=environment)
        with log.open("a", buffering=1) as output:
            child = subprocess.Popen(argv, cwd=source_root, env=environment,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        publish(phase, optimizer_steps=info["progress"]["optimizer_steps"], command=argv)
        result = monitor_process(
            child, root / "training/metrics.jsonl", log, run=run, target_epochs=100,
            seats_per_epoch=info["train_seats_per_epoch"],
            initial_completed=resume_completed(info["progress"], info["train_seats_per_epoch"]),
            cursor_path=root / ".notebook" / f"wandb_cursor_{mode}.json", on_poll=sync_small,
        )
        child = None
        return result

    try:
        info = inspect_checkpoint(checkpoint, manifest, verify_file=False)
        publish("preparing", optimizer_steps=info["progress"]["optimizer_steps"], settings=settings)
        run, logging = open_wandb(root, drive, settings)
        mode = logging["mode"]
        cursor = root / ".notebook" / f"wandb_cursor_{mode}.json"
        remote_cursor = drive / cursor.relative_to(root)
        if not cursor.exists() and remote_cursor.exists():
            copy_verified(remote_cursor, cursor)
        pending = pilot_remaining(info)
        if pending:
            before = info
            result = train(trainer_command(info, source_root=source_root, root=root, drive=drive,
                                           settings=settings, pilot_updates=pending), info, "pilot")
            info = inspect_checkpoint(checkpoint, manifest, verify_file=False)
            verify_pilot(before, info, pending)
            publish("pilot_passed", optimizer_steps=info["progress"]["optimizer_steps"],
                    successful_pilot_updates=pending, logging_status=result["logging_status"])
        if resume_completed(info["progress"], info["train_seats_per_epoch"]) < 100 * info["train_seats_per_epoch"]:
            result = train(trainer_command(info, source_root=source_root, root=root, drive=drive,
                                           settings=settings), info, "training")
            info = inspect_checkpoint(checkpoint, manifest, verify_file=False)
        else:
            result = {"logging_status": "accepted", "already_complete": True}
        if info["progress"]["epoch"] != 100 or info["progress"]["next_episode"] != 0:
            raise RuntimeError("Trainer exited before the requested total 100 epochs completed")
        run.finish()
        last_sync = 0.0
        sync_small()
        publish("complete", epochs=100, optimizer_steps=info["progress"]["optimizer_steps"],
                logging_status=result["logging_status"], online_logging=mode == "online")
        return info
    except BaseException as error:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        try:
            publish("failed", error_type=type(error).__name__, error=str(error), training_log_tail=log_tail(log))
        except Exception:
            traceback.print_exc()
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/bc100"))
    parser.add_argument("--drive-run", type=Path,
                        default=Path("/content/drive/MyDrive/cr_rl/runs/state_global_event040_20261110/bc100_cosine"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args(argv)
    coordinate(args.root, args.drive_run, profile_settings(args.settings))


if __name__ == "__main__":
    main()
