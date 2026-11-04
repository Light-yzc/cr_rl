"""Run a bounded, explicitly exploratory PPO trial in the prepared Colab kernel."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-on-complete", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size not in (4, 8, 16, 32, 64):
        parser.error("Microbatch must divide the unchanged effective batch of 64")
    root = Path("/content/cr-rl-lab")
    run = root / "artifacts/121m_ppo_pilot_20261104"
    backup = Path("/content/drive/MyDrive/cr_rl/runs/121m_ppo_pilot_20261104")
    source = root / "artifacts/121m_bc_repair_20261104/repaired/bc.pt"
    sys.path.insert(0, str(root / "src"))
    from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256, sync_logs

    if not backup.parent.is_dir() or not os.environ.get("TBE_RUNTIME_ADDR"):
        raise RuntimeError("Use the existing Colab kernel with Drive mounted")
    if run.exists() and not args.resume:
        raise RuntimeError("Pilot directory exists; inspect it instead of launching a duplicate")
    receipt = json.loads(
        (backup.parent / "121m_bc_repair_20261104/repaired/bc/latest.json").read_text()
    )
    if file_sha256(source) != receipt["sha256"] or receipt["optimizer_steps"] != 5096:
        raise RuntimeError("BC source differs from the verified 5096 checkpoint")
    run.mkdir(parents=True, exist_ok=args.resume)
    command = [
        sys.executable,
        "scripts/train_selfplay.py",
        "--v2",
        "--init-from-bc",
        str(source),
        "--critic-mode",
        "independent",
        "--critic-model",
        "same",
        "--critic-learning-rate",
        "1e-4",
        "--critic-epochs",
        "4",
        "--critic-warmup-updates",
        "2",
        "--critic-warmup-games",
        "4",
        "--envs",
        "2",
        "--steps",
        "1536",
        "--epochs",
        "2",
        "--batch-size",
        str(args.batch_size),
        "--grad-accumulation",
        str(64 // args.batch_size),
        "--learning-rate",
        "2e-5",
        "--gamma",
        "0.9995",
        "--gae-lambda",
        "0.98",
        "--tower-shaping",
        "--princess-reward",
        "0.15",
        "--king-reward",
        "0.20",
        "--crown-reward",
        "0.15",
        "--overflow-penalty",
        "0.001",
        "--reference-kl-coef",
        "0.01",
        "--target-kl",
        "0.02",
        "--decks",
        "configs/decks_initial3.yaml",
        "--updates",
        "6",
        "--out",
        str(run),
        "--backup-dir",
        str(backup),
        "--progress-jsonl",
        str(run / "progress.jsonl"),
        "--device",
        "cuda",
    ]
    config = dict(
        purpose="User-authorized exploratory PPO; formal BC promotion gate has not passed",
        source_checkpoint_sha256=receipt["sha256"],
        source_bc_step=5096,
        command=command,
        budget_seconds=4 * 3600,
        kernel_pid=os.getpid(),
        deck_scope="Three validated decks: giant/sparky, hog cycle, graveyard control",
        created_unix=time.time(),
        evaluate_on_complete=args.evaluate_on_complete,
    )
    if args.resume:
        from cr_rl.training.runtime_supervision import process_alive

        previous_status = json.loads((run / "job_status.json").read_text())
        if process_alive(previous_status.get("child_pid")):
            raise RuntimeError("Existing training process is alive; refusing duplicate resume")
        previous = json.loads((run / "run_config.json").read_text())
        manifest = json.loads((backup / "latest.json").read_text())
        generation = manifest["update"]
        if not 0 < generation < 6:
            raise RuntimeError("No unfinished bounded pilot to resume")
        record = next(row for row in manifest["files"] if row["path"] == manifest["checkpoint"])
        checkpoint = run / "ppo_latest.pt"
        if not checkpoint.exists() or file_sha256(checkpoint) != record["sha256"]:
            copy_verified(backup / manifest["checkpoint"], checkpoint)
        if file_sha256(checkpoint) != record["sha256"]:
            raise RuntimeError("Resume checkpoint differs from published generation")
        pool = json.loads((backup / manifest["pool"]).read_text())
        if file_sha256(backup / manifest["pool"]) != manifest["pool_sha256"]:
            raise RuntimeError("Published opponent pool checksum differs")
        for entry in pool["entries"]:
            name = Path(entry["checkpoint"]).name
            destination = run / "opponents" / name
            copy_verified(backup / "opponents" / name, destination)
            entry["checkpoint"] = str(destination.resolve())
        atomic_json(run / "opponent_pool.json", pool)
        init = command.index("--init-from-bc")
        command[init : init + 2] = ["--resume", str(checkpoint)]
        command[command.index("--updates") + 1] = str(6 - generation)
        config.update(
            resumed_from_update=generation,
            resume_checkpoint_sha256=record["sha256"],
            created_unix=previous["created_unix"],
            restarted_unix=time.time(),
            budget_seconds=max(1, 4 * 3600 - (time.time() - previous["created_unix"])),
            restart_semantics="Model/optimizer/RNG restored; unfinished live games discarded",
            efficiency_changes="Prepared immutable inputs; same effective batch 64",
        )
        copy_verified(
            run / "run_config.json", backup / f"run_config.before_resume_{generation}.json"
        )
    atomic_json(run / "run_config.json", config)
    copy_verified(run / "run_config.json", backup / "run_config.json")
    copy_verified(root / "configs/decks_initial3.yaml", backup / "league.yaml")
    copy_verified(Path(__file__), backup / "tools/colab_start_ppo_trial.py")
    if args.evaluate_on_complete:
        request = dict(
            heldout_pair_start=16,
            heldout_pairs=16,
            chunk_pairs=4,
            trained_deck_pairs=3,
            opponent="exact BC initialization",
            wait_bias=0,
        )
        atomic_json(run / "evaluation_request.json", request)
        copy_verified(run / "evaluation_request.json", backup / "evaluation_request.json")
        copy_verified(
            root / "scripts/evaluate_ppo_pilot.py", backup / "tools/evaluate_ppo_pilot.py"
        )
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    phase, failure = "starting", None
    with (run / "train.log").open("a") as log:
        child = subprocess.Popen(
            command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        began = time.monotonic()
        try:
            while True:
                rc = child.poll()
                latest = None
                progress = run / "progress.jsonl"
                if progress.exists():
                    rows = progress.read_text().splitlines()
                    for row in reversed(rows):
                        try:
                            latest = json.loads(row)
                            break
                        except json.JSONDecodeError:
                            continue
                phase = "running" if rc is None else ("completed" if rc == 0 else "failed")
                status = dict(
                    phase=phase,
                    kernel_pid=os.getpid(),
                    child_pid=child.pid,
                    observed_unix=time.time(),
                    exit_code=rc,
                    latest_update=latest,
                )
                atomic_json(run / "job_status.json", status)
                sync_logs(run, backup)
                print(json.dumps(status), flush=True)
                if rc is not None:
                    if rc:
                        raise RuntimeError(f"PPO exited {rc}; inspect train.log")
                    manifest = json.loads((backup / "latest.json").read_text())
                    if manifest["update"] != 6:
                        raise RuntimeError("Final PPO update backup is incomplete")
                    if args.evaluate_on_complete:
                        with (run / "evaluation_controller.log").open("a") as evaluation_log:
                            child = subprocess.Popen(
                                [
                                    sys.executable,
                                    "scripts/evaluate_ppo_pilot.py",
                                    "--run",
                                    str(run),
                                    "--backup",
                                    str(backup),
                                ],
                                cwd=root,
                                env=env,
                                stdout=evaluation_log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        while child.poll() is None:
                            phase = "evaluating"
                            evaluation_status = run / "evaluation/status.json"
                            details = (
                                json.loads(evaluation_status.read_text())
                                if evaluation_status.exists()
                                else {}
                            )
                            atomic_json(
                                run / "job_status.json",
                                dict(
                                    phase=phase,
                                    child_pid=child.pid,
                                    kernel_pid=os.getpid(),
                                    observed_unix=time.time(),
                                    evaluation=details,
                                ),
                            )
                            sync_logs(run, backup)
                            print(json.dumps(dict(phase=phase, evaluation=details)), flush=True)
                            if time.monotonic() - began > config["budget_seconds"]:
                                raise TimeoutError("Evaluation exceeded remaining pilot budget")
                            time.sleep(30)
                        if child.returncode:
                            raise RuntimeError(
                                "Post-PPO evaluation failed; inspect evaluation_controller.log"
                            )
                        marker = run / "evaluation/completed.json"
                        if file_sha256(marker) != file_sha256(backup / "evaluation/completed.json"):
                            raise RuntimeError("Evaluation completion marker is not backed up")
                        phase = "completed"
                        atomic_json(
                            run / "job_status.json",
                            dict(
                                phase=phase,
                                child_pid=child.pid,
                                kernel_pid=os.getpid(),
                                observed_unix=time.time(),
                                updates=6,
                                evaluation_completed=True,
                            ),
                        )
                        sync_logs(run, backup)
                    break
                if time.monotonic() - began > config["budget_seconds"]:
                    raise TimeoutError("Bounded pilot budget elapsed")
                time.sleep(30)
        except BaseException as error:
            failure = type(error).__name__
            if child.poll() is None:
                import signal

                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            atomic_json(
                run / "job_status.json",
                dict(
                    phase="failed",
                    error_type=failure,
                    observed_unix=time.time(),
                    child_pid=child.pid,
                    exit_code=child.returncode,
                ),
            )
            sync_logs(run, backup)
            raise
        finally:
            # Release only after a successful, backed-up bounded run. A failure
            # keeps the runtime available for inspection and recovery.
            if phase == "completed" and failure is None:
                atomic_json(
                    backup / "pilot_completed.json", dict(updates=6, observed_unix=time.time())
                )
                print("Six PPO updates backed up; releasing the bounded runtime.", flush=True)
                from google.colab import runtime

                runtime.unassign()


if __name__ == "__main__":
    main()
