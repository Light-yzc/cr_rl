"""Audit the next frozen stage and calibrate its completed checkpoint on Colab."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


@contextmanager
def paused_next_trainer(run, completed_stage):
    """Use a PID handle, and wait for a stable subsequent trainer before sharing GPU."""
    while True:
        status = json.loads((run / "status.json").read_text())
        if status["phase"] == "failed":
            raise RuntimeError("BC failed; retain its checkpoint and inspect training logs")
        if status["phase"] == "completed":
            yield None
            return
        if status["phase"] == "bc_training" and status["stage"] > completed_stage:
            pid = status["training_pid"]
            process = Path(f"/proc/{pid}")
            try:
                handle = os.pidfd_open(pid)
            except ProcessLookupError:
                time.sleep(5)
                continue
            stopped = False
            try:
                command = (process / "cmdline").read_bytes()
                if b"train_bc.py" not in command or os.fsencode(str(run)) not in command:
                    raise RuntimeError("Trainer identity differs; refusing to signal it")
                signal.pidfd_send_signal(handle, signal.SIGSTOP)
                stopped = True
                for _ in range(50):
                    if (process / "stat").read_text().split()[2] in ("T", "t"):
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Trainer did not acknowledge the GPU pause")
                yield pid
            finally:
                if stopped:
                    try:
                        signal.pidfd_send_signal(handle, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                os.close(handle)
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--drive-run", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=2)
    args = parser.parse_args()
    if args.stage < 2:
        parser.error("Follow-up requires a new stage after the tested first stage")
    run = args.run.resolve()
    stage = run / f"stage_{args.stage:04d}"
    out = run / f"stage_{args.stage:04d}_followup"
    drive = args.drive_run / out.name
    out.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")

    def publish(phase, **details):
        state = dict(phase=phase, stage=args.stage, observed_unix=time.time(), **details)
        atomic_json(out / "status.json", state)
        copy_verified(out / "status.json", drive / "status.json")
        print(json.dumps(state), flush=True)

    def execute(command):
        subprocess.run([sys.executable, *command], env=env, check=True)

    try:
        publish("waiting_stage_data", next_final_pair_start=13, ppo_started=False)
        while not (stage / "training_config.json").exists():
            status = json.loads((run / "status.json").read_text())
            if status["phase"] in ("failed", "completed"):
                raise RuntimeError("BC ended before the requested new stage became available")
            time.sleep(20)
        publish("auditing_coverage")
        execute(
            [
                "scripts/audit_bc_coverage.py",
                "--before",
                str(run / "stage_0001/data"),
                "--after",
                str(stage / "data"),
                "--out",
                str(out / "coverage.json"),
            ]
        )
        copy_verified(out / "coverage.json", drive / "coverage.json")
        publish("waiting_completed_checkpoint")
        planned_epochs = json.loads((stage / "training_plan.json").read_text())["epochs"]
        while True:
            record = args.drive_run / stage.name / "bc/latest.json"
            if record.exists():
                latest = json.loads(record.read_text())
                if not latest["mid_epoch"] and latest["epoch"] >= planned_epochs:
                    source = args.drive_run / stage.name / "bc" / latest["path"]
                    if file_sha256(source) != latest["sha256"]:
                        raise ValueError("Completed Drive checkpoint integrity check failed")
                    break
            if json.loads((run / "status.json").read_text())["phase"] == "failed":
                raise RuntimeError("BC failed before completing this stage")
            time.sleep(20)
        pinned = out / "candidate.pt"
        copy_verified(source, pinned)
        atomic_json(out / "checkpoint_source.json", {**latest, "path": str(source)})
        copy_verified(out / "checkpoint_source.json", drive / "checkpoint_source.json")
        with paused_next_trainer(run, args.stage) as pid:
            publish("natural_calibration", paused_trainer_pid=pid)
            execute(
                [
                    "scripts/calibrate_wait_bias.py",
                    "--checkpoint",
                    str(pinned),
                    "--dataset",
                    str(stage / "data"),
                    "--out",
                    str(out / "report.json"),
                    "--reuse-cache-dir",
                    str(run / "array_cache"),
                    "--cache-dir",
                    str(out / "private_arrays"),
                    "--samples",
                    "4096",
                ]
            )
            execute(
                [
                    "scripts/calibrate_action_windows.py",
                    "--calibration-dir",
                    str(out),
                    "--reuse-cache-dir",
                    str(run / "array_cache"),
                ]
            )
        for name in ("report.json", "report.samples.npz", "action_windows.json"):
            copy_verified(out / name, drive / name)
        publish(
            "completed",
            ppo_started=False,
            next_final_pair_start=13,
            next="Review new calibration and cadence before testing fresh paired decks",
        )
    except Exception:
        publish("failed", error=traceback.format_exc(), ppo_started=False)
        raise


if __name__ == "__main__":
    main()
