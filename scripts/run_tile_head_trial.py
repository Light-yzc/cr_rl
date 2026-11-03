"""Checkpoint-safe full-model BC comparison, followed by the selected full-data run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified
from cr_rl.training.tile_trial import select_tile_trial

ROOT = Path(__file__).resolve().parents[1]
DRIVE = Path("/content/drive/MyDrive/cr_rl/runs")


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-queue-pid", type=int, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--run-name", default="119m_tile_fix_trial_20261103")
    parser.add_argument("--old-run-name", default="119m_all_bc_20261103")
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--manual-selection", action="store_true")
    args = parser.parse_args()
    output = ROOT / "artifacts" / args.run_name
    destination = DRIVE / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    old_local = ROOT / "artifacts" / args.old_run_name
    old_drive = DRIVE / args.old_run_name
    dataset = old_local / "stage_0001/data"
    status = {"phase": "waiting_acceptance", "run_name": args.run_name, "success": False}

    def sync(case=None):
        status["heartbeat_unix"] = time.time()
        atomic_json(output / "status.json", status)
        roots = [(output, destination)]
        if case:
            roots.append((case, destination / case.name))
        for local, remote in roots:
            for path in [*local.glob("*.json"), *local.glob("*.log")]:
                snapshot = local / ".snapshots" / path.name
                snapshot.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, snapshot)
                copy_verified(snapshot, remote / path.name)

    def latest():
        entries = []
        for path in old_drive.glob("stage_*/bc/latest.json"):
            data = json.loads(path.read_text())
            entries.append((int(data["optimizer_steps"]), path.parent / data["path"], data))
        if not entries:
            raise RuntimeError("No verified old BC checkpoint is available")
        return max(entries, key=lambda row: row[0])

    def run(command, local):
        atomic_json(local / "command.json", {"command": command})
        with (local / "train.log").open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status["training_pid"] = process.pid
            while process.poll() is None:
                tracking = local / "bc.wandb.json"
                if tracking.exists():
                    status["wandb"] = json.loads(tracking.read_text())["url"]
                sync(local)
                time.sleep(20)
        if process.returncode:
            raise RuntimeError((local / "train.log").read_text()[-3000:])
        sync(local)

    try:
        while not args.acceptance.exists():
            sync()
            time.sleep(10)
        if not json.loads(args.acceptance.read_text()).get("passed"):
            raise RuntimeError("Placement migration integration acceptance failed")
        parent = output / "parent.pt"
        if not (output / "parent.json").exists():
            baseline_step = latest()[0]
            status.update(phase="waiting_checkpoint", baseline_step=baseline_step)
            while True:
                step, path, record = latest()
                if step > baseline_step or not alive(args.old_queue_pid):
                    copied = copy_verified(path, parent)
                    if copied["sha256"] == record["sha256"]:
                        break
                sync()
                time.sleep(5)
            proc = Path(f"/proc/{args.old_queue_pid}/cmdline")
            if alive(args.old_queue_pid):
                if "scripts/run_all_bc.py" not in proc.read_text():
                    raise RuntimeError("Old queue identity changed; refusing to stop another job")
                os.killpg(args.old_queue_pid, signal.SIGTERM)
            copy_verified(parent, destination / "parent.pt")
            atomic_json(output / "parent.json", {"source": str(path), **record})
        for path in dataset.glob("*.json"):
            copy_verified(path, destination / "data" / path.name)

        cases = {}
        for name in ("control", "fixed"):
            local = output / name
            local.mkdir(exist_ok=True)
            status.update(phase="bc_comparison", case=name)
            command = [
                sys.executable,
                "scripts/train_bc.py",
                str(dataset),
                "--model",
                "production",
                "--device",
                "cuda",
                "--epochs",
                "1",
                "--samples-per-epoch",
                str(args.samples),
                "--stable-split",
                "--stream-cache-gib",
                "8",
                "--cache-dir",
                str(old_local / "array_cache"),
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
                "--wandb-name",
                args.run_name + "_" + name,
                "--holdout-decks",
                "configs/decks_eval.yaml",
                "--out",
                str(local / "bc.pt"),
                "--backup-dir",
                str(destination / name / "bc"),
            ]
            if (local / "bc.pt").exists():
                command += ["--resume", str(local / "bc.pt")]
            else:
                command += ["--init-from-bc", str(parent), "--carry-optimizer"]
                if name == "fixed":
                    command += ["--upgrade-tile-head", "--tile-learning-rate", "3e-4"]
            run(command, local)
            cases[name] = {
                "metrics": json.loads((local / "bc.json").read_text()),
                "data": json.loads((local / "bc.data.json").read_text()),
                "wandb": json.loads((local / "bc.wandb.json").read_text())["url"],
            }
        a, b = cases["control"], cases["fixed"]
        assert a["data"]["dataset_sha256"] == b["data"]["dataset_sha256"]
        assert a["data"]["validation_rows_sha256"] == b["data"]["validation_rows_sha256"]
        assert (
            a["metrics"]["history"][-1]["sample_indices_sha256"]
            == b["metrics"]["history"][-1]["sample_indices_sha256"]
        )
        atomic_json(
            output / "comparison.json",
            {
                "same_parent": True,
                "same_training_samples": True,
                "same_validation_samples": True,
                "training_draws_per_case": args.samples,
                "cases": cases,
            },
        )
        status["phase"] = "comparison_ready"
        selection = output / "selection.json"
        if not args.manual_selection:
            atomic_json(
                selection, select_tile_trial(json.loads((output / "comparison.json").read_text()))
            )
        while not selection.exists():
            sync()
            time.sleep(10)
        choice = json.loads(selection.read_text())["choice"]
        if choice not in cases:
            raise ValueError("Selection must name one of the two tested checkpoints")
        continuation_name = "119m_all_bc_tilefix_20261103"
        local = ROOT / "artifacts" / continuation_name
        local.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "scripts/run_all_bc.py",
            "--run-name",
            continuation_name,
            "--initial-checkpoint",
            str(output / choice / "bc.pt"),
        ]
        status.update(phase="bc_training", selected=choice, continuation=continuation_name)
        with (output / "continuation.log").open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                path = local / "status.json"
                if path.exists():
                    following = json.loads(path.read_text())
                    status["continuation_phase"] = following["phase"]
                    status["wandb"] = following.get("wandb")
                sync()
                time.sleep(20)
        following = json.loads((local / "status.json").read_text())
        if not following.get("success"):
            raise RuntimeError("Full-data continuation failed; inspect its published status")
        status.update(phase="completed", success=True)
    except Exception:
        message = traceback.format_exc()
        key = os.environ.get("WANDB_API_KEY")
        status.update(phase="failed", error=message.replace(key, "[REDACTED]") if key else message)
    finally:
        sync()


if __name__ == "__main__":
    main()
