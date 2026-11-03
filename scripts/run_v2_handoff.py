"""Finish an active clean-slate BC run, then validate and train persistent v2 PPO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cr_rl.evaluation.promotion import verify_promotion
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256
from cr_rl.training.selection import select_bc_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def completed_bc_checkpoint(run, status):
    path = Path(status.get("final_checkpoint", run / "bc.pt")).resolve()
    if not path.is_relative_to(run.resolve()) or not path.is_file():
        raise ValueError("Completed BC checkpoint is missing or outside its run")
    return path


def audit_updates(rows):
    if len(rows) < 3 or not any(row["optimizer_steps"] > 0 for row in rows):
        raise ValueError("Production actor has not updated after critic warmup")
    if not any(row["critic_optimizer_steps"] > 0 for row in rows):
        raise ValueError("Production critic has not updated")
    if not any(not row["actor_enabled"] for row in rows):
        raise ValueError("Missing frozen-actor warmup")
    if not sum(row["terminal_games"] for row in rows):
        raise ValueError("No complete production training games")
    if any(row["faulted_samples"] for row in rows):
        raise ValueError("Faulted production samples require review")
    for previous, following in zip(rows, rows[1:], strict=False):
        if previous["episode_steps_at_update_end"] != following["episode_steps_at_update_start"]:
            raise ValueError("Simulator episode continuity failed")
    return {
        "updates": len(rows),
        "actor_steps": sum(row["optimizer_steps"] for row in rows),
        "critic_steps": sum(row["critic_optimizer_steps"] for row in rows),
        "terminal_games": sum(row["terminal_games"] for row in rows),
        "continuous_boundaries": len(rows) - 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", required=True)
    parser.add_argument("--promotion-report", required=True, type=Path)
    parser.add_argument("--ppo-updates", type=int, default=6)
    parser.add_argument("--suffix", default="_handoff")
    args = parser.parse_args()
    if args.ppo_updates < 3:
        parser.error("Need warmup and at least one actor update")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Supply W&B credentials through the existing training environment")
    bc = ROOT / "artifacts" / args.bc_run
    if not args.suffix.replace("_", "").replace("-", "").isalnum():
        parser.error("suffix may contain only letters, digits, underscores and hyphens")
    out = ROOT / "artifacts" / (args.bc_run + args.suffix)
    ppo_name = args.bc_run + "_ppo"
    ppo = ROOT / "artifacts" / ppo_name
    drive_root = Path("/content/drive/MyDrive/cr_rl/runs")
    drive = drive_root / out.name
    if (out / "status.json").exists():
        raise ValueError("Handoff already exists; inspect its worker before relaunching")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
    status = {"phase": "waiting_for_bc", "worker_pid": os.getpid(), "bc_run": args.bc_run}

    def sync():
        status["heartbeat_unix"] = time.time()
        atomic_json(out / "status.json", status)
        copy_verified(out / "status.json", drive / "status.json")

    def run(command, phase):
        status["phase"] = phase
        sync()
        log_path = out / f"{phase}.log"
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = child.pid
            while child.poll() is None:
                sync()
                time.sleep(20)
        copy_verified(log_path, drive / log_path.name)
        status.pop("child_pid", None)
        if child.returncode:
            raise RuntimeError(f"{phase} exited {child.returncode}; see {log_path.name}")

    try:
        sync()
        deadline = time.monotonic() + 4 * 3600
        while True:
            current = json.loads((bc / "status.json").read_text())
            if current["phase"] == "failed":
                raise RuntimeError("BC failed; inspect its retained checkpoint and log")
            if current["phase"] == "completed":
                if not current.get("success"):
                    raise RuntimeError("BC did not pass its completion audit")
                break
            pid = current.get("worker_pid")
            if not pid and (bc / "worker.json").exists():
                pid = json.loads((bc / "worker.json").read_text())["pid"]
            process = Path(f"/proc/{pid}/stat")
            if not process.exists() or process.read_text().split()[2] == "Z":
                raise RuntimeError("BC supervisor exited before publishing completion")
            if time.monotonic() > deadline:
                raise TimeoutError("BC wait budget ended; no PPO started")
            sync()
            time.sleep(30)

        final = completed_bc_checkpoint(bc, current)
        stage = final.parent.relative_to(bc.resolve())
        selected, selection = select_bc_checkpoint(final, drive_root / args.bc_run / stage / "bc")
        status["selection"] = selection
        # Promotion requires the exact calibrated weights to beat both frozen
        # historical baselines on the same fresh 8+ paired deck cohort.
        selected, promotion = verify_promotion(args.promotion_report)
        status["promotion"] = promotion
        copy_verified(args.promotion_report, drive / "promotion_report.json")
        run(
            [
                sys.executable,
                "scripts/run_colab_training.py",
                "--v2",
                "--model",
                "production",
                "--updates",
                str(args.ppo_updates),
                "--run-name",
                ppo_name,
                "--init-from-bc",
                str(selected),
                "--decks",
                "configs/decks.yaml",
                "--critic-mode",
                "independent",
                "--critic-model",
                "same",
                "--critic-warmup-updates",
                "2",
                "--critic-warmup-games",
                "4",
            ],
            "ppo",
        )
        rows = [json.loads(line) for line in (ppo / "metrics.jsonl").read_text().splitlines()]
        audit = audit_updates(rows)
        if audit["updates"] != args.ppo_updates:
            raise ValueError("PPO stopped before its requested update count")
        # Independent readback of the published actor/critic checkpoint and export.
        published = drive_root / ppo_name
        manifest = json.loads((published / "latest.json").read_text())
        for record in manifest["files"]:
            path = published / record["path"]
            if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
                raise ValueError(f"Drive integrity failure: {record['path']}")
        run(
            [
                sys.executable,
                "scripts/evaluate_policy.py",
                "--policy",
                str(ppo / "actor_export.pt"),
                "--opponent",
                str(selected),
                "--decks",
                "configs/decks_eval.yaml",
                "--pairs",
                "2",
                "--out",
                str(out / "ppo_evaluation"),
            ],
            "ppo_closed_loop",
        )
        for path in (out / "ppo_evaluation").iterdir():
            if path.is_file():
                copy_verified(path, drive / "ppo_evaluation" / path.name)
        status.update(
            phase="completed",
            passed=True,
            production=audit,
            ppo_evaluation=json.loads((out / "ppo_evaluation/evaluation.json").read_text()),
            limitation="Four paired games per comparison; no reliable strength estimate",
        )
        sync()
    except Exception:
        error = traceback.format_exc()
        secret = os.environ.get("WANDB_API_KEY")
        if secret:
            error = error.replace(secret, "[REDACTED]")
        status.update(phase="failed", passed=False, error=error)
        sync()
        raise


if __name__ == "__main__":
    main()
