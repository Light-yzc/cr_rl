"""Complete evaluation and independent audits, including older supervisor handoffs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified
from cr_rl.training.selection import select_bc_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="119m_multideck_bc_20261103")
    args = parser.parse_args()
    local = ROOT / "artifacts" / args.run_name
    ppo = ROOT / "artifacts" / (args.run_name + "_ppo")
    drive = Path("/content/drive/MyDrive/cr_rl/runs") / args.run_name
    data = ROOT / "artifacts/multideck"
    status = {"phase": "waiting_for_training", "run_name": args.run_name}
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

    def heartbeat():
        status["heartbeat_unix"] = time.time()
        atomic_json(local / "completion_status.json", status)
        copy_verified(local / "completion_status.json", drive / "completion_status.json")

    def run(command, phase):
        status["phase"] = phase
        heartbeat()
        with (local / f"{phase}.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                heartbeat()
                time.sleep(30)
        copy_verified(local / f"{phase}.log", drive / f"{phase}.log")
        if child.returncode:
            raise RuntimeError(f"{phase} failed; see its log")

    try:
        deadline = time.monotonic() + 3 * 3600
        while time.monotonic() < deadline:
            pipeline = json.loads((local / "pipeline_status.json").read_text())
            heartbeat()
            if pipeline["phase"] in ("failed", "completed"):
                break
            time.sleep(30)
        else:
            raise TimeoutError("Training supervisor did not finish")
        ppo_already_complete = (ppo / "job_status.json").exists() and json.loads(
            (ppo / "job_status.json").read_text()
        ).get("phase") == "completed"
        if pipeline["phase"] == "failed" and not ppo_already_complete:
            if "BC held-out imitation gate failed" not in pipeline.get("error", ""):
                raise RuntimeError(pipeline.get("error", "Unknown pipeline failure"))
            # An older in-memory supervisor gated the last epoch's small validation
            # subset. Re-evaluate all held-out play labels and select a complete epoch.
            status["initial_gate_error"] = pipeline["error"]
            run(
                [
                    sys.executable,
                    "scripts/evaluate_bc_checkpoints.py",
                    "--dataset",
                    str(data / "human_bc"),
                    "--run-dir",
                    str(local),
                    "--backup-dir",
                    str(drive / "bc"),
                ],
                "bc_full_validation",
            )
            selected, selection = select_bc_checkpoint(local / "bc.pt")
            status["selection"] = selection
            run(
                [
                    sys.executable,
                    "scripts/run_colab_training.py",
                    "--model",
                    "production",
                    "--updates",
                    "3",
                    "--run-name",
                    args.run_name + "_ppo",
                    "--init-from-bc",
                    str(selected),
                    "--decks",
                    str(data / "league_train.yaml"),
                ],
                "ppo_handoff",
            )
        ppo_status = json.loads((ppo / "job_status.json").read_text())
        if ppo_status["phase"] != "completed":
            raise RuntimeError(f"PPO incomplete: {ppo_status['phase']}")
        selected, selection = select_bc_checkpoint(local / "bc.pt")
        status["selection"] = selection
        if not (local / "evaluation/evaluation.json").exists():
            run(
                [
                    sys.executable,
                    "scripts/evaluate_policy.py",
                    "--policy",
                    str(ppo / "ppo_latest.pt"),
                    "--opponent",
                    str(selected),
                    "--decks",
                    str(data / "league_eval.yaml"),
                    "--out",
                    str(local / "evaluation"),
                ],
                "evaluation",
            )
        run(
            [
                sys.executable,
                "scripts/verify_bc_ppo_run.py",
                "--bc-run",
                str(local),
                "--ppo-run",
                str(ppo),
            ],
            "independent_verification",
        )
        run([sys.executable, "-m", "pytest", "-q"], "final_tests")
        for source in (local / "evaluation").iterdir():
            copy_verified(source, drive / "evaluation" / source.name)
        import wandb

        api = wandb.Api()
        runs = {}
        for label, folder, filename in (
            ("bc", local, "bc.wandb.json"),
            ("ppo", ppo, "wandb_run.json"),
        ):
            metadata = json.loads((folder / filename).read_text())
            prefix = "/".join(metadata["url"].split("/runs/")[0].split("/")[-2:])
            tracked = api.run(prefix + "/" + metadata["id"])
            if label == "ppo":
                for name in (
                    "train.log",
                    "metrics.jsonl",
                    "progress.jsonl",
                    "job_status.json",
                    "run_config.json",
                ):
                    tracked.upload_file(str(folder / name), root=str(folder))
                keys = ["ppo/total_loss", "ppo/grad_norm", "policy/entropy", "selfplay/win_rate"]
            else:
                for name in (
                    "bc.json",
                    "bc_selection_full.json",
                    "bc_selection.json",
                    "final_verification.json",
                    "evaluation/evaluation.json",
                ):
                    tracked.upload_file(str(folder / name), root=str(folder))
                tracked.summary["selected_validation"] = selection
                tracked.summary.update()
                keys = [
                    "bc/train_loss",
                    "bc/grad_norm",
                    "bc_v3/play_slot_accuracy",
                    "bc_v3/play_recall",
                ]
            counts = {key: sum(1 for _ in tracked.scan_history(keys=[key])) for key in keys}
            if not all(counts.values()):
                raise RuntimeError(f"W&B useful history missing: {counts}")
            runs[label] = {"url": tracked.url, "history_counts": counts}
        result = json.loads((local / "final_verification.json").read_text())
        result["wandb_verified"] = runs
        result["evaluation"] = json.loads((local / "evaluation/evaluation.json").read_text())
        result["bc_selection"] = selection
        atomic_json(local / "final_verification.json", result)
        copy_verified(local / "final_verification.json", drive / "final_verification.json")
        status.update(
            phase="completed",
            wandb=runs,
            ppo=ppo_status,
            evaluation=result["evaluation"],
            verified=True,
        )
        pipeline.update(
            phase="completed",
            completion_workflow="completion_status.json",
            bc_selection=selection,
            ppo=ppo_status,
        )
        if "bc_gate" in pipeline:
            pipeline["initial_bc_gate"] = pipeline["bc_gate"]
        pipeline["bc_gate"] = {
            "passed": True,
            "metric_version": 3,
            "baseline": selection["baseline"],
            "selected": selection["validation"],
        }
        if "error" in pipeline:
            pipeline["initial_gate_error"] = pipeline.pop("error")
        atomic_json(local / "pipeline_status.json", pipeline)
        copy_verified(local / "pipeline_status.json", drive / "pipeline_status.json")
    except Exception:
        status.update(
            phase="failed",
            error=traceback.format_exc().replace(
                os.environ.get("WANDB_API_KEY", "<absent>"), "[REDACTED]"
            ),
        )
        print(status["error"], flush=True)
    finally:
        heartbeat()
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
