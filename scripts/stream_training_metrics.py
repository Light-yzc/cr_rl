"""Mirror real JSONL training events to W&B independently of the trainer."""

import argparse
import json
import time
from pathlib import Path

import wandb

from cr_rl.training.persistence import atomic_json, copy_verified
from cr_rl.training.telemetry import TrainingTelemetry, rollout_value_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--min-update", type=int, default=7)
    parser.add_argument("--name", default="122m_ppo_continue_20261104")
    args = parser.parse_args()
    record = args.run / "wandb_run.json"
    options = (
        {"id": json.loads(record.read_text())["id"], "resume": "allow"} if record.exists() else {}
    )
    with wandb.init(
        project="cr-rl", name=args.name, job_type="ppo-v2", **options
    ) as run:
        atomic_json(record, {"id": run.id, "url": run.url})
        copy_verified(record, args.backup / "wandb_run.json")
        telemetry = TrainingTelemetry(run, 3072)
        position = 0
        summaries = set()
        began = time.monotonic()
        while time.monotonic() - began < 4 * 3600:
            with (args.run / "progress.jsonl").open() as events:
                events.seek(position)
                while True:
                    start = events.tell()
                    line = events.readline()
                    if not line or not line.endswith("\n"):
                        position = start
                        break
                    event = json.loads(line)
                    if event.get("update", 0) >= args.min_update:
                        telemetry.log(event)
            for line in (args.run / "metrics.jsonl").read_text().splitlines(keepends=True):
                if not line.endswith("\n"):
                    continue
                summary = json.loads(line)
                update = summary["update"]
                if update < args.min_update or update in summaries:
                    continue
                # Only analyze a generation once the trainer has published it.
                manifest = args.backup / f"checkpoints/update_{update:06d}.manifest.json"
                if not manifest.exists():
                    continue
                telemetry.log_summary(summary)
                diagnostics = rollout_value_diagnostics(
                    args.run / "rollouts" / f"rollout_{update:06d}.npz"
                )
                telemetry.log(
                    dict(
                        phase="targets",
                        update=update,
                        learner_samples_total=summary["learner_samples_total"],
                        **diagnostics,
                    )
                )
                atomic_json(args.backup / f"value_diagnostics_{update:06d}.json", diagnostics)
                summaries.add(update)
            state = json.loads((args.backup / "continuation_status.json").read_text())
            if state["phase"] in ("failed", "training_completed"):
                run.summary["controller_phase"] = state["phase"]
                return
            time.sleep(10)
        raise TimeoutError("Metric mirror fixed budget elapsed")


if __name__ == "__main__":
    main()
