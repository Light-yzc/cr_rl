"""Publish non-secret training diagnostics to an existing W&B run."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Pass WANDB_API_KEY through the environment")
    deadline = time.monotonic() + 3 * 3600
    while args.wait_pid:
        stat = Path(f"/proc/{args.wait_pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Log publication timed out waiting for training")
        time.sleep(10)
    run = wandb.Api().run(args.run)
    for name in (
        "train.log",
        "metrics.jsonl",
        "progress.jsonl",
        "job_status.json",
        "run_config.json",
    ):
        source = args.out / name
        if source.exists():
            run.upload_file(str(source), root=str(args.out))
            print("Published diagnostic file:", name, flush=True)


if __name__ == "__main__":
    main()
