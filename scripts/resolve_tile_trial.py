"""Backfill automatic selection for an already-running legacy trial controller."""

import argparse
import json
import time
from pathlib import Path

from cr_rl.training.persistence import atomic_json, copy_verified
from cr_rl.training.tile_trial import select_tile_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        source = args.trial / "comparison.json"
        if source.exists():
            result = select_tile_trial(json.loads(source.read_text()))
            target = args.trial / "selection.json"
            if not target.exists():
                atomic_json(target, result)
            copy_verified(
                target, Path("/content/drive/MyDrive/cr_rl/runs") / args.trial.name / target.name
            )
            print(json.dumps(result), flush=True)
            return
        status = args.trial / "status.json"
        if status.exists() and json.loads(status.read_text())["phase"] == "failed":
            raise RuntimeError("Trial failed; no checkpoint selected")
        time.sleep(5)
    raise TimeoutError("Trial did not finish within the selection deadline")


if __name__ == "__main__":
    main()
