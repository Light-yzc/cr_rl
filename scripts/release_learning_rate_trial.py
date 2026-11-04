"""Release the same runtime after the complete LR trial and verified branch backups."""

import argparse
import json
import os
import time
from pathlib import Path

from finish_and_release import release_runtime
from release_continued_ppo import live_handoff
from run_learning_rate_trial import verify_trial_completion

from cr_rl.training.persistence import atomic_json
from cr_rl.training.runtime_supervision import process_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--grace-seconds", type=int, default=900)
    parser.add_argument("--handoff-request", type=Path)
    args = parser.parse_args()
    if not os.environ.get("TBE_RUNTIME_ADDR"):
        raise RuntimeError("Same-runtime management endpoint required")
    atomic_json(args.backup / "trial_release_ready.json", dict(pid=os.getpid()))
    began = time.monotonic()
    completed_since = None
    while time.monotonic() - began < 6 * 3600:
        p = args.backup / "trial_status.json"
        state = json.loads(p.read_text()) if p.exists() else {}
        if state.get("phase") in ("failed", "blocked_by_source_failure", "skipped_source_improved"):
            return
        if state.get("phase") == "completed" and not process_alive(state.get("pid")):
            request_path = args.handoff_request or args.backup / "confirmation_handoff.json"
            if request_path.exists():
                request = json.loads(request_path.read_text())
                if live_handoff(request, args.backup):
                    verified = verify_trial_completion(args.backup)
                    if live_handoff(request, args.backup):
                        atomic_json(
                            args.backup / "trial_release_handoff.json",
                            dict(request=request, source_verified=verified),
                        )
                        return
            if completed_since is None:
                completed_since = time.monotonic()
            if time.monotonic() - completed_since < args.grace_seconds:
                time.sleep(15)
                continue
            verified = verify_trial_completion(args.backup)
            atomic_json(args.backup / "trial_release_verified.json", verified)
            release_runtime()
            return
        time.sleep(30)
    raise TimeoutError("LR trial release guard elapsed")


if __name__ == "__main__":
    main()
