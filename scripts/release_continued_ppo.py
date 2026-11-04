"""Release only this bounded L4 after final training and all 64 games are backed up."""

import argparse
import json
import os
import time
from pathlib import Path

from finish_and_release import release_runtime

from cr_rl.training.persistence import atomic_json, file_sha256
from cr_rl.training.runtime_supervision import process_alive


def live_handoff(request, current_backup, alive=process_alive, command=None):
    """Accept only a different project run with a real controller and release guard."""
    if command is None:

        def command(pid):
            return Path(f"/proc/{pid}/cmdline").read_bytes()

    try:
        target = Path(request["backup"]).resolve()
        current = current_backup.resolve()
        pids = (request["controller_pid"], request["release_guard_pid"])
        if target in (current, current.parent) or not target.is_relative_to(current.parent):
            return False
        if pids[0] == pids[1] or any(not alive(pid) for pid in pids):
            return False
        return all(str(target).encode() in command(pid) for pid in pids)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def verify_completion(backup, expected_generation=14):
    evaluation = json.loads((backup / "evaluation_status.json").read_text())
    if evaluation["phase"] != "completed" or process_alive(evaluation.get("pid")):
        raise ValueError("Evaluation is unfinished or still running")
    summary_path = backup / "learning_curve_summary.json"
    if file_sha256(summary_path) != evaluation["summary_sha256"]:
        raise ValueError("Final evaluation summary SHA differs")
    summary = json.loads(summary_path.read_text())
    comparisons = summary["comparisons"]
    if (
        len(comparisons) != 4
        or sum(r["games"] for r in comparisons.values()) != 64
        or any(r["native_parity_games"] != r["games"] for r in comparisons.values())
    ):
        raise ValueError("Registered evaluation is not complete")
    manifest = json.loads((backup / "latest.json").read_text())
    if manifest["update"] != expected_generation:
        raise ValueError("Final training generation differs")
    if file_sha256(backup / manifest["pool"]) != manifest["pool_sha256"]:
        raise ValueError("Final opponent pool SHA differs")
    for row in manifest["files"]:
        path = backup / row["path"]
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError("Final training backup integrity failed")
    audits = list((backup / "learning_curve").glob("*/pairs_*/execution_audit.json"))
    if len(audits) != 8:
        raise ValueError("Missing native evaluation chunks")
    for path in audits:
        audit = json.loads(path.read_text())
        if len(audit["games"]) != 8 or audit["evaluation_sha256"] != file_sha256(
            path.parent / "evaluation.json"
        ):
            raise ValueError("Native audit does not match backed-up evaluation")
        for index, game in enumerate(audit["games"]):
            if not game["native_outcome_parity"] or game["replay_sha256"] != file_sha256(
                path.parent / f"match_{index:03d}.json"
            ):
                raise ValueError("Backed-up native replay integrity failed")
    return dict(
        generation=expected_generation, games=64, summary_sha256=evaluation["summary_sha256"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--final-generation", type=int, default=14)
    parser.add_argument("--handoff-grace-seconds", type=int, default=0)
    parser.add_argument("--handoff-request", type=Path)
    args = parser.parse_args()
    if args.handoff_grace_seconds < 0:
        parser.error("Handoff grace must be nonnegative")
    if not os.environ.get("TBE_RUNTIME_ADDR"):
        raise RuntimeError("Same-runtime management endpoint is required")
    began = time.monotonic()
    completed_since = None
    atomic_json(
        args.backup / "release_guard_ready.json",
        dict(
            pid=os.getpid(),
            final_generation=args.final_generation,
            handoff_grace_seconds=args.handoff_grace_seconds,
        ),
    )
    while time.monotonic() - began < 5 * 3600:
        state = json.loads((args.backup / "evaluation_status.json").read_text())
        if state["phase"] in ("failed", "blocked_by_training_failure"):
            return  # Preserve failures for inspection, never release an unverified run.
        if state["phase"] == "completed" and not process_alive(state.get("pid")):
            if args.handoff_request and args.handoff_request.exists():
                request = json.loads(args.handoff_request.read_text())
                if live_handoff(request, args.backup):
                    verified = verify_completion(args.backup, args.final_generation)
                    if live_handoff(request, args.backup):
                        atomic_json(
                            args.backup / "release_handoff.json",
                            dict(request=request, source_verified=verified),
                        )
                        return
            if completed_since is None:
                completed_since = time.monotonic()
            if time.monotonic() - completed_since < args.handoff_grace_seconds:
                time.sleep(15)
                continue
            verified = verify_completion(args.backup, args.final_generation)
            atomic_json(
                args.backup / "release_verified.json", dict(verified, observed_unix=time.time())
            )
            time.sleep(15)
            release_runtime()
            return
        time.sleep(45)
    raise TimeoutError("Release guard elapsed; inspect existing runtime")


if __name__ == "__main__":
    main()
