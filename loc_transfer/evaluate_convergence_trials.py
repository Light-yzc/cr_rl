"""Run the fixed held-out comparison for completed convergence branches."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "artifacts/ppo_convergence_trials_20261108"
SOURCE = (ROOT / "artifacts/local_dual_v100").resolve()
DECKS = ROOT / "configs/decks_expansion12.yaml"
V100 = (
    "GPU-95f9c3e4-242f-9098-3c63-dc4dc1557d69",
    "GPU-cb00ba20-7367-bd99-c5d6-0824adaaa5b2",
)
BRANCHES = ("baseline", "adaptive_lr", "longer_gae")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_one(branch: str, opponent_name: str, gpu: str) -> dict:
    branch_dir = TRIALS / branch
    opponent = (
        ROOT / "artifacts/v100_handoff/opponents/opponent_000000.pt"
        if opponent_name == "bc"
        else SOURCE / "actor_export.pt"
    )
    out = branch_dir / "evaluation" / opponent_name
    out.mkdir(parents=True, exist_ok=True)
    report = out / "evaluation.json"
    if report.exists():
        row = json.loads(report.read_text())
        if (
            row.get("games") == 16
            and row.get("invalid_games") == 0
            and row.get("pair_start") == 32
            and row.get("pair_count") == 8
        ):
            return {"branch": branch, "opponent": opponent_name, "reused": True, **row}
        raise RuntimeError(f"Existing incomplete evaluation: {out}")
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_policy.py"),
        "--policy", str(branch_dir / "actor_export.pt"),
        "--opponent", str(opponent),
        "--decks", str(DECKS),
        "--out", str(out),
        "--pairs", "8", "--pair-start", "32",
        "--sampling-seed", "20261201", "--pairing-seed", "20261201",
        "--simulator-seed-base", "11000000", "--device", "cuda:0",
        "--frame-cache", "--forced-wait-fastpath",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    with (out / "evaluation.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not report.exists():
        raise RuntimeError(f"Evaluation failed: {branch}/{opponent_name}")
    row = json.loads(report.read_text())
    if row.get("games") != 16 or row.get("invalid_games") != 0:
        raise RuntimeError(f"Evaluation incomplete: {branch}/{opponent_name}")
    return {"branch": branch, "opponent": opponent_name, "reused": False, **row}


def main() -> None:
    jobs = [(branch, opponent) for branch in BRANCHES for opponent in ("bc", "current")]
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_one, branch, opponent, V100[index % len(V100)]): (branch, opponent)
            for index, (branch, opponent) in enumerate(jobs)
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["branch"], row["opponent"]))
    result = []
    for row in rows:
        games = row["games"]
        result.append({
            "branch": row["branch"], "opponent": row["opponent"],
            "games": games, "wins": row["wins"], "losses": row["losses"],
            "draws": row["draws"], "invalid_games": row["invalid_games"],
            "score": (row["wins"] + 0.5 * row["draws"]) / games,
            "policy_sha256": row["policy_sha256"],
            "opponent_sha256": row["opponent_sha256"], "reused": row["reused"],
        })
    (TRIALS / "evaluation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
