"""Real-data v2 mid-epoch BC resume, BC handoff, and restored PPO candidate gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from cr_rl.data.bc_snapshot import read_published
from cr_rl.training.persistence import atomic_json, copy_verified, publish_update, restore_run

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ppo-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drive", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    manifest, metadata, _ = read_published(args.dataset)
    subset = args.out / "subset"
    subset.mkdir()
    copy_verified(args.dataset / manifest["metadata"], subset / "metadata.json")
    selected = manifest["shards"][:8]
    manifest = {
        **manifest,
        "metadata": "metadata.json",
        "shards": [
            {**shard, "path": str((args.dataset / shard["path"]).resolve())} for shard in selected
        ],
        "rows": sum(shard["rows"] for shard in selected),
        "acceptance_subset": True,
    }
    atomic_json(subset / "dataset.manifest.json", manifest)
    full, resumed = args.out / "bc_full", args.out / "bc_resumed"
    full.mkdir()
    resumed.mkdir()
    common = [
        sys.executable,
        "scripts/train_bc.py",
        str(subset),
        "--v2",
        "--model",
        "tiny",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--samples-per-epoch",
        "512",
        "--batch-size",
        "8",
        "--grad-accumulation",
        "8",
        "--stable-split",
        "--natural-validation-samples",
        "128",
        "--validation-samples",
        "128",
        "--checkpoint-every-steps",
        "1",
        "--stream-cache-gib",
        "1",
        "--seed",
        "20261103",
    ]
    checkpoint = args.out / "bc_mid.pt"
    with (args.out / "bc_full.log").open("w") as log:
        child = subprocess.Popen(
            [*common, "--out", str(full / "bc.pt"), "--backup-dir", str(args.drive / "bc_full")],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 600
        captured = None
        while child.poll() is None:
            path = full / "bc.pt"
            if captured is None and path.exists():
                state = torch.load(path, map_location="cpu", weights_only=False)
                if state.get("epoch_progress"):
                    captured = state
                    torch.save(state, checkpoint)
            if time.monotonic() > deadline:
                child.terminate()
                child.wait(timeout=30)
                raise TimeoutError("Mid-epoch checkpoint acceptance exceeded its budget")
            time.sleep(0.1)
    if child.returncode or captured is None:
        raise RuntimeError("BC did not produce a usable mid-epoch checkpoint; inspect bc_full.log")
    with (args.out / "bc_resume.log").open("w") as log:
        subprocess.run(
            [*common, "--resume", str(checkpoint), "--out", str(resumed / "bc.pt")],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    reference = torch.load(full / "bc.pt", map_location="cpu", weights_only=False)
    restored = torch.load(resumed / "bc.pt", map_location="cpu", weights_only=False)
    assert reference["config"]["representation_version"] == 2
    assert reference["parent_bc_sha256"] is None
    assert all(
        torch.equal(value, restored["model"][key]) for key, value in reference["model"].items()
    )
    assert reference["baseline"] == restored["baseline"] == captured["baseline"]
    assert (
        reference["natural_baseline"]
        == restored["natural_baseline"]
        == captured["natural_baseline"]
    )
    assert reference["optimizer_steps"] == restored["optimizer_steps"]
    for index, state in reference["optimizer"]["state"].items():
        assert all(
            torch.equal(value, restored["optimizer"]["state"][index][key])
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        )
    report = {
        "passed": False,
        "bc_mid_epoch_step": captured["optimizer_steps"],
        "bc_final_step": restored["optimizer_steps"],
        "bc_model_bitwise_equal": True,
        "bc_optimizer_bitwise_equal": True,
        "initial_baselines_preserved": True,
        "scope": "Tiny real-data acceptance; eight shards and reduced snapshot gates",
    }
    handoff = args.out / "handoff"
    with (args.out / "handoff.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "scripts/train_selfplay.py",
                "--init-from-bc",
                str(resumed / "bc.pt"),
                "--model",
                "tiny",
                "--device",
                "cpu",
                "--envs",
                "1",
                "--steps",
                "32",
                "--updates",
                "2",
                "--epochs",
                "1",
                "--critic-epochs",
                "1",
                "--out",
                str(handoff),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    anchor = torch.load(handoff / "opponent_000000.pt", map_location="cpu", weights_only=False)
    assert all(
        torch.equal(value, anchor["model"][key])
        for key, value in restored["model"].items()
        if not key.startswith("critic.")
    )
    report["bc_to_ppo_actor_exact"] = True
    published = publish_update(args.ppo_source, args.drive / "published", 32)
    moved = args.out / "restored_ppo"
    restore_run(args.drive / "published", moved)
    state = torch.load(moved / "ppo_latest.pt", map_location="cpu", weights_only=False)
    state["admission"].update(
        interval_samples=1,
        min_games=1,
        min_score=0,
        games=1,
        score_sum=1,
        last_admitted_samples=0,
        last_failed_candidate_samples=0,
    )
    torch.save(state, moved / "ppo_latest.pt")
    with (args.out / "restored_ppo.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "scripts/train_selfplay.py",
                "--resume",
                str(moved / "ppo_latest.pt"),
                "--model",
                "tiny",
                "--device",
                "cpu",
                "--envs",
                "1",
                "--steps",
                "64",
                "--updates",
                "1",
                "--epochs",
                "1",
                "--critic-epochs",
                "1",
                "--out",
                str(moved),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    summary = json.loads((moved / "summary.json").read_text())
    gate = json.loads((moved / "candidate_gates/update_000033/gate.json").read_text())
    assert summary["update"] == 33 and summary["optimizer_steps"] > 0
    assert summary["critic_optimizer_steps"] > 0 and gate["passed"]
    assert summary["episode_steps_at_update_start"] == [0]
    pool = json.loads((moved / "opponent_pool.json").read_text())
    assert any(entry["policy_id"] == 33 for entry in pool["entries"])
    report.update(
        passed=True,
        ppo_restored_update=33,
        candidate_complete_game_gate=True,
        published_update=published["update"],
        simulator_reset_only_on_process_restart=True,
    )
    atomic_json(args.out / "report.json", report)
    for p in args.out.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".log", ".pt"):
            copy_verified(p, args.drive / "acceptance" / p.relative_to(args.out))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
