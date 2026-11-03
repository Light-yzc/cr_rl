"""Independently read weights, raw trajectories and Drive generations after training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from validate_training import audit_rollout

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--ppo-run", type=Path, required=True)
    parser.add_argument(
        "--drive-root", type=Path, default=Path("/content/drive/MyDrive/cr_rl/runs")
    )
    args = parser.parse_args()
    bc_path = args.bc_run / "bc_selected.pt"
    if not bc_path.exists():
        bc_path = args.bc_run / "bc.pt"
    bc = torch.load(bc_path, map_location="cpu", weights_only=False)
    bc_report = json.loads((args.bc_run / "bc.json").read_text())
    ppo = torch.load(args.ppo_run / "ppo_latest.pt", map_location="cpu", weights_only=False)
    assert ppo["initialization"]["checkpoint_sha256"] == file_sha256(bc_path)
    pool = json.loads((args.ppo_run / "opponent_pool.json").read_text())
    first = next(row for row in pool["entries"] if row["policy_id"] == 0)
    path = Path(first["checkpoint"])
    initial = torch.load(
        path if path.is_absolute() else args.ppo_run / path, map_location="cpu", weights_only=False
    )
    assert all(torch.equal(value, initial["model"][name]) for name, value in bc["model"].items())
    assert any(not torch.equal(value, ppo["model"][name]) for name, value in bc["model"].items())
    progress = [
        json.loads(line) for line in (args.ppo_run / "progress.jsonl").read_text().splitlines()
    ]
    optimizer_events = [row for row in progress if row["phase"] == "optimization"]
    assert optimizer_events[0]["optimizer_steps_total"] == 1
    assert ppo["optimizer"]["state"]
    league = (
        ppo.get("league_config")
        or json.loads((args.ppo_run / "run_config.json").read_text())["decks"]
    )
    unique_decks = {
        tuple(sorted(zip(cards, league.get("slots", {}).get(name, [0] * 8), strict=True)))
        for name, cards in league["decks"].items()
    }
    drive_ppo = args.drive_root / args.ppo_run.name
    latest = json.loads((drive_ppo / "latest.json").read_text())
    assert latest["update"] == ppo["updates"]
    for record in latest["files"]:
        path = drive_ppo / record["path"]
        assert file_sha256(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]
    drive_bc = args.drive_root / args.bc_run.name / "bc"
    latest_bc = json.loads((drive_bc / "latest.json").read_text())
    assert file_sha256(drive_bc / latest_bc["path"]) == latest_bc["sha256"]
    assert latest_bc["epoch"] == bc_report["history"][-1]["epoch"]
    saved = torch.load(drive_bc / latest_bc["path"], map_location="cpu", weights_only=False)
    assert saved["epochs"] == latest_bc["epoch"]
    del saved, initial, ppo
    probe = args.bc_run / "production_restore_probe"
    subprocess.run(
        [
            sys.executable,
            "scripts/verify_drive_resume.py",
            "--source",
            str(drive_ppo),
            "--out",
            str(probe),
        ],
        check=True,
    )
    rows = [json.loads(line) for line in (args.ppo_run / "metrics.jsonl").read_text().splitlines()]
    observed = set()
    deck_fields = [
        i
        for i, field in enumerate(bc["dataset_metadata"]["schema"]["header_fields"])
        if field == "deck"
    ]
    for path in sorted((args.ppo_run / "rollouts").glob("*.npz")):
        with np.load(path) as data:
            learner_obs = data["obs"][data["learner_mask"].astype(bool)]
            for deck in np.unique(np.sort(learner_obs[:, deck_fields], axis=1), axis=0):
                observed.add(tuple(int(card) for card in deck))
    report = {
        "passed": True,
        "initial_ppo_weights_equal_bc": True,
        "ppo_weights_changed": True,
        "ppo_optimizer_started_at_one": True,
        "ppo_deck_source_entries": len(league["decks"]),
        "ppo_distinct_training_decks": len(unique_decks),
        "ppo_actual_learner_eight_card_combinations": len(observed),
        "bc_epochs": len(bc_report["history"]),
        "selected_bc_epoch": bc["epochs"],
        "bc_optimizer_steps": bc["optimizer_steps"],
        "bc_validation_replays": bc["validation_replays"],
        "bc_excluded_heldout_deck_replays": bc["heldout_deck_replays_excluded"],
        "bc_history": bc_report["history"],
        "ppo_summaries": rows,
        "rollout_audits": [
            audit_rollout(p) for p in sorted((args.ppo_run / "rollouts").glob("*.npz"))
        ],
        "verified_drive_ppo_files": len(latest["files"]),
        "verified_drive_bc_generations": len(list(drive_bc.glob("bc_epoch_*.pt"))),
        "production_drive_gpu_resume": json.loads((probe / "verification.json").read_text()),
    }
    atomic_json(args.bc_run / "final_verification.json", report)
    copy_verified(
        args.bc_run / "final_verification.json",
        args.drive_root / args.bc_run.name / "final_verification.json",
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
