"""Audit and evaluate the bounded exploratory PPO run against its exact BC start."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from cr_rl.evaluation.cohort import heldout_pairings
from cr_rl.evaluation.paired import paired_outcomes
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def summarize_matches(matches):
    pairs = {}
    for row in matches:
        if not row.get("valid"):
            raise ValueError("Invalid evaluation game")
        pair = pairs.setdefault(row["pair"], {})
        seat = row["learner_seat"]
        if seat in pair or seat not in (0, 1) or row["reward"] not in (-1, 0, 1):
            raise ValueError("Invalid paired-game identity or reward")
        pair[seat] = row
    if not pairs:
        raise ValueError("No paired games")
    scores = []
    for seats in pairs.values():
        if set(seats) != {0, 1}:
            raise ValueError("Missing paired seat swap")
        a, b = seats[0], seats[1]
        if any(a[key] != b[key] for key in ("seed", "deck0", "deck1")):
            raise ValueError("Physical decks or seed differ within a pair")
        scores.append(((a["reward"] + 1) / 2 + (b["reward"] + 1) / 2) / 2)
    scores = np.asarray(scores)
    rng = np.random.default_rng(20261104)
    bootstrap = scores[rng.integers(len(scores), size=(10000, len(scores)))].mean(1)
    low, high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    paired = paired_outcomes({"matchups": matches})
    return dict(
        games=len(matches),
        pairs=len(pairs),
        wins=sum(x["reward"] > 0 for x in matches),
        losses=sum(x["reward"] < 0 for x in matches),
        draws=sum(x["reward"] == 0 for x in matches),
        score=float(scores.mean()),
        paired_bootstrap_95_interval=[low, high],
        paired=paired,
        evidence_of_improvement=(low > 0.5 and paired["swept_pair_sign_test_one_sided_p"] <= 0.05),
        limitation=(
            "Small fixed-seed sample; paired bootstrap interval is exploratory, "
            "not proof of general strength"
        ),
    )


def validate_chunk(result, *, policy_sha, opponent_sha, decks_sha, code_sha, expected_pairs, start):
    expected = {
        "policy_sha256": policy_sha,
        "opponent_sha256": opponent_sha,
        "deck_pool_sha256": decks_sha,
        "evaluation_code_sha256": code_sha,
        "evaluation_code_file_sha256_at_end": code_sha,
        "wait_logit_bias": 0,
        "opponent_wait_logit_bias": 0,
        "pair_start": start,
        "pair_count": len(expected_pairs),
        "sampling_seed": 20261103 + start,
        "games": len(expected_pairs) * 2,
        "invalid_games": 0,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Evaluation chunk differs from registered weights/cohort/protocol")
    actual = {(m["pair"], m["learner_seat"]): m for m in result["matchups"]}
    if len(actual) != len(result["matchups"]) or len(actual) != len(expected_pairs) * 2:
        raise ValueError("Missing or duplicated evaluation identities")
    for pair, deck0, deck1 in expected_pairs:
        for seat in (0, 1):
            game = actual.get((pair, seat))
            if not game or any(
                game.get(key) != value
                for key, value in {
                    "seed": 1000000 + pair,
                    "deck0": deck0,
                    "deck1": deck1,
                    "valid": True,
                }.items()
            ):
                raise ValueError("Evaluation physical decks/seeds or validity differ")
            if game["info"]["fault"] or game["info"]["truncated"]:
                raise ValueError("Native evaluation fault or truncation")
    return result["matchups"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, backup = args.run, args.backup
    out = run / "evaluation"
    out.mkdir(exist_ok=True)
    manifest = json.loads((backup / "latest.json").read_text())
    if manifest["update"] != 6:
        raise ValueError("Evaluate only the completed six-update pilot")
    for record in manifest["files"]:
        path = backup / record["path"]
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Published file integrity failure: {record['path']}")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    if [x["update"] for x in rows] != list(range(1, 7)):
        raise ValueError("Missing or duplicated PPO update records")
    if any(x["faulted_samples"] for x in rows) or not sum(x["optimizer_steps"] for x in rows):
        raise ValueError("Faulted rollout or Actor never updated")
    if not sum(x["terminal_games"] for x in rows):
        raise ValueError("No complete training games")
    if any(not math.isfinite(value) for row in rows for value in row["metrics"].values()):
        raise ValueError("Nonfinite training metric")
    restart_boundaries = []
    for path in sorted(backup.glob("run_config.before_resume_*.json")):
        generation = int(path.stem.rsplit("_", 1)[-1])
        restart_boundaries.append(generation)
    for before, after in zip(rows, rows[1:], strict=False):
        if before["episode_steps_at_update_end"] != after["episode_steps_at_update_start"]:
            if before["update"] not in restart_boundaries or any(
                after["episode_steps_at_update_start"]
            ):
                raise ValueError("Unexplained simulator continuity failure")
    audit = dict(
        updates=6,
        actor_steps=sum(x["optimizer_steps"] for x in rows),
        critic_steps=sum(x["critic_optimizer_steps"] for x in rows),
        complete_games=sum(x["terminal_games"] for x in rows),
        faulted_samples=0,
        declared_process_restart_boundaries=restart_boundaries,
        warmup_final=rows[-1]["critic_warmup"],
    )
    atomic_json(out / "training_audit.json", audit)
    export = next(row for row in manifest["files"] if row["path"].endswith(".actor.pt"))
    if file_sha256(run / "actor_export.pt") != export["sha256"]:
        raise ValueError("Local evaluated Actor differs from published update 6 export")
    source = root / "artifacts/121m_bc_repair_20261104/repaired/bc.pt"
    initial = json.loads((run / "run_config.json").read_text())["source_checkpoint_sha256"]
    if file_sha256(source) != initial:
        raise ValueError("BC opponent differs from the actual initialization")
    protocol = dict(
        policy_sha256=file_sha256(run / "actor_export.pt"),
        opponent_sha256=initial,
        heldout_pair_start=16,
        heldout_pairs=16,
        chunk_pairs=4,
        trained_deck_pairs=3,
        wait_bias=0,
        sampling_seed=20261103,
        primary="heldout score against exact unmodified BC initialization",
        no_automatic_formal_promotion=True,
        evaluation_code_sha256=file_sha256(root / "scripts/evaluate_policy.py"),
        deck_pool_sha256={
            name: file_sha256(root / name)
            for name in ("configs/decks_eval.yaml", "configs/decks_initial3.yaml")
        },
        improvement_rule="paired bootstrap lower bound >0.5 AND one-sided sweep sign-test p<=0.05",
        native_replay_execution_audit=True,
    )
    atomic_json(out / "protocol.json", protocol)
    copy_verified(out / "protocol.json", backup / "evaluation/protocol.json")
    cohorts = [
        ("heldout", "configs/decks_eval.yaml", range(16, 32, 4), 4),
        ("trained", "configs/decks_initial3.yaml", [0], 3),
    ]
    summaries = {}
    for name, decks, starts, count in cohorts:
        matches, native_games = [], []
        for start in starts:
            target = out / name / f"pairs_{start:03d}"
            report = target / "evaluation.json"
            if not report.exists():
                target.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "scripts/evaluate_policy.py",
                    "--policy",
                    str(run / "actor_export.pt"),
                    "--opponent",
                    str(source),
                    "--decks",
                    decks,
                    "--pairs",
                    str(count),
                    "--pair-start",
                    str(start),
                    "--frame-cache",
                    "--device",
                    "cuda",
                    "--out",
                    str(target),
                ]
                with (target / "evaluation.log").open("a") as log:
                    child = subprocess.Popen(
                        command, cwd=root, stdout=log, stderr=subprocess.STDOUT
                    )
                    began = time.monotonic()
                    while child.poll() is None:
                        status = dict(
                            phase="evaluating",
                            cohort=name,
                            pair_start=start,
                            child_pid=child.pid,
                            pid=os.getpid(),
                            observed_unix=time.time(),
                        )
                        atomic_json(out / "status.json", status)
                        copy_verified(out / "status.json", backup / "evaluation/status.json")
                        print(json.dumps(status), flush=True)
                        if time.monotonic() - began > 1200:
                            child.terminate()
                            child.wait(timeout=30)
                            raise TimeoutError("Evaluation chunk budget elapsed")
                        time.sleep(15)
                if child.returncode:
                    raise RuntimeError(f"Evaluation failed: {name}/{start}")
            result = json.loads(report.read_text())
            expected_pairs = heldout_pairings(
                list(yaml.safe_load((root / decks).read_text())["decks"]), count, start
            )
            matches.extend(
                validate_chunk(
                    result,
                    policy_sha=protocol["policy_sha256"],
                    opponent_sha=initial,
                    decks_sha=protocol["deck_pool_sha256"][decks],
                    code_sha=protocol["evaluation_code_sha256"],
                    expected_pairs=expected_pairs,
                    start=start,
                )
            )
            execution = target / "execution_audit.json"
            if not execution.exists():
                with (target / "execution_audit.log").open("a") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/audit_policy_replays.py",
                            "--evaluation",
                            str(report),
                            "--out",
                            str(execution),
                            "--java-home",
                            "/usr/lib/jvm/java-21-openjdk-amd64",
                        ],
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=300,
                    )
            native = json.loads(execution.read_text())
            if native["evaluation_sha256"] != file_sha256(report) or not all(
                game["native_outcome_parity"] for game in native["games"]
            ):
                raise ValueError("Native replay execution audit differs from evaluation")
            native_games.extend(native["games"])
            archive = target / "all_matches.jsonl"
            if not archive.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/prepare_replay_archive.py",
                        str(target),
                        "--policy-label",
                        "PPO update6",
                        "--opponent-label",
                        "BC repaired5096",
                    ],
                    cwd=root,
                    check=True,
                    timeout=60,
                )
            for path in target.iterdir():
                if path.is_file():
                    copy_verified(path, backup / "evaluation" / path.relative_to(out))
        summaries[name] = summarize_matches(matches)
        summaries[name]["matchups"] = matches
        summaries[name]["execution"] = {
            role: dict(
                mean_successful_cards=float(
                    np.mean([game[role]["successful_card_plays"] for game in native_games])
                ),
                mean_paid_card_elixir=float(
                    np.mean([game[role]["paid_card_elixir_total"] for game in native_games])
                ),
                mean_refused_card_commands=float(
                    np.mean([game[role]["refused_card_commands"] for game in native_games])
                ),
                mean_full_elixir_wait_seconds=float(
                    np.mean(
                        [
                            game["decision_audit"][role]["full_elixir_wait_legal_seconds"]
                            for game in native_games
                        ]
                    )
                ),
                max_continuous_full_elixir_wait_seconds=max(
                    game["decision_audit"][role]["longest_full_elixir_wait_legal_seconds"]
                    for game in native_games
                ),
                successful_card_histogram={
                    card: sum(
                        game[role]["successful_card_histogram"].get(card, 0)
                        for game in native_games
                    )
                    for card in sorted(
                        {
                            card
                            for game in native_games
                            for card in game[role]["successful_card_histogram"]
                        }
                    )
                },
            )
            for role in ("learner", "opponent")
        }
    result = dict(
        training_audit=audit,
        protocol=protocol,
        comparisons=summaries,
        interpretation="Frozen exploratory evaluation, not a passed BC promotion gate",
    )
    atomic_json(out / "summary.json", result)
    for name in ["summary.json", "training_audit.json"]:
        copy_verified(out / name, backup / "evaluation" / name)
    atomic_json(
        out / "completed.json",
        dict(observed_unix=time.time(), summary_sha256=file_sha256(out / "summary.json")),
    )
    copy_verified(out / "completed.json", backup / "evaluation/completed.json")
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "matchups"} for k, v in summaries.items()}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
