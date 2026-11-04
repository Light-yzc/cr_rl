"""Fixed continuation learning curve; never select a checkpoint using test wins."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from evaluate_ppo_pilot import summarize_matches

from cr_rl.evaluation.cohort import paired_match_specs
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def verify_result(result, policy_sha, opponent_sha, protocol, start, names):
    expected = dict(
        policy_sha256=policy_sha,
        opponent_sha256=opponent_sha,
        deck_pool_sha256=protocol["decks_sha256"],
        evaluation_code_sha256=protocol["evaluation_code_sha256"],
        evaluation_code_file_sha256_at_end=protocol["evaluation_code_sha256"],
        wait_logit_bias=0,
        opponent_wait_logit_bias=0,
        simulator_seed_base=protocol["simulator_seed_base"],
        pairing_seed=protocol["pairing_seed"],
        sampling_seed=protocol["sampling_seed"] + start,
        pair_start=start,
        pair_count=4,
        games=8,
        invalid_games=0,
    )
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("Continuation evaluation differs from frozen protocol")
    specs = paired_match_specs(
        names,
        4,
        start,
        simulator_seed_base=protocol["simulator_seed_base"],
        pairing_seed=protocol["pairing_seed"],
    )
    actual = {(m["pair"], m["learner_seat"]): m for m in result["matchups"]}
    if len(actual) != 8 or len(result["matchups"]) != 8:
        raise ValueError("Incomplete or duplicated paired evaluation")
    for spec in specs:
        game = actual.get((spec["pair"], spec["learner_seat"]))
        if (
            not game
            or not game["valid"]
            or any(game[k] != spec[k] for k in ("seed", "deck0", "deck1"))
        ):
            raise ValueError("Continuation physical seed/deck/seat mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--generations", type=int, nargs=2, default=[10, 14])
    parser.add_argument("--simulator-seed-base", type=int, default=6000000)
    parser.add_argument("--sampling-seed", type=int, default=20261105)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (args.backup / "evaluation_protocol.json").exists():
        raise ValueError("Evaluator already registered; inspect existing worker")
    protocol = dict(
        generations=args.generations,
        pair_starts=[16, 20],
        pairs_per_chunk=4,
        simulator_seed_base=args.simulator_seed_base,
        sampling_seed=args.sampling_seed,
        pairing_seed=20261103,
        decks="configs/decks_eval.yaml",
        decks_sha256=file_sha256(root / "configs/decks_eval.yaml"),
        evaluation_code_sha256=file_sha256(root / "scripts/evaluate_policy.py"),
        opponents=["generation6", "bc5096"],
        games=64,
        selection=f"Report both frozen generations; final {args.generations[-1]} fixed in advance",
        observed_unix=time.time(),
    )
    atomic_json(args.backup / "evaluation_protocol.json", protocol)
    began = time.monotonic()

    def status(phase, **details):
        atomic_json(
            args.backup / "evaluation_status.json",
            dict(phase=phase, pid=os.getpid(), observed_unix=time.time(), **details),
        )

    while True:
        state = json.loads((args.backup / "continuation_status.json").read_text())
        if state["phase"] == "failed":
            status("blocked_by_training_failure")
            return
        if state["phase"] == "training_completed":
            break
        if time.monotonic() - began > 4 * 3600:
            raise TimeoutError("Continuation did not finish in the registered waiting budget")
        status("waiting_training")
        time.sleep(30)
    out = args.run / "learning_curve"
    out.mkdir(parents=True, exist_ok=False)
    original_manifest = json.loads((args.original / "latest.json").read_text())
    original_actor = next(r for r in original_manifest["files"] if r["path"].endswith(".actor.pt"))
    bc_sha = json.loads((args.original / "run_config.json").read_text())["source_checkpoint_sha256"]
    opponents = {
        "generation6": (args.original / original_actor["path"], original_actor["sha256"]),
        "bc5096": (
            args.original.parent / "121m_bc_repair_20261104/repaired/bc/bc_latest.pt",
            bc_sha,
        ),
    }
    names = list(yaml.safe_load((root / protocol["decks"]).read_text())["decks"])
    comparisons = {}
    try:
        for generation in protocol["generations"]:
            manifest = json.loads(
                (args.backup / f"checkpoints/update_{generation:06d}.manifest.json").read_text()
            )
            actor = next(r for r in manifest["files"] if r["path"].endswith(".actor.pt"))
            policy = args.backup / actor["path"]
            if file_sha256(policy) != actor["sha256"]:
                raise ValueError("Frozen candidate SHA mismatch")
            for opponent_name, (opponent, opponent_sha) in opponents.items():
                if file_sha256(opponent) != opponent_sha:
                    raise ValueError("Frozen baseline SHA mismatch")
                matches, native = [], []
                for start in protocol["pair_starts"]:
                    target = out / f"generation{generation}_vs_{opponent_name}" / f"pairs_{start}"
                    target.mkdir(parents=True)
                    report = target / "evaluation.json"
                    status(
                        "evaluating",
                        generation=generation,
                        opponent=opponent_name,
                        pair_start=start,
                    )
                    command = [
                        sys.executable,
                        "scripts/evaluate_policy.py",
                        "--policy",
                        str(policy),
                        "--opponent",
                        str(opponent),
                        "--decks",
                        protocol["decks"],
                        "--pairs",
                        "4",
                        "--pair-start",
                        str(start),
                        "--sampling-seed",
                        str(protocol["sampling_seed"]),
                        "--simulator-seed-base",
                        str(protocol["simulator_seed_base"]),
                        "--pairing-seed",
                        str(protocol["pairing_seed"]),
                        "--frame-cache",
                        "--device",
                        "cuda",
                        "--out",
                        str(target),
                    ]
                    with (target / "evaluation.log").open("w") as log:
                        subprocess.run(
                            command,
                            cwd=root,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=1200,
                        )
                    result = json.loads(report.read_text())
                    verify_result(result, actor["sha256"], opponent_sha, protocol, start, names)
                    execution = target / "execution_audit.json"
                    with (target / "audit.log").open("w") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                "scripts/audit_policy_replays.py",
                                "--evaluation",
                                str(report),
                                "--out",
                                str(execution),
                                "--java-home",
                                os.environ["JAVA_HOME"],
                            ],
                            cwd=root,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=300,
                        )
                    audit = json.loads(execution.read_text())
                    if (
                        audit["evaluation_sha256"] != file_sha256(report)
                        or not all(g["native_outcome_parity"] for g in audit["games"])
                        or len(audit["games"]) != 8
                    ):
                        raise ValueError("Native replay audit does not match frozen evaluation")
                    matches.extend(result["matchups"])
                    native.extend(audit["games"])
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/prepare_replay_archive.py",
                            str(target),
                            "--policy-label",
                            f"PPO generation{generation}",
                            "--opponent-label",
                            opponent_name,
                        ],
                        cwd=root,
                        check=True,
                        timeout=60,
                    )
                    for path in target.iterdir():
                        if path.is_file():
                            copy_verified(
                                path, args.backup / "learning_curve" / path.relative_to(out)
                            )
                summary = summarize_matches(matches)
                summary["policy_sha256"], summary["opponent_sha256"] = actor["sha256"], opponent_sha
                summary["mean_full_elixir_wait_seconds"] = float(
                    sum(
                        g["decision_audit"]["learner"]["full_elixir_wait_legal_seconds"]
                        for g in native
                    )
                    / len(native)
                )
                summary["native_parity_games"] = len(native)
                comparisons[f"generation{generation}_vs_{opponent_name}"] = summary
        atomic_json(
            args.backup / "learning_curve_summary.json",
            dict(protocol=protocol, comparisons=comparisons),
        )
        status("completed", summary_sha256=file_sha256(args.backup / "learning_curve_summary.json"))
    except BaseException as error:
        status("failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
