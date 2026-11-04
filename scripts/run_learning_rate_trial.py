"""Conditional, fixed-budget Actor-LR control trial after the history evaluation."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
from evaluate_continued_ppo import verify_result
from evaluate_ppo_pilot import summarize_matches

from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256
from cr_rl.training.runtime_supervision import process_alive


def comparisons():
    return [
        dict(
            name="candidate_vs_control",
            policy="candidate",
            opponent="control",
            starts=[0, 4, 8, 12],
        ),
        dict(name="candidate_vs_source", policy="candidate", opponent="source", starts=[16, 20]),
        dict(name="control_vs_source", policy="control", opponent="source", starts=[16, 20]),
        dict(name="candidate_vs_bc", policy="candidate", opponent="bc", starts=[16, 20]),
        dict(name="control_vs_bc", policy="control", opponent="bc", starts=[16, 20]),
    ]


def trial_needed(summary):
    rows = summary["comparisons"]
    return not all(
        rows[f"generation22_vs_{opponent}"]["evidence_of_improvement"]
        for opponent in ("generation6", "bc5096")
    )


def development_preference(rows):
    direct = rows["candidate_vs_control"]["score"]
    if (
        direct >= 0.55
        and rows["candidate_vs_source"]["score"] >= 0.5
        and rows["candidate_vs_bc"]["score"] >= rows["control_vs_bc"]["score"] - 0.0625
    ):
        preferred = "candidate"
    elif direct <= 0.45 and rows["control_vs_source"]["score"] >= 0.5:
        preferred = "control"
    else:
        preferred = "inconclusive"
    return dict(
        preferred=preferred, promotion="none; development trial needs independent confirmation"
    )


def verify_trial_completion(backup):
    status = json.loads((backup / "trial_status.json").read_text())
    if status["phase"] != "completed" or process_alive(status.get("pid")):
        raise ValueError("LR trial is not completed")
    result = json.loads((backup / "trial_results.json").read_text())
    if file_sha256(backup / "trial_results.json") != status["summary_sha256"]:
        raise ValueError("Trial summary checksum mismatch")
    if set(result["comparisons"]) != {r["name"] for r in comparisons()}:
        raise ValueError("Missing trial comparison")
    if sum(r["games"] for r in result["comparisons"].values()) != 96:
        raise ValueError("Incomplete fixed 96-game trial")
    if any(r["native_parity_games"] != r["games"] for r in result["comparisons"].values()):
        raise ValueError("Trial native replay audits missing")
    protocol = json.loads((backup / "trial_protocol.json").read_text())
    if result["protocol"] != protocol:
        raise ValueError("Trial protocol changed")
    if (
        protocol["comparisons"] != comparisons()
        or protocol["games"] != 96
        or protocol["source_generation"] != 22
        or protocol["target_generation"] != 24
        or protocol["updates_per_branch"] != 2
        or protocol["actor_learning_rates"] != dict(control=2e-5, candidate=1e-5)
    ):
        raise ValueError("Trial budget differs from the declared control comparison")
    expected_actors = {}
    for branch in ("control", "candidate"):
        base = backup / branch
        manifest = json.loads((base / "latest.json").read_text())
        if (
            manifest["update"] != 24
            or file_sha256(base / manifest["pool"]) != manifest["pool_sha256"]
        ):
            raise ValueError("Branch checkpoint or pool mismatch")
        for row in manifest["files"]:
            path = base / row["path"]
            if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
                raise ValueError("Branch backup checksum mismatch")
        exported = next(r for r in manifest["files"] if r["path"].endswith(".actor.pt"))
        expected_actors[branch] = exported["sha256"]
    binding = json.loads((backup / "source_binding.json").read_text())
    if file_sha256(backup / "source_binding.json") != result["source_binding_sha256"]:
        raise ValueError("Source binding checksum mismatch")
    expected_actors.update(source=binding["actor_sha256"], bc=binding["bc_sha256"])
    root = Path(__file__).resolve().parents[1]
    names = list(yaml.safe_load((root / protocol["decks"]).read_text())["decks"])
    for comparison in comparisons():
        name = comparison["name"]
        stored = result["comparisons"][name]
        policy_sha, opponent_sha = (
            expected_actors[comparison["policy"]],
            expected_actors[comparison["opponent"]],
        )
        if stored["policy_sha256"] != policy_sha or stored["opponent_sha256"] != opponent_sha:
            raise ValueError("Comparison was not against the frozen declared policies")
        matches = []
        for start in comparison["starts"]:
            directory = backup / "evaluation" / name / f"pairs_{start}"
            report_path = directory / "evaluation.json"
            report = json.loads(report_path.read_text())
            verify_result(report, policy_sha, opponent_sha, protocol, start, names)
            audit = json.loads((directory / "execution_audit.json").read_text())
            if audit["evaluation_sha256"] != file_sha256(report_path) or len(audit["games"]) != 8:
                raise ValueError("Native audit provenance mismatch")
            for index, (game, match) in enumerate(
                zip(audit["games"], report["matchups"], strict=True)
            ):
                if (
                    not game["native_outcome_parity"]
                    or any(game[k] != match[k] for k in ("pair", "learner_seat", "reward"))
                    or file_sha256(directory / f"match_{index:03d}.json") != game["replay_sha256"]
                ):
                    raise ValueError("Native replay checksum or outcome mismatch")
            matches.extend(report["matchups"])
        if any(stored[k] != v for k, v in summarize_matches(matches).items()):
            raise ValueError("Trial aggregate does not match its actual games")
    if result["decision"] != development_preference(result["comparisons"]):
        raise ValueError("Trial preference differs from the frozen decision rule")
    return dict(final_generation=24, games=96, summary_sha256=status["summary_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.out.exists() or args.backup.exists():
        raise ValueError("Inspect existing trial instead of restarting it")
    args.out.mkdir(parents=True)
    args.backup.mkdir(parents=True)
    protocol = dict(
        source_generation=22,
        target_generation=24,
        updates_per_branch=2,
        actor_learning_rates=dict(control=2e-5, candidate=1e-5),
        budget="equal environment decisions and effective batch; report actual learner counts",
        comparisons=comparisons(),
        games=96,
        simulator_seed_base=8000000,
        sampling_seed=20261107,
        pairing_seed=20261103,
        decks="configs/decks_eval.yaml",
        decks_sha256=file_sha256(root / "configs/decks_eval.yaml"),
        evaluation_code_sha256=file_sha256(root / "scripts/evaluate_policy.py"),
        trigger="Run only if final22 lacks fixed development evidence against generation6 or BC",
        promotion="No production promotion from this development trial",
        observed_unix=time.time(),
    )
    atomic_json(args.backup / "trial_protocol.json", protocol)
    active = None
    began = None

    def status(phase, **details):
        atomic_json(
            args.backup / "trial_status.json",
            dict(
                phase=phase,
                pid=os.getpid(),
                child_pid=active.pid if active else None,
                observed_unix=time.time(),
                **details,
            ),
        )

    def run(command, log, phase, **details):
        nonlocal active
        mirror = None
        with log.open("w") as stream:
            active = subprocess.Popen(
                command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
        while active.poll() is None:
            if time.monotonic() - began > args.budget_seconds:
                raise TimeoutError("Fixed LR-trial budget reached")
            status(phase, **details)
            if phase == "training_branch" and mirror is None and os.environ.get("WANDB_API_KEY"):
                branch = details["branch"]
                local = args.out / branch / "state"
                remote = args.backup / branch
                state_path = remote / "continuation_status.json"
                branch_state = json.loads(state_path.read_text()) if state_path.exists() else {}
                if (
                    branch_state.get("phase") in ("numeric_audit", "training", "training_completed")
                    and (local / "progress.jsonl").exists()
                ):
                    with (remote / "metric_mirror.log").open("w") as metrics:
                        mirror = subprocess.Popen(
                            [
                                sys.executable,
                                "scripts/stream_training_metrics.py",
                                "--run",
                                str(local),
                                "--backup",
                                str(remote),
                                "--min-update",
                                "23",
                                "--name",
                                f"122m_lr_{branch}_20261105",
                            ],
                            cwd=root,
                            stdout=metrics,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    atomic_json(remote / "metric_mirror_pid.json", mirror.pid)
            time.sleep(15)
        if active.returncode:
            raise RuntimeError(f"Trial child failed; inspect {log}")
        if mirror is not None:
            try:
                mirror.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass

    try:
        wait_started = time.monotonic()
        while True:
            p = args.source / "evaluation_status.json"
            source_state = json.loads(p.read_text()) if p.exists() else {}
            if source_state.get("phase") in ("failed", "blocked_by_training_failure"):
                status("blocked_by_source_failure")
                return
            if source_state.get("phase") == "completed" and not process_alive(
                source_state.get("pid")
            ):
                break
            if time.monotonic() - wait_started > 4 * 3600:
                raise TimeoutError("Source evaluation did not finish")
            status("waiting_source_evaluation")
            time.sleep(30)
        source_summary = json.loads((args.source / "learning_curve_summary.json").read_text())
        if (
            file_sha256(args.source / "learning_curve_summary.json")
            != source_state["summary_sha256"]
        ):
            raise ValueError("Source evaluation checksum mismatch")
        if not trial_needed(source_summary):
            status(
                "skipped_source_improved",
                reason="Prefer independent confirmation over an unnecessary LR change",
            )
            return
        began = time.monotonic()
        source_manifest = json.loads((args.source / "latest.json").read_text())
        if source_manifest["update"] != 22:
            raise ValueError("Expected fixed generation22 source")
        export = next(r for r in source_manifest["files"] if r["path"].endswith(".actor.pt"))
        actors = {"source": (args.source / export["path"], export["sha256"])}
        run_config = json.loads((args.source / "run_config.json").read_text())
        actors["bc"] = (
            args.source.parent / "121m_bc_repair_20261104/repaired/bc/bc_latest.pt",
            run_config["source_checkpoint_sha256"],
        )
        atomic_json(
            args.backup / "source_binding.json",
            dict(
                source_backup=str(args.source),
                source_manifest_sha256=file_sha256(args.source / "latest.json"),
                actor_sha256=export["sha256"],
                bc_sha256=actors["bc"][1],
            ),
        )
        for branch, lr in protocol["actor_learning_rates"].items():
            local = args.out / branch
            remote = args.backup / branch
            command = [
                sys.executable,
                "scripts/continue_ppo_training.py",
                "--original",
                str(args.source),
                "--out",
                str(local),
                "--backup",
                str(remote),
                "--source-generation",
                "22",
                "--additional-updates",
                "2",
                "--resume-actor-learning-rate",
                str(lr),
                "--new-wandb-run",
                "--frame-encoder-fp32",
                "--budget-seconds",
                "2400",
            ]
            run(command, args.backup / f"{branch}_controller.log", "training_branch", branch=branch)
            manifest = json.loads((remote / "latest.json").read_text())
            if manifest["update"] != 24:
                raise ValueError("Branch budget did not complete")
            summary = json.loads((remote / "summary.json").read_text())
            if summary["learning_rates"]["actor"] != [lr]:
                raise ValueError("Requested Actor LR was not actually applied")
            actor = next(r for r in manifest["files"] if r["path"].endswith(".actor.pt"))
            actors[branch] = (remote / actor["path"], actor["sha256"])
        names = list(yaml.safe_load((root / protocol["decks"]).read_text())["decks"])
        results = {}
        for comparison in protocol["comparisons"]:
            policy, policy_sha = actors[comparison["policy"]]
            opponent, opponent_sha = actors[comparison["opponent"]]
            if file_sha256(policy) != policy_sha or file_sha256(opponent) != opponent_sha:
                raise ValueError("Frozen evaluation weights changed")
            matches = []
            native = []
            for start in comparison["starts"]:
                target = args.out / "evaluation" / comparison["name"] / f"pairs_{start}"
                target.mkdir(parents=True)
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
                run(
                    command,
                    target / "evaluation.log",
                    "evaluating",
                    comparison=comparison["name"],
                    pair_start=start,
                )
                report = target / "evaluation.json"
                data = json.loads(report.read_text())
                verify_result(data, policy_sha, opponent_sha, protocol, start, names)
                audit_path = target / "execution_audit.json"
                run(
                    [
                        sys.executable,
                        "scripts/audit_policy_replays.py",
                        "--evaluation",
                        str(report),
                        "--out",
                        str(audit_path),
                        "--java-home",
                        os.environ["JAVA_HOME"],
                    ],
                    target / "audit.log",
                    "auditing",
                    comparison=comparison["name"],
                    pair_start=start,
                )
                audit = json.loads(audit_path.read_text())
                if (
                    audit["evaluation_sha256"] != file_sha256(report)
                    or len(audit["games"]) != 8
                    or not all(g["native_outcome_parity"] for g in audit["games"])
                ):
                    raise ValueError("Trial native replay mismatch")
                matches.extend(data["matchups"])
                native.extend(audit["games"])
                for path in target.iterdir():
                    if path.is_file():
                        copy_verified(
                            path,
                            args.backup / "evaluation" / path.relative_to(args.out / "evaluation"),
                        )
            result = summarize_matches(matches)
            result.update(
                policy_sha256=policy_sha,
                opponent_sha256=opponent_sha,
                native_parity_games=len(native),
            )
            results[comparison["name"]] = result
        decision = development_preference(results)
        atomic_json(
            args.backup / "trial_results.json",
            dict(
                protocol=protocol,
                comparisons=results,
                decision=decision,
                source_binding_sha256=file_sha256(args.backup / "source_binding.json"),
            ),
        )
        journal = args.backup / "traj.md"
        header = (
            (root / "traj.md").read_text()
            if (root / "traj.md").exists()
            else "# Optimization trajectory\n"
        )
        lines = [
            "\n\n## Actor LR development trial (automatic record)\n",
            "Both branches resumed generation22, two updates each; control2e-5 / candidate1e-5.\n",
            "| Comparison | W-L-D | Score | Paired 95% interval |\n",
            "|---|---:|---:|---:|\n",
        ]
        for name, row in results.items():
            lines.append(
                f"| {name} | {row['wins']}-{row['losses']}-{row['draws']} | "
                f"{row['score']:.4f} | {row['paired_bootstrap_95_interval']} |\n"
            )
        lines.append(
            f"\nDevelopment preference: {decision['preferred']}. {decision['promotion']}.\n"
        )
        journal.write_text(header + "".join(lines))
        status("completed", summary_sha256=file_sha256(args.backup / "trial_results.json"))
    except BaseException as error:
        if active and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=10)
        status("failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
