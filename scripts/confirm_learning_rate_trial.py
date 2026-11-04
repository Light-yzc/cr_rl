"""Fresh-cohort confirmation of a development nominee; never auto-promote it."""

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
from finish_and_release import release_runtime
from run_learning_rate_trial import development_preference, verify_trial_completion

from cr_rl.evaluation.strategy_audit import summarize_strategy
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256
from cr_rl.training.runtime_supervision import process_alive


def confirmation_decision(comparisons):
    required = {"selected_vs_stable", "selected_vs_parent", "selected_vs_bc"}
    if set(comparisons) != required:
        raise ValueError("Missing confirmation opponent")
    checks = {}
    for name, row in comparisons.items():
        if row["games"] != 32 or row["pairs"] != 16 or row["native_parity_games"] != 32:
            raise ValueError("Confirmation requires all 16 paired matchups per opponent")
        # Conservative family-wise sign-test threshold; overlapping deck pairs
        # still limit generalization, and bootstrap intervals are descriptive.
        checks[name] = bool(
            row["paired_bootstrap_95_interval"][0] > 0.5
            and row["paired"]["swept_pair_sign_test_one_sided_p"] <= 0.05 / 3
        )
    return dict(
        eligible_for_independent_review=all(checks.values()),
        opponent_checks=checks,
        promotion="none; require independent backup/replay readback before registry change",
        rule="All three comparisons: paired bootstrap lower95>0.5 and sweep sign-test p<=0.05/3",
    )


def exported_actor(backup, generation):
    manifest = json.loads(
        (backup / f"checkpoints/update_{generation:06d}.manifest.json").read_text()
    )
    if manifest["update"] != generation:
        raise ValueError("Frozen source generation mismatch")
    row = next(r for r in manifest["files"] if r["path"].endswith(".actor.pt"))
    path = backup / row["path"]
    if file_sha256(path) != row["sha256"]:
        raise ValueError("Frozen actor checksum mismatch")
    return dict(path=str(path), sha256=row["sha256"], generation=generation)


def verify_saved_confirmation(backup, alive=process_alive):
    state = json.loads((backup / "confirmation_status.json").read_text())
    if state["phase"] != "completed" or alive(state["pid"]):
        raise ValueError("Confirmation is unfinished")
    result_path = backup / "confirmation_results.json"
    if file_sha256(result_path) != state["summary_sha256"]:
        raise ValueError("Confirmation summary changed")
    result = json.loads(result_path.read_text())
    protocol = json.loads((backup / "confirmation_protocol.json").read_text())
    binding = json.loads((backup / "frozen_policies.json").read_text())
    if result["protocol"] != protocol or result["policies_sha256"] != file_sha256(
        backup / "frozen_policies.json"
    ):
        raise ValueError("Confirmation protocol or policy binding changed")
    if (
        protocol["pair_starts"] != [24, 28, 32, 36]
        or protocol["games"] != 96
        or protocol["simulator_seed_base"] != 9000000
        or protocol["sampling_seed"] != 20261108
        or protocol["pairing_seed"] != 20261103
    ):
        raise ValueError("Confirmation cohort differs from preregistration")
    for policy in binding.values():
        if file_sha256(Path(policy["path"])) != policy["sha256"]:
            raise ValueError("Frozen confirmation actor changed")
    root = Path(__file__).resolve().parents[1]
    names = list(yaml.safe_load((root / protocol["decks"]).read_text())["decks"])
    for name, expected in result["comparisons"].items():
        opponent = name.removeprefix("selected_vs_")
        matches = []
        chunks = []
        for start in protocol["pair_starts"]:
            directory = backup / "evaluation" / name / f"pairs_{start}"
            report_path = directory / "evaluation.json"
            report = json.loads(report_path.read_text())
            verify_result(
                report,
                binding["selected"]["sha256"],
                binding[opponent]["sha256"],
                protocol,
                start,
                names,
            )
            audit = json.loads((directory / "execution_audit.json").read_text())
            if audit["evaluation_sha256"] != file_sha256(report_path) or len(audit["games"]) != 8:
                raise ValueError("Native confirmation audit changed")
            for index, (game, match) in enumerate(
                zip(audit["games"], report["matchups"], strict=True)
            ):
                if (
                    not game["native_outcome_parity"]
                    or any(game[k] != match[k] for k in ("pair", "learner_seat", "reward"))
                    or game["replay_sha256"] != file_sha256(directory / f"match_{index:03d}.json")
                ):
                    raise ValueError("Native confirmation replay mismatch")
            matches.extend(report["matchups"])
            chunks.append((report, audit))
        if any(expected[k] != v for k, v in summarize_matches(matches).items()):
            raise ValueError("Confirmation summary does not match actual games")
        if expected["strategy"] != summarize_strategy(chunks):
            raise ValueError("Confirmation strategy diagnostics do not match native executions")
    if result["decision"] != confirmation_decision(result["comparisons"]):
        raise ValueError("Confirmation decision changed")
    return dict(
        games=96,
        summary_sha256=state["summary_sha256"],
        registry_not_modified_by_this_evaluator=True,
    )


def release_guard(args):
    if not os.environ.get("TBE_RUNTIME_ADDR"):
        raise RuntimeError("Same-runtime management endpoint required")
    atomic_json(args.backup / "confirmation_release_ready.json", dict(pid=os.getpid()))
    began = time.monotonic()
    completed_since = None
    while time.monotonic() - began < 6 * 3600:
        p = args.backup / "confirmation_status.json"
        state = json.loads(p.read_text()) if p.exists() else {}
        if state.get("phase") in (
            "failed",
            "blocked_by_development_failure",
            "skipped_no_preference",
        ):
            return
        if state.get("phase") == "completed" and not process_alive(state["pid"]):
            if completed_since is None:
                completed_since = time.monotonic()
            receipt = args.backup / "confirmation_cpu_readback.json"
            if time.monotonic() - completed_since < 900 or not receipt.exists():
                time.sleep(15)
                continue
            independent = json.loads(receipt.read_text())
            if (
                independent.get("summary_sha256") != state["summary_sha256"]
                or independent.get("passed") is not True
            ):
                raise ValueError("Independent confirmation receipt mismatch")
            verified = verify_saved_confirmation(args.backup)
            atomic_json(args.backup / "confirmation_release_verified.json", verified)
            release_runtime()
            return
        time.sleep(30)
    raise TimeoutError("Confirmation release guard elapsed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--history-run", type=Path)
    parser.add_argument("--lr-trial", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--release-only", action="store_true")
    args = parser.parse_args()
    if args.release_only:
        return release_guard(args)
    if not all((args.history_run, args.lr_trial, args.out)):
        parser.error("History run, LR trial and output required")
    if args.out.exists() or args.backup.exists():
        raise ValueError("Inspect existing confirmation instead of restarting")
    args.out.mkdir(parents=True)
    args.backup.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    registry_path = args.history_run.parent / "stable_policy.json"
    registered_stable = json.loads(registry_path.read_text())
    if registered_stable["generation"] != 6:
        raise ValueError("Expected the previously verified generation6 reference")
    protocol = dict(
        games=96,
        pair_starts=[24, 28, 32, 36],
        simulator_seed_base=9000000,
        sampling_seed=20261108,
        pairing_seed=20261103,
        decks="configs/decks_eval.yaml",
        decks_sha256=file_sha256(root / "configs/decks_eval.yaml"),
        evaluation_code_sha256=file_sha256(root / "scripts/evaluate_policy.py"),
        stable_registry_sha256=file_sha256(registry_path),
        stable_actor_sha256=registered_stable["actor_sha256"],
        nomination="Frozen development preference; no nominee if inconclusive",
        criterion="All three lower95>0.5 and sweep p<=0.05/3; no automatic registry change",
        limitations=(
            "Fresh seeds and pairs disjoint from this LR development cohort; "
            "deck identities and some pairs appeared in earlier studies"
        ),
        created_unix=time.time(),
    )
    atomic_json(args.backup / "confirmation_protocol.json", protocol)
    active = None
    began = None

    def status(phase, **details):
        atomic_json(
            args.backup / "confirmation_status.json",
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
        with log.open("w") as stream:
            active = subprocess.Popen(
                command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
        while active.poll() is None:
            if time.monotonic() - began > 3600:
                raise TimeoutError("Confirmation evaluation budget reached")
            status(phase, **details)
            time.sleep(15)
        if active.returncode:
            raise RuntimeError("Confirmation subprocess failed")

    try:
        started = time.monotonic()
        while True:
            state = json.loads((args.lr_trial / "trial_status.json").read_text())
            if state["phase"] in ("failed", "blocked_by_source_failure"):
                status("blocked_by_development_failure")
                return
            if state["phase"] in ("completed", "skipped_source_improved") and not process_alive(
                state["pid"]
            ):
                break
            if time.monotonic() - started > 4 * 3600:
                raise TimeoutError("Development trial waiting budget elapsed")
            status("waiting_development")
            time.sleep(30)
        if state["phase"] == "completed":
            verify_trial_completion(args.lr_trial)
            development = json.loads((args.lr_trial / "trial_results.json").read_text())
            preference = development_preference(development["comparisons"])["preferred"]
            if preference == "inconclusive":
                status("skipped_no_preference")
                return
            selected = exported_actor(args.lr_trial / preference, 24)
            parent = exported_actor(args.history_run, 22)
        else:
            preference = "source_history22"
            selected = exported_actor(args.history_run, 22)
            parent = exported_actor(args.history_run.parent / "122m_ppo_continue_20261104", 14)
        began = time.monotonic()
        stable_path = args.history_run.parent / "stable_policy.json"
        stable = json.loads(stable_path.read_text())
        if (
            stable["generation"] != 6
            or file_sha256(stable_path) != protocol["stable_registry_sha256"]
            or stable["actor_sha256"] != protocol["stable_actor_sha256"]
            or file_sha256(Path(stable["actor"])) != stable["actor_sha256"]
        ):
            raise ValueError("Stable reference changed since registration")
        config = json.loads((args.history_run / "run_config.json").read_text())
        policies = dict(
            selected=selected,
            parent=parent,
            stable=dict(path=stable["actor"], sha256=stable["actor_sha256"], generation=6),
            bc=dict(
                path=str(
                    args.history_run.parent / "121m_bc_repair_20261104/repaired/bc/bc_latest.pt"
                ),
                sha256=config["source_checkpoint_sha256"],
            ),
        )
        for policy in policies.values():
            if file_sha256(Path(policy["path"])) != policy["sha256"]:
                raise ValueError("Frozen confirmation policy mismatch")
        atomic_json(args.backup / "frozen_policies.json", policies)
        atomic_json(
            args.backup / "nomination.json",
            dict(preference=preference, stable_registry_sha256=file_sha256(stable_path)),
        )
        names = list(yaml.safe_load((root / protocol["decks"]).read_text())["decks"])
        results = {}
        for opponent in ("stable", "parent", "bc"):
            name = "selected_vs_" + opponent
            chunks = []
            matches = []
            for start in protocol["pair_starts"]:
                target = args.out / "evaluation" / name / f"pairs_{start}"
                target.mkdir(parents=True)
                command = [
                    sys.executable,
                    "scripts/evaluate_policy.py",
                    "--policy",
                    policies["selected"]["path"],
                    "--opponent",
                    policies[opponent]["path"],
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
                    comparison=name,
                    pair_start=start,
                )
                report = target / "evaluation.json"
                data = json.loads(report.read_text())
                verify_result(
                    data, selected["sha256"], policies[opponent]["sha256"], protocol, start, names
                )
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
                    comparison=name,
                    pair_start=start,
                )
                audit = json.loads(audit_path.read_text())
                if (
                    audit["evaluation_sha256"] != file_sha256(report)
                    or len(audit["games"]) != 8
                    or not all(g["native_outcome_parity"] for g in audit["games"])
                ):
                    raise ValueError("Confirmation native audit failed")
                chunks.append((data, audit))
                matches.extend(data["matchups"])
                for path in target.iterdir():
                    if path.is_file():
                        copy_verified(
                            path,
                            args.backup / "evaluation" / path.relative_to(args.out / "evaluation"),
                        )
            row = summarize_matches(matches)
            row.update(native_parity_games=len(matches), strategy=summarize_strategy(chunks))
            results[name] = row
        decision = confirmation_decision(results)
        atomic_json(
            args.backup / "confirmation_results.json",
            dict(
                protocol=protocol,
                comparisons=results,
                decision=decision,
                policies_sha256=file_sha256(args.backup / "frozen_policies.json"),
            ),
        )
        previous_journal = args.lr_trial / "traj.md"
        journal = (
            previous_journal.read_text()
            if previous_journal.exists()
            else (root / "traj.md").read_text()
            if (root / "traj.md").exists()
            else "# Optimization trajectory\n"
        )
        journal += "\n\n## Independent confirmation (automatic record)\n"
        for name, row in results.items():
            journal += (
                f"- {name}: {row['wins']}W/{row['losses']}L/{row['draws']}D, "
                f"score={row['score']:.4f}, paired95={row['paired_bootstrap_95_interval']}.\n"
            )
        journal += "\nDecision: " + json.dumps(decision) + ". Registry remains unchanged.\n"
        (args.backup / "traj.md").write_text(journal)
        status("completed", summary_sha256=file_sha256(args.backup / "confirmation_results.json"))
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
