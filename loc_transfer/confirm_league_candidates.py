"""Confirm the two strongest non-anchor league candidates on fresh paired games.

This is an explicit evaluation job.  It never edits the screening evidence or
the prepared baseline pool.  Each worker owns one fixed V100 and evaluates its
candidate's two fresh chunks sequentially, keeping peak host memory bounded.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "artifacts/ppo_convergence_trials_20261108/opponent_screen"
OUT = ROOT / "artifacts/ppo_convergence_trials_20261108/opponent_confirm"
BASELINE_POOL = ROOT / "artifacts/ppo_convergence_trials_20261108/baseline/opponent_pool.json"
BASE = (ROOT / "artifacts/local_dual_v100").resolve()
ACTOR = BASE / "actor_export.pt"
V100_GPUS = (
    "GPU-95f9c3e4-242f-9098-3c63-dc4dc1557d69",
    "GPU-cb00ba20-7367-bd99-c5d6-0824adaaa5b2",
)
GENERATIONS = (0, 6, 119, 130, 133, 136, 139, 142)


def policy_id(generation: int) -> int:
    return 0 if generation == 0 else (1 << 30) + generation


PAIRING_SEED = 20261201
SAMPLING_SEED = 20261202
SIMULATOR_SEED_BASE = 11010000
PAIR_STARTS = (8, 16)
PAIRS_PER_CHUNK = 8
GAMES_PER_CHUNK = PAIRS_PER_CHUNK * 2
TOTAL_GAMES = GAMES_PER_CHUNK * len(PAIR_STARTS)
PAIRED_PROTOCOL = "fixed physical decks/seeds, paired policy-seat swap, stochastic sampling"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _report_valid(report: dict, *, candidate_sha: str, opponent: Path, start: int) -> bool:
    """Check a complete evaluator result before allowing it to be reused."""
    if not isinstance(report, dict):
        return False
    if (
        report.get("policy_sha256") != candidate_sha
        or report.get("opponent_sha256") != digest(opponent)
        or report.get("pairing_seed") != PAIRING_SEED
        or report.get("sampling_seed") != SAMPLING_SEED + start
        or report.get("simulator_seed_base") != SIMULATOR_SEED_BASE
        or report.get("pair_start") != start
        or report.get("pair_count") != PAIRS_PER_CHUNK
        or report.get("games") != GAMES_PER_CHUNK
        or report.get("invalid_games") != 0
        or report.get("protocol") != PAIRED_PROTOCOL
        or not is_sha256(report.get("deck_pool_sha256"))
        or report.get("wait_logit_bias") != 0
        or report.get("opponent_wait_logit_bias") != 0
        or not is_sha256(report.get("evaluation_code_sha256"))
        or report.get("evaluation_code_sha256") != report.get("evaluation_code_file_sha256_at_end")
    ):
        return False
    matches = report.get("matchups")
    if not isinstance(matches, list) or len(matches) != GAMES_PER_CHUNK:
        return False
    outcomes = []
    paired = {}
    identities = set()
    for match in matches:
        if not isinstance(match, dict) or match.get("valid") is not True:
            return False
        if match.get("reward") not in (-1, 0, 1):
            return False
        info = match.get("info")
        if not isinstance(info, dict) or any(
            info.get(key) != 0 for key in ("fault", "truncated", "illegal_0", "illegal_1")
        ):
            return False
        seat = match.get("learner_seat")
        if seat not in (0, 1):
            return False
        physical = (match.get("pair"), match.get("seed"), match.get("deck0"), match.get("deck1"))
        pair, seed, deck0, deck1 = physical
        if (
            not isinstance(pair, int)
            or not isinstance(seed, int)
            or pair < start
            or pair >= start + PAIRS_PER_CHUNK
            or seed != SIMULATOR_SEED_BASE + pair
            or not isinstance(deck0, str)
            or not isinstance(deck1, str)
            or deck0 == deck1
        ):
            return False
        identity = (report.get("sampling_seed"), *physical, seat)
        if identity in identities:
            return False
        identities.add(identity)
        paired.setdefault(physical, []).append(seat)
        outcomes.append(match["reward"])
    if len(paired) != PAIRS_PER_CHUNK or any(sorted(seats) != [0, 1] for seats in paired.values()):
        return False
    counts = (
        sum(x > 0 for x in outcomes),
        sum(x < 0 for x in outcomes),
        sum(x == 0 for x in outcomes),
    )
    return counts == tuple(report.get(key) for key in ("wins", "losses", "draws"))


def _move_partial(target: Path) -> None:
    if not target.exists():
        return
    stamp = time.time_ns()
    target.rename(target.with_name(f"{target.name}.interrupted_{stamp}"))


def _chunk(candidate: dict, start: int, gpu: str, candidate_sha: str) -> dict:
    generation = candidate["generation"]
    opponent = Path(candidate["checkpoint"]).resolve()
    target = OUT / f"gen_{generation:03d}" / f"part_{start:03d}"
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "evaluation.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            report = None
        if _report_valid(report, candidate_sha=candidate_sha, opponent=opponent, start=start):
            return {
                "start": start,
                "report": str(report_path),
                "reused": True,
                "games": report["games"],
            }
        _move_partial(target)
        target.mkdir(parents=True, exist_ok=True)
    elif list(target.glob("match_*.json")):
        _move_partial(target)
        target.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_policy.py"),
        "--policy",
        str(ACTOR),
        "--opponent",
        str(opponent),
        "--decks",
        str(ROOT / "configs/decks_expansion12.yaml"),
        "--out",
        str(target),
        "--pairs",
        str(PAIRS_PER_CHUNK),
        "--pair-start",
        str(start),
        "--sampling-seed",
        str(SAMPLING_SEED),
        "--pairing-seed",
        str(PAIRING_SEED),
        "--simulator-seed-base",
        str(SIMULATOR_SEED_BASE),
        "--device",
        "cuda:0",
        "--frame-cache",
        "--forced-wait-fastpath",
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        CONFIRMATION_MEMORY_MAX_GIB="4",
    )
    with (target / "evaluation.log").open("w") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not report_path.exists():
        raise RuntimeError(
            f"confirmation evaluation failed for generation {generation}, pair-start {start}"
        )
    report = json.loads(report_path.read_text())
    if not _report_valid(report, candidate_sha=candidate_sha, opponent=opponent, start=start):
        raise RuntimeError(
            f"invalid confirmation report for generation {generation}, pair-start {start}"
        )
    return {"start": start, "report": str(report_path), "reused": False, "games": report["games"]}


def _candidate_worker(candidate: dict, gpu: str, candidate_sha: str) -> dict:
    chunks = [_chunk(candidate, start, gpu, candidate_sha) for start in PAIR_STARTS]
    reports = [json.loads(Path(chunk["report"]).read_text()) for chunk in chunks]
    deck_hashes = {report["deck_pool_sha256"] for report in reports}
    code_hashes = {
        (report["evaluation_code_sha256"], report["evaluation_code_file_sha256_at_end"])
        for report in reports
    }
    if len(deck_hashes) != 1 or len(code_hashes) != 1:
        raise RuntimeError(
            f"confirmation chunks disagree on frozen evaluation inputs for generation {candidate['generation']}"
        )
    physical = [
        (match["pair"], match["seed"], match["deck0"], match["deck1"])
        for report in reports
        for match in report["matchups"]
    ]
    if len(set(physical)) != len(physical):
        raise RuntimeError(
            f"confirmation chunks repeat a physical deck/seed matchup for generation {candidate['generation']}"
        )
    rewards = [match["reward"] for report in reports for match in report["matchups"]]
    games = len(rewards)
    score = (sum(x < 0 for x in rewards) + 0.5 * sum(x == 0 for x in rewards)) / games
    return {
        **candidate,
        "reports": [chunk["report"] for chunk in chunks],
        "reused_chunks": sum(chunk["reused"] for chunk in chunks),
        "games": games,
        "learner_wins": sum(x > 0 for x in rewards),
        "learner_losses": sum(x < 0 for x in rewards),
        "draws": sum(x == 0 for x in rewards),
        "opponent_score": score,
        "confirmed": score >= 0.55 and games == TOTAL_GAMES,
    }


def _load_screening(screening: Path, protocol: Path, actor: Path) -> tuple[list[dict], str]:
    if not screening.exists() or not protocol.exists():
        raise RuntimeError("screening is incomplete; wait for all eight candidates")
    rows = json.loads(screening.read_text())
    screen_protocol = json.loads(protocol.read_text())
    candidate_sha = digest(actor)
    if (
        screen_protocol.get("candidate_update") != 182
        or screen_protocol.get("candidate_sha256") != candidate_sha
    ):
        raise RuntimeError("screening candidate does not match frozen update 182 actor")
    if set(screen_protocol.get("generations", [])) != set(GENERATIONS) or len(rows) != len(
        GENERATIONS
    ):
        raise RuntimeError("screening must contain exactly the eight registered generations")
    checked = []
    for row in rows:
        generation = row.get("generation")
        if generation not in GENERATIONS or row.get("valid") is not True:
            raise RuntimeError("screening contains an invalid or unexpected candidate")
        report_path = Path(row["evaluation"]).resolve()
        opponent = Path(row["checkpoint"]).resolve()
        if not report_path.exists() or not opponent.exists():
            raise RuntimeError("screening checkpoint/report is missing")
        report = json.loads(report_path.read_text())
        # Screening has four pairs at pair-start zero; retain its strict protocol and hashes.
        if (
            report.get("policy_sha256") != candidate_sha
            or report.get("opponent_sha256") != digest(opponent)
            or report.get("pair_start") != 0
            or report.get("pair_count") != 4
            or report.get("games") != 8
            or report.get("invalid_games") != 0
            or report.get("protocol") != PAIRED_PROTOCOL
            or report.get("wait_logit_bias") != 0
            or report.get("opponent_wait_logit_bias") != 0
            or not is_sha256(report.get("deck_pool_sha256"))
            or not is_sha256(report.get("evaluation_code_sha256"))
            or report.get("evaluation_code_sha256")
            != report.get("evaluation_code_file_sha256_at_end")
        ):
            raise RuntimeError(f"screening protocol/hash invalid for generation {generation}")
        matches = report.get("matchups")
        if not isinstance(matches, list) or len(matches) != 8:
            raise RuntimeError(f"screening games are incomplete for generation {generation}")
        pairs = {}
        outcomes = []
        for match in matches:
            if match.get("valid") is not True or match.get("reward") not in (-1, 0, 1):
                raise RuntimeError(f"screening contains invalid game for generation {generation}")
            info = match.get("info")
            if not isinstance(info, dict) or any(
                info.get(key) != 0 for key in ("fault", "truncated", "illegal_0", "illegal_1")
            ):
                raise RuntimeError(f"screening contains faulted game for generation {generation}")
            physical = (
                match.get("pair"),
                match.get("seed"),
                match.get("deck0"),
                match.get("deck1"),
            )
            if match.get("learner_seat") not in (0, 1):
                raise RuntimeError(f"screening seat swap is incomplete for generation {generation}")
            pair, seed, deck0, deck1 = physical
            if (
                not isinstance(pair, int)
                or not isinstance(seed, int)
                or pair < 0
                or pair >= 4
                or seed != 11000000 + pair
                or not isinstance(deck0, str)
                or not isinstance(deck1, str)
                or deck0 == deck1
            ):
                raise RuntimeError(
                    f"screening physical matchup is invalid for generation {generation}"
                )
            pairs.setdefault(physical, []).append(match["learner_seat"])
            outcomes.append(match["reward"])
        if len(pairs) != 4 or any(sorted(seats) != [0, 1] for seats in pairs.values()):
            raise RuntimeError(f"screening seat swap is incomplete for generation {generation}")
        counts = (
            sum(x > 0 for x in outcomes),
            sum(x < 0 for x in outcomes),
            sum(x == 0 for x in outcomes),
        )
        if counts != tuple(report.get(key) for key in ("wins", "losses", "draws")):
            raise RuntimeError(
                f"screening aggregate differs from games for generation {generation}"
            )
        checked.append(
            {
                "generation": generation,
                "policy_id": policy_id(generation),
                "checkpoint": str(opponent),
                "screening_score": (report["losses"] + 0.5 * report["draws"]) / report["games"],
                "anchor": generation == 0 or row.get("reference") == "BC_anchor",
            }
        )
    if len({row["generation"] for row in checked}) != len(GENERATIONS):
        raise RuntimeError("screening repeats a generation")
    non_anchor = [row for row in checked if not row["anchor"]]
    non_anchor.sort(key=lambda row: (-row["screening_score"], row["generation"]))
    if len(non_anchor) < 2:
        raise RuntimeError("screening has fewer than two non-anchor candidates")
    return non_anchor[:2], candidate_sha


def _write_diverse_pool(
    rows: list[dict], candidate_sha: str, baseline_path: Path, out: Path
) -> str:
    from cr_rl.opponents.league import OpponentEntry, OpponentPool

    if not baseline_path.exists():
        raise RuntimeError(f"prepared baseline pool is missing: {baseline_path}")
    pool = OpponentPool.load(baseline_path)
    for row in rows:
        if not row["confirmed"]:
            continue
        existing = next(
            (entry for entry in pool.entries if entry.policy_id == row["policy_id"]), None
        )
        if existing is not None:
            if Path(existing.checkpoint).resolve() != Path(row["checkpoint"]).resolve():
                raise RuntimeError("baseline policy id points at a different checkpoint")
            if existing.role != "historical":
                raise RuntimeError("confirmed expert must be historical in the pool")
            continue
        pool.add(
            OpponentEntry(
                policy_id=row["policy_id"],
                checkpoint=str(Path(row["checkpoint"]).resolve()),
                generation=row["generation"],
                role="historical",
            )
        )
    experts = [
        {
            "policy_id": row["policy_id"],
            "specialty": f"independent_counter_{row['generation']}",
            "reports": row["reports"],
        }
        for row in rows
        if row["confirmed"]
    ]
    manifest = {
        "schema_version": 1,
        "evaluated_against_version": 182,
        "candidate_sha256": candidate_sha,
        "candidate_checkpoint": str(ACTOR.resolve()),
        "max_age_versions": 20,
        "recent_slots": 2,
        "expert_slots": 3,
        "min_games": 32,
        "min_opponent_score": 0.55,
        "experts": experts,
    }
    pool.configure_diverse_retention(manifest, learner_version=182, verify_files=True)
    pool.prune(learner_version=182)
    # The prepared baseline and the trial branches are siblings.  Keep the
    # generated branch beside ``baseline`` so it can be selected independently.
    target = out.parent / "diverse_pool" / "opponent_pool.json"
    if target.exists():
        target.rename(target.with_name(f"{target.name}.previous_{time.time_ns()}"))
    pool.save(target)
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", type=Path, default=SCREEN / "screening.json")
    parser.add_argument("--screening-protocol", type=Path, default=SCREEN / "protocol.json")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--baseline-pool", type=Path, default=BASELINE_POOL)
    parser.add_argument(
        "--fixture", action="store_true", help="validate constants and print the two chunk plan"
    )
    args = parser.parse_args()
    if args.fixture:
        print(
            json.dumps(
                {
                    "generations": GENERATIONS,
                    "pair_starts": PAIR_STARTS,
                    "games_per_candidate": TOTAL_GAMES,
                    "gpu_uuids": V100_GPUS,
                    "pairing_seed": PAIRING_SEED,
                    "sampling_seed": SAMPLING_SEED,
                },
                indent=2,
            )
        )
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    candidates, candidate_sha = _load_screening(args.screening, args.screening_protocol, ACTOR)
    queue = Queue()
    for gpu in V100_GPUS:
        queue.put(gpu)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_candidate_worker, candidate, queue.get(), candidate_sha): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["generation"])
    protocol = {
        "schema_version": 1,
        "status": "complete",
        "candidate_update": 182,
        "candidate_sha256": candidate_sha,
        "candidate_checkpoint": str(ACTOR.resolve()),
        "selected_top2": [row["generation"] for row in candidates],
        "pairing_seed": PAIRING_SEED,
        "sampling_seed_base": SAMPLING_SEED,
        "simulator_seed_base": SIMULATOR_SEED_BASE,
        "pair_starts": PAIR_STARTS,
        "pair_count_per_chunk": PAIRS_PER_CHUNK,
        "games_per_candidate": TOTAL_GAMES,
        "gpu_uuids": V100_GPUS,
        "max_concurrent_evaluators": 2,
        "memory_max_gib_per_service": 4,
        "started_unix": time.time(),
    }
    atomic_json(args.out / "confirmation_protocol.json", protocol)
    confirmation = {
        "protocol": protocol,
        "candidates": rows,
        "confirmed": [row for row in rows if row["confirmed"]],
    }
    atomic_json(args.out / "confirmation.json", confirmation)
    confirmed = confirmation["confirmed"]
    if not confirmed:
        stale_manifest = args.out / "expert_manifest.json"
        if stale_manifest.exists():
            stale_manifest.rename(
                stale_manifest.with_name(f"{stale_manifest.name}.previous_{time.time_ns()}")
            )
        atomic_json(
            args.out / "no_experts.json",
            {
                "reason": "Neither selected non-anchor candidate met 32-game opponent_score >= 0.55 gate",
                "confirmation": str(args.out / "confirmation.json"),
            },
        )
        return 0
    manifest_path = args.out / "expert_manifest.json"
    # Build the pool only after confirmation.json is durable and all experts pass the gate.
    pool_path = _write_diverse_pool(confirmed, candidate_sha, args.baseline_pool, args.out)
    manifest = {
        "schema_version": 1,
        "evaluated_against_version": 182,
        "candidate_sha256": candidate_sha,
        "candidate_checkpoint": str(ACTOR.resolve()),
        "max_age_versions": 20,
        "recent_slots": 2,
        "expert_slots": 3,
        "min_games": 32,
        "min_opponent_score": 0.55,
        "experts": [
            {
                "policy_id": row["policy_id"],
                "specialty": f"independent_counter_{row['generation']}",
                "reports": row["reports"],
            }
            for row in confirmed
        ],
    }
    atomic_json(manifest_path, manifest)
    confirmation["diverse_pool"] = pool_path
    atomic_json(args.out / "confirmation.json", confirmation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
