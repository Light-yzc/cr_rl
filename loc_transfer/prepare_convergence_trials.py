"""Prepare immutable, single-factor PPO trials without importing Torch or training.

The stopped update-182 source is read only. Initial weights are shared through
symlinks; a trainer must publish replacements atomically, as selfplay_v2 does.
Existing protocols and registered branches are never rewritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_UPDATE = 182
BRANCHES = ("baseline", "adaptive_lr", "longer_gae", "diverse_pool")


def encoded_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_pool(pool: dict, source: Path) -> dict:
    normalized = copy.deepcopy(pool)
    entries = normalized["entries"]
    if not entries or len({entry["policy_id"] for entry in entries}) != len(entries):
        raise ValueError("Source opponent pool is empty or has duplicate policy IDs")
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = source / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        if not checkpoint.is_file():
            raise ValueError(f"Opponent checkpoint is not a file: {checkpoint}")
        entry["checkpoint"] = str(checkpoint)
    if sum(entry.get("role") == "anchor" for entry in entries) != 1:
        raise ValueError("Source pool must have exactly one frozen BC anchor")
    if not any(entry.get("role") == "historical" for entry in entries):
        raise ValueError("Source pool is missing historical PPO opponents")
    return normalized


def inspect_source(source: Path, audit_path: Path) -> tuple[dict, dict]:
    """Stream hashes, read JSON, and reject stale or inconsistent audit evidence."""
    summary_path = source / "summary.json"
    summary = read_json(summary_path)
    audit = read_json(audit_path)
    if summary.get("update") != SOURCE_UPDATE or audit.get("update") != SOURCE_UPDATE:
        raise ValueError("Both source summary and checkpoint audit must describe update 182")
    if audit.get("passed") is not True or audit.get("finite_tensors", 0) < 1:
        raise ValueError("The update-182 finite-checkpoint audit did not pass")
    if audit.get("completed_games") != summary.get("completed_training_games_total"):
        raise ValueError("Source summary and audit have different completed-game counts")
    mixture = {"current": 0.3, "historical": 0.5, "anchor": 0.2}
    if summary.get("role_mixture") != mixture:
        raise ValueError("Source is not the preregistered 30/50/20 role league")
    if summary.get("learning_rates") != {"actor": [1e-5], "critic": [1e-4]}:
        raise ValueError("Source optimizer learning rates differ from the preregistered baseline")
    checkpoint = source / "checkpoints" / "update_000182.pt"
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != audit.get("sha256"):
        raise ValueError("Immutable update-182 checkpoint SHA-256 differs from the passed audit")
    export = source / "actor_export.pt"
    export_hash = digest(export)
    current_pool = normalize_pool(read_json(source / "opponent_pool.json"), source)
    pool_path = source / "checkpoints" / "update_000182.pool.json"
    pool = normalize_pool(read_json(pool_path), source)
    if pool != current_pool or len(pool["entries"]) != audit.get("pool_entries_resolve"):
        raise ValueError("Source current/snapshot opponent pools or audit entry counts differ")
    if read_json(summary_path) != summary or read_json(audit_path) != audit:
        raise ValueError("Source/audit changed during preparation; source must remain stopped")
    source_record = {
        "run_dir": str(source),
        "update": SOURCE_UPDATE,
        "summary": {"path": str(summary_path), "sha256": digest(summary_path)},
        "passed_checkpoint_audit": {"path": str(audit_path), "sha256": digest(audit_path)},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "bytes": checkpoint.stat().st_size,
        },
        "actor_export_182": {
            "path": str(export),
            "sha256": export_hash,
            "bytes": export.stat().st_size,
            "binding": "Stopped source summary is update 182; source must never resume in place",
        },
        "pool": {"path": str(pool_path), "sha256": digest(pool_path)},
        "normalized_pool_sha256": hashlib.sha256(encoded_json(pool).encode()).hexdigest(),
        "learner_samples_total": summary["learner_samples_total"],
        "completed_training_games_total": summary["completed_training_games_total"],
    }
    return source_record, pool


def make_protocol(source_record: dict, pool: dict, output: Path) -> dict:
    common = {
        "v2": True,
        "device": "cuda:0",
        "critic_device": "cuda:1",
        "amp_dtype": "fp16",
        "envs": 4,
        "env_threads": 4,
        "torch_threads": 4,
        "torch_interop_threads": 1,
        "steps": 1536,
        "updates": 3,
        "epochs": 2,
        "critic_epochs": 1,
        "batch_size": 8,
        "actor_batch_size": 8,
        "critic_batch_size": 16,
        "grad_accumulation": 8,
        "critic_model": "same",
        "critic_mode": "independent",
        "critic_warmup_updates": 2,
        "critic_warmup_games": 4,
        "gamma": 0.9995,
        "gae_lambda": 0.993,
        "critic_gae_lambda": 0.995,
        "tower_shaping": True,
        "princess_reward": 0.05,
        "king_reward": 0.05,
        "crown_reward": 0.15,
        "overflow_penalty": 0,
        "reference_kl_coef": 0.01,
        "target_kl": 0.02,
        "target_card_kl": 0.015,
        "target_tile_kl": 0.06,
        "max_initial_logp_error": 0.05,
        "actor_choice_only": True,
        "frame_encoder_fp32": True,
        "role_league": True,
        "disable_holding_prior": True,
        "compact_host_memory": True,
        "balanced_critic": True,
        "dual_rollout": True,
        "incremental_inference": True,
        "forced_wait_fastpath": True,
        "cache_reference": True,
        "defer_scalar_metrics": True,
        "card_entropy_coef": 0.001,
        "tile_entropy_coef": 0.001,
        "opponent_cache_size": 4,
        "training_opponent_cache_size": 2,
        "keep_checkpoints": 1,
        "keep_periodic_checkpoints": 0,
        "history_snapshot_interval": 20000,
        "min_training_games": 1000000000,
        "decks": str(ROOT / "configs/decks_expansion12.yaml"),
    }
    branches = {
        "baseline": {"enabled": True, "training_overrides": {}},
        "adaptive_lr": {
            "enabled": True,
            "training_overrides": {
                "adaptive_actor_lr": True,
                "adaptive_actor_lr_min": 2.5e-6,
                "adaptive_actor_lr_max": 1e-5,
            },
        },
        "longer_gae": {"enabled": True, "training_overrides": {"gae_lambda": 0.997}},
        "diverse_pool": {
            "enabled": False,
            "training_overrides": {},
            "planned_single_factor": "Confirmed league diversity manifest only",
            "manifest": None,
            "activation": (
                "Requires independently confirmed frozen opponents and a separate immutable "
                "activation receipt binding the manifest SHA-256; do not edit this protocol"
            ),
        },
    }
    anchor = next(entry["checkpoint"] for entry in pool["entries"] if entry["role"] == "anchor")
    return {
        "schema_version": 1,
        "experiment": "ppo_convergence_trials_20261108",
        "source": source_record,
        "output_dir": str(output),
        "common_training_args": common,
        "role_mixture": {"current": 0.3, "historical": 0.5, "anchor": 0.2},
        "initial_learning_rates": {"actor": 1e-5, "critic": 1e-4},
        "branches": branches,
        "budget": {
            "updates": 3,
            "envs": 4,
            "steps_per_env_per_update": 1536,
            "simulator_steps_per_branch": 18432,
            "expected_final_update": 185,
            "learner_sample_count": "Measured separately; role/episode composition changes it",
            "old_game_stop_override": (
                "Override saved min_training_games=1258 with 1000000000 in all branches; "
                "the fixed 3-update budget controls stopping"
            ),
        },
        "controlled_conditions": {
            "initial_pool": "Identical normalized source pool; scores/statistics unchanged",
            "resume": "Same model/optimizer/scaler/RNG; simulator resets on every branch start",
            "warmup_and_admission_state": "Restore unchanged from the common checkpoint",
            "history_snapshot_rule": "Existing history admission clock and interval=20000",
            "history_candidate_gate": (
                "Unchanged: one paired evaluation (2 valid complete games); "
                "numerical/legal gate, not a strength gate"
            ),
            "hard_kl_limits": "Card=.015, Tile=.06, joint=.02; unchanged in every branch",
            "bc_interleaving": False,
            "ppg": False,
        },
        "storage": {
            "source_read_only": True,
            "preserve_source_updates": [181, 182],
            "initial_weight_sharing": "Absolute symlinks, replaced atomically by the trainer",
            "trial_keep_checkpoints": 1,
            "trial_keep_periodic_checkpoints": 0,
            "rollout_recent": 2,
            "opponent_retention": "Keep all files referenced by retained pools/evaluations",
            "source_disk": "Existing Intel mount; no writes or cleanup",
            "trial_disk": "Default output is the workspace HDD; do not use GALAX",
        },
        "evaluation": {
            "pair_start": 32,
            "pairs": 8,
            "games_per_opponent_per_branch": 16,
            "sampling_seed": 20261201,
            "pairing_seed": 20261201,
            "simulator_seed_base": 11000000,
            "swap_seats": True,
            "opponents": {
                "BC_anchor": anchor,
                "frozen_182": source_record["actor_export_182"]["path"],
            },
            "decks": common["decks"],
            "scope": "Fixed small pilot; no statistical-significance or convergence claim",
            "compare": [
                "win/draw/loss and invalid games",
                "actual successful card costs and >=5-cost plays",
                "full-elixir legal WAIT duration",
                "Actor committed/planned samples, rejected groups and KL",
                "wall-clock throughput, AMP overflows and peak memory",
            ],
            "screening_separation": "Final test pairs32-39 are excluded from opponent selection",
        },
    }


def protocol_markdown(protocol: dict, protocol_hash: str) -> str:
    return f"""# PPO convergence pilot — 2026-11-08

Protocol SHA-256: `{protocol_hash}`. Preparation does not launch training.

All four directories start from audited update 182 with shared weight symlinks.
The stopped Intel source, including updates 181/182, remains read only; branch
data goes to the workspace HDD. Do not resume or clean the source in place.

| Branch | Single changed factor | Enabled |
| --- | --- | --- |
| baseline | None | Yes |
| adaptive_lr | Next-update Actor LR feedback, bounds 2.5e-6–1e-5 | Yes |
| longer_gae | Actor GAE λ .993 → .997 | Yes |
| diverse_pool | Confirmed frozen-opponent diversity manifest | No; pending confirmation |

Budget: **3 updates × 4 environments × 1536 steps**, expected updates 183–185.
The inherited game stop of 1258 must be overridden with 1000000000 for every
branch, otherwise source count 1268 would stop training immediately. Critic λ
stays .995, role mixture stays 30/50/20, and the existing history clock, snapshot
interval 20000 and two-game legality gate remain unchanged. Baseline/adaptive/
longer use identical source pool scores. No BC interleaving or PPG is included.

Keep 1 trial checkpoint, 0 periodic checkpoints and the default 2 recent rollouts.
Keep opponent files needed by retained pools/evaluations. Trainers must replace
the initial weight symlinks atomically and apply cleanup only inside their branch.

Final testing uses **pair_start=32, pairs=8 (16 games)** per branch against each
of the BC anchor and frozen 182. Use the same decks, swapped seats and fixed
sampling/pairing seed 20261201, simulator seed base 11000000. Pairs 32–39 must not
be used to select opponents. Report wins/draws/losses, actual costs, full-elixir
WAIT, KL/sample use, throughput, overflows and memory. This is a small pilot;
it cannot establish statistical significance or long-run convergence.

The diversity branch starts disabled with the same pool. Activation requires a
separate immutable receipt binding an independently confirmed manifest hash;
leave this preregistration intact. Re-running preparation preserves every
registered branch, including any weights or metrics written after training.

Full paths, source hashes and exact shared trainer settings are in `protocol.json`.
"""


def require_same_or_absent(path: Path, expected: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink in immutable preparation metadata: {path}")
    if path.exists() and path.read_text(encoding="utf-8") != expected:
        raise ValueError(f"Refusing to change existing protocol/preparation metadata: {path}")


def write_once(path: Path, text: str) -> None:
    require_same_or_absent(path, text)
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)


def prepare(source: Path, output: Path, audit_path: Path) -> dict:
    source = source.resolve(strict=True)
    output = output.resolve()
    audit_path = audit_path.resolve(strict=True)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Trial output and read-only source must be separate directory trees")
    source_record, pool = inspect_source(source, audit_path)
    protocol = make_protocol(source_record, pool, output)
    protocol_text = encoded_json(protocol)
    protocol_hash = hashlib.sha256(protocol_text.encode()).hexdigest()
    markdown = protocol_markdown(protocol, protocol_hash)
    pool_text = encoded_json(pool)
    receipts = {}
    for branch, settings in protocol["branches"].items():
        receipts[branch] = encoded_json(
            {
                "branch": branch,
                "protocol_sha256": protocol_hash,
                "source_update": SOURCE_UPDATE,
                "enabled": settings["enabled"],
                "training_args": {
                    **protocol["common_training_args"],
                    **settings["training_overrides"],
                },
                "initial_pool_sha256": source_record["normalized_pool_sha256"],
                "initial_links": {
                    "ppo_latest.pt": source_record["checkpoint"]["path"],
                    "actor_export.pt": source_record["actor_export_182"]["path"],
                },
            }
        )
    # Check all conflicts before writing anything. Existing branch contents are
    # training-owned; never restore their initial symlinks or reset their pools.
    require_same_or_absent(output / "protocol.json", protocol_text)
    require_same_or_absent(output / "protocol.md", markdown)
    for branch, receipt in receipts.items():
        target = output / branch
        if target.is_symlink():
            raise ValueError(f"Trial branch cannot be a symlink: {target}")
        if target.exists():
            if not target.is_dir() or not (target / "trial.json").is_file():
                raise ValueError(f"Refusing to adopt or overwrite an unregistered branch: {target}")
            require_same_or_absent(target / "trial.json", receipt)
    output.mkdir(parents=True, exist_ok=True)
    write_once(output / "protocol.json", protocol_text)
    write_once(output / "protocol.md", markdown)
    results = {}
    for branch, receipt in receipts.items():
        target = output / branch
        if target.exists():
            results[branch] = "existing_preserved"
            continue
        stage = Path(tempfile.mkdtemp(prefix=f".{branch}.prepare-", dir=output))
        try:
            (stage / "trial.json").write_text(receipt, encoding="utf-8")
            (stage / "opponent_pool.json").write_text(pool_text, encoding="utf-8")
            for filename, link_target in read_json(stage / "trial.json")["initial_links"].items():
                (stage / filename).symlink_to(link_target)
            if target.exists():
                raise FileExistsError(f"Branch appeared during preparation: {target}")
            stage.rename(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        results[branch] = "prepared" if protocol["branches"][branch]["enabled"] else "disabled"
    return {"output": str(output), "protocol_sha256": protocol_hash, "branches": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/local_dual_v100")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/ppo_convergence_trials_20261108"
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts/local_dual_v100_checks/checkpoint_000182_audit.json",
    )
    args = parser.parse_args()
    print(encoded_json(prepare(args.source, args.out, args.audit)), end="")


if __name__ == "__main__":
    main()
