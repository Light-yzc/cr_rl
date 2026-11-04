"""Fixed-seed, paired-seat evaluation against one frozen checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from train_selfplay import _forward, _policy_inputs, _windows

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.evaluation.cohort import paired_match_specs
from cr_rl.models import ActorCritic, ActorPolicy, ModelConfig
from cr_rl.training.inference import FrozenInference
from cr_rl.training.persistence import atomic_json, file_sha256


def load_policy(path, device, wait_logit_bias=0.0):
    if not math.isfinite(wait_logit_bias):
        raise ValueError("WAIT logit bias must be finite")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("model_architecture_version") != 2:
        raise ValueError("Evaluation requires architecture-v2 weights")
    cfg = ModelConfig(**state["config"])
    model = (ActorPolicy if state.get("policy_only") else ActorCritic)(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    model.checkpoint_calibration = state.get("calibration")
    model.checkpoint_optimizer_steps = state.get("optimizer_steps")
    # Calibrate only action timing. Conditional card and placement probabilities
    # remain unchanged; training checkpoints on disk are never modified.
    with torch.no_grad():
        model.wait_actor.bias.add_(wait_logit_bias)
    return model, cfg


def main():
    evaluation_code_sha256 = file_sha256(Path(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--decks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=20261103)
    parser.add_argument("--simulator-seed-base", type=int, default=1000000)
    parser.add_argument("--pairing-seed", type=int, default=20261103)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-cache", action="store_true")
    parser.add_argument("--wait-logit-bias", type=float, default=0.0)
    parser.add_argument("--opponent-wait-logit-bias", type=float, default=0.0)
    args = parser.parse_args()
    if (args.out / "evaluation.json").exists() or list(args.out.glob("match_*.json")):
        parser.error("Use a fresh evaluation directory to preserve frozen evidence")
    torch.manual_seed(args.sampling_seed + args.pair_start)
    model, cfg = load_policy(args.policy, args.device, args.wait_logit_bias)
    opponent, other_cfg = load_policy(args.opponent, args.device, args.opponent_wait_logit_bias)
    input_fields = ("context_len", "header_dim", "entity_feature_dim", "action_count", "vocab_size")
    input_cfg = {k: getattr(cfg, k) for k in input_fields}
    opponent_cfg = {k: getattr(other_cfg, k) for k in input_fields}
    if input_cfg != opponent_cfg:
        raise ValueError("Policy and frozen opponent configs differ")
    league = yaml.safe_load(args.decks.read_text())
    names = list(league["decks"])
    if len(names) < 2:
        raise ValueError("Need multiple held-out decks")
    matchups = paired_match_specs(
        names,
        args.pairs,
        args.pair_start,
        simulator_seed_base=args.simulator_seed_base,
        pairing_seed=args.pairing_seed,
    )
    count = args.pairs * 2
    seeds, deck0, deck1, forms0, forms1, learner_seats = [], [], [], [], [], []
    for match in matchups:
        a, b = match["deck0"], match["deck1"]
        seeds.append(match["seed"])
        learner_seats.append(match["learner_seat"])
        deck0.append(league["decks"][a])
        deck1.append(league["decks"][b])
        forms0.append(league.get("slots", {}).get(a, [0] * 8))
        forms1.append(league.get("slots", {}).get(b, [0] * 8))
    args.out.mkdir(parents=True, exist_ok=True)
    active = np.ones(count, dtype=bool)
    started = time.perf_counter()
    with CrForgeVector(count, count, independent=True, arena="Arena_L") as env:
        env.set_autoreset(False)
        env.reset(seeds, deck0, deck1, forms0, forms1)
        cached = args.frame_cache or cfg.representation_version == 2
        adapter = CrForgeObservationAdapter(
            env.vector.schema, env.vector.vocabulary_size, entity_bucket=16 if cached else 0
        )
        inference = (
            {
                id(policy): FrozenInference(policy, adapter, args.device)
                for policy in (model, opponent)
            }
            if cached
            else {}
        )
        histories = [deque(maxlen=cfg.context_len) for _ in range(count * 2)]
        decision_audit = [
            [
                {
                    "decisions": 0,
                    "wait_decisions": 0,
                    "card_requests": 0,
                    "ability_requests": 0,
                    "full_elixir_wait_legal_seconds": 0.0,
                    "longest_full_elixir_wait_legal_seconds": 0.0,
                }
                for _ in range(2)
            ]
            for _ in range(count)
        ]
        full_wait_run = np.zeros((count, 2), dtype=np.int64)
        elixir_index = env.vector.schema.header["elixir"][0]
        for step in range(2000):
            for i in range(count):
                for seat in (0, 1):
                    histories[i * 2 + seat].append(env.observations[i, seat].copy())
            actions = np.zeros((count, 2), dtype=np.int32)
            for policy, seats in (
                (model, learner_seats),
                (opponent, [1 - seat for seat in learner_seats]),
            ):
                raw = np.stack([env.observations[i, seat] for i, seat in enumerate(seats)])
                legal = np.stack([env.masks[i, seat] for i, seat in enumerate(seats)])
                windows = [histories[i * 2 + seat] for i, seat in enumerate(seats)]
                if cached:
                    history, time_mask = _windows(
                        windows, cfg.context_len, env.vector.schema.seat_ints
                    )
                    chosen, _, _ = inference[id(policy)].sample(history, time_mask, legal)
                else:
                    chosen, _, _ = _forward(
                        policy, _policy_inputs(raw, legal, windows, adapter), args.device
                    )
                if not np.all(legal[np.arange(count), chosen]):
                    raise RuntimeError("Illegal evaluation action")
                for i, seat in enumerate(seats):
                    actions[i, seat] = chosen[i]
                    if active[i]:
                        audit = decision_audit[i][seat]
                        action = int(chosen[i])
                        audit["decisions"] += 1
                        audit["wait_decisions"] += int(action == 0)
                        audit["card_requests"] += int(0 < action < 2305)
                        audit["ability_requests"] += int(action >= 2305)
                        full_wait = (
                            action == 0
                            and raw[i, elixir_index] >= 100000
                            and legal[i, 1:2305].any()
                        )
                        full_wait_run[i, seat] = full_wait_run[i, seat] + 1 if full_wait else 0
                        if full_wait:
                            audit["full_elixir_wait_legal_seconds"] += 0.25
                            audit["longest_full_elixir_wait_legal_seconds"] = max(
                                audit["longest_full_elixir_wait_legal_seconds"],
                                float(full_wait_run[i, seat]) * 0.25,
                            )
            env.step(actions)
            for i, done in enumerate(env.dones.copy()):
                if not done:
                    continue
                if active[i]:
                    info = env.vector.info(i)
                    valid = not info["fault"] and not info["truncated"]
                    matchups[i].update(
                        steps=step + 1,
                        valid=valid,
                        info=info,
                        reward=float(env.rewards[i, learner_seats[i]]),
                        decision_audit={
                            "learner": decision_audit[i][learner_seats[i]],
                            "opponent": decision_audit[i][1 - learner_seats[i]],
                            "scope": (
                                "Active-game decisions before env.step; "
                                "requests are not confirmed placements"
                            ),
                        },
                    )
                    (args.out / f"match_{i:03d}.json").write_text(
                        env.vector.lanes[i].replay_json(0)
                    )
                    active[i] = False
                env.vector.reset_env(i, seeds[i])
                histories[i * 2].clear()
                histories[i * 2 + 1].clear()
            if step % 128 == 0:
                print(json.dumps({"eval_step": step, "remaining": int(active.sum())}), flush=True)
            if not active.any():
                break
    for i in np.flatnonzero(active):
        matchups[i].update(valid=False, reason="evaluation decision limit")
    valid = [row for row in matchups if row["valid"]]
    rewards = [row["reward"] for row in valid]
    result = {
        "policy_sha256": file_sha256(args.policy),
        "opponent_sha256": file_sha256(args.opponent),
        "deck_pool_sha256": file_sha256(args.decks),
        "evaluation_code_sha256": evaluation_code_sha256,
        "evaluation_code_file_sha256_at_end": file_sha256(Path(__file__)),
        "wait_logit_bias": args.wait_logit_bias,
        "policy_checkpoint_calibration": model.checkpoint_calibration,
        "policy_optimizer_steps": model.checkpoint_optimizer_steps,
        "opponent_wait_logit_bias": args.opponent_wait_logit_bias,
        "protocol": "fixed physical decks/seeds, paired policy-seat swap, stochastic sampling",
        "deck_selection": f"unique pairs from seeded held-out pool (seed {args.pairing_seed})",
        "pairing_seed": args.pairing_seed,
        "simulator_seed_base": args.simulator_seed_base,
        "pair_start": args.pair_start,
        "pair_count": args.pairs,
        "sampling_seed": args.sampling_seed + args.pair_start,
        "sampling_protocol": (
            "Fixed batch size and pair start across candidates; not batch-size invariant"
        ),
        "games": len(valid),
        "invalid_games": len(matchups) - len(valid),
        "wins": sum(r > 0 for r in rewards),
        "losses": sum(r < 0 for r in rewards),
        "draws": sum(r == 0 for r in rewards),
        "matchups": matchups,
        "seconds": time.perf_counter() - started,
        "limitation": "Small fixed evaluation; not a statistically reliable strength estimate",
    }
    atomic_json(args.out / "evaluation.json", result)
    print(json.dumps(result), flush=True)
    if not valid or result["invalid_games"]:
        raise RuntimeError("Evaluation has unfinished, truncated or faulted matches")


if __name__ == "__main__":
    main()
