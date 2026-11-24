"""Head-to-head paired A/B evaluator for TestTimeLookaheadSearch vs baseline argmax."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import yaml

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.evaluation.cohort import paired_match_specs
from cr_rl.models.state_policy import state_model_from_checkpoint, state_value_model_from_checkpoint
from cr_rl.training.test_time_search import LookaheadConfig, LookaheadPolicyStream
from scripts.evaluate_state_policy import PolicyStream, load_frozen_policy


def run_paired_ab(args) -> dict:
    device = torch.device(args.device)
    ckpt = torch.load(args.policy, map_location="cpu", weights_only=False, mmap=True)
    actor = state_model_from_checkpoint(ckpt, device=device).eval()
    critic = None
    if not args.no_critic and "critic" in ckpt and ckpt["critic"] is not None:
        critic = state_value_model_from_checkpoint(ckpt, device=device).eval()
    del ckpt

    opp_model, _ = load_frozen_policy(args.opponent, device)
    deck_doc = yaml.safe_load(Path(args.decks).read_text(encoding="utf-8"))
    decks_map = deck_doc["decks"]
    slots_map = deck_doc.get("slots", {})
    specs = paired_match_specs(list(decks_map), args.pairs, simulator_seed_base=args.seed)
    if args.max_games:
        specs = specs[: args.max_games]

    cfg = LookaheadConfig(
        max_candidates=args.candidates,
        rollouts_per_candidate=args.rollouts,
        horizon_steps=args.horizon_steps,
        trigger_elixir=args.trigger_elixir,
        threads=args.threads,
    )
    max_lanes = cfg.max_candidates * cfg.rollouts_per_candidate

    with (
        CrForgeVector(1, 1, independent=True, arena="Arena_L", decision_ticks=8) as play_env,
        CrForgeVector(max_lanes, min(max_lanes, cfg.threads), independent=True, arena="Arena_L", decision_ticks=8) as search_env,
    ):
        play_env.set_autoreset(False)
        search_env.set_autoreset(False)
        schema = play_env.vector.schema
        adapter = CrForgeObservationAdapter(schema, play_env.vector.vocabulary_size, entity_bucket=16)

        results = {}
        for mode in ("baseline_argmax", "lookahead_search"):
            wins = losses = draws = 0
            crown_diffs = []
            sparky_id = play_env.vector.card_id("ZapMachine")
            sparky_requests = 0
            t0 = time.perf_counter()

            stream = (
                LookaheadPolicyStream(
                    actor, adapter, device, critic=critic, search_env=search_env, config=cfg, precision=args.precision
                )
                if mode == "lookahead_search"
                else PolicyStream(actor, adapter, device, precision=args.precision, action_mode="argmax")
            )
            opp_stream = PolicyStream(opp_model, adapter, device, precision=args.precision, action_mode="argmax")

            for idx, sp in enumerate(specs):
                seat = int(sp["learner_seat"])
                opp_seat = 1 - seat
                d0_name, d1_name = sp["deck0"], sp["deck1"]
                d0 = decks_map[d0_name]
                d1 = decks_map[d1_name]
                s0 = slots_map.get(d0_name, [0] * 8)
                s1 = slots_map.get(d1_name, [0] * 8)
                cand_deck = d0_name if seat == 0 else d1_name
                opp_deck = d1_name if seat == 0 else d0_name
                seed = int(sp["seed"])

                play_env.vector.reset_env(0, seed, d0, d1, s0, s1)
                opp_stream.reset()
                if mode == "lookahead_search":
                    stream.bind_match(seed=seed, deck0=d0, deck1=d1, slots0=s0, slots1=s1, seat=seat)
                else:
                    stream.reset()

                while not play_env.dones[0]:
                    raw0 = play_env.observations[0, seat].copy()
                    leg0 = play_env.masks[0, seat].copy()
                    raw1 = play_env.observations[0, opp_seat].copy()
                    leg1 = play_env.masks[0, opp_seat].copy()

                    if mode == "lookahead_search":
                        a_cand, _ = stream.act(raw0, leg0, opp_raw=raw1, opp_legal=leg1)
                    else:
                        a_cand, _ = stream.sample(raw0, leg0)
                    a_opp, _ = opp_stream.sample(raw1, leg1)

                    if 0 < a_cand < 2305:
                        slot = (a_cand - 1) // 576
                        if int(raw0[schema.header["hand_card"][slot]]) == sparky_id:
                            sparky_requests += 1

                    acts = [0, 0]
                    acts[seat] = a_cand
                    acts[opp_seat] = a_opp
                    if mode == "lookahead_search":
                        stream.record_step_actions(acts[0], acts[1])
                    play_env.vector.actions[0, 0] = acts[0]
                    play_env.vector.actions[0, 1] = acts[1]
                    play_env.vector.step_lanes([0])

                info = play_env.vector.info(0)
                obs_end = play_env.observations[0, seat]
                cd = int(schema.field(obs_end, "crowns") - schema.field(obs_end, "crowns_enemy"))
                crown_diffs.append(cd)
                if info["winner"] == seat:
                    wins += 1
                elif info["winner"] == opp_seat:
                    losses += 1
                else:
                    draws += 1
                print(
                    f"[{mode}] game {idx+1}/{len(specs)} ({cand_deck} vs {opp_deck}, seat={seat}): "
                    f"W={wins} L={losses} D={draws} cd={cd:+d}",
                    flush=True,
                )

            elapsed = time.perf_counter() - t0
            entry = dict(
                games=len(specs),
                wins=wins,
                losses=losses,
                draws=draws,
                win_rate=wins / max(1, len(specs)),
                mean_crown_difference=float(np.mean(crown_diffs)) if crown_diffs else 0.0,
                sparky_requests=sparky_requests,
                wall_seconds=round(elapsed, 2),
            )
            if mode == "lookahead_search":
                entry["search_stats"] = dict(stream.stats)
            results[mode] = entry

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("=== PAIRED A/B SUMMARY ===")
    print(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=Path, required=True, help="ppo_latest.pt (with critic) or actor_export.pt")
    p.add_argument("--opponent", type=Path, required=True)
    p.add_argument("--decks", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", default="fp16")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--horizon-steps", type=int, default=10)
    p.add_argument("--trigger-elixir", type=float, default=6.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--out", type=Path)
    run_paired_ab(p.parse_args())


if __name__ == "__main__":
    main()
