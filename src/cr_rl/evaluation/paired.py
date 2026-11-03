"""Descriptive, seat-balanced outcomes and an explicitly limited exact sign test."""

from __future__ import annotations

import math
from collections import defaultdict


def paired_outcomes(evaluation: dict, *, minimum_pairs: int = 1) -> dict:
    groups = defaultdict(list)
    for match in evaluation["matchups"]:
        if not match.get("valid"):
            raise ValueError("Paired analysis requires every game to be complete and valid")
        groups[int(match["pair"])].append(match)
    if len(groups) < minimum_pairs:
        raise ValueError(f"Need at least {minimum_pairs} complete deck pairs")
    outcomes = {
        "swept_wins_2_0": 0,
        "split_pairs_1_1": 0,
        "swept_losses_0_2": 0,
        "win_draw_pairs": 0,
        "loss_draw_pairs": 0,
        "draw_draw_pairs": 0,
    }
    rows = []
    positives = negatives = 0
    for pair, matches in sorted(groups.items()):
        if len(matches) != 2 or {m["learner_seat"] for m in matches} != {0, 1}:
            raise ValueError("Every pair must contain exactly one game per policy seat")
        if any(matches[0].get(k) != matches[1].get(k) for k in ("seed", "deck0", "deck1")):
            raise ValueError("Pair games must use the same physical decks and simulator seed")
        rewards = [float(m["reward"]) for m in matches]
        if any(r not in (-1.0, 0.0, 1.0) for r in rewards):
            raise ValueError("Pair rewards must be terminal win/draw/loss values")
        wins, losses = sum(r > 0 for r in rewards), sum(r < 0 for r in rewards)
        key = (
            "swept_wins_2_0"
            if wins == 2
            else "swept_losses_0_2"
            if losses == 2
            else "split_pairs_1_1"
            if wins == losses == 1
            else "win_draw_pairs"
            if wins == 1
            else "loss_draw_pairs"
            if losses == 1
            else "draw_draw_pairs"
        )
        outcomes[key] += 1
        score = sum(rewards) / 2
        positives += score > 0
        negatives += score < 0
        rows.append(
            {
                "pair": pair,
                "rewards": rewards,
                "net_reward_per_game": score,
                "outcome": key,
                "seed": matches[0].get("seed"),
                "deck0": matches[0].get("deck0"),
                "deck1": matches[0].get("deck1"),
            }
        )
    # Primary requested contrast: 2-0 versus 0-2. Ignore split pairs and draws.
    decisive = outcomes["swept_wins_2_0"] + outcomes["swept_losses_0_2"]
    swept_wins = outcomes["swept_wins_2_0"]
    p = sum(math.comb(decisive, k) for k in range(swept_wins, decisive + 1)) / 2**decisive
    return {
        "pairs": len(rows),
        "games": 2 * len(rows),
        **outcomes,
        "net_swept_pairs": swept_wins - outcomes["swept_losses_0_2"],
        "mean_paired_reward": sum(r["net_reward_per_game"] for r in rows) / len(rows),
        "positive_net_pairs_including_draws": positives,
        "negative_net_pairs_including_draws": negatives,
        "decisive_swept_pairs": decisive,
        "swept_pair_sign_test_one_sided_p": p,
        "sign_test_null": "equal sweep-win and sweep-loss probability, conditional on a sweep",
        "rows": rows,
        "limitations": [
            "A 1-1 split does not prove deck countering; stochastic play can also split a pair.",
            "Few decisive sweeps give low power. With no sweeps p=1, not evidence of equality.",
            "The sign test assumes independent comparable pairs; overlapping decks and one "
            "simulator seed per pair limit that assumption. Freeze bias/cohort before final tests.",
        ],
    }
