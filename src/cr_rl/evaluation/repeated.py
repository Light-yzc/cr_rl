"""Aggregate registered seed repetitions without counting repeats as new deck pairs."""

import math

import numpy as np

from cr_rl.evaluation.cohort import paired_match_specs


def summarize_repeated_evaluations(protocol, names, evaluations):
    starts = list(
        range(
            protocol["pair_start"],
            protocol["pair_start"] + protocol["pair_count"],
            protocol["chunk_pairs"],
        )
    )
    blocks = protocol["blocks"]
    expected = {(b["simulator_seed_base"], start): b for b in blocks for start in starts}
    if len(expected) != len(blocks) * len(starts):
        raise ValueError("Repeated simulator seed blocks")
    chunks, scores = set(), {}
    wins = losses = draws = 0
    for result in evaluations:
        key = result["simulator_seed_base"], result["pair_start"]
        if key not in expected or key in chunks:
            raise ValueError("Unexpected or duplicated evaluation chunk")
        block = expected[key]
        count = min(
            protocol["chunk_pairs"], protocol["pair_start"] + protocol["pair_count"] - key[1]
        )
        required = dict(
            policy_sha256=protocol["policy_sha256"],
            opponent_sha256=protocol["opponent_sha256"],
            deck_pool_sha256=protocol["deck_pool_sha256"],
            pairing_seed=protocol["pairing_seed"],
            sampling_seed=block["sampling_seed"] + key[1],
            pair_count=count,
            games=2 * count,
            invalid_games=0,
            wait_logit_bias=0,
            opponent_wait_logit_bias=0,
        )
        if any(result.get(k) != v for k, v in required.items()):
            raise ValueError("Evaluation does not match frozen protocol")
        planned = paired_match_specs(
            names, count, key[1], simulator_seed_base=key[0], pairing_seed=protocol["pairing_seed"]
        )
        actual = {(m["pair"], m["learner_seat"]): m for m in result["matchups"]}
        if len(actual) != len(planned) or len(actual) != len(result["matchups"]):
            raise ValueError("Missing or duplicated seat swap")
        pairs = {}
        for spec in planned:
            game = actual.get((spec["pair"], spec["learner_seat"]))
            if game is None or any(game.get(k) != v for k, v in spec.items()):
                raise ValueError("Physical seed, cards or seat differ")
            if not game.get("valid") or game["reward"] not in (-1, 0, 1):
                raise ValueError("Invalid game or terminal reward")
            pairs.setdefault(spec["pair"], []).append((game["reward"] + 1) / 2)
            wins += game["reward"] > 0
            losses += game["reward"] < 0
            draws += game["reward"] == 0
        for pair, values in pairs.items():
            scores.setdefault(pair, []).append(float(np.mean(values)))
        chunks.add(key)
    if chunks != set(expected) or any(len(v) != len(blocks) for v in scores.values()):
        raise ValueError("Registered repetitions are incomplete; no interim inference")
    values = np.array([np.mean(scores[pair]) for pair in sorted(scores)])
    rng = np.random.default_rng(20261104)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975]).tolist()
    positives, negatives = int((values > 0.5).sum()), int((values < 0.5).sum())
    decisive = positives + negatives
    sign_p = sum(math.comb(decisive, k) for k in range(positives, decisive + 1)) / 2**decisive
    return dict(
        games=wins + losses + draws,
        wins=wins,
        losses=losses,
        draws=draws,
        distinct_deck_pairs=len(values),
        repetitions_per_pair=len(blocks),
        score=float(values.mean()),
        deck_pair_cluster_bootstrap_95_interval=[low, high],
        positive_pairs=positives,
        negative_pairs=negatives,
        paired_sign_test_one_sided_p=sign_p,
        evidence_of_improvement=low > 0.5 and sign_p <= 0.05,
        per_pair=[
            dict(pair=p, score=float(np.mean(scores[p])), seed_scores=scores[p])
            for p in sorted(scores)
        ],
        limitation=(
            "Fixed overlapping deck pairs; exploratory replication across seeds, "
            "not independent new decks"
        ),
    )
