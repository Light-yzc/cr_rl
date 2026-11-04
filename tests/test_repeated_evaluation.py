import copy

import pytest

from cr_rl.evaluation.cohort import paired_match_specs
from cr_rl.evaluation.repeated import summarize_repeated_evaluations


def fixtures():
    names = ["a", "b", "c"]
    protocol = dict(
        pair_start=0,
        pair_count=2,
        chunk_pairs=1,
        policy_sha256="p",
        opponent_sha256="o",
        deck_pool_sha256="d",
        pairing_seed=20261103,
        blocks=[
            dict(simulator_seed_base=base, sampling_seed=7 + i)
            for i, base in enumerate([2000000, 3000000, 4000000, 5000000])
        ],
    )
    results = []
    for block in protocol["blocks"]:
        for start in range(2):
            matches = paired_match_specs(
                names, 1, start, simulator_seed_base=block["simulator_seed_base"]
            )
            for game in matches:
                game.update(valid=True, reward=1 if start == 0 else -1)
            results.append(
                dict(
                    policy_sha256="p",
                    opponent_sha256="o",
                    deck_pool_sha256="d",
                    simulator_seed_base=block["simulator_seed_base"],
                    pair_start=start,
                    pairing_seed=20261103,
                    sampling_seed=block["sampling_seed"] + start,
                    pair_count=1,
                    games=2,
                    invalid_games=0,
                    wait_logit_bias=0,
                    opponent_wait_logit_bias=0,
                    matchups=matches,
                )
            )
    return protocol, names, results


def test_repeated_seeds_are_clustered_as_two_deck_pairs_not_eight_independent_pairs():
    protocol, names, results = fixtures()
    result = summarize_repeated_evaluations(protocol, names, results)
    assert result["games"] == 16
    assert result["distinct_deck_pairs"] == 2
    assert result["repetitions_per_pair"] == 4
    assert result["score"] == 0.5
    assert result["deck_pair_cluster_bootstrap_95_interval"] == [0, 1]
    assert not result["evidence_of_improvement"]


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "seed", "deck", "sampling", "weights", "seat"]
)
def test_no_repeated_seed_conclusion_from_incomplete_or_mismatched_protocol(change):
    protocol, names, results = fixtures()
    if change == "missing":
        results.pop()
    elif change == "duplicate":
        results.append(copy.deepcopy(results[0]))
    elif change == "seed":
        results[0]["matchups"][0]["seed"] += 1
    elif change == "deck":
        results[0]["matchups"][0]["deck0"] = "unexpected"
    elif change == "sampling":
        results[0]["sampling_seed"] += 1
    elif change == "weights":
        results[0]["policy_sha256"] = "unexpected"
    else:
        results[0]["matchups"].pop()
    with pytest.raises(ValueError):
        summarize_repeated_evaluations(protocol, names, results)
