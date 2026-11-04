import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pilot_eval", Path(__file__).parents[1] / "scripts/evaluate_ppo_pilot.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def matches(rewards):
    return [
        dict(
            pair=i // 2,
            learner_seat=i % 2,
            reward=r,
            valid=True,
            seed=i // 2 + 1,
            deck0="A",
            deck1="B",
        )
        for i, r in enumerate(rewards)
    ]


def test_pair_summary_keeps_swapped_seats_as_the_resampling_unit():
    result = module.summarize_matches(matches([1, -1] * 16))
    assert result["games"] == 32 and result["pairs"] == 16
    assert result["score"] == 0.5
    assert result["paired_bootstrap_95_interval"] == [0.5, 0.5]
    assert not result["evidence_of_improvement"]
    assert module.summarize_matches(matches([1, 1] * 16))["evidence_of_improvement"]
    assert module.summarize_matches(matches([0, 0] * 16))["score"] == 0.5


@pytest.mark.parametrize("change", ["missing", "duplicate", "invalid", "seed"])
def test_pair_summary_rejects_invalid_or_unpaired_evidence(change):
    rows = matches([1, -1])
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "invalid":
        rows[0]["valid"] = False
    else:
        rows[1]["seed"] += 1
    with pytest.raises(ValueError):
        module.summarize_matches(rows)


def test_four_sweeps_are_not_declared_improvement_from_bootstrap_alone():
    result = module.summarize_matches(matches([1, 1] * 4 + [1, -1] * 12))
    assert result["paired_bootstrap_95_interval"][0] > 0.5
    assert result["paired"]["swept_pair_sign_test_one_sided_p"] == 1 / 16
    assert not result["evidence_of_improvement"]


def chunk():
    rows = matches([1, -1])
    for row in rows:
        row["seed"] = 1000000
        row["info"] = dict(fault=0, truncated=False)
    report = dict(
        policy_sha256="p",
        opponent_sha256="o",
        deck_pool_sha256="d",
        evaluation_code_sha256="c",
        evaluation_code_file_sha256_at_end="c",
        wait_logit_bias=0,
        opponent_wait_logit_bias=0,
        pair_start=0,
        pair_count=1,
        sampling_seed=20261103,
        games=2,
        invalid_games=0,
        matchups=rows,
    )
    arguments = dict(
        policy_sha="p",
        opponent_sha="o",
        decks_sha="d",
        code_sha="c",
        expected_pairs=[(0, "A", "B")],
        start=0,
    )
    return report, arguments


def test_registered_chunk_accepts_exact_physical_pair_and_source_hashes():
    report, arguments = chunk()
    assert module.validate_chunk(report, **arguments) == report["matchups"]


@pytest.mark.parametrize(
    "change", ["weights", "decks", "bias", "code", "seed", "physical_deck", "fault"]
)
def test_registered_chunk_rejects_wrong_evidence(change):
    report, arguments = chunk()
    if change == "weights":
        report["policy_sha256"] = "different"
    elif change == "decks":
        report["deck_pool_sha256"] = "different"
    elif change == "bias":
        report["wait_logit_bias"] = 1.5
    elif change == "code":
        report["evaluation_code_file_sha256_at_end"] = "different"
    elif change == "seed":
        report["matchups"][0]["seed"] += 1
    elif change == "physical_deck":
        report["matchups"][0]["deck0"] = "different"
    else:
        report["matchups"][0]["info"]["fault"] = 1
    with pytest.raises(ValueError):
        module.validate_chunk(report, **arguments)
