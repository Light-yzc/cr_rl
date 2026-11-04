import importlib.util
from pathlib import Path

import pytest

from cr_rl.evaluation.cohort import paired_match_specs


def test_continuation_eval_rejects_changed_weights_seeds_bias_and_duplicate_seats(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "continued_eval", scripts / "evaluate_continued_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [f"deck{i}" for i in range(5)]
    protocol = dict(
        decks_sha256="deck-sha",
        evaluation_code_sha256="code-sha",
        simulator_seed_base=6000000,
        sampling_seed=20261105,
        pairing_seed=20261103,
    )
    result = dict(
        policy_sha256="actor-sha",
        opponent_sha256="baseline-sha",
        deck_pool_sha256="deck-sha",
        evaluation_code_sha256="code-sha",
        evaluation_code_file_sha256_at_end="code-sha",
        wait_logit_bias=0,
        opponent_wait_logit_bias=0,
        simulator_seed_base=6000000,
        pairing_seed=20261103,
        sampling_seed=20261105,
        pair_start=0,
        pair_count=4,
        games=8,
        invalid_games=0,
        matchups=[
            dict(r, valid=True)
            for r in paired_match_specs(
                names, 4, 0, simulator_seed_base=6000000, pairing_seed=20261103
            )
        ],
    )
    module.verify_result(result, "actor-sha", "baseline-sha", protocol, 0, names)
    for key, value in [
        ("policy_sha256", "wrong"),
        ("wait_logit_bias", 1),
        ("simulator_seed_base", 1),
    ]:
        with pytest.raises(ValueError):
            module.verify_result(
                dict(result, **{key: value}), "actor-sha", "baseline-sha", protocol, 0, names
            )
    duplicate = dict(result, matchups=[result["matchups"][0]] * 8)
    with pytest.raises(ValueError):
        module.verify_result(duplicate, "actor-sha", "baseline-sha", protocol, 0, names)
