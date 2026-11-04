import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from cr_rl.evaluation.cohort import paired_match_specs
from cr_rl.evaluation.strategy_audit import summarize_strategy
from cr_rl.training.persistence import file_sha256


def load_module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "policy_confirmation", scripts / "confirm_learning_rate_trial.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_requires_complete_fresh_comparisons_and_corrected_p(monkeypatch):
    module = load_module(monkeypatch)
    rows = {
        f"selected_vs_{name}": dict(
            games=32,
            pairs=16,
            native_parity_games=32,
            paired_bootstrap_95_interval=[0.55, 0.9],
            paired=dict(swept_pair_sign_test_one_sided_p=0.01),
        )
        for name in ("stable", "parent", "bc")
    }
    assert module.confirmation_decision(rows)["eligible_for_independent_review"]
    rows["selected_vs_bc"]["paired"]["swept_pair_sign_test_one_sided_p"] = 0.02
    assert not module.confirmation_decision(rows)["eligible_for_independent_review"]
    rows["selected_vs_bc"]["paired"]["swept_pair_sign_test_one_sided_p"] = 0.01
    rows["selected_vs_parent"]["paired_bootstrap_95_interval"][0] = 0.5
    assert not module.confirmation_decision(rows)["eligible_for_independent_review"]
    rows["selected_vs_parent"]["games"] = 31
    with pytest.raises(ValueError, match="all 16"):
        module.confirmation_decision(rows)
    del rows["selected_vs_parent"]
    with pytest.raises(ValueError, match="Missing"):
        module.confirmation_decision(rows)


def test_confirmation_replays_outcomes_and_actor_bindings_are_verified(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    root = Path(__file__).parents[1]
    names = list(yaml.safe_load((root / "configs/decks_eval.yaml").read_text())["decks"])

    def write(name, data):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        return p

    binding = {}
    for name in ("selected", "stable", "parent", "bc"):
        p = tmp_path / f"{name}.pt"
        p.write_bytes(name.encode())
        binding[name] = dict(path=str(p), sha256=file_sha256(p))
    binding_path = write("frozen_policies.json", binding)
    protocol = dict(
        games=96,
        pair_starts=[24, 28, 32, 36],
        simulator_seed_base=9000000,
        sampling_seed=20261108,
        pairing_seed=20261103,
        decks="configs/decks_eval.yaml",
        decks_sha256=file_sha256(root / "configs/decks_eval.yaml"),
        evaluation_code_sha256=file_sha256(root / "scripts/evaluate_policy.py"),
    )
    write("confirmation_protocol.json", protocol)
    results = {}
    side = dict(
        successful_card_plays=1,
        paid_card_elixir_total=3,
        successful_card_histogram={"card": 1},
        core_cost_thresholds={"5": dict(count=0)},
        elixir_before_successful_card=dict(mean=4),
        elixir_after_successful_card=dict(mean=1),
    )
    for opponent in ("stable", "parent", "bc"):
        name = "selected_vs_" + opponent
        matches = []
        chunks = []
        for start in protocol["pair_starts"]:
            specs = paired_match_specs(
                names, 4, start, simulator_seed_base=9000000, pairing_seed=20261103
            )
            games = [dict(s, valid=True, reward=1) for s in specs]
            directory = f"evaluation/{name}/pairs_{start}"
            report = dict(
                matchups=games,
                policy_sha256=binding["selected"]["sha256"],
                opponent_sha256=binding[opponent]["sha256"],
                deck_pool_sha256=protocol["decks_sha256"],
                evaluation_code_sha256=protocol["evaluation_code_sha256"],
                evaluation_code_file_sha256_at_end=protocol["evaluation_code_sha256"],
                wait_logit_bias=0,
                opponent_wait_logit_bias=0,
                simulator_seed_base=9000000,
                pairing_seed=20261103,
                sampling_seed=20261108 + start,
                pair_start=start,
                pair_count=4,
                games=8,
                invalid_games=0,
            )
            report_path = write(directory + "/evaluation.json", report)
            native = []
            for index, game in enumerate(games):
                replay = write(directory + f"/match_{index:03d}.json", dict(fixture=index))
                native.append(
                    dict(
                        game,
                        native_outcome_parity=True,
                        replay_sha256=file_sha256(replay),
                        learner=side,
                        opponent=side,
                        active_simulation_seconds=60,
                        decision_audit=dict(
                            learner=dict(full_elixir_wait_legal_seconds=0),
                            opponent=dict(full_elixir_wait_legal_seconds=0),
                        ),
                    )
                )
            audit = dict(evaluation_sha256=file_sha256(report_path), games=native)
            write(directory + "/execution_audit.json", audit)
            chunks.append((report, audit))
            matches.extend(games)
        row = module.summarize_matches(matches)
        row.update(native_parity_games=32, strategy=summarize_strategy(chunks))
        results[name] = row
    result_path = write(
        "confirmation_results.json",
        dict(
            protocol=protocol,
            policies_sha256=file_sha256(binding_path),
            comparisons=results,
            decision=module.confirmation_decision(results),
        ),
    )
    write(
        "confirmation_status.json",
        dict(phase="completed", pid=999, summary_sha256=file_sha256(result_path)),
    )
    assert module.verify_saved_confirmation(tmp_path, alive=lambda pid: False)["games"] == 96
    p = tmp_path / "stable.pt"
    original = p.read_bytes()
    p.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="actor"):
        module.verify_saved_confirmation(tmp_path, alive=lambda pid: False)
    p.write_bytes(original)
    replay = tmp_path / "evaluation/selected_vs_stable/pairs_24/match_000.json"
    replay.write_text("{}")
    with pytest.raises(ValueError, match="replay"):
        module.verify_saved_confirmation(tmp_path, alive=lambda pid: False)
