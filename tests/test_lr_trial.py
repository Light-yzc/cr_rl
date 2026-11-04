import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from cr_rl.evaluation.cohort import paired_match_specs
from cr_rl.training.persistence import file_sha256


def load_module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "lr_trial", scripts / "run_learning_rate_trial.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trial_protocol_is_fixed_and_preference_does_not_promote(monkeypatch):
    module = load_module(monkeypatch)
    rows = module.comparisons()
    assert sum(len(r["starts"]) * 8 for r in rows) == 96
    assert rows[0]["policy"] == "candidate" and rows[0]["opponent"] == "control"
    summary = dict(
        comparisons={
            f"generation22_vs_{name}": dict(evidence_of_improvement=True)
            for name in ("generation6", "bc5096")
        }
    )
    assert not module.trial_needed(summary)
    summary["comparisons"]["generation22_vs_generation6"]["evidence_of_improvement"] = False
    assert module.trial_needed(summary)
    scores = {r["name"]: dict(score=0.5) for r in rows}
    scores["candidate_vs_control"]["score"] = 0.625
    decision = module.development_preference(scores)
    assert decision["preferred"] == "candidate" and decision["promotion"].startswith("none")
    scores["candidate_vs_source"]["score"] = 0.375
    assert module.development_preference(scores)["preferred"] == "inconclusive"


def test_trial_release_recomputes_games_and_rejects_changed_weights_or_replays(
    tmp_path, monkeypatch
):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module, "process_alive", lambda pid: False)
    root = Path(__file__).parents[1]
    names = list(yaml.safe_load((root / "configs/decks_eval.yaml").read_text())["decks"])

    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    actors = dict(source="a" * 64, bc="b" * 64)
    for branch in ("control", "candidate"):
        files = []
        for relative in ("checkpoints/update_000024.pt", "exports/update_000024.actor.pt"):
            p = tmp_path / branch / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(branch.encode())
            files.append(dict(path=relative, sha256=file_sha256(p), bytes=p.stat().st_size))
        pool = write(f"{branch}/pool.json", {})
        write(
            f"{branch}/latest.json",
            dict(update=24, pool="pool.json", pool_sha256=file_sha256(pool), files=files),
        )
        actors[branch] = files[-1]["sha256"]
    binding = write(
        "source_binding.json", dict(actor_sha256=actors["source"], bc_sha256=actors["bc"])
    )
    protocol = dict(
        source_generation=22,
        target_generation=24,
        updates_per_branch=2,
        actor_learning_rates=dict(control=2e-5, candidate=1e-5),
        comparisons=module.comparisons(),
        games=96,
        simulator_seed_base=8000000,
        sampling_seed=20261107,
        pairing_seed=20261103,
        decks="configs/decks_eval.yaml",
        decks_sha256=file_sha256(root / "configs/decks_eval.yaml"),
        evaluation_code_sha256=file_sha256(root / "scripts/evaluate_policy.py"),
    )
    write("trial_protocol.json", protocol)
    results = {}
    for comparison in protocol["comparisons"]:
        matches = []
        for start in comparison["starts"]:
            specs = paired_match_specs(
                names,
                4,
                start,
                simulator_seed_base=protocol["simulator_seed_base"],
                pairing_seed=protocol["pairing_seed"],
            )
            games = [dict(s, valid=True, reward=1) for s in specs]
            directory = f"evaluation/{comparison['name']}/pairs_{start}"
            report = write(
                directory + "/evaluation.json",
                dict(
                    matchups=games,
                    policy_sha256=actors[comparison["policy"]],
                    opponent_sha256=actors[comparison["opponent"]],
                    deck_pool_sha256=protocol["decks_sha256"],
                    evaluation_code_sha256=protocol["evaluation_code_sha256"],
                    evaluation_code_file_sha256_at_end=protocol["evaluation_code_sha256"],
                    wait_logit_bias=0,
                    opponent_wait_logit_bias=0,
                    simulator_seed_base=protocol["simulator_seed_base"],
                    pairing_seed=protocol["pairing_seed"],
                    sampling_seed=protocol["sampling_seed"] + start,
                    pair_start=start,
                    pair_count=4,
                    games=8,
                    invalid_games=0,
                ),
            )
            native = []
            for i, game in enumerate(games):
                p = write(directory + f"/match_{i:03d}.json", {"fixture": i})
                native.append(dict(game, native_outcome_parity=True, replay_sha256=file_sha256(p)))
            write(
                directory + "/execution_audit.json",
                dict(evaluation_sha256=file_sha256(report), games=native),
            )
            matches.extend(games)
        result = module.summarize_matches(matches)
        result.update(
            policy_sha256=actors[comparison["policy"]],
            opponent_sha256=actors[comparison["opponent"]],
            native_parity_games=len(matches),
        )
        results[comparison["name"]] = result
    result_path = write(
        "trial_results.json",
        dict(
            protocol=protocol,
            comparisons=results,
            source_binding_sha256=file_sha256(binding),
            decision=module.development_preference(results),
        ),
    )
    write("trial_status.json", dict(phase="completed", summary_sha256=file_sha256(result_path)))
    assert module.verify_trial_completion(tmp_path)["games"] == 96
    checkpoint = tmp_path / "control/checkpoints/update_000024.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        module.verify_trial_completion(tmp_path)
    checkpoint.write_bytes(original)
    replay = tmp_path / "evaluation/candidate_vs_control/pairs_0/match_000.json"
    replay.write_text("{}")
    with pytest.raises(ValueError, match="Native replay"):
        module.verify_trial_completion(tmp_path)
