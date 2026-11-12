import copy
import hashlib
import json

import pytest

from cr_rl.opponents.diversity import DiversityRetention, PAIRED_PROTOCOL
from cr_rl.opponents.league import OpponentEntry, OpponentPool


LEARNER_SHA = "a" * 64


def fixture(tmp_path, *, expert_ids=(2, 4, 6), games=32, learner_wins=8):
    entries = []
    for index in range(11):
        checkpoint = tmp_path / f"opponent_{index}.pt"
        checkpoint.write_bytes(f"immutable-policy-{index}".encode())
        entries.append(
            OpponentEntry(
                index, str(checkpoint), index,
                role="anchor" if index == 0 else "historical",
                games=1, score_sum=1,
            )
        )
    reports = {}
    for index in expert_ids:
        matches = []
        for game in range(games):
            matches.append(dict(
                pair=game // 2, seed=9000 + game // 2,
                learner_seat=game % 2, deck0="hog", deck1="golem",
                valid=True, reward=1 if game < learner_wins else -1,
                info={"fault": 0, "truncated": 0, "illegal_0": 0, "illegal_1": 0},
            ))
        report = dict(
            policy_sha256=LEARNER_SHA,
            opponent_sha256=hashlib.sha256(f"immutable-policy-{index}".encode()).hexdigest(),
            deck_pool_sha256="d" * 64,
            evaluation_code_sha256="e" * 64,
            evaluation_code_file_sha256_at_end="e" * 64,
            wait_logit_bias=0, opponent_wait_logit_bias=0,
            protocol=PAIRED_PROTOCOL,
            games=games, invalid_games=0, wins=learner_wins,
            losses=games-learner_wins, draws=0, sampling_seed=42,
            matchups=matches,
        )
        path = tmp_path / f"evaluation_{index}.json"
        path.write_text(json.dumps(report))
        reports[index] = path
    manifest = dict(
        schema_version=1, evaluated_against_version=182,
        candidate_sha256=LEARNER_SHA, max_age_versions=20,
        experts=[dict(policy_id=i, specialty=f"style-{i}", reports=[str(reports[i])])
                 for i in expert_ids],
    )
    return OpponentPool(entries, max_entries=8), manifest, reports


def ids(pool):
    return {entry.policy_id for entry in pool.entries}


def test_disabled_retention_preserves_existing_archive_selection(tmp_path):
    pool, _, _ = fixture(tmp_path)
    pool.prune()
    assert ids(pool) == {0, 1, 3, 6, 7, 8, 9, 10}
    assert pool.diversity_metadata(182) == {"enabled": False}
    path = pool.save(tmp_path / "pool.json")
    assert "diversity_retention" not in json.loads(path.read_text())


def test_confirmed_specialists_survive_old_age_without_growing_pool(tmp_path):
    pool, manifest, _ = fixture(tmp_path)
    state = pool.configure_diverse_retention(manifest, learner_version=182)
    assert all(record["games"] == 32 for record in state["experts"])
    assert all(record["opponent_score"] == 0.75 for record in state["experts"])
    evicted = pool.prune(learner_version=182)
    assert ids(pool) == {0, 1, 2, 4, 6, 8, 9, 10}
    assert {entry.policy_id for entry in evicted} == {3, 5, 7}
    assert len(pool.entries) == pool.max_entries == 8
    assert pool.diversity_metadata(182)["confirmed_expert_ids"] == [2, 4, 6]
    # Independent retention does not rewrite online PFSP or inflate training games.
    assert all(entry.games == 1 and entry.score_sum == 1 for entry in pool.entries)


def test_live_matches_and_anchor_override_soft_specialty_quotas(tmp_path):
    pool, manifest, _ = fixture(tmp_path)
    pool.configure_diverse_retention(manifest, learner_version=182)
    pool.prune(protected_ids=[5, 6, 7, 8], learner_version=182)
    assert ids(pool) == {0, 2, 5, 6, 7, 8, 9, 10}
    assert 4 not in ids(pool)  # Lower-priority expert yields when all slots are occupied.


def test_impossible_mandatory_capacity_fails_without_deleting_entries(tmp_path):
    pool, manifest, _ = fixture(tmp_path)
    pool.configure_diverse_retention(manifest, learner_version=182)
    pool.max_entries = 3
    before = ids(pool)
    with pytest.raises(RuntimeError, match="active opponents"):
        pool.prune(protected_ids=[5, 6], learner_version=182)
    assert ids(pool) == before


def test_same_specialty_keeps_only_one_soft_expert_and_expiry_removes_protection(tmp_path):
    pool, manifest, _ = fixture(tmp_path)
    for row in manifest["experts"]:
        row["specialty"] = "counterpush"
    pool.configure_diverse_retention(manifest, learner_version=182)
    pool.prune(learner_version=182)
    # Equal evidence chooses the stable smallest policy identity, rather than three copies.
    assert 2 in ids(pool)
    assert len(pool.diversity_retention.active_experts(pool.entries, 202)) > 0
    assert pool.diversity_retention.active_experts(pool.entries, 203) == []
    assert pool.diversity_retention.active_experts(pool.entries, 181) == []
    assert pool.diversity_metadata(203)["confirmed_expert_ids"] == []


def test_configured_retention_survives_pool_restore_with_report_provenance(tmp_path):
    pool, manifest, _ = fixture(tmp_path)
    pool.configure_diverse_retention(manifest, learner_version=182)
    pool.prune(learner_version=182)
    restored = OpponentPool.load(pool.save(tmp_path / "pool.json"))
    assert restored.diversity_retention.state_dict() == pool.diversity_retention.state_dict()
    assert restored.diversity_metadata(182) == pool.diversity_metadata(182)
    assert ids(restored) == ids(pool)


def test_tiny_online_sample_cannot_certify_an_expert(tmp_path):
    pool, manifest, _ = fixture(tmp_path, expert_ids=(2,), games=8, learner_wins=0)
    with pytest.raises(ValueError, match="games"):
        pool.configure_diverse_retention(manifest, learner_version=182)
    assert pool.diversity_retention is None


@pytest.mark.parametrize("bad_case", ["duplicate", "unpaired", "invalid", "wrong_counts",
                                     "wrong_learner", "wrong_checkpoint", "not_hard",
                                     "wait_bias", "opponent_bias", "missing_bias",
                                     "wrong_protocol", "code_changed", "missing_deck_sha",
                                     "truthy_valid", "missing_info", "missing_flag"])
def test_independent_evidence_is_verified_before_mutating_pool(tmp_path, bad_case):
    pool, manifest, paths = fixture(tmp_path, expert_ids=(2,))
    path = paths[2]
    report = json.loads(path.read_text())
    if bad_case == "duplicate":
        manifest["experts"][0]["reports"].append(str(path))
    elif bad_case == "unpaired":
        report["matchups"][1]["learner_seat"] = 0
    elif bad_case == "invalid":
        report["matchups"][0]["info"]["truncated"] = 1
    elif bad_case == "wrong_counts":
        report["wins"] += 1
    elif bad_case == "wrong_learner":
        report["policy_sha256"] = "b" * 64
    elif bad_case == "wrong_checkpoint":
        (tmp_path / "opponent_2.pt").write_bytes(b"different-weights")
    elif bad_case == "not_hard":
        # Strong-looking online counters cannot override actual independent outcomes.
        pool.entries[2].score_sum = 0
        for game in report["matchups"]:
            game["reward"] *= -1
        report["wins"], report["losses"] = report["losses"], report["wins"]
    elif bad_case == "wait_bias":
        report["wait_logit_bias"] = 1.0
    elif bad_case == "opponent_bias":
        report["opponent_wait_logit_bias"] = 1.0
    elif bad_case == "missing_bias":
        report.pop("wait_logit_bias")
    elif bad_case == "wrong_protocol":
        report["protocol"] = "unpaired"
    elif bad_case == "code_changed":
        report["evaluation_code_file_sha256_at_end"] = "f" * 64
    elif bad_case == "missing_deck_sha":
        report.pop("deck_pool_sha256")
    elif bad_case == "truthy_valid":
        report["matchups"][0]["valid"] = 1
    elif bad_case == "missing_info":
        report["matchups"][0].pop("info")
    elif bad_case == "missing_flag":
        report["matchups"][0]["info"].pop("illegal_1")
    path.write_text(json.dumps(report))
    before = copy.deepcopy(pool.entries)
    with pytest.raises(ValueError):
        pool.configure_diverse_retention(manifest, learner_version=182)
    assert pool.diversity_retention is None
    assert pool.entries == before


def test_confirmation_minimum_and_same_frozen_candidate_cannot_be_bypassed(tmp_path):
    pool, manifest, _ = fixture(tmp_path)
    manifest["min_games"] = 8
    with pytest.raises(ValueError, match="min_games"):
        pool.configure_diverse_retention(manifest, learner_version=182)
    manifest.pop("min_games")
    pool.configure_diverse_retention(manifest, learner_version=182)
    state = pool.diversity_retention.state_dict()
    state["experts"][0]["candidate_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="evidence gate"):
        DiversityRetention.from_state(state)


@pytest.mark.parametrize("changed_field", ["deck_pool_sha256", "evaluation_code_sha256"])
def test_reports_for_one_expert_must_share_frozen_protocol(tmp_path, changed_field):
    pool, manifest, paths = fixture(tmp_path, expert_ids=(2,))
    report = json.loads(paths[2].read_text())
    report["sampling_seed"] += 1  # Independent stochastic repetition of physical pairs.
    report[changed_field] = "f" * 64
    if changed_field == "evaluation_code_sha256":
        report["evaluation_code_file_sha256_at_end"] = "f" * 64
    path = tmp_path / "second_report.json"
    path.write_text(json.dumps(report))
    manifest["experts"][0]["reports"].append(str(path))
    with pytest.raises(ValueError, match="different evaluation deck pools|changed evaluation code"):
        pool.configure_diverse_retention(manifest, learner_version=182)
    assert pool.diversity_retention is None


def test_different_experts_cannot_mix_evaluation_deck_catalogues(tmp_path):
    pool, manifest, paths = fixture(tmp_path, expert_ids=(2, 4))
    report = json.loads(paths[4].read_text())
    report["deck_pool_sha256"] = "f" * 64
    paths[4].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different evaluation deck pools"):
        pool.configure_diverse_retention(manifest, learner_version=182)


def test_optional_candidate_checkpoint_is_resolved_and_hashed_before_admission(tmp_path):
    pool, manifest, _ = fixture(tmp_path, expert_ids=(2,))
    candidate = tmp_path / "learner.pt"
    candidate.write_bytes(b"frozen-learner")
    manifest["candidate_checkpoint"] = candidate.name
    path = tmp_path / "experts.json"
    path.write_text(json.dumps(manifest))
    # A valid-looking report digest cannot replace the actual frozen file hash.
    with pytest.raises(ValueError, match="Candidate checkpoint differs"):
        pool.configure_diverse_retention(path, learner_version=182, verify_files=False)
    assert pool.diversity_retention is None

    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    manifest["candidate_sha256"] = candidate_sha
    report_path = tmp_path / "evaluation_2.json"
    report = json.loads(report_path.read_text())
    report["policy_sha256"] = candidate_sha
    report_path.write_text(json.dumps(report))
    manifest["experts"][0]["reports"] = [report_path.name]
    path.write_text(json.dumps(manifest))
    state = pool.configure_diverse_retention(path, learner_version=182)
    assert state["candidate_sha256"] == candidate_sha
    assert "candidate_checkpoint" not in state
