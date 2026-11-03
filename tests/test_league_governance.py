import numpy as np

from cr_rl.opponents.admission import SnapshotAdmission
from cr_rl.opponents.league import OpponentEntry, OpponentPool


def test_conditional_ema_forgets_stale_learner_and_survives_restore(tmp_path):
    pool = OpponentPool(
        [OpponentEntry(0, str(tmp_path / "a.pt"), 0)],
        ema_decay=0.5,
        version_decay=0.9,
        max_entries=3,
    )
    for _ in range(8):
        pool.record(0, 1, "hog", "golem", learner_version=4)
        pool.record(0, -1, "golem", "hog", learner_version=4)
    entry = pool.entries[0]
    assert pool.score_for(entry, "hog", "golem", 4) > 0.99
    assert pool.score_for(entry, "golem", "hog", 4) < 0.01
    assert abs(pool.score_for(entry, "hog", "golem", 104) - 0.5) < 1e-4
    restored = OpponentPool.load(pool.save(tmp_path / "pool.json"))
    assert restored.entries[0].matchup_scores == entry.matchup_scores
    assert restored.max_entries == 3 and restored.ema_decay == 0.5
    assert restored.entries[0].games == 16


def test_pool_pruning_preserves_anchor_live_opponent_and_latest():
    entries = [
        OpponentEntry(i, f"p{i}.pt", i, role="anchor" if i == 0 else "main") for i in range(5)
    ]
    pool = OpponentPool(entries, max_entries=3)
    evicted = pool.prune(protected_ids=[3])
    assert [entry.policy_id for entry in evicted] == [1, 2]
    assert [entry.policy_id for entry in pool.entries] == [0, 3, 4]


def test_admission_requires_sample_interval_and_results_then_resets():
    gate = SnapshotAdmission(interval_samples=100, min_games=4, min_score=0.75)
    assert not gate.ready(1000)
    for result in (1, 1, -1, 1):
        gate.record(result)
    assert not gate.ready(99)
    assert gate.ready(100)
    restored = SnapshotAdmission(**gate.state_dict())
    restored.admitted(100)
    assert not restored.ready(1000)
    assert restored.last_admitted_samples == 100 and restored.games == 0


def test_failed_candidate_waits_another_sample_interval_before_retry():
    gate = SnapshotAdmission(interval_samples=100, min_games=1, min_score=0)
    gate.record(1)
    assert gate.ready(100)
    gate.last_failed_candidate_samples = 100
    assert not gate.ready(199)
    assert SnapshotAdmission(**gate.state_dict()).ready(200)


def test_pfsp_uses_requested_deck_pair():
    pool = OpponentPool([OpponentEntry(i, f"p{i}.pt", i) for i in range(3)], ema_decay=0)
    pool.record(0, 1, "hog", "golem")
    pool.record(1, -1, "hog", "golem")
    pool.record(0, -1, "golem", "hog")
    pool.record(1, 1, "golem", "hog")
    rng = np.random.default_rng(15)
    draws = [
        pool.sample_league(rng, 2, 0, 1, 0, own_deck="hog", enemy_deck="golem").policy_id
        for _ in range(200)
    ]
    assert draws.count(1) > 170
