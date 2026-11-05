import numpy as np

from cr_rl.opponents.admission import SnapshotAdmission
from cr_rl.opponents.league import OpponentEntry, OpponentPool
from cr_rl.opponents.role_sampling import RoleMixture, historical_policy_id
from cr_rl.training.episode_trace import on_policy_seats


def test_role_draws_separate_bc_history_and_current_and_restore(tmp_path):
    pool = OpponentPool(
        [
            OpponentEntry(0, "bc.pt", 0, role="anchor"),
            *[
                OpponentEntry(historical_policy_id(g), f"g{g}.pt", g, role="historical")
                for g in (6, 10, 14)
            ],
        ],
        ema_decay=0,
    )
    pool.record(historical_policy_id(6), -1, "a", "b", 14)
    pool.record(historical_policy_id(10), 1, "a", "b", 14)
    config = RoleMixture(**RoleMixture().state_dict())
    rng = np.random.default_rng(184)
    counts = dict(current=0, historical=0, anchor=0)
    opponents = []
    for _ in range(10000):
        entry, requested, actual = config.sample(pool, rng, "a", "b", 14)
        assert requested == actual
        counts[actual] += 1
        if entry:
            assert entry.role == actual
            opponents.append(entry.policy_id)
    for role, fraction in config.state_dict().items():
        assert abs(counts[role] / 10000 - fraction) < 0.025
    assert opponents.count(historical_policy_id(6)) > 5 * opponents.count(historical_policy_id(10))
    # Generation14 frozen opponent cannot contribute its seat as current behavior14.
    np.testing.assert_array_equal(
        on_policy_seats(np.array([[14, historical_policy_id(14)]]), 14, [0]), [[1, 0]]
    )
    restored = OpponentPool.load(pool.save(tmp_path / "pool.json"))
    assert [e.role for e in restored.entries] == [e.role for e in pool.entries]


def test_missing_history_is_explicit_fallback_and_history_gate_does_not_promote():
    pool = OpponentPool([OpponentEntry(0, "bc.pt", 0, role="anchor")])
    entry, requested, actual = RoleMixture(0, 1, 0).sample(
        pool, np.random.default_rng(0), "a", "b", 14
    )
    assert entry.policy_id == 0 and requested == "historical" and actual == "anchor"
    gate = SnapshotAdmission(interval_samples=20000, last_admitted_samples=67701, purpose="history")
    assert not gate.ready(87700) and gate.ready(87701)
    gate.admitted(87701)
    assert not SnapshotAdmission(**gate.state_dict()).ready(87702)
    assert not SnapshotAdmission(interval_samples=20000).ready(87701)


def test_reserved_history_ids_prune_by_generation():
    pool = OpponentPool(
        [
            OpponentEntry(0, "bc.pt", 0, role="anchor"),
            *[
                OpponentEntry(historical_policy_id(g), f"g{g}.pt", g, role="historical")
                for g in (6, 10, 14, 20)
            ],
        ],
        max_entries=3,
    )
    pool.prune([historical_policy_id(6)])
    assert {e.generation for e in pool.entries} == {0, 6, 20}


def test_branch_snapshot_has_unique_retained_identity_and_restores(tmp_path):
    from cr_rl.opponents.role_sampling import frozen_policy_version, next_history_snapshot_id

    old = OpponentEntry(historical_policy_id(14), "old14.pt", 14, role="historical")
    anchor = OpponentEntry(0, "bc.pt", 0, role="anchor")
    later = OpponentEntry(historical_policy_id(22), "old22.pt", 22, role="historical")
    entries = [anchor, old, later]
    new_id = next_history_snapshot_id(entries, 14)
    assert new_id == historical_policy_id(23)
    new = OpponentEntry(new_id, "new14.pt", 14, role="historical")
    pool = OpponentPool([*entries, new])
    assert frozen_policy_version(old) != frozen_policy_version(new)
    assert frozen_policy_version(anchor) == historical_policy_id(0)
    for entry in (old, new):
        np.testing.assert_array_equal(
            on_policy_seats(np.array([[14, frozen_policy_version(entry)]]), 14, [0]), [[1, 0]]
        )
    restored = OpponentPool.load(pool.save(tmp_path / "pool.json"))
    assert [frozen_policy_version(e) for e in restored.entries] == [
        frozen_policy_version(e) for e in pool.entries
    ]
    assert next_history_snapshot_id(entries, 15) == historical_policy_id(15)
