from types import SimpleNamespace

import numpy as np
import pytest

from cr_rl.training.game_budget import TrainingGameCounter


def row(
    ids=(1,), *, starts=True, terminal=True, truncated=False, fault=0, valid=True, learner=True
):
    count = len(ids)
    return dict(
        episode_ids=np.asarray(ids),
        episode_start=np.full(count, starts),
        terminated=np.full(count, terminal),
        truncated=np.full(count, truncated),
        fault_codes=np.full(count, fault),
        valid=np.full((count, 2), valid),
        learner_mask=np.full((count, 2), learner),
    )


def test_both_learner_seats_and_repeated_terminal_count_one_game():
    counter = TrainingGameCounter(origin_update=18)
    assert counter.record([row()]) == 1
    assert counter.record([row(starts=False)]) == 0
    assert counter.count == 1
    assert counter.record([row(ids=(2, 3))]) == 2
    assert counter.count == 3


def test_cross_update_episode_and_checkpoint_resume_keep_exact_identities():
    counter = TrainingGameCounter(origin_update=18)
    assert counter.record([row(ids=(5,), terminal=False)]) == 0
    restored = TrainingGameCounter(counter.state_dict())
    assert restored.record([row(ids=(5,), starts=False)]) == 1
    assert restored.origin_update == 18
    assert restored.state_dict()["completed_episode_ids"] == [5]
    again = TrainingGameCounter(restored.state_dict())
    assert again.record([row(ids=(5,), starts=False)]) == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(fault=1),
        dict(truncated=True),
        dict(valid=False),
        dict(learner=False),
        dict(terminal=False),
    ],
)
def test_faults_truncations_invalid_nonlearner_and_partial_games_do_not_count(change):
    counter = TrainingGameCounter()
    assert counter.record([row(**change)]) == 0
    assert counter.count == 0


def test_terminal_fragment_and_tampered_saved_counts_are_rejected():
    counter = TrainingGameCounter()
    with pytest.raises(ValueError, match="fragment"):
        counter.record([row(starts=False)])
    counter.record([row()])
    saved = counter.state_dict()
    saved["completed_games"] = 2
    with pytest.raises(ValueError, match="differs"):
        TrainingGameCounter(saved)
    with pytest.raises(ValueError, match="Duplicated"):
        TrainingGameCounter(dict(started_episode_ids=[1, 1]))
    with pytest.raises(ValueError, match="nonnegative"):
        counter.record([row(ids=(-1,))])


def test_continuous_run_stops_only_after_a_complete_optimizer_checkpoint(monkeypatch):
    from cr_rl.training import selfplay_v2

    runs = []

    class Runner:
        def __init__(self, args):
            self.game_counter = SimpleNamespace(count=0)

        def run_update(self):
            runs.append("published")
            self.game_counter.count += 2

        def close(self):
            runs.append("closed")

    monkeypatch.setattr(selfplay_v2, "PersistentSelfplay", Runner)
    selfplay_v2.run(
        SimpleNamespace(
            batch_size=8,
            grad_accumulation=8,
            critic_mode="independent",
            updates=10,
            min_training_games=5,
        )
    )
    assert runs == ["published", "published", "published", "closed"]
