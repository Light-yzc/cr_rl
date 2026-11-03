from cr_rl.training.telemetry import TrainingTelemetry


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logs = []
        self.definitions = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, values):
        self.logs.append(values)


def test_independent_value_updates_have_their_own_optimizer_axis():
    run = FakeRun()
    telemetry = TrainingTelemetry(run, 64)
    telemetry.log(
        {
            "phase": "critic_optimization",
            "update": 1,
            "critic_optimizer_steps": 12,
            "value_loss": 0.4,
            "grad_norm": 1.2,
            "learning_rate": 1e-4,
        }
    )
    assert run.logs[-1]["charts/critic_optimizer_steps"] == 12
    assert run.logs[-1]["critic/value_loss"] == 0.4
    assert "ppo/value_loss" not in run.logs[-1]


def test_progress_does_not_create_constant_learning_charts():
    run = FakeRun()
    telemetry = TrainingTelemetry(run, 3072)
    telemetry.log(
        {
            "phase": "rollout",
            "update": 1,
            "rollout_step": 0,
            "learner_samples_total": 0,
            "seconds": 1,
            "rollout_total": 1536,
        }
    )
    assert run.logs == []
    assert run.summary["progress"]["update"] == 1
    telemetry.log(
        {
            "phase": "policy",
            "update": 1,
            "learner_samples_total": 128,
            "entropy": 2.5,
            "play_fraction": 0.2,
            "wait_when_play_legal": None,
        }
    )
    assert run.logs[-1] == {
        "charts/learner_samples": 128,
        "policy/entropy": 2.5,
        "policy/play_fraction": 0.2,
    }


def test_ppo_uses_optimizer_axis_and_does_not_fabricate_missing_metrics():
    run = FakeRun()
    telemetry = TrainingTelemetry(run, 3072)
    telemetry.log(
        {
            "phase": "optimization",
            "update": 2,
            "optimizer_steps": 3,
            "optimizer_steps_total": 23,
            "loss": 0.4,
            "grad_norm": 1.2,
        }
    )
    assert run.logs[-1]["charts/optimizer_steps"] == 23
    assert run.logs[-1]["ppo/total_loss"] == 0.4
    assert run.logs[-1]["ppo/grad_norm"] == 1.2
    assert "ppo/entropy" not in run.logs[-1]


def test_faulted_games_do_not_inflate_win_rate():
    run = FakeRun()
    telemetry = TrainingTelemetry(run, 3072)
    for reward, fault in [(1, 0), (-1, 0), (1, 1)]:
        telemetry.log(
            {
                "phase": "episode",
                "update": 1,
                "reward": reward,
                "episode_length": 100,
                "fault": fault,
                "learner_deck": "hog",
            }
        )
    assert telemetry.episodes == 2
    assert run.logs[-1]["selfplay/win_rate"] == 0.5
    assert run.summary["per_deck_training_outcomes"]["hog"] == {
        "games": 2,
        "wins": 1,
        "losses": 1,
        "draws": 0,
    }


def test_rollout_throughput_uses_actual_two_seat_samples_after_variable_updates():
    run = FakeRun()
    telemetry = TrainingTelemetry(run, 3072)
    telemetry.log(
        {
            "phase": "rollout",
            "update": 3,
            "rollout_step": 64,
            "learner_samples_total": 11000,
            "rollout_samples": 256,
            "seconds": 8,
        }
    )
    assert run.logs[-1]["performance/rollout_samples_per_second"] == 32
