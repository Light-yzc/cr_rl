"""Step and outcome gates for admitting distinct historical snapshots."""

from dataclasses import asdict, dataclass


@dataclass
class SnapshotAdmission:
    interval_samples: int = 250_000
    min_games: int = 20
    min_score: float = 0.55
    last_admitted_samples: int = 0
    games: int = 0
    score_sum: float = 0.0
    last_failed_candidate_samples: int = 0

    def __post_init__(self):
        if self.interval_samples < 1 or self.min_games < 1 or not 0 <= self.min_score <= 1:
            raise ValueError("Invalid snapshot admission gate")

    def record(self, raw_result):
        if raw_result not in (-1, 0, 1):
            raise ValueError("Snapshot gates use raw win/draw/loss results")
        self.games += 1
        self.score_sum += (raw_result + 1) / 2

    def ready(self, learner_samples):
        return (
            learner_samples - max(self.last_admitted_samples, self.last_failed_candidate_samples)
            >= self.interval_samples
            and self.games >= self.min_games
            and self.score_sum / self.games >= self.min_score
        )

    def admitted(self, learner_samples):
        if not self.ready(learner_samples):
            raise ValueError("Snapshot admission gate has not passed")
        self.last_admitted_samples = learner_samples
        self.games, self.score_sum = 0, 0.0

    def state_dict(self):
        return asdict(self)
