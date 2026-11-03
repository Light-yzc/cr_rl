from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OpponentEntry:
    policy_id: int
    checkpoint: str
    generation: int
    games: int = 0
    score_sum: float = 0.0
    deck_id: str = "default"
    role: str = "main"
    ema_score: float | None = None
    last_learner_version: int = 0
    matchup_scores: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        if self.ema_score is not None:
            return self.ema_score
        return (self.score_sum + 1.0) / (self.games + 2.0)


class OpponentPool:
    """Small frozen-policy league with prioritized fictitious self-play sampling."""

    def __init__(
        self,
        entries: list[OpponentEntry] | None = None,
        uniform_mix: float = 0.1,
        max_entries: int | None = None,
        ema_decay: float | None = None,
        version_decay: float = 0.99,
    ):
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError("uniform_mix must be in [0, 1]")
        self.entries = list(entries or [])
        self.uniform_mix = float(uniform_mix)
        if max_entries is not None and max_entries < 2:
            raise ValueError("A bounded pool needs at least two slots")
        if ema_decay is not None and not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0 < version_decay <= 1:
            raise ValueError("version_decay must be in (0, 1]")
        self.max_entries, self.ema_decay, self.version_decay = max_entries, ema_decay, version_decay

    def add(self, entry: OpponentEntry) -> None:
        if any(item.policy_id == entry.policy_id for item in self.entries):
            raise ValueError(f"duplicate policy_id {entry.policy_id}")
        self.entries.append(entry)

    def record(
        self, policy_id: int, score: float, own_deck=None, enemy_deck=None, learner_version=0
    ) -> None:
        if score not in (-1.0, 0.0, 1.0):
            raise ValueError("score is from the learner perspective and must be -1, 0, or 1")
        for entry in self.entries:
            if entry.policy_id == policy_id:
                entry.games += 1
                entry.score_sum += (score + 1.0) / 2.0
                if self.ema_decay is not None:
                    value = (score + 1.0) / 2.0
                    previous = self.score_for(entry, learner_version=learner_version)
                    entry.ema_score = self.ema_decay * previous + (1 - self.ema_decay) * value
                    entry.last_learner_version = learner_version
                    if own_deck is not None and enemy_deck is not None:
                        key = json.dumps([own_deck, enemy_deck], separators=(",", ":"))
                        row = entry.matchup_scores.setdefault(
                            key, {"score": 0.5, "games": 0, "version": learner_version}
                        )
                        previous = 0.5 + (row["score"] - 0.5) * self.version_decay ** max(
                            0, learner_version - row["version"]
                        )
                        row.update(
                            score=self.ema_decay * previous + (1 - self.ema_decay) * value,
                            games=row["games"] + 1,
                            version=learner_version,
                        )
                return
        raise KeyError(policy_id)

    def score_for(self, entry, own_deck=None, enemy_deck=None, learner_version=None):
        score, version = entry.score, entry.last_learner_version
        if own_deck is not None and enemy_deck is not None:
            key = json.dumps([own_deck, enemy_deck], separators=(",", ":"))
            if key in entry.matchup_scores:
                row = entry.matchup_scores[key]
                score, version = row["score"], row["version"]
        if learner_version is not None and self.ema_decay is not None:
            score = 0.5 + (score - 0.5) * self.version_decay ** max(0, learner_version - version)
        return score

    def prune(self, protected_ids=()):
        """Return evicted records; checkpoint deletion requires the caller's live-match check."""
        evicted = []
        if self.max_entries is None:
            return evicted
        protected = set(protected_ids)
        if self.entries:
            protected.add(max(entry.policy_id for entry in self.entries))
        if len(self.entries) <= self.max_entries:
            return evicted
        protected.update(entry.policy_id for entry in self.entries if entry.role == "anchor")
        if len(protected) > self.max_entries:
            raise RuntimeError("Pool capacity is smaller than its protected live opponents")
        ordered = sorted(self.entries, key=lambda entry: entry.generation)
        # Default capacity 8: four recent versions plus four temporally spaced
        # archives. Live-match references and anchors take precedence.
        for entry in reversed(ordered[-min(4, self.max_entries // 2) :]):
            if len(protected) < self.max_entries:
                protected.add(entry.policy_id)
        archives = [entry for entry in ordered if entry.policy_id not in protected]
        slots = self.max_entries - len(protected)
        if slots:
            for index in np.linspace(0, len(archives) - 1, slots, dtype=int):
                protected.add(archives[index].policy_id)
        evicted = [entry for entry in ordered if entry.policy_id not in protected]
        self.entries = [entry for entry in self.entries if entry.policy_id in protected]
        return evicted

    def probabilities(self, alpha: float = 1.0) -> np.ndarray:
        if not self.entries:
            return np.zeros(0, dtype=np.float64)
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        # PFSP: focus on opponents the learner does not reliably beat. The small uniform
        # component prevents the league from forgetting old strategic niches.
        scores = np.asarray([entry.score for entry in self.entries], dtype=np.float64)
        priority = np.maximum(1.0 - scores, 1e-3) ** alpha
        priority /= priority.sum()
        return (1.0 - self.uniform_mix) * priority + self.uniform_mix / len(priority)

    def sample(self, rng: np.random.Generator, alpha: float = 1.0) -> OpponentEntry:
        if not self.entries:
            raise RuntimeError("cannot sample from an empty opponent pool")
        index = int(rng.choice(len(self.entries), p=self.probabilities(alpha)))
        return self.entries[index]

    def sample_league(
        self,
        rng: np.random.Generator,
        current_policy_id: int,
        current_fraction: float = 0.35,
        historical_pfsp_fraction: float = 0.50,
        forgotten_fraction: float = 0.15,
        alpha: float = 1.0,
        own_deck=None,
        enemy_deck=None,
        learner_version=None,
    ) -> OpponentEntry:
        """AlphaStar-inspired mix of current self-play, historical PFSP and forgotten foes."""
        fractions = (current_fraction, historical_pfsp_fraction, forgotten_fraction)
        if any(value < 0.0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
            raise ValueError("league sampling fractions must be non-negative and sum to one")
        current = next((x for x in self.entries if x.policy_id == current_policy_id), None)
        if current is None:
            raise KeyError(f"current policy {current_policy_id} is missing from the league")
        historical = [x for x in self.entries if x.policy_id != current_policy_id]
        if not historical:
            return current
        draw = float(rng.random())
        if draw < current_fraction:
            return current
        if draw < current_fraction + historical_pfsp_fraction:
            scores = np.asarray(
                [
                    self.score_for(entry, own_deck, enemy_deck, learner_version)
                    for entry in historical
                ],
                dtype=np.float64,
            )
            weights = np.maximum(1.0 - scores, 1e-3) ** alpha
            weights /= weights.sum()
            weights = (1 - self.uniform_mix) * weights + self.uniform_mix / len(weights)
            return historical[int(rng.choice(len(historical), p=weights))]
        return historical[int(rng.integers(0, len(historical)))]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for entry in self.entries:
            row = asdict(entry)
            checkpoint = Path(entry.checkpoint).resolve()
            if checkpoint.is_relative_to(target.parent.resolve()):
                row["checkpoint"] = checkpoint.relative_to(target.parent.resolve()).as_posix()
            entries.append(row)
        target.write_text(
            json.dumps(
                {
                    "uniform_mix": self.uniform_mix,
                    "entries": entries,
                    "max_entries": self.max_entries,
                    "ema_decay": self.ema_decay,
                    "version_decay": self.version_decay,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> OpponentPool:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in raw["entries"]:
            checkpoint = Path(row["checkpoint"])
            if not checkpoint.is_absolute():
                row["checkpoint"] = str((Path(path).resolve().parent / checkpoint).resolve())
        return cls(
            [OpponentEntry(**row) for row in raw["entries"]],
            raw["uniform_mix"],
            max_entries=raw.get("max_entries"),
            ema_decay=raw.get("ema_decay"),
            version_decay=raw.get("version_decay", 0.99),
        )


def solve_zero_sum_meta(
    payoff: np.ndarray, iterations: int = 20_000, epsilon: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, float]:
    """Approximate Nash strategies for a finite two-player zero-sum payoff matrix.

    Rows are player 0 policies, columns player 1 policies, and entries are player 0
    expected scores. Regret matching returns average strategies, not the last iterate.
    """
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("payoff must be a non-empty finite matrix")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rows, cols = matrix.shape
    row_regret = np.zeros(rows, dtype=np.float64)
    col_regret = np.zeros(cols, dtype=np.float64)
    row_sum = np.zeros(rows, dtype=np.float64)
    col_sum = np.zeros(cols, dtype=np.float64)
    uniform_row = np.full(rows, 1.0 / rows)
    uniform_col = np.full(cols, 1.0 / cols)
    for _ in range(iterations):
        row_positive = np.maximum(row_regret, 0.0)
        col_positive = np.maximum(col_regret, 0.0)
        row = row_positive / row_positive.sum() if row_positive.sum() > epsilon else uniform_row
        col = col_positive / col_positive.sum() if col_positive.sum() > epsilon else uniform_col
        row_sum += row
        col_sum += col
        row_regret += matrix @ col - float(row @ matrix @ col)
        col_regret += -(row @ matrix) + float(row @ matrix @ col)
    row_strategy = row_sum / iterations
    col_strategy = col_sum / iterations
    row_value = float(row_strategy @ matrix @ col_strategy)
    row_exploit = float(np.max(matrix @ col_strategy) - row_value)
    col_exploit = float(row_value - np.min(row_strategy @ matrix))
    return row_strategy, col_strategy, 0.5 * (row_exploit + col_exploit)
