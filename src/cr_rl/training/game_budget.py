"""Count complete learner games once across updates and checkpoint resumes."""

from __future__ import annotations

import numpy as np


class TrainingGameCounter:
    def __init__(self, state=None, *, origin_update=0):
        state = state or {}
        self.origin_update = state.get("origin_update", origin_update)
        self.started = self._ids(state.get("started_episode_ids", []))
        self.completed = self._ids(state.get("completed_episode_ids", []))
        if not self.completed.issubset(self.started):
            raise ValueError("Completed training games lack observed episode starts")
        if state.get("completed_games", len(self.completed)) != len(self.completed):
            raise ValueError("Saved training-game count differs from episode identities")

    @staticmethod
    def _ids(values):
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("Training episode identities must be nonnegative integers")
        if len(values) != len(set(values)):
            raise ValueError("Duplicated training episode identities")
        return set(values)

    def record(self, snapshots):
        before = len(self.completed)
        for row in snapshots:
            episodes = np.asarray(row["episode_ids"])
            starts = np.asarray(row["episode_start"], dtype=bool)
            if episodes.ndim != 1 or starts.shape != episodes.shape:
                raise ValueError("Game identities/starts must have one value per environment")
            if not np.issubdtype(episodes.dtype, np.integer) or (episodes < 0).any():
                raise ValueError("Training episode identities must be nonnegative integers")
            self.started.update(int(value) for value in episodes[starts])
            valid = np.asarray(row["valid"], dtype=bool).all(-1)
            learner = np.asarray(row["learner_mask"], dtype=bool).any(-1)
            finished = (
                np.asarray(row["terminated"], dtype=bool)
                & ~np.asarray(row["truncated"], dtype=bool)
                & (np.asarray(row["fault_codes"]) == 0)
                & valid
                & learner
            )
            for value in episodes[finished]:
                identity = int(value)
                if identity not in self.started:
                    raise ValueError("Refuse to count a terminal fragment without its start")
                self.completed.add(identity)
        return len(self.completed) - before

    @property
    def count(self):
        return len(self.completed)

    def state_dict(self):
        return dict(
            version=1,
            origin_update=self.origin_update,
            completed_games=self.count,
            started_episode_ids=sorted(self.started),
            completed_episode_ids=sorted(self.completed),
            scope="Complete fault-free learner episodes; both seats count as one game",
        )
