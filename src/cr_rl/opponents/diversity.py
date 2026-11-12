"""Bounded retention of opponents supported by independent paired evaluations.

These records influence which frozen opponents survive pruning. They do not
change the current/history/anchor mixture or the existing deck-specific PFSP.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PAIRED_PROTOCOL = "fixed physical decks/seeds, paired policy-seat swap, stochastic sampling"


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha256(value, field):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA256 digest") from error
    return value.lower()


def _integer(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class OpponentEvidence:
    policy_id: int
    specialty: str
    games: int
    opponent_score: float
    evaluated_against_version: int
    candidate_sha256: str
    opponent_sha256: str
    report_paths: tuple[str, ...]
    report_sha256: tuple[str, ...]

    @classmethod
    def from_state(cls, state):
        fields = dict(state)
        for name in ("report_paths", "report_sha256"):
            fields[name] = tuple(fields[name])
        value = cls(**fields)
        _integer(value.policy_id, "policy_id")
        _integer(value.games, "games", 32)
        _integer(value.evaluated_against_version, "evaluated_against_version")
        if not isinstance(value.specialty, str) or not value.specialty.strip():
            raise ValueError("Expert specialty must be a nonempty label")
        if not math.isfinite(value.opponent_score) or not 0 <= value.opponent_score <= 1:
            raise ValueError("Invalid independent opponent score")
        _sha256(value.candidate_sha256, "candidate_sha256")
        _sha256(value.opponent_sha256, "opponent_sha256")
        if not value.report_paths or len(value.report_paths) != len(value.report_sha256):
            raise ValueError("Evaluation report provenance is incomplete")
        for digest in value.report_sha256:
            _sha256(digest, "report_sha256")
        return value


@dataclass(frozen=True)
class DiversityRetention:
    evaluated_against_version: int
    candidate_sha256: str
    experts: tuple[OpponentEvidence, ...]
    max_age_versions: int = 20
    recent_slots: int = 2
    expert_slots: int = 3
    min_games: int = 32
    min_opponent_score: float = 0.55

    def __post_init__(self):
        _integer(self.evaluated_against_version, "evaluated_against_version")
        _sha256(self.candidate_sha256, "candidate_sha256")
        _integer(self.max_age_versions, "max_age_versions")
        _integer(self.recent_slots, "recent_slots")
        _integer(self.expert_slots, "expert_slots")
        _integer(self.min_games, "min_games", 32)
        if not math.isfinite(self.min_opponent_score) or not 0.5 <= self.min_opponent_score <= 1:
            raise ValueError("min_opponent_score must be between 0.5 and 1")
        ids = set()
        for expert in self.experts:
            if expert.policy_id in ids:
                raise ValueError("Duplicate independently evaluated expert policy_id")
            ids.add(expert.policy_id)
            if (
                expert.evaluated_against_version != self.evaluated_against_version
                or expert.candidate_sha256 != self.candidate_sha256
                or expert.games < self.min_games
                or expert.opponent_score < self.min_opponent_score
            ):
                raise ValueError("Expert does not satisfy the configured independent evidence gate")

    @classmethod
    def from_state(cls, state):
        fields = dict(state)
        if fields.pop("schema_version", None) != 1:
            raise ValueError("Unsupported diversity retention schema")
        fields["experts"] = tuple(OpponentEvidence.from_state(x) for x in fields["experts"])
        return cls(**fields)

    def state_dict(self):
        return {"schema_version": 1, **asdict(self)}

    def active_experts(self, entries, learner_version):
        _integer(learner_version, "learner_version")
        age = learner_version - self.evaluated_against_version
        if age < 0 or age > self.max_age_versions:
            return []
        eligible = {entry.policy_id for entry in entries if entry.role == "historical"}
        return sorted(
            (expert for expert in self.experts if expert.policy_id in eligible),
            key=lambda x: (-x.opponent_score, -x.games, x.policy_id),
        )

    @classmethod
    def from_manifest(cls, manifest, entries, *, learner_version, verify_files=True):
        """Validate actual frozen-policy results, never online aggregate win rates.

        Evaluation reports must have used the learner as ``--policy`` and the
        proposed expert as ``--opponent``. Every complete physical matchup must
        contain both learner seats. Repeated report/match identities do not count.
        """
        base = Path.cwd()
        if isinstance(manifest, (str, Path)):
            path = Path(manifest).resolve()
            base = path.parent
            manifest = json.loads(path.read_text())
        fields = dict(manifest)
        if fields.pop("schema_version", None) != 1:
            raise ValueError("Unsupported diversity manifest schema")
        proposed = fields.pop("experts")
        version = _integer(fields["evaluated_against_version"], "evaluated_against_version")
        candidate_sha = _sha256(fields["candidate_sha256"], "candidate_sha256")
        fields["candidate_sha256"] = candidate_sha
        candidate_path = fields.pop("candidate_checkpoint", None)
        if candidate_path is not None:
            path = Path(candidate_path)
            path = path.resolve() if path.is_absolute() else (base / path).resolve()
            if _digest(path) != candidate_sha:
                raise ValueError("Candidate checkpoint differs from the evaluated learner")
        if learner_version < version:
            raise ValueError("Cannot apply expert evidence from a future learner")
        by_id = {entry.policy_id: entry for entry in entries}
        evidence = []
        deck_pool_sha = None
        for row in proposed:
            policy_id = _integer(row["policy_id"], "policy_id")
            entry = by_id.get(policy_id)
            if entry is None or entry.role != "historical":
                raise ValueError("A proposed expert must be an existing historical opponent")
            specialty = row["specialty"]
            if not isinstance(specialty, str) or not specialty.strip():
                raise ValueError("Expert specialty must be a nonempty label")
            paths, digests, identities, outcomes = [], [], set(), []
            opponent_sha = None
            evaluation_code_sha = None
            for raw_path in row["reports"]:
                path = Path(raw_path)
                path = path.resolve() if path.is_absolute() else (base / path).resolve()
                report = json.loads(path.read_text())
                if report.get("policy_sha256") != candidate_sha:
                    raise ValueError("Evaluation learner differs from the frozen candidate")
                if (
                    report.get("wait_logit_bias") != 0
                    or report.get("opponent_wait_logit_bias") != 0
                    or report.get("protocol") != PAIRED_PROTOCOL
                ):
                    raise ValueError("Expert evidence requires unmodified paired policy evaluation")
                deck_sha = _sha256(report.get("deck_pool_sha256"), "deck_pool_sha256")
                if deck_pool_sha is not None and deck_pool_sha != deck_sha:
                    raise ValueError("Expert evidence uses different evaluation deck pools")
                deck_pool_sha = deck_sha
                code_sha = _sha256(report.get("evaluation_code_sha256"), "evaluation_code_sha256")
                code_end_sha = _sha256(
                    report.get("evaluation_code_file_sha256_at_end"),
                    "evaluation_code_file_sha256_at_end",
                )
                if code_sha != code_end_sha or (
                    evaluation_code_sha is not None and evaluation_code_sha != code_sha
                ):
                    raise ValueError("Expert evidence uses changed evaluation code")
                evaluation_code_sha = code_sha
                digest = _sha256(report["opponent_sha256"], "opponent_sha256")
                if opponent_sha is not None and opponent_sha != digest:
                    raise ValueError("One expert has reports for different frozen opponents")
                opponent_sha = digest
                matches = report["matchups"]
                if (
                    report.get("invalid_games") != 0
                    or report.get("games") != len(matches)
                    or not matches
                ):
                    raise ValueError("Expert evidence contains invalid or incomplete games")
                pairs = {}
                rewards = []
                for match in matches:
                    seat = match["learner_seat"]
                    reward = match.get("reward")
                    info = match.get("info")
                    if (
                        seat not in (0, 1)
                        or match.get("valid") is not True
                        or reward not in (-1, 0, 1)
                        or not isinstance(info, dict)
                        or any(info.get(key) != 0 for key in
                               ("fault", "truncated", "illegal_0", "illegal_1"))
                    ):
                        raise ValueError("Expert evidence contains invalid game outcomes")
                    physical = (match["pair"], match["seed"], match["deck0"], match["deck1"])
                    identity = (report.get("sampling_seed"), *physical, seat)
                    if identity in identities:
                        raise ValueError("Repeated independent evaluation game")
                    identities.add(identity)
                    pairs.setdefault(physical, []).append(seat)
                    rewards.append(reward)
                if any(sorted(seats) != [0, 1] for seats in pairs.values()):
                    raise ValueError("Expert evidence must include both policy-seat swaps")
                actual_counts = (
                    sum(x > 0 for x in rewards),
                    sum(x < 0 for x in rewards),
                    sum(x == 0 for x in rewards),
                )
                if actual_counts != tuple(report.get(k) for k in ("wins", "losses", "draws")):
                    raise ValueError("Evaluation aggregate and individual outcomes differ")
                outcomes.extend(rewards)
                paths.append(str(path))
                digests.append(_digest(path))
            if not outcomes:
                raise ValueError("Expert requires independent evaluation reports")
            if verify_files and _digest(entry.checkpoint) != opponent_sha:
                raise ValueError("Expert checkpoint differs from the evaluated weights")
            evidence.append(
                OpponentEvidence.from_state(
                    dict(
                        policy_id=policy_id,
                        specialty=specialty.strip(),
                        games=len(outcomes),
                        opponent_score=sum((1 - reward) / 2 for reward in outcomes) / len(outcomes),
                        evaluated_against_version=version,
                        candidate_sha256=candidate_sha,
                        opponent_sha256=opponent_sha,
                        report_paths=tuple(paths),
                        report_sha256=tuple(digests),
                    )
                )
            )
        return cls(experts=tuple(evidence), **fields)

    def select(self, entries, capacity, protected_ids, learner_version):
        """Live matches/anchors/latest are hard; quotas yield to their capacity.

        Remaining preference order: two recent versions, up to three distinct
        confirmed specialties, an old archive, a new candidate, spaced archives.
        """
        by_id = {entry.policy_id: entry for entry in entries}
        selected = set(protected_ids) & by_id.keys()
        selected.update(entry.policy_id for entry in entries if entry.role == "anchor")
        ordered = sorted(entries, key=lambda entry: (entry.generation, entry.policy_id))
        if ordered:
            selected.add(ordered[-1].policy_id)
        if len(selected) > capacity:
            raise RuntimeError("Pool capacity cannot fit active opponents, anchors and latest snapshot")
        if len(entries) <= capacity:
            return set(by_id)

        def add(policy_id):
            if len(selected) < capacity:
                selected.add(policy_id)

        historical = [entry for entry in ordered if entry.role != "anchor"]
        for entry in list(reversed(historical))[: self.recent_slots]:
            add(entry.policy_id)
        experts = self.active_experts(entries, learner_version)
        specialties = {expert.specialty for expert in experts if expert.policy_id in selected}
        for expert in experts:
            if len(specialties) >= self.expert_slots:
                break
            if expert.specialty not in specialties and len(selected) < capacity:
                add(expert.policy_id)
                specialties.add(expert.specialty)
        remaining = [entry for entry in historical if entry.policy_id not in selected]
        if remaining:
            add(remaining[0].policy_id)  # Preserve an older reference against forgetting.
        remaining = [entry for entry in historical if entry.policy_id not in selected]
        if remaining:
            add(remaining[-1].policy_id)  # One recent candidate can be independently tested.
        remaining = [entry for entry in ordered if entry.policy_id not in selected]
        slots = capacity - len(selected)
        if slots:
            for index in np.linspace(0, len(remaining) - 1, slots, dtype=int):
                add(remaining[index].policy_id)
        return selected
