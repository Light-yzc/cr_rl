"""Audit requested card costs and legal opportunities from saved learner rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit(path):
    metadata = json.loads(Path(str(path) + ".json").read_text())
    fields = metadata["schema"]["header_fields"]

    def columns(name):
        return [i for i, field in enumerate(fields) if field == name]

    with np.load(path) as data:
        eligible = data["learner_mask"].astype(bool) & data["valid"].astype(bool)
        obs = data["obs"][eligible]
        actions = data["actions"][eligible]
        masks = data["masks"][eligible]
    costs = obs[:, columns("hand_cost")]
    cards = obs[:, columns("hand_card")]
    pending = obs[:, columns("hand_pending")]
    available = (obs[:, columns("elixir")[0]] - obs[:, columns("elixir_reserved")[0]]) / 10000
    legal_slots = masks[:, 1:2305].reshape(-1, 4, 576).any(-1)
    choice = masks.sum(-1) > 1
    high = (cards >= 0) & (pending == 0) & (costs >= 5)
    high_legal = (high & legal_slots).any(-1)
    high_unaffordable = (high & (costs > available[:, None])).any(-1)
    play = (actions > 0) & (actions < 2305)
    slots = ((actions - 1) // 576).clip(0, 3)
    chosen = np.take_along_axis(costs, slots[:, None], -1)[:, 0]
    choices = int(choice.sum())
    result = {
        "rollout": path.name,
        "valid_learner_rows": len(actions),
        "choice_rows": choices,
        "card_requests": int(play.sum()),
        "requested_card_cost_mean": float(chosen[play].mean()) if play.any() else None,
        "requests_by_cost": {
            str(int(c)): int((play & (chosen == c)).sum()) for c in np.unique(chosen[play])
        },
        "high_cost_requests": int((play & (chosen >= 5)).sum()),
        "high_cost_legal_rows": int(high_legal.sum()),
        "high_cost_chosen_when_legal": int((high_legal & play & (chosen >= 5)).sum()),
        "wait_when_high_cost_legal": int((high_legal & (actions == 0)).sum()),
        "low_cost_chosen_when_high_cost_legal": int((high_legal & play & (chosen <= 3)).sum()),
        "high_cost_in_hand_but_unaffordable_choice_rows": int((choice & high_unaffordable).sum()),
        "available_elixir_at_least_five_rows": int((available >= 5).sum()),
        "available_elixir_percentiles": np.percentile(available, [25, 50, 75, 95]).tolist(),
        "scope": (
            "Saved valid learner decisions; requests, not confirmed deployments; "
            "correlated frames, not independent games"
        ),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = [audit(path) for path in args.rollouts]
    report = json.dumps(results, indent=2)
    if args.out:
        args.out.write_text(report + "\n")
    print(report)
