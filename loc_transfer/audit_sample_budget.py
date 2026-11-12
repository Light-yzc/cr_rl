"""Compare unique on-policy decisions with optimizer steps and sample presentations."""

import argparse
import json
from pathlib import Path


def audit(run, count=6):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    rows = rows[-count:]
    output = []
    for row in rows:
        usage = row["sample_utilization"]
        unique = usage["actor_eligible_unique"]
        committed = usage["actor_samples_committed"]
        output.append(
            {
                "update": row["update"],
                "new_learner_rows": row["valid_learner_samples"],
                "new_actor_choice_rows": unique,
                "actor_choice_fraction": unique / row["valid_learner_samples"],
                "actor_optimizer_steps": row["optimizer_steps"],
                "actor_sample_presentations": committed,
                "actor_presentations_per_eligible_unique": committed / unique if unique else None,
                "actor_unique_committed": usage["actor_unique_samples_committed"],
                "critic_optimizer_steps": row["critic_optimizer_steps"],
                "critic_sample_presentations": usage["critic_samples_committed"],
                "completed_games": row["completed_training_games"],
                "high_cost_requests": row["decision_coverage"]["at_least_five_cost_requests"],
                "actor_kl_stopped": row["actor_kl_stopped"],
            }
        )
    totals = {
        key: sum(row[key] for row in output)
        for key in (
            "new_learner_rows",
            "new_actor_choice_rows",
            "actor_optimizer_steps",
            "actor_sample_presentations",
            "critic_optimizer_steps",
            "critic_sample_presentations",
            "completed_games",
            "high_cost_requests",
        )
    }
    totals["actor_presentations_per_eligible_unique"] = (
        totals["actor_sample_presentations"] / totals["new_actor_choice_rows"]
    )
    totals["critic_presentations_per_new_learner_row"] = (
        totals["critic_sample_presentations"] / totals["new_learner_rows"]
    )
    return {
        "updates": output,
        "totals": totals,
        "scope": (
            "Unique decision rows are correlated, not independent trajectories; "
            "context windows are not extra samples"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.run, args.count), indent=2)
    if args.out:
        args.out.write_text(report + "\n")
    print(report)
