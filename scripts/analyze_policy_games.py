"""Summarize matched native execution evidence without treating requests as plays."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def analyze(report):
    if not report.get("games"):
        raise ValueError("No native game evidence")
    grouped = defaultdict(list)
    seen = set()
    worst = []
    for game in report["games"]:
        match, execution = game["match"], game["execution"]
        identity = game["cohort"], match["pair"], match["learner_seat"]
        if identity in seen or not match.get("valid") or not execution["native_outcome_parity"]:
            raise ValueError("Duplicate, invalid, or non-parity game")
        seen.add(identity)
        if execution["pair"] != match["pair"] or execution["learner_seat"] != match["learner_seat"]:
            raise ValueError("Native audit does not match policy seat/pair")
        seat = match["learner_seat"]
        deck = match[f"deck{seat}"]
        row = dict(
            cohort=game["cohort"],
            deck=deck,
            opponent_deck=match[f"deck{1 - seat}"],
            pair=match["pair"],
            seat=seat,
            seed=match["seed"],
            reward=match["reward"],
            active_seconds=execution["active_simulation_seconds"],
        )
        for role in ("learner", "opponent"):
            native, decision = execution[role], execution["decision_audit"][role]
            row[role] = dict(
                cards=native["successful_card_plays"],
                paid=native["paid_card_elixir_total"],
                refused=native["refused_card_commands"],
                cards_per_active_minute=native["successful_cards_per_active_minute"],
                full_wait=decision["full_elixir_wait_legal_seconds"],
                longest_full_wait=decision["longest_full_elixir_wait_legal_seconds"],
                card_histogram=native["successful_card_histogram"],
                high_cost=native["core_cost_thresholds"],
            )
        grouped[game["cohort"], deck].append(row)
        worst.append(row)
    results = []
    for (cohort, deck), rows in sorted(grouped.items()):
        roles = {}
        for role in ("learner", "opponent"):
            hist = Counter()
            for row in rows:
                hist.update(row[role]["card_histogram"])
            roles[role] = dict(
                mean_successful_cards=mean(row[role]["cards"] for row in rows),
                mean_paid_card_elixir=mean(row[role]["paid"] for row in rows),
                successful_cards_per_active_minute=mean(
                    row[role]["cards_per_active_minute"] for row in rows
                ),
                mean_full_elixir_wait_seconds=mean(row[role]["full_wait"] for row in rows),
                max_continuous_full_elixir_wait_seconds=max(
                    row[role]["longest_full_wait"] for row in rows
                ),
                successful_cards=dict(hist),
                successful_cost_at_least_5=sum(
                    row[role]["high_cost"]["5"]["count"] for row in rows
                ),
            )
        results.append(
            dict(
                cohort=cohort,
                deck=deck,
                games=len(rows),
                wins=sum(row["reward"] > 0 for row in rows),
                losses=sum(row["reward"] < 0 for row in rows),
                draws=sum(row["reward"] == 0 for row in rows),
                roles=roles,
            )
        )
    return dict(
        source_summary_sha256=report["source_summary_sha256"],
        game_count=len(seen),
        by_own_deck=results,
        longest_wait_games=sorted(worst, key=lambda row: row["learner"]["full_wait"], reverse=True)[
            :10
        ],
        limitation=(
            "Descriptive small samples with different opponents/durations; "
            "not causal attribution or deck ranking"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.source.read_text()))
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "games": result["game_count"],
                "decks": len(result["by_own_deck"]),
                "out": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
