"""Descriptive native execution summaries by policy seat and actual deck."""

from collections import Counter, defaultdict


def _side_summary(rows):
    cards = Counter()
    plays = paid = active_seconds = 0
    high = 0
    waits = []
    before = after = 0.0
    for game, side, _ in rows:
        record = game[side]
        count = record["successful_card_plays"]
        plays += count
        paid += record["paid_card_elixir_total"]
        active_seconds += game["active_simulation_seconds"]
        cards.update(record["successful_card_histogram"])
        high += record["core_cost_thresholds"]["5"]["count"]
        if count:
            before += record["elixir_before_successful_card"]["mean"] * count
            after += record["elixir_after_successful_card"]["mean"] * count
        audit = (game.get("decision_audit") or {}).get(side)
        if audit is not None:
            waits.append(audit["full_elixir_wait_legal_seconds"])
    if sum(cards.values()) != plays:
        raise ValueError("Successful-card histogram differs from executed count")
    return dict(
        games=len(rows),
        successful_card_plays=plays,
        successful_card_histogram=dict(cards),
        paid_elixir_per_successful_card=paid / plays if plays else None,
        elixir_before_successful_card=before / plays if plays else None,
        elixir_after_successful_card=after / plays if plays else None,
        successful_cards_per_active_minute=plays * 60 / active_seconds if active_seconds else None,
        cost_at_least5_fraction=high / plays if plays else None,
        games_with_wait_audit=len(waits),
        mean_full_elixir_wait_legal_seconds=sum(waits) / len(waits)
        if len(waits) == len(rows) and rows
        else None,
    )


def summarize_strategy(evaluations_and_audits):
    overall = []
    by_deck = defaultdict(list)
    seen = set()
    for evaluation, audit in evaluations_and_audits:
        matches = {(m["pair"], m["learner_seat"]): m for m in evaluation["matchups"]}
        if len(matches) != len(evaluation["matchups"]) or len(audit["games"]) != len(matches):
            raise ValueError("Duplicate or missing evaluation identities")
        for game in audit["games"]:
            key = game["pair"], game["learner_seat"]
            match = matches[key]
            identity = (match["seed"], *key, match["deck0"], match["deck1"])
            if identity in seen or not game["native_outcome_parity"] or not match["valid"]:
                raise ValueError("Duplicate, invalid or non-native game")
            if game["reward"] != match["reward"]:
                raise ValueError("Native and evaluation outcomes differ")
            seen.add(identity)
            overall.append((game, "learner", match["reward"]))
            for side, seat, score in (
                ("learner", match["learner_seat"], match["reward"]),
                ("opponent", 1 - match["learner_seat"], -match["reward"]),
            ):
                by_deck[side, match[f"deck{seat}"]].append((game, side, score))
    result = {}
    for side in ("learner", "opponent"):
        rows = [(game, side, score if side == "learner" else -score) for game, _, score in overall]
        summary = _side_summary(rows)
        summary["by_deck"] = {}
        for (policy, deck), items in by_deck.items():
            if policy != side:
                continue
            row = _side_summary(items)
            row.update(
                wins=sum(s > 0 for _, _, s in items),
                losses=sum(s < 0 for _, _, s in items),
                draws=sum(s == 0 for _, _, s in items),
            )
            summary["by_deck"][deck] = row
        result[side] = summary
    result["limitation"] = (
        "Descriptive executed-action statistics; card use is not a win condition "
        "or promotion gate; small per-deck game counts"
    )
    return result
