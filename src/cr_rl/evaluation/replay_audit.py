"""Summaries of native command outcomes; queued commands are never called successes."""

from __future__ import annotations

from collections import Counter
from statistics import mean


def _stats(values):
    values = list(values)
    return {
        "mean": mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_executions(native: dict, seat: int) -> dict:
    if not native.get("native_outcome_parity"):
        raise ValueError("Native execution audit has not passed uninstrumented replay parity")
    commands = [c for c in native["commands"] if c["side"] == seat]
    cards = [c for c in commands if c["kind"] != "ABILITY"]
    executed = [c for c in cards if c["status"] == "executed"]
    abilities = [c for c in commands if c["kind"] == "ABILITY"]
    paid = sum(c.get("spent_elixir", 0.0) for c in commands)
    if abs(paid - native["total_spent_elixir_by_side"][seat]) > 1e-6:
        raise ValueError("Command-level payments differ from native side's total spent elixir")
    duration = native["duration_seconds"]
    active = native["active_simulation_seconds"]
    hist = Counter(str(c["spent_elixir"]) for c in executed)
    high = {
        str(threshold): {
            "count": sum(c["spent_elixir"] >= threshold for c in executed),
            "fraction": sum(c["spent_elixir"] >= threshold for c in executed) / len(executed)
            if executed
            else None,
            "elixir_before": _stats(
                c["elixir_before"] for c in executed if c["spent_elixir"] >= threshold
            ),
        }
        for threshold in (5, 7)
    }
    return {
        "queued_card_commands": len(cards),
        "successful_plays": len(executed),
        "successful_card_plays": len(executed),
        "refused_card_commands": sum(c["status"] == "refused" for c in cards),
        "unrun_card_commands": sum(c["status"] == "queued_unrun" for c in cards),
        "queued_ability_commands": len(abilities),
        "successful_ability_commands": sum(c["status"] == "executed" for c in abilities),
        "refused_ability_commands": sum(c["status"] == "refused" for c in abilities),
        "unrun_ability_commands": sum(c["status"] == "queued_unrun" for c in abilities),
        "mean_paid_elixir_per_successful_card": mean(c["spent_elixir"] for c in executed)
        if executed
        else None,
        "elixir_before_successful_card": _stats(c["elixir_before"] for c in executed),
        "elixir_after_successful_card": _stats(c["elixir_after"] for c in executed),
        "paid_card_cost_histogram": dict(sorted(hist.items(), key=lambda kv: float(kv[0]))),
        "core_cost_thresholds": high,
        "successful_card_histogram": dict(Counter(c["card"] for c in executed)),
        "paid_card_elixir_total": sum(c["spent_elixir"] for c in executed),
        "paid_all_commands_elixir_total": paid,
        "successful_cards_per_active_minute": len(executed) * 60 / active if active else None,
        "successful_cards_per_180_wall_seconds": len(executed) * 180 / duration
        if duration
        else None,
        "elixir_observation_definition": "Immediately before/after the individual native command "
        "in its run pass, before entity regeneration/income; includes dynamic item payments.",
    }
