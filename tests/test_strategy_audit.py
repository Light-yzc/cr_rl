import copy

import pytest

from cr_rl.evaluation.strategy_audit import summarize_strategy


def test_strategy_uses_executed_cards_and_weights_by_actual_plays():
    def native(count, cost):
        return dict(
            successful_card_plays=count,
            paid_card_elixir_total=count * cost,
            successful_card_histogram={"card": count},
            core_cost_thresholds={"5": dict(count=count if cost >= 5 else 0)},
            elixir_before_successful_card=dict(mean=cost + 1),
            elixir_after_successful_card=dict(mean=1),
        )

    matches = [
        dict(
            pair=0,
            learner_seat=seat,
            seed=12,
            deck0="deckA",
            deck1="deckB",
            valid=True,
            reward=1 if seat == 0 else -1,
        )
        for seat in (0, 1)
    ]
    games = [
        dict(
            pair=0,
            learner_seat=seat,
            reward=match["reward"],
            native_outcome_parity=True,
            active_simulation_seconds=60,
            learner=native(1 if seat == 0 else 3, 2 if seat == 0 else 5),
            opponent=native(2, 3),
            decision_audit=dict(
                learner=dict(full_elixir_wait_legal_seconds=seat * 2),
                opponent=dict(full_elixir_wait_legal_seconds=1),
            ),
        )
        for seat, match in enumerate(matches)
    ]
    pairs = [(dict(matchups=matches), dict(games=games))]
    result = summarize_strategy(pairs)
    assert result["learner"]["paid_elixir_per_successful_card"] == 4.25
    assert result["learner"]["successful_card_plays"] == 4
    assert result["learner"]["by_deck"]["deckA"]["wins"] == 1
    assert result["opponent"]["by_deck"]["deckA"]["wins"] == 1
    assert result["learner"]["mean_full_elixir_wait_legal_seconds"] == 1
    assert result["learner"]["cost_at_least5_fraction"] == 0.75
    del games[0]["decision_audit"]
    assert summarize_strategy(pairs)["learner"]["mean_full_elixir_wait_legal_seconds"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_strategy(pairs + pairs)
    bad = copy.deepcopy(pairs)
    bad[0][1]["games"][0]["learner"]["successful_card_plays"] = 2
    with pytest.raises(ValueError, match="histogram"):
        summarize_strategy(bad)
