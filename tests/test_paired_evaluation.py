import pytest

from cr_rl.evaluation.paired import paired_outcomes
from cr_rl.evaluation.replay_audit import summarize_executions


def cohort(outcomes):
    return {
        "matchups": [
            dict(
                pair=i,
                learner_seat=seat,
                seed=i,
                deck0=f"a{i}",
                deck1=f"b{i}",
                valid=True,
                reward=reward,
            )
            for i, rewards in enumerate(outcomes)
            for seat, reward in enumerate(rewards)
        ]
    }


def test_paired_sign_test_counts_sweeps_not_deck_split_wins():
    result = paired_outcomes(cohort([(1, 1)] * 5 + [(1, -1)] * 3), minimum_pairs=8)
    assert result["swept_wins_2_0"] == 5
    assert result["split_pairs_1_1"] == 3
    assert result["swept_pair_sign_test_one_sided_p"] == 1 / 32
    assert paired_outcomes(cohort([(1, -1)]))["swept_pair_sign_test_one_sided_p"] == 1


def test_invalid_or_unpaired_games_cannot_prove_promotion():
    data = cohort([(1, 1)])
    data["matchups"][1]["seed"] = 99
    with pytest.raises(ValueError, match="same physical"):
        paired_outcomes(data)
    with pytest.raises(ValueError, match="at least"):
        paired_outcomes(cohort([(1, 1)]), minimum_pairs=8)


def test_queued_or_refused_cards_are_not_successes():
    native = dict(
        native_outcome_parity=True,
        duration_seconds=184,
        active_simulation_seconds=180,
        total_spent_elixir_by_side=[5, 0],
        commands=[
            dict(
                side=0,
                kind="CARD",
                status="executed",
                spent_elixir=5,
                elixir_before=8,
                elixir_after=3,
                card="Giant",
            ),
            dict(side=0, kind="CARD", status="refused", spent_elixir=0),
            dict(side=0, kind="CARD", status="queued_unrun", spent_elixir=0),
        ],
    )
    result = summarize_executions(native, 0)
    assert result["queued_card_commands"] == 3
    assert result["successful_plays"] == 1
    assert result["mean_paid_elixir_per_successful_card"] == 5
    assert result["core_cost_thresholds"]["5"]["count"] == 1
    native["native_outcome_parity"] = False
    with pytest.raises(ValueError, match="parity"):
        summarize_executions(native, 0)
