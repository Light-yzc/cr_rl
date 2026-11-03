import pytest

from scripts.evaluate_cycle_stage import choose_timing_bias


def evaluation(wait=0, longest=0, wins=True):
    return dict(
        games=4,
        invalid_games=0,
        matchups=[
            dict(
                pair=i // 2,
                seed=i // 2,
                deck0=f"a{i // 2}",
                deck1=f"b{i // 2}",
                valid=True,
                learner_seat=i % 2,
                reward=1 if wins else -1,
                decision_audit={
                    "learner": dict(
                        full_elixir_wait_legal_seconds=wait,
                        longest_full_elixir_wait_legal_seconds=longest,
                    )
                },
            )
            for i in range(4)
        ],
    )


def test_bias_selection_does_not_chase_tuning_wins_or_allow_stalling():
    calibration = dict(
        grid=[
            dict(bias=0, binary_timing_nll=0.4),
            dict(bias=1.5, binary_timing_nll=0.15),
            dict(bias=2.5, binary_timing_nll=0.10),
        ]
    )
    games = {
        0: evaluation(wins=True),
        1.5: evaluation(wait=1, longest=1, wins=False),
        2.5: evaluation(wait=11.5, longest=6.5, wins=True),
    }
    assert choose_timing_bias(calibration, games) == 1.5


def test_no_valid_bias_cannot_be_promoted():
    calibration = dict(grid=[dict(bias=2.5, binary_timing_nll=0.1)])
    with pytest.raises(ValueError, match="No predeclared"):
        choose_timing_bias(calibration, {2.5: evaluation(wait=3)})
    incomplete = evaluation()
    incomplete["matchups"].pop()
    with pytest.raises(ValueError, match="exactly one"):
        choose_timing_bias(calibration, {2.5: incomplete})
