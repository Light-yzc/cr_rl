import numpy as np

from cr_rl.opponents.league import OpponentEntry, OpponentPool, solve_zero_sum_meta


def test_pfsp_prioritizes_hard_opponents_and_keeps_uniform_floor():
    pool = OpponentPool(
        [
            OpponentEntry(0, "easy.pt", 0, games=20, score_sum=18.0),
            OpponentEntry(1, "hard.pt", 0, games=20, score_sum=2.0),
        ],
        uniform_mix=0.1,
    )
    probabilities = pool.probabilities()
    assert probabilities[1] > probabilities[0]
    assert np.all(probabilities > 0)
    pool.record(1, 1.0)
    assert pool.entries[1].games == 21


def test_league_sampling_includes_current_historical_pfsp_and_forgotten_mix():
    pool = OpponentPool(
        [
            OpponentEntry(0, "current.pt", 2),
            OpponentEntry(1, "hard.pt", 1, games=20, score_sum=1.0),
            OpponentEntry(2, "forgotten.pt", 0, games=20, score_sum=19.0),
        ]
    )
    rng = np.random.default_rng(42)
    samples = [pool.sample_league(rng, current_policy_id=0).policy_id for _ in range(5000)]
    assert abs(samples.count(0) / len(samples) - 0.35) < 0.03
    assert samples.count(1) > samples.count(2)


def test_regret_matching_finds_rock_paper_scissors_uniform_meta_strategy():
    payoff = np.asarray([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=np.float64)
    row, col, exploitability = solve_zero_sum_meta(payoff, iterations=30_000)
    np.testing.assert_allclose(row, np.full(3, 1 / 3), atol=0.02)
    np.testing.assert_allclose(col, np.full(3, 1 / 3), atol=0.02)
    assert exploitability < 0.02
