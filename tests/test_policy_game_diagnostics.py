import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "diagnostic", Path(__file__).parents[1] / "scripts/analyze_policy_games.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report():
    native = dict(
        successful_card_plays=2,
        paid_card_elixir_total=8,
        refused_card_commands=1,
        successful_cards_per_active_minute=2,
        successful_card_histogram={"Giant": 1, "Knight": 1},
        core_cost_thresholds={"5": {"count": 1}},
    )
    decision = dict(full_elixir_wait_legal_seconds=4, longest_full_elixir_wait_legal_seconds=2)
    match = dict(
        pair=4, learner_seat=1, seed=1000004, deck0="opponent", deck1="own", reward=-1, valid=True
    )
    execution = dict(
        pair=4,
        learner_seat=1,
        native_outcome_parity=True,
        active_simulation_seconds=60,
        learner=copy.deepcopy(native),
        opponent=copy.deepcopy(native),
        decision_audit=dict(learner=decision, opponent=decision),
    )
    return dict(
        source_summary_sha256="frozen",
        games=[dict(cohort="heldout", match=match, execution=execution)],
    )


def test_native_diagnostics_use_owned_deck_and_confirmed_plays():
    result = module.analyze(report())
    row = result["by_own_deck"][0]
    assert row["deck"] == "own" and row["wins"] == 0 and row["losses"] == 1
    assert row["roles"]["learner"]["mean_successful_cards"] == 2
    assert row["roles"]["learner"]["successful_cards"] == {"Giant": 1, "Knight": 1}
    assert row["roles"]["learner"]["successful_cost_at_least_5"] == 1


@pytest.mark.parametrize("change", ["duplicate", "parity", "seat"])
def test_native_diagnostics_reject_untrustworthy_evidence(change):
    data = report()
    if change == "duplicate":
        data["games"].append(copy.deepcopy(data["games"][0]))
    elif change == "parity":
        data["games"][0]["execution"]["native_outcome_parity"] = False
    else:
        data["games"][0]["execution"]["learner_seat"] = 0
    with pytest.raises(ValueError):
        module.analyze(data)
