"""Validate the independent JVM verification tool's evidence requirements."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.verify_late_champion_fix import (
    validate_ability_result,
    validate_labelled_replay,
    validate_lookup_probe,
)


def lookup(**changes):
    return {
        "state": 2,
        "alive_copies": 1,
        "available": True,
        "try_given": True,
        "try_refusal": "NONE",
        **changes,
    }


def test_broken_proof_requires_alive_ready_closed_and_no_champion_refusal():
    broken = lookup(available=False, try_given=False, try_refusal="NO_CHAMPION")
    validate_lookup_probe(broken, expect="broken")
    for changes in ({"alive_copies": 0}, {"state": 1}, {"try_refusal": "COOLDOWN"}):
        with pytest.raises(AssertionError):
            validate_lookup_probe({**broken, **changes}, expect="broken")


def test_fixed_proof_requires_both_native_mask_and_readonly_try():
    validate_lookup_probe(lookup(), expect="fixed")
    for changes in ({"available": False}, {"try_given": False}, {"try_refusal": "NO_CHAMPION"}):
        with pytest.raises(AssertionError):
            validate_lookup_probe(lookup(**changes), expect="fixed")


def test_native_result_requires_actual_success_and_exact_elixir_payment():
    result = {
        "kind": "ability",
        "success": True,
        "ability_outcome_code": 0,
        "elixir_before": 60000,
        "elixir_after": 40000,
    }
    validate_ability_result(result, 2)
    for changes in (
        {"success": False},
        {"ability_outcome_code": 1006},
        {"elixir_after": 50000},
    ):
        with pytest.raises(AssertionError):
            validate_ability_result({**result, **changes}, 2)


def labelled(**changes):
    masks = np.ones((2, 2307), dtype=np.uint8)
    return SimpleNamespace(
        **{
            "counts": {"refusal": None},
            "actions": np.array([0, 2305]),
            "flags": np.zeros(2, dtype=np.int32),
            "masks": masks,
            "drive": {"reproduced": True, "state_reproduced": True},
            **changes,
        }
    )


def test_export_proof_requires_unflagged_legal_ability_and_native_state_reproduction():
    validate_labelled_replay(labelled())
    masked = labelled()
    masked.masks[1, 2305] = 0
    for replay in (
        masked,
        labelled(flags=np.array([0, 4])),
        labelled(actions=np.array([0, 0])),
        labelled(actions=np.array([2305, 2305])),
        labelled(counts={"refusal": "unknown late unit"}),
        labelled(drive={"reproduced": True, "state_reproduced": False}),
    ):
        with pytest.raises(AssertionError):
            validate_labelled_replay(replay)
