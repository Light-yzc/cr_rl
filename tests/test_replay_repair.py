import numpy as np
import pytest

from cr_rl.data.replay_repair import (
    RepairConfig,
    RepairDecision,
    RepairLedger,
    align_tick,
    decode_play_action,
    encode_play_action,
    nearest_legal_action,
    plan_hand_swap,
    repair_action,
)


def test_tick_alignment_accepts_only_the_declared_tolerance():
    config = RepairConfig(decision_ticks=5, time_tolerance_ticks=2)
    assert align_tick(12, config).simulator_tick == 10
    assert align_tick(13, config).simulator_tick == 15
    assert align_tick(7, config).accepted
    assert not align_tick(7, config, mode="ceil").accepted
    assert align_tick(-1, config).reason == "negative_tick"
    with pytest.raises(ValueError, match="mode"):
        align_tick(1, config, mode="floor")


@pytest.mark.parametrize("slot", range(4))
def test_native_action_codec_roundtrips_each_hand_slot(slot):
    for x, y in ((0, 0), (17, 31), (7, 10)):
        action = encode_play_action(slot, x, y)
        assert decode_play_action(action) == (slot, x, y)
    with pytest.raises(ValueError, match="slot"):
        encode_play_action(4, 1, 1)
    with pytest.raises(ValueError, match="placement"):
        decode_play_action(2305)


def test_nearest_legal_tile_preserves_slot_and_bounds_distance():
    mask = np.zeros(2307, dtype=bool)
    requested = encode_play_action(2, 7, 10)
    candidate = encode_play_action(2, 8, 11)
    mask[candidate] = True
    mask[encode_play_action(1, 7, 10)] = True
    assert nearest_legal_action(requested, mask, 2) == (candidate, 2)
    assert nearest_legal_action(requested, mask, 1) is None
    assert nearest_legal_action(2305, mask, None) is None
    with pytest.raises(ValueError, match="2307"):
        nearest_legal_action(requested, mask[:, None], 2)


def test_nearest_tile_tie_is_deterministic_and_unlimited_is_explicit():
    mask = np.zeros(2307, dtype=bool)
    requested = encode_play_action(0, 7, 10)
    lower = encode_play_action(0, 6, 10)
    mask[lower] = mask[encode_play_action(0, 8, 10)] = True
    assert nearest_legal_action(requested, mask, None) == (lower, 1)
    with pytest.raises(ValueError, match="negative"):
        nearest_legal_action(requested, mask, -1)


def test_repair_action_never_invents_hand_or_elixir():
    mask = np.ones(2307, dtype=bool)
    config = RepairConfig()
    assert repair_action(1, mask, config=config, hand_contains_card=False).category == "not_in_hand"
    assert (
        repair_action(1, mask, config=config, elixir_legal=False).category
        == "elixir_insufficient"
    )


def test_tile_repair_is_explicit_and_recorded():
    mask = np.zeros(2307, dtype=bool)
    requested = encode_play_action(3, 4, 4)
    candidate = encode_play_action(3, 5, 4)
    mask[candidate] = True
    decision = repair_action(
        requested,
        mask,
        config=RepairConfig(allow_tile_snap=True, max_tile_shift=1),
        hand_contains_card=True,
        elixir_legal=True,
    )
    assert decision == RepairDecision(
        requested, candidate, "tile_snap", False, True, True, 1, "same-slot nearest legal tile"
    )
    ledger = RepairLedger()
    ledger.add(decision, seat=0, source_tick=10)
    summary = ledger.summary()
    assert summary["repaired"] == 1
    assert summary["categories"] == {"tile_snap": 1}
    summary["records"][0]["category"] = "changed"
    assert ledger.summary()["categories"] == {"tile_snap": 1}
    with pytest.raises(ValueError, match="seat"):
        ledger.add(decision, seat=2, source_tick=10)


def test_hand_repair_budget_can_be_unlimited_or_explicitly_bounded():
    assert RepairConfig(allow_hand_replacement=True).max_hand_replacements is None
    assert RepairConfig(allow_hand_replacement=True, max_hand_replacements=10)
    with pytest.raises(ValueError, match="positive replacement budget"):
        RepairConfig(allow_hand_replacement=True, max_hand_replacements=0)


def test_hand_swap_preserves_runtime_cards_and_exact_remaining_queue_order():
    plan = plan_hand_swap([0, 1, 2, 3], [4, 5, 6, 7], 6, preferred_slot=2)
    assert plan["places"] == [2, 6]
    assert plan["cards"] == [6, 2]
    assert plan["after_slots"] == [0, 1, 6, 3]
    assert plan["after_queue"] == [4, 5, 2, 7]
    assert sorted(plan["after_slots"] + plan["after_queue"]) == list(range(8))
    unchanged = plan_hand_swap([0, 1, 2, 3], [4, 5, 6, 7], 2)
    assert not unchanged["changed"] and unchanged["slot"] == 2


def test_hand_swap_skips_empty_pending_and_refuses_pending_requested_card():
    pending = [False, True, False, False]
    plan = plan_hand_swap([-1, 1, 2, 3], [4, 5, 6, 7, 0], 6, pending)
    assert plan["slot"] == 2
    assert plan["after_slots"][0] == -1
    assert plan["after_slots"][1] == 1
    for preferred in (0, 1):
        with pytest.raises(ValueError, match="empty or pending"):
            plan_hand_swap([-1, 1, 2, 3], [4, 5, 6, 7, 0], 6, pending, preferred)
    with pytest.raises(ValueError, match="Requested card is pending"):
        plan_hand_swap([0, 1, 2, 3], [4, 5, 6, 7], 1, pending)
    with pytest.raises(ValueError, match="No nonempty"):
        plan_hand_swap([0, 1, 2, 3], [4, 5, 6, 7], 4, [True] * 4)


def test_hand_swap_handles_champion_absence_without_inventing_the_card():
    plan = plan_hand_swap([0, 1, 2, 3], [4, 5, 6], 4)
    assert sorted(plan["after_slots"] + plan["after_queue"]) == list(range(7))
    with pytest.raises(ValueError, match="not present"):
        plan_hand_swap([0, 1, 2, 3], [4, 5, 6], 7)
    with pytest.raises(ValueError, match="distinct"):
        plan_hand_swap([0, 1, 2, 3], [4, 5, 6, 0], 4)
