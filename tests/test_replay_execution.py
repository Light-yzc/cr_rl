"""Exercise repair scheduling/online labels without a JVM or a training GPU."""

from types import SimpleNamespace

import numpy as np
import pytest

from cr_rl.data.replay_execution import (
    ExecutionConfig,
    execute_repaired_events,
    source_event_keys,
)

CARDS = list("ABCDEFGH")


def event(card="A", side="team", *, x=10500, y=10500):
    return {
        "kind": "play_card",
        "side": side,
        "card_key": card,
        "coordinates": {"native_world_units": {"x": x, "y": y}},
    }


class FakeVector:
    def __init__(
        self,
        *,
        open_at=0,
        terminal_at=None,
        play_delay=20,
        run_refusal=False,
        truncated=False,
        ability_delay=20,
        submission_refusal_seat=None,
    ):
        self.tick = 0
        self.open_at = open_at
        self.terminal_at = terminal_at
        self.run_refusal = run_refusal
        self.truncated = truncated
        self.ability_delay = ability_delay
        self.submission_refusal_seat = submission_refusal_seat
        self.last_illegal = [0, 0]
        self.dones = np.zeros(1, bool)
        self.actions = np.zeros((1, 2), np.int32)
        self.observations = np.zeros((1, 2, 15), np.int32)
        self.observations[0, :, :4] = np.arange(4)
        self.observations[0, :, 8:10] = -1
        self.masks = np.zeros((1, 2, 2307), bool)
        self.schema = SimpleNamespace(
            header={
                "hand_card": (0, 1, 2, 3),
                "hand_pending": (4, 5, 6, 7),
                "ability_champion": (8, 9),
                "last_run_refusal": (10,),
                "last_ability_run_refusal": (11,),
                "ability_command_ticks": (13, 14),
            },
            play_delay_ticks=play_delay,
            play=lambda slot, x, y: 1 + slot * 576 + y * 18 + x,
        )
        self.vocabulary = CARDS
        self.pending = []
        self.submissions = []
        self.executions = []
        self.pending_abilities = []
        self.ability_executions = []
        self.refresh()

    def card_id(self, name):
        return CARDS.index(name)

    def info(self, env):
        assert env == 0
        return {
            "tick": self.tick,
            "fault": 0,
            "illegal_0": self.last_illegal[0],
            "illegal_1": self.last_illegal[1],
            "truncated": int(self.truncated and self.dones[0]),
        }

    def refresh(self):
        self.observations[0, :, 12] = self.tick
        self.masks[:] = False
        self.masks[:, :, 0] = True
        if self.tick >= self.open_at:
            for side in (0, 1):
                for slot in range(4):
                    if not self.observations[0, side, 4 + slot]:
                        self.masks[0, side, 1 + slot * 576 : 1 + (slot + 1) * 576] = True
                for slot in (0, 1):
                    if (
                        self.observations[0, side, 8 + slot] >= 0
                        and not self.observations[0, side, 13 + slot]
                    ):
                        self.masks[0, side, 2305 + slot] = True

    def step(self):
        self.last_illegal = [0, 0]
        for side, action in enumerate(self.actions[0]):
            if action:
                assert self.masks[0, side, action]
                if side == self.submission_refusal_seat:
                    self.last_illegal[side] = 1
                    continue
                self.submissions.append((self.tick, side, int(action)))
                if action >= 2305:
                    slot = int(action) - 2305
                    self.pending_abilities.append((self.tick + self.ability_delay, side, slot))
                else:
                    slot = (int(action) - 1) // 576
                    self.pending.append((self.tick + self.schema.play_delay_ticks, side, slot))
                    self.observations[0, side, 4 + slot] = 1
        self.tick += 5
        self.observations[0, :, 10:12] = 0
        for due, side, slot in list(self.pending):
            if due <= self.tick:
                self.pending.remove((due, side, slot))
                self.observations[0, side, 4 + slot] = 0
                if self.run_refusal:
                    self.observations[0, side, 10] = 7
                else:
                    self.executions.append((due, side, slot))
        for due, side, slot in list(self.pending_abilities):
            self.observations[0, side, 13 + slot] = max(0, due - self.tick)
            if due <= self.tick:
                self.pending_abilities.remove((due, side, slot))
                self.ability_executions.append((due, side, slot))
        if self.terminal_at is not None and self.tick >= self.terminal_at:
            self.dones[0] = True
        self.refresh()


def run(vector, events, **settings):
    decks = settings.pop("decks", [CARDS, CARDS])
    deck_keys = settings.pop("deck_keys", [CARDS, CARDS])
    runtime = settings.pop("runtime_decks", decks)
    native_observer = settings.pop("native_observer", None)
    return execute_repaired_events(
        vector,
        events,
        decks,
        deck_keys,
        runtime,
        ExecutionConfig(**settings),
        native_observer=native_observer,
    )


class FakeNativeObserver:
    """Emit each accepted submission and delayed native result exactly once."""

    def __init__(self, vector, *, failures=None, match_decided_at=None, results_missing=()):
        self.vector = vector
        self.failures = failures or {}
        self.match_decided_at = match_decided_at
        self.results_missing = set(results_missing)
        self.commands = []
        self.result_names = set()

    def poll(self):
        vector = self.vector
        new_commands = []
        for index in range(len(self.commands), len(vector.submissions)):
            tick, seat, action = vector.submissions[index]
            ability = action >= 2305
            command = {
                "name": f"cmd{index}",
                "kind": "ability" if ability else "card",
                "seat": seat,
                "given_tick": tick,
                "run_tick": tick
                + (vector.ability_delay if ability else vector.schema.play_delay_ticks),
                "card": CARDS[(action - 1) // 576] if not ability else "A",
            }
            self.commands.append(command)
            new_commands.append(command.copy())
        results, pending = [], []
        for index, command in enumerate(self.commands):
            if command["name"] in self.result_names:
                continue
            if command["run_tick"] > vector.tick:
                pending.append(command.copy())
                continue
            if index in self.results_missing:
                # A command was observed as submitted but disappeared from the
                # pending queue without an observable native result.
                self.result_names.add(command["name"])
                continue
            failure = self.failures.get(index)
            results.append(
                {
                    "name": command["name"],
                    "kind": command["kind"],
                    "seat": command["seat"],
                    "tick": command["run_tick"],
                    "success": failure is None,
                    "retryable": bool(failure and failure.get("retryable")),
                    "refusal": failure.get("refusal") if failure else None,
                    "card": command["card"],
                }
            )
            self.result_names.add(command["name"])
        return {
            "new_commands": new_commands,
            "new_results": results,
            "pending": pending,
            "pending_count": len(pending),
            "match_decided": (
                self.match_decided_at is not None and vector.tick >= self.match_decided_at
            ),
        }


class MappedAbilityObserver(FakeNativeObserver):
    """Model native source-card identity independently of its champion unit row."""

    def __init__(self, vector, slots):
        super().__init__(vector)
        self.slots = slots
        self.seats_requested = []

    def ability_slots(self, seat=None):
        self.seats_requested.append(seat)
        return [dict(row) for row in self.slots if seat is None or row["seat"] == seat]


def native_ability_slot(
    slot, card, champion, *, seat=0, available=True, state_name="ready", unit_id=42
):
    return {
        "seat": seat,
        "slot": slot,
        "action": 2305 + slot,
        "card": card,
        "card_candidates": [card],
        "champion": champion,
        "native_unit_name": "cmd0_0" if unit_id >= 0 else None,
        "unit_id": unit_id,
        "state_name": state_name,
        "cooldown_ms": 0,
        "available": available,
        "mapping_source": "native_deck_index",
    }


def ability_decks(*cards):
    return [list(cards) + CARDS[len(cards) :], CARDS]


class WinnerObserver(FakeNativeObserver):
    def __init__(self, vector, *, winner, match_decided_at, match_ended_at=None):
        super().__init__(vector, match_decided_at=match_decided_at)
        self.winner = winner
        self.match_ended_at = match_decided_at if match_ended_at is None else match_ended_at

    def poll(self):
        result = super().poll()
        result["winner"] = self.winner
        result["crowns"] = [1, 0] if self.winner == 0 else [0, 1]
        result["match_ended"] = self.vector.tick >= self.match_ended_at
        result["end_timer_ms"] = 0 if result["match_ended"] else -1
        result["tiebreak_ms"] = 0 if result["match_ended"] else 1000
        return result


def test_fifo_serializes_same_seat_and_submits_both_seats_together():
    vector = FakeVector()
    stats, arrays, _, _ = run(
        vector,
        {0: [event("A"), event("B"), event("C", "opponent")]},
        repair_time=True,
    )
    assert [(tick, seat) for tick, seat, _ in vector.submissions] == [(0, 0), (0, 1), (5, 0)]
    assert stats["submitted_commands"] == 3
    assert len(vector.executions) == 3
    assert stats["source_event_stream_completed"]
    assert arrays["actions"].shape == arrays["ticks"].shape == arrays["seats"].shape
    np.testing.assert_array_equal(arrays["seats"], np.tile([0, 1], len(arrays["seats"]) // 2))


def test_closed_mask_waits_then_uses_live_pre_action_state_and_drains_commands():
    vector = FakeVector(open_at=10)
    stats, arrays, _, _ = run(vector, {0: [event()]}, repair_time=True)
    assert vector.submissions[0][0] == 10
    assert vector.executions[0][0] == 30
    rows = np.flatnonzero(arrays["actions"] > 0)
    assert arrays["ticks"][rows].tolist() == [10]
    np.testing.assert_array_equal(arrays["observations"][:, 12], arrays["ticks"])
    assert arrays["masks"][rows, arrays["actions"][rows]].all()
    assert stats["last_simulator_tick"] >= 30
    assert stats["source_event_stream_completed"]


def test_hand_repair_uses_runtime_index_and_captures_mutated_state(monkeypatch):
    vector = FakeVector()
    calls = []

    def swap(actual_vector, side, index, *, return_record):
        assert actual_vector is vector and return_record
        calls.append(index)
        vector.observations[0, side, 0] = vector.card_id("E")
        vector.refresh()
        return {"before_slots": [0, 1, 2, 3], "after_slots": [4, 1, 2, 3]}

    monkeypatch.setattr("cr_rl.data.opening_cycle.force_card_into_hand", swap)
    runtime = [list(reversed(CARDS)), CARDS]
    stats, arrays, _, _ = run(vector, {0: [event("E")]}, repair_hand=True, runtime_decks=runtime)
    assert calls == [3]  # E's runtime deck index, not the source index 4.
    assert arrays["observations"][0, 0] == 4
    assert arrays["actions"][0] != 0
    assert stats["repair_summary"]["hand_replacements"] == 1
    assert stats["dataset_kind"] == "repaired"


def test_terminal_before_delayed_command_runs_is_not_full_execution():
    vector = FakeVector(terminal_at=5, play_delay=20)
    stats, _, _, _ = run(vector, {0: [event()]})
    assert stats["submitted_commands"] == 1
    assert not vector.executions and vector.pending
    assert not stats["all_source_events_executed"]
    assert not stats["source_event_stream_completed"]


def test_global_budget_with_unsettled_pending_command_is_not_complete():
    vector = FakeVector(play_delay=1000)
    stats, _, _, _ = run(vector, {0: [event()]}, max_delay_ticks=0)
    assert not vector.executions and vector.pending
    assert not stats["all_source_events_executed"]
    assert not stats["source_event_stream_completed"]


def test_early_terminal_accounts_for_every_unattempted_source_event():
    vector = FakeVector(terminal_at=5)
    stats, _, _, _ = run(vector, {0: [event()], 30: [event("B")]})
    final = [
        record
        for record in stats["repair_records"]
        if record["category"] in ("execution", "rejected")
    ]
    assert {record["event_id"] for record in final} == {0, 1}
    remaining = next(record for record in final if record["event_id"] == 1)
    assert remaining["reason"] == "terminal_before_event"
    assert not stats["source_event_stream_completed"]


def test_native_run_refusal_invalidates_corresponding_action_supervision():
    vector = FakeVector(run_refusal=True)
    stats, arrays, _, _ = run(vector, {0: [event()]})
    assert stats["native_run_refusals"]
    assert not stats["all_source_events_executed"]
    acted = arrays["actions"] > 0
    assert acted.any()
    assert (arrays["flags"][acted] != 0).all()


def test_time_limit_is_recorded_separately_from_native_terminal():
    vector = FakeVector(terminal_at=5, truncated=True)
    stats, _, _, _ = run(vector, {0: [event()]})
    assert not stats["simulator_terminated"]
    assert not stats["source_event_stream_completed"]


def test_drain_waits_for_ability_commands_as_well_as_hand_pending():
    vector = FakeVector(ability_delay=80)
    vector.observations[0, 0, 8] = 0
    vector.refresh()
    ability = {"kind": "ability", "side": "team", "card_key": "A"}
    stats, _, _, _ = run(vector, {0: [ability]})
    assert vector.ability_executions == [(80, 0, 0)]
    assert stats["source_event_stream_completed"]


def test_out_of_bounds_coordinates_cannot_bypass_declared_tile_shift_limit():
    vector = FakeVector()
    stats, _, _, _ = run(vector, {0: [event(x=-5500)]}, repair_tile=True, max_tile_shift=2)
    assert stats["submitted_commands"] == 0
    assert not stats["source_event_stream_completed"]


def test_source_event_keys_preserves_ability_candidates_without_fabricating_card():
    assert source_event_keys({"kind": "ability", "ability_source_candidates": ["B", "A", "B"]}) == [
        "B",
        "A",
    ]
    assert source_event_keys({"kind": "ability", "card_key": "C"}) == ["C"]
    assert source_event_keys({"kind": "play_card", "ability_source_candidates": ["A"]}) == []


def test_multi_candidate_ability_uses_only_unique_matching_legal_slot():
    vector = FakeVector()
    vector.observations[0, 0, 8] = 1  # Only B is an available champion.
    vector.refresh()
    ability = {
        "kind": "ability",
        "side": "team",
        "ability_source_candidates": ["A", "B"],
        "ability_source_authoritative": False,
    }
    stats, arrays, _, _ = run(vector, {0: [ability]})
    assert stats["submitted_commands"] == 1
    assert arrays["actions"][0] == 2305
    changes = stats["repair_records"]
    assert any(row["category"] == "ability_source_resolution" for row in changes)


def test_ambiguous_two_legal_champions_does_not_guess_an_ability():
    vector = FakeVector()
    vector.observations[0, 0, 8:10] = [0, 1]
    vector.refresh()
    ability = {
        "kind": "ability",
        "side": "team",
        "ability_source_candidates": ["A", "B"],
    }
    stats, _, _, _ = run(vector, {0: [ability]})
    assert stats["submitted_commands"] == 0
    assert not stats["source_event_stream_completed"]


@pytest.mark.parametrize(
    ("card", "champion"),
    [("Goblinstein", "goblinstein_doctor"), ("Giant", "GiantHero")],
)
def test_native_card_mapping_resolves_different_champion_unit_names(card, champion):
    vector = FakeVector()
    vector.vocabulary = CARDS + [champion]
    vector.observations[0, 0, 8] = len(CARDS)
    vector.refresh()
    observer = MappedAbilityObserver(vector, [native_ability_slot(0, card, champion)])
    ability = {"kind": "ability", "side": "team", "card_key": card.lower()}
    stats, arrays, _, _ = run(
        vector,
        {0: [ability]},
        decks=ability_decks(card),
        deck_keys=[[card.lower()] + CARDS[1:], CARDS],
        native_observer=observer,
    )
    assert observer.seats_requested == [0]
    assert stats["native_successful_commands"] == 1
    assert stats["all_source_events_executed"]
    assert arrays["actions"][0] == 2305
    execution = next(row for row in stats["repair_records"] if row["category"] == "execution")
    assert execution["ability_slot"]["card"] == card
    assert execution["ability_slot"]["champion"] == champion
    resolution = next(
        row for row in stats["repair_records"] if row["category"] == "ability_source_resolution"
    )
    assert resolution["resolved_card"] == card
    assert resolution["native_champion"] == champion
    assert resolution["mapping_source"] == "native_deck_index"


def test_native_multi_source_candidates_choose_unique_available_ability_slot():
    vector = FakeVector()
    vector.observations[0, 0, 8:10] = [0, 1]
    vector.refresh()
    observer = MappedAbilityObserver(
        vector,
        [
            native_ability_slot(0, "Goblinstein", "goblinstein_doctor", available=False),
            native_ability_slot(1, "Giant", "GiantHero"),
        ],
    )
    ability = {
        "kind": "ability",
        "side": "team",
        "ability_source_candidates": ["goblinstein", "giant"],
        "ability_source_authoritative": False,
    }
    stats, arrays, _, _ = run(
        vector,
        {0: [ability]},
        decks=ability_decks("Goblinstein", "Giant"),
        deck_keys=[["goblinstein", "giant"] + CARDS[2:], CARDS],
        native_observer=observer,
    )
    assert stats["all_source_events_executed"]
    assert stats["native_successful_commands"] == 1
    assert arrays["actions"][0] == 2306
    execution = next(row for row in stats["repair_records"] if row["category"] == "execution")
    assert execution["ability_slot"]["card"] == "Giant"


def test_native_multiple_matching_legal_ability_slots_are_rejected_with_diagnostics():
    vector = FakeVector()
    vector.observations[0, 0, 8:10] = [0, 1]
    vector.refresh()
    observer = MappedAbilityObserver(
        vector,
        [
            native_ability_slot(0, "Goblinstein", "goblinstein_doctor"),
            native_ability_slot(1, "Giant", "GiantHero"),
        ],
    )
    ability = {
        "kind": "ability",
        "side": "team",
        "ability_source_candidates": ["goblinstein", "giant"],
    }
    stats, _, _, _ = run(
        vector,
        {0: [ability]},
        decks=ability_decks("Goblinstein", "Giant"),
        deck_keys=[["goblinstein", "giant"] + CARDS[2:], CARDS],
        native_observer=observer,
    )
    assert stats["submitted_commands"] == 0
    rejected = next(row for row in stats["repair_records"] if row["category"] == "rejected")
    assert rejected["reason"] == "ability_unavailable_or_ambiguous"
    assert rejected["detail"]["ability_failure"] == "multiple_legal_slots"
    assert rejected["detail"]["matching_slots"] == 2
    assert rejected["detail"]["legal_matching_slots"] == 2


@pytest.mark.parametrize(("state_name", "unit_id"), [("absent", -1), ("cooldown", 42)])
def test_native_matched_but_closed_ability_keeps_identity_and_gate_diagnostics(state_name, unit_id):
    vector = FakeVector()
    vector.observations[0, 0, 8] = 0
    vector.refresh()
    slot = native_ability_slot(
        0,
        "Goblinstein",
        "goblinstein_doctor",
        available=False,
        state_name=state_name,
        unit_id=unit_id,
    )
    observer = MappedAbilityObserver(vector, [slot])
    ability = {"kind": "ability", "side": "team", "card_key": "goblinstein"}
    stats, _, _, _ = run(
        vector,
        {0: [ability]},
        decks=ability_decks("Goblinstein"),
        deck_keys=[["goblinstein"] + CARDS[1:], CARDS],
        native_observer=observer,
        repair_time=True,
        max_delay_ticks=10,
    )
    assert not stats["all_source_events_executed"]
    assert stats["submitted_commands"] == 0
    rejected = next(row for row in stats["repair_records"] if row["category"] == "rejected")
    assert rejected["last_wait_reason"] == "ability_unavailable_or_ambiguous"
    assert rejected["detail"]["ability_failure"] == "matching_slot_closed"
    assert rejected["detail"]["matching_slots"] == 1
    assert rejected["detail"]["legal_matching_slots"] == 0
    assert rejected["detail"]["native_slots"] == [slot]


def test_declared_source_duration_continues_after_last_event_and_pending_drain():
    vector = FakeVector()
    stats, arrays, _, _ = run(vector, {0: [event()]}, source_end_tick=100)
    assert stats["last_simulator_tick"] >= 100
    assert stats["reached_declared_end"] is True
    assert stats["source_event_stream_completed"] is True
    assert stats["simulator_terminated"] is False
    assert arrays["ticks"].max() == 95


def test_native_submission_refusal_is_attributed_and_masks_its_action_row():
    vector = FakeVector(submission_refusal_seat=0)
    stats, arrays, _, omitted = run(vector, {0: [event()]})
    assert stats["submitted_commands"] == 0
    assert stats["skipped"]["native_submission_refused"] == 1
    assert omitted == [(0, 0)]
    record = next(row for row in stats["repair_records"] if row["category"] == "execution")
    assert record["event_id"] == 0 and record["seat"] == 0
    assert record["accepted"] is False
    acted = arrays["actions"] > 0
    assert (arrays["flags"][acted] != 0).all()


def test_observer_counts_confirmed_native_execution_separately_from_submission():
    vector = FakeVector()
    observer = FakeNativeObserver(vector)
    stats, _, executed, omitted = run(vector, {0: [event()]}, native_observer=observer)
    assert stats["submitted_commands"] == stats["native_successful_commands"] == 1
    assert stats["execution_confirmation"] == "native_command_results"
    assert stats["source_event_stream_completed"]
    assert executed == [(0, False)] and omitted == []
    record = next(row for row in stats["repair_records"] if row["category"] == "execution")
    assert record["native_command_name"] == "cmd0"
    assert record["execution_status"] == "confirmed"
    assert record["actual_run_tick"] == 20


def test_observer_pending_terminal_is_never_counted_as_native_success():
    vector = FakeVector(terminal_at=5)
    stats, arrays, executed, _ = run(
        vector, {0: [event()]}, native_observer=FakeNativeObserver(vector)
    )
    assert stats["submitted_commands"] == 1
    assert stats["native_successful_commands"] == stats["used"] == 0
    assert stats["missing_source_events"] == 1
    assert not stats["all_source_events_executed"]
    assert not stats["pending_commands_settled"]
    assert executed == []
    assert arrays["flags"][0] != 0


def test_retry_waits_for_confirmation_then_retries_before_later_same_seat_event():
    vector = FakeVector()
    observer = FakeNativeObserver(
        vector, failures={0: {"refusal": "not_enough_elixir", "retryable": True}}
    )
    stats, arrays, executed, omitted = run(
        vector,
        {0: [event("A")], 5: [event("B")]},
        native_observer=observer,
        repair_time=True,
    )
    assert [(tick, action // 576) for tick, _, action in vector.submissions] == [
        (0, 0),
        (20, 0),
        (40, 1),
    ]
    assert stats["submitted_commands"] == 3
    assert stats["native_successful_commands"] == stats["used"] == 2
    assert stats["coverage"] == 1.0
    assert stats["source_event_stream_completed"]
    assert len(executed) == 2 and omitted == []
    assert stats["native_run_refusals"][0]["retried"] is True
    failed = (arrays["ticks"] == 0) & (arrays["seats"] == 0)
    retry = (arrays["ticks"] == 20) & (arrays["seats"] == 0)
    assert (arrays["flags"][failed] != 0).all()
    assert (arrays["flags"][retry] == 0).all()
    records = [row for row in stats["repair_records"] if row["category"] == "execution"]
    assert [row["event_id"] for row in records] == [0, 0, 1]
    assert [row["execution_status"] for row in records] == [
        "native_run_refused",
        "confirmed",
        "confirmed",
    ]


def test_exact_mode_allows_later_legal_combo_while_first_native_command_is_pending():
    vector = FakeVector()
    stats, _, _, omitted = run(
        vector,
        {0: [event("A")], 5: [event("B")]},
        native_observer=FakeNativeObserver(vector),
    )
    assert [tick for tick, _, _ in vector.submissions] == [0, 5]
    assert stats["native_successful_commands"] == 2
    assert stats["source_event_stream_completed"] and not omitted


def test_nonretryable_native_failure_masks_the_original_submission_row():
    vector = FakeVector()
    observer = FakeNativeObserver(
        vector, failures={0: {"refusal": "match_over", "retryable": False}}
    )
    stats, arrays, executed, omitted = run(
        vector, {0: [event()]}, native_observer=observer, repair_time=True
    )
    assert stats["submitted_commands"] == 1
    assert stats["native_successful_commands"] == 0
    assert not stats["all_source_events_executed"]
    assert executed == [] and omitted == [(0, 0)]
    assert arrays["actions"][0] > 0 and arrays["flags"][0] != 0
    record = next(row for row in stats["repair_records"] if row["category"] == "execution")
    assert record["native_success"] is False
    assert record["actual_run_tick"] == 20
    assert record["execution_status"] == "native_run_refused"


def test_native_decided_without_episode_done_is_reported_as_early_terminal():
    vector = FakeVector(play_delay=5)
    observer = FakeNativeObserver(vector, match_decided_at=10)
    stats, _, _, _ = run(vector, {0: [event()], 30: [event("B")]}, native_observer=observer)
    assert not vector.dones[0]
    assert stats["simulator_match_decided"] and stats["simulator_terminated"]
    assert not stats["simulator_episode_done"]
    assert stats["early_terminal_before_last_source_event"]
    assert not stats["source_event_stream_completed"]
    assert stats["last_simulator_tick"] == 10
    remaining = [
        row
        for row in stats["repair_records"]
        if row["category"] == "rejected" and row["event_id"] == 1
    ]
    assert remaining[0]["reason"] == "terminal_before_event"


@pytest.mark.parametrize("winner", [0, 1])
def test_normal_native_early_result_counts_combined_completion_only(winner):
    vector = FakeVector(play_delay=5)
    observer = WinnerObserver(vector, winner=winner, match_decided_at=10)
    stats, _, _, _ = run(vector, {0: [event()], 30: [event("B")]}, native_observer=observer)
    assert not vector.dones[0]  # Native result can precede the end-screen episode flag.
    assert stats["native_winner"] == winner
    assert stats["normal_simulator_terminal"]
    assert stats["source_or_native_terminal_completed"]
    assert stats["trajectory_completion_kind"] == "normal_simulator_terminal"
    assert not stats["all_source_events_executed"]
    assert stats["native_successful_commands"] == 1
    assert stats["missing_source_events"] == 1


def test_native_decided_without_a_winner_is_not_normal_completion():
    vector = FakeVector(play_delay=5)
    stats, _, _, _ = run(
        vector,
        {0: [event()], 30: [event("B")]},
        native_observer=WinnerObserver(vector, winner=-1, match_decided_at=10),
    )
    assert stats["simulator_match_decided"]
    assert stats["native_winner"] == -1
    assert not stats["normal_simulator_terminal"]
    assert not stats["source_or_native_terminal_completed"]
    assert stats["trajectory_completion_kind"] == "incomplete"


def test_decided_and_default_winner_during_tiebreak_is_not_normal_completion():
    vector = FakeVector(terminal_at=10, play_delay=5)
    stats, _, _, _ = run(
        vector,
        {0: [event()], 30: [event("B")]},
        native_observer=WinnerObserver(vector, winner=0, match_decided_at=10, match_ended_at=100),
    )
    assert stats["simulator_match_decided"]
    assert stats["native_winner"] == 0
    assert not stats["normal_simulator_terminal"]
    assert not stats["source_or_native_terminal_completed"]


def test_native_decided_waits_until_match_ended_before_normal_completion():
    vector = FakeVector(play_delay=5)
    stats, _, _, _ = run(
        vector,
        {0: [event()], 30: [event("B")]},
        native_observer=WinnerObserver(vector, winner=0, match_decided_at=10, match_ended_at=20),
    )
    assert stats["last_simulator_tick"] == 20
    assert stats["normal_simulator_terminal"]
    assert stats["source_or_native_terminal_completed"]
    assert not stats["all_source_events_executed"]


def test_winner_without_native_decided_is_not_normal_completion():
    vector = FakeVector(terminal_at=10, play_delay=5)
    stats, _, _, _ = run(
        vector,
        {0: [event()], 30: [event("B")]},
        native_observer=WinnerObserver(vector, winner=0, match_decided_at=100),
    )
    assert stats["simulator_episode_done"]
    assert not stats["simulator_match_decided"]
    assert not stats["normal_simulator_terminal"]
    assert not stats["source_or_native_terminal_completed"]


def test_truncated_native_result_is_not_normal_completion():
    vector = FakeVector(terminal_at=10, play_delay=5, truncated=True)
    stats, _, _, _ = run(
        vector,
        {0: [event()], 30: [event("B")]},
        native_observer=WinnerObserver(vector, winner=0, match_decided_at=10),
    )
    assert stats["simulator_match_decided"] and stats["simulator_truncated"]
    assert not stats["normal_simulator_terminal"]
    assert not stats["source_or_native_terminal_completed"]


def test_faulted_native_result_is_not_normal_completion():
    class FaultedVector(FakeVector):
        def info(self, env):
            return {**super().info(env), "fault": int(self.tick >= 10)}

    vector = FaultedVector(play_delay=5)
    stats, _, _, _ = run(
        vector,
        {0: [event()], 30: [event("B")]},
        native_observer=WinnerObserver(vector, winner=0, match_decided_at=10),
    )
    assert stats["simulator_fault"]
    assert not stats["normal_simulator_terminal"]
    assert not stats["source_or_native_terminal_completed"]


def test_complete_source_stream_counts_without_a_native_terminal_result():
    vector = FakeVector(play_delay=5)
    stats, _, _, _ = run(
        vector,
        {0: [event()]},
        native_observer=WinnerObserver(vector, winner=-1, match_decided_at=100),
    )
    assert stats["all_source_events_executed"]
    assert not stats["normal_simulator_terminal"]
    assert stats["source_or_native_terminal_completed"]
    assert stats["trajectory_completion_kind"] == "all_source_events"


def test_unconfirmed_command_cannot_be_supervised_when_native_pending_disappears():
    vector = FakeVector()
    observer = FakeNativeObserver(vector, match_decided_at=45, results_missing={1})
    stats, arrays, _, _ = run(
        vector,
        {0: [event("A")], 25: [event("B")]},
        native_observer=observer,
    )
    assert stats["submitted_commands"] == 2 and stats["native_successful_commands"] == 1
    assert not stats["source_event_stream_completed"]
    first = (arrays["ticks"] == 0) & (arrays["seats"] == 0)
    unknown = (arrays["ticks"] == 25) & (arrays["seats"] == 0)
    assert (arrays["flags"][first] == 0).all()
    assert (arrays["flags"][unknown] != 0).all()
    assert not stats["pending_commands_settled"]
