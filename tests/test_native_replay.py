from types import SimpleNamespace

import numpy as np
import pytest

from cr_rl.data.native_replay import NativeReplayObserver, native_refusal, placement_refusal


def play(name="cmd0", code=0, placed=True):
    result = SimpleNamespace(code=lambda: 0 if placed else 23, placed=lambda: placed,
                             x=lambda: 10, y=lambda: 20)
    return SimpleNamespace(
        name=lambda: name, matchCode=lambda: code, units=lambda: [],
        result=lambda: result if code == 0 else None,
        side=lambda: 0, tick=lambda: 110, x=lambda: 9, y=lambda: 19,
    )


def observer(results=None):
    result = NativeReplayObserver.__new__(NativeReplayObserver)
    result._commands = {"cmd0": {"card": "Knight"}}
    result._unit_cards = {"cmd0_0": "Knight"}
    result._command_cursor, result._play_cursor, result._ability_cursor = 0, 0, 0
    result.env_index = 0
    result.standard = SimpleNamespace(
        getCommands=lambda: [], getPlays=lambda: results or [], getAbilityUses=lambda: []
    )
    result.env = SimpleNamespace(battle=lambda: result.standard)
    result.match = SimpleNamespace(
        getWinner=lambda: 1,
        crowns=lambda side: [1, 2][side],
        isEnded=lambda: True,
        getTiebreakMs=lambda: 0,
        getEndTimerMs=lambda: 1,
    )
    result._decided = SimpleNamespace(invoke=lambda match: True)
    result._pending_records = lambda: []
    result.vector = SimpleNamespace(dones=[False], info=lambda index: {"tick": 115})
    return result


def test_native_run_codes_are_not_schema_submission_refusal_names():
    assert native_refusal(4) == "match_over"
    assert native_refusal(22) == "start_cooldown"
    assert native_refusal(13) == "not_enough_elixir"
    assert native_refusal(1005, ability=True) == "cooldown"
    assert native_refusal(1005) == "unknown_native_run_code_1005"
    assert placement_refusal(4) == "unknown_placement_code_4"
    assert placement_refusal(19) == "cell_not_placeable"
    assert placement_refusal(23) == "no_position"


def test_native_card_outcome_distinguishes_submission_and_placement():
    reader = observer()
    ok = reader._card_result(play())
    assert ok["success"] and ok["result_placed"] and ok["card"] == "Knight"
    over = reader._card_result(play(code=4))
    assert not over["success"] and over["refusal"] == "match_over" and not over["retryable"]
    elixir = reader._card_result(play(code=13))
    assert not elixir["success"] and elixir["retryable"]
    position = reader._card_result(play(placed=False))
    assert not position["success"] and position["result_code"] == 23
    assert position["refusal"] == "no_position" and position["retryable"]


def test_native_ability_outcome_retains_its_separate_code_and_payment():
    reader = observer()
    use = SimpleNamespace(
        name=lambda: "cmd1", unit=lambda: "cmd0_0", side=lambda: 0, tick=lambda: 140,
        outcome=lambda: SimpleNamespace(code=lambda: 1050, requested=lambda: [],
                                        elixirBefore=lambda: 5000, elixirAfter=lambda: 5000),
    )
    row = reader._ability_result(use)
    assert row["ability_outcome_code"] == 1050 and not row["success"]
    assert row["card"] == "Knight" and row["retryable"]
    assert row["elixir_after"] == row["elixir_before"] == 5000


def test_poll_emits_actual_outcome_once_and_sees_decided_before_done():
    reader = observer([play(code=4)])
    first = reader.poll()
    assert len(first["new_results"]) == 1
    assert first["match_decided"] and not first["simulator_done"]
    assert first["winner"] == 1 and first["crowns"] == [1, 2]
    assert first["match_ended"] and first["end_timer_ms"] == 1
    assert not first["new_results"][0]["success"]
    assert reader.poll()["new_results"] == []
    reader.env.battle = lambda: object()
    with pytest.raises(ValueError, match="reset"):
        reader.poll()


def ability_observer(card, champion_name, *, index=1, available=True, state=2):
    reader = observer()
    champion = SimpleNamespace(
        name=lambda: champion_name,
        ability=lambda: SimpleNamespace(name=lambda: champion_name + "_ability"),
    )
    controller = SimpleNamespace(
        getChampion=lambda: champion,
        getDeckIndex=lambda: index,
        getState=lambda: state,
        getCooldownMs=lambda: 0,
        getCharges=lambda: -1,
        getDeployIndex=lambda: 17,
    )
    names = ["Knight", card, "Zap", "Archer", "Giant", "Minions", "Fireball", "IceSpirit"]
    deck = [SimpleNamespace(name=lambda name=name: name) for name in names]
    own = SimpleNamespace(
        deck=lambda: deck,
        item=lambda i: SimpleNamespace(spell=lambda: deck[i]),
    )
    world = SimpleNamespace(
        getRecords=lambda: SimpleNamespace(
            cardChampion=lambda name: champion if name == card else None
        ),
        kingTower=lambda side: SimpleNamespace(
            championSlot=lambda slot: controller if side == 0 and slot == 1 else None
        ),
    )
    reader.standard.getWorld = lambda: world
    reader.match.side = lambda side: own
    reader.env.abilityUnit = (
        lambda side, slot: "projectile_spawned_unit" if side == slot == 0 else None
    )
    reader.vector = SimpleNamespace(
        schema=SimpleNamespace(header={"ability_unit": (0, 1)}),
        observations=np.full((1, 2, 2), -1, np.int32),
        masks=np.zeros((1, 2, 2307), bool),
    )
    reader.vector.masks[0, 0, 2305] = available
    reader.vector.observations[0, 0, 0] = 42 if available else -1
    reader._unit_cards = {}  # Delayed projectile spawn need not occur in Play.units.
    return reader


@pytest.mark.parametrize(
    "card,champion",
    [
        ("Goblinstein", "goblinstein_doctor"),
        ("BarbLog", "BarbLogBarbarianHero"),
        ("Tombstone", "TombstoneHero_Monster_Passive"),
    ],
)
def test_ability_slot_uses_native_source_card_index_instead_of_unit_name_aliases(card, champion):
    reader = ability_observer(card, champion)
    row = reader.ability_slots(0)[0]
    assert row["card"] == card and row["card_candidates"] == [card]
    assert row["champion"] == champion
    assert row["mapping_source"] == "native_deck_index"
    assert row["deck_index"] == 1 and row["action"] == 2305
    assert row["available"] and row["state_name"] == "ready"
    assert row["native_unit_name"] == "projectile_spawned_unit"
    assert not reader._unit_cards


def test_native_deck_index_remains_authoritative_when_next_card_form_changes():
    reader = ability_observer("BarbLog", "BarbLogBarbarianHero")
    # The card currently resolves to an evolved/non-champion form, but the
    # controller still follows the champion created by the preceding hero play.
    reader.standard.getWorld().getRecords = lambda: SimpleNamespace(cardChampion=lambda _: None)
    row = reader.ability_slots(0)[0]
    assert row["card"] == "BarbLog" and row["available"]
    assert row["mapping_source"] == "native_deck_index"


def test_mapping_without_a_live_unit_does_not_call_it_dead():
    reader = ability_observer(
        "Tombstone", "TombstoneHero_Monster_Passive", available=False, state=1
    )
    row = reader.ability_slots(0)[0]
    assert row["card"] == "Tombstone"
    assert row["unit_id"] == -1 and not row["available"]
    assert row["state_name"] == "absent"
    assert "dead" not in row  # Absent also includes not yet spawned/deployed.


def test_missing_deck_index_uses_native_card_champion_mapping_without_guessing():
    reader = ability_observer("Goblinstein", "goblinstein_doctor", index=-1)
    row = reader.ability_slots(0)[0]
    assert row["card"] == "Goblinstein" and row["deck_index"] == 1
    assert row["mapping_source"] == "native_card_champion_scan"
    reader.standard.getWorld().getRecords = lambda: SimpleNamespace(cardChampion=lambda _: None)
    row = reader.ability_slots(0)[0]
    assert row["card"] is None and row["card_candidates"] == []
    assert row["mapping_source"] == "unresolved"


def test_ability_mapping_reads_both_seats_and_rejects_invalid_seat_or_reset():
    reader = ability_observer("Goblinstein", "goblinstein_doctor")
    assert [(row["seat"], row["slot"]) for row in reader.ability_slots()] == [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]
    with pytest.raises(ValueError, match="seat"):
        reader.ability_slots(2)
    reader.env.battle = lambda: object()
    with pytest.raises(ValueError, match="reset"):
        reader.ability_slots(0)
