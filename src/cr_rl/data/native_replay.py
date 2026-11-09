"""Read actual command outcomes from the pinned crforge JVM without mutation.

The submission mask and ``last_refusal`` use a different code space from native
``Play.matchCode``. In particular native run code 4 means MATCH_OVER, while
schema submission refusal 4 means START_COOLDOWN. This reader reports the native
command identity and result so delayed refusals cannot be counted as execution.
"""

from __future__ import annotations

from collections import Counter

from cr_rl.data.opening_cycle import _native_env_match

CARD_RUN_CODES = {
    0: "ok",
    4: "match_over",
    8: "king_dead",
    9: "not_in_hand",
    13: "not_enough_elixir",
    22: "start_cooldown",
}
PLACEMENT_CODES = {
    0: "ok",
    15: "point_left_of_map",
    16: "point_above_map",
    17: "point_right_of_map",
    18: "point_below_map",
    19: "cell_not_placeable",
    23: "no_position",
}
ABILITY_RUN_CODES = {
    0: "ok",
    8: "not_champion",
    10: "clone",
    1004: "king_gone",
    1005: "cooldown",
    1006: "no_champion",
    1009: "pending",
    1011: "frozen",
    1013: "disabled",
    1014: "no_charges",
    1050: "not_enough_elixir",
}


def native_refusal(code, *, ability=False):
    """Interpret a native *run* code; never pass schema submission codes here."""
    codes = ABILITY_RUN_CODES if ability else CARD_RUN_CODES
    return codes.get(int(code), f"unknown_native_run_code_{int(code)}")


def placement_refusal(code):
    """CardPlacement.Result uses MapCheck/placement codes, not match codes."""
    return PLACEMENT_CODES.get(int(code), f"unknown_placement_code_{int(code)}")


def _private_field(java_object, name):
    result = java_object.getClass().getDeclaredField(name)
    result.setAccessible(True)
    return result


def _record_methods(java_class, names):
    methods = {}
    for name in names:
        method = java_class.getDeclaredMethod(name)
        method.setAccessible(True)
        methods[name] = method
    return methods


class NativeReplayObserver:
    """Incremental observer bound to one live match and one vector lane.

    Construct after reset, then call ``poll()`` after every ``vector.step()``.
    Every actual play/ability outcome is emitted exactly once. ``new_commands``
    reports accepted submissions, distinct from ``new_results``. ``pending``
    includes native card and ability waiting lists, even if hand_pending alone
    is empty. A reset invalidates this observer; make a new one after reset.

    No field is changed. Private reflection handles are resolved once at init
    and reused. This is deliberately pinned to crforge 7ee59b39 and fails if
    those implementation contracts change.
    """

    def __init__(self, vector, env_index=0, *, include_existing=False):
        self.vector = vector
        self.env_index = int(env_index)
        self.env, self.match = _native_env_match(vector, self.env_index)
        self.standard = self.env.battle()
        self._waiting = _private_field(self.env, "waiting")
        self._ability_waiting = _private_field(self.env, "abilityWaiting")
        classes = {
            str(cls.getSimpleName()): cls for cls in self.env.getClass().getDeclaredClasses()
        }
        self._waiting_methods = _record_methods(
            classes["Waiting"], ("name", "side", "card", "x", "y", "runTick")
        )
        self._ability_methods = _record_methods(
            classes["AbilityWaiting"], ("name", "side", "slot", "runTick")
        )
        self._decided = self.match.getClass().getDeclaredMethod("decided")
        self._decided.setAccessible(True)
        self._commands = {}
        self._unit_cards = {}
        self._command_cursor = self._play_cursor = self._ability_cursor = 0
        # Prime original card identity for ability units created before attach.
        for command in self.standard.getCommands():
            self._commands[str(command.name())] = self._command_record(command)
        for play in self.standard.getPlays():
            self._remember_units(play)
        if not include_existing:
            self._command_cursor = len(self.standard.getCommands())
            self._play_cursor = len(self.standard.getPlays())
            self._ability_cursor = len(self.standard.getAbilityUses())

    def _command_record(self, command):
        name = str(command.name())
        card = str(command.card()) if command.card() is not None else None
        unit = str(command.unit()) if command.unit() is not None else None
        return {
            "name": name,
            "kind": "ability" if str(command.kind()) == "ABILITY" else "card",
            "seat": int(command.side()),
            "given_tick": int(command.given()),
            "run_tick": int(command.tick()),
            "card": card or self._unit_cards.get(unit),
            "unit": unit,
        }

    def _remember_units(self, play):
        command = self._commands.get(str(play.name()))
        card = command["card"] if command else None
        for unit in play.units():
            self._unit_cards[str(unit.name())] = card or str(unit.getData().name())

    def _card_result(self, play):
        self._remember_units(play)
        name = str(play.name())
        command = self._commands.get(name, {})
        result = play.result()
        match_code = int(play.matchCode())
        result_code = None if result is None else int(result.code())
        placed = bool(result is not None and result.placed())
        success = match_code == 0 and placed
        reason = None
        if not success:
            if match_code:
                reason = native_refusal(match_code)
            elif result_code is not None:
                reason = placement_refusal(result_code)
            else:
                reason = "missing_placement_result"
        return {
            "kind": "card",
            "name": name,
            "seat": int(play.side()),
            "tick": int(play.tick()),
            "card": command.get("card"),
            "match_code": match_code,
            "result_code": result_code,
            "result_placed": placed,
            "success": success,
            "refusal": reason,
            "retryable": reason
            in {
                "not_in_hand", "not_enough_elixir", "start_cooldown", "no_position",
                "point_left_of_map", "point_above_map", "point_right_of_map",
                "point_below_map", "cell_not_placeable",
            },
            "requested_x": int(play.x()),
            "requested_y": int(play.y()),
            "placed_x": int(result.x()) if placed else None,
            "placed_y": int(result.y()) if placed else None,
        }

    def _ability_result(self, use):
        name, unit = str(use.name()), str(use.unit())
        outcome = use.outcome()
        code = int(outcome.code())
        requested = list(outcome.requested())
        card = self._unit_cards.get(unit)
        if card is None and requested:
            card = str(requested[0].getData().name())
        reason = None if code == 0 else native_refusal(code, ability=True)
        return {
            "kind": "ability",
            "name": name,
            "seat": int(use.side()),
            "tick": int(use.tick()),
            "card": card,
            "unit": unit,
            "ability_outcome_code": code,
            "success": code == 0,
            "refusal": reason,
            "retryable": reason
            in {"cooldown", "no_champion", "pending", "frozen", "not_enough_elixir"},
            "elixir_before": int(outcome.elixirBefore()),
            "elixir_after": int(outcome.elixirAfter()),
            "requested_unit_names": [str(entity.name()) for entity in requested],
        }

    def ability_slots(self, seat=None):
        """Read source deck-card identities for native champion ability slots.

        A champion slot names a UnitData row, not the source card row.  The
        native deck pass retains its originating runtime deck index.  This
        authoritative link handles secondary groups, linked projectile champions
        and listed units without handwritten name aliases or Play.units scans.
        If that index is unavailable, expose all matching native deck
        candidates rather than guessing one. The next card form can change
        after a hero play, so it never overrides an explicit native deck index.

        ``available`` is the native legal mask.  A mapped but unavailable slot
        does not imply death: inspect state/cooldown/charges/unit_id separately.
        No gameplay state is changed and observation rows are not regenerated.
        """
        if seat is not None and int(seat) not in (0, 1):
            raise ValueError("Ability slot seat must be zero or one")
        if self.env.battle() != self.standard:
            raise ValueError("NativeReplayObserver cannot span a simulator reset")
        world = self.standard.getWorld()
        records = world.getRecords()
        state_names = {
            0: "unknown", 1: "absent", 2: "ready", 3: "deploying",
            4: "limited_availability", 6: "no_charges", 7: "pending", 8: "cooldown",
            9: "not_enough_elixir", 10: "casting", 11: "disabled",
            12: "temporarily_unavailable", 13: "not_yet_available",
        }
        result = []
        for side in (range(2) if seat is None else (int(seat),)):
            own = self.match.side(side)
            deck = own.deck()

            def champion_of(index, own=own):
                spell = own.item(index).spell()
                data = records.cardChampion(str(spell.name()))
                return None if data is None else str(data.name())

            king = world.kingTower(side)
            for slot in range(2):
                controller = king.championSlot(slot + 1)
                champion = None if controller is None else controller.getChampion()
                name = None if champion is None else str(champion.name())
                index = -1 if controller is None else int(controller.getDeckIndex())
                matching = []
                mapping_source = "unresolved"
                if name is not None:
                    if 0 <= index < len(deck):
                        matching = [index]
                        mapping_source = "native_deck_index"
                    else:
                        matching = [i for i in range(len(deck)) if champion_of(i) == name]
                        if matching:
                            mapping_source = "native_card_champion_scan"
                candidates = list(dict.fromkeys(str(deck[i].name()) for i in matching))
                ability = None if champion is None else champion.ability()
                state = 0 if controller is None else int(controller.getState())
                unit_name = self.env.abilityUnit(side, slot)
                unit_fields = self.vector.schema.header.get("ability_unit", ())
                unit_id = (
                    int(self.vector.observations[self.env_index, side, unit_fields[slot]])
                    if len(unit_fields) > slot else -1
                )
                result.append(
                    {
                        "seat": side,
                        "slot": slot,
                        "action": 2305 + slot,
                        "card": candidates[0] if len(candidates) == 1 else None,
                        "card_candidates": candidates,
                        "deck_index": matching[0] if len(matching) == 1 else None,
                        "controller_deck_index": index,
                        "champion": name,
                        "ability": None if ability is None else str(ability.name()),
                        "native_unit_name": None if unit_name is None else str(unit_name),
                        "unit_id": unit_id,
                        "state": state,
                        "state_name": state_names.get(state, f"unknown_state_{state}"),
                        "cooldown_ms": 0 if controller is None else int(controller.getCooldownMs()),
                        "charges": 0 if controller is None else int(controller.getCharges()),
                        "deploy_index": (
                            -1 if controller is None else int(controller.getDeployIndex())
                        ),
                        "available": bool(self.vector.masks[self.env_index, side, 2305 + slot]),
                        "mapping_source": mapping_source,
                    }
                )
        return result

    def _pending_records(self):
        records = []
        for field, methods, kind in (
            (self._waiting, self._waiting_methods, "card"),
            (self._ability_waiting, self._ability_methods, "ability"),
        ):
            for entry in field.get(self.env):
                raw = {name: method.invoke(entry) for name, method in methods.items()}
                name = str(raw["name"])
                row = {
                    "kind": kind,
                    "name": name,
                    "seat": int(raw["side"]),
                    "run_tick": int(raw["runTick"]),
                    "card": self._commands.get(name, {}).get("card"),
                }
                if kind == "card":
                    row.update(
                        card_vocabulary_id=int(raw["card"]), x=int(raw["x"]), y=int(raw["y"])
                    )
                else:
                    row["ability_slot"] = int(raw["slot"])
                records.append(row)
        return records

    def poll(self):
        if self.env.battle() != self.standard:
            raise ValueError("NativeReplayObserver cannot span a simulator reset")
        commands = self.standard.getCommands()
        plays, abilities = self.standard.getPlays(), self.standard.getAbilityUses()
        if (
            len(commands) < self._command_cursor
            or len(plays) < self._play_cursor
            or len(abilities) < self._ability_cursor
        ):
            raise ValueError("Native command/result history shrank unexpectedly")
        new_commands = []
        for index in range(self._command_cursor, len(commands)):
            command = self._command_record(commands[index])
            self._commands[command["name"]] = command
            new_commands.append(command)
        new_results = [
            self._card_result(plays[index]) for index in range(self._play_cursor, len(plays))
        ]
        new_results.extend(
            self._ability_result(abilities[index])
            for index in range(self._ability_cursor, len(abilities))
        )
        order = {name: index for index, name in enumerate(self._commands)}
        new_results.sort(key=lambda row: (row["tick"], order.get(row["name"], -1)))
        self._command_cursor, self._play_cursor, self._ability_cursor = (
            len(commands),
            len(plays),
            len(abilities),
        )
        pending = self._pending_records()
        counts = Counter(row["kind"] for row in pending)
        return {
            "new_commands": new_commands,
            "new_results": new_results,
            "pending": pending,
            "pending_count": len(pending),
            "pending_card_count": counts["card"],
            "pending_ability_count": counts["ability"],
            "match_decided": bool(self._decided.invoke(self.match)),
            "winner": int(self.match.getWinner()),
            "crowns": [int(self.match.crowns(side)) for side in (0, 1)],
            "match_ended": bool(self.match.isEnded()),
            "tiebreak_ms": int(self.match.getTiebreakMs()),
            "end_timer_ms": int(self.match.getEndTimerMs()),
            "simulator_done": bool(self.vector.dones[self.env_index]),
            "tick": int(self.vector.info(self.env_index)["tick"]),
            "source": "pinned_native_command_outcomes",
        }
