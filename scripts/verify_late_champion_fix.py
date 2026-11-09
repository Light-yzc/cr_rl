"""Independently verify the late-spawned champion command lookup in a real JVM.

Run baseline and patched installs in separate processes; JPype cannot switch jars.
The positive cases use only legal actions, then export and label/re-drive those
native replays. Deliberately injected negative entities are tested afterwards and
never included in the exported replay or a BC dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

PATCH_MARKER = "cr_rl_late_champion_units_v1"
DEFAULT_INSTALL = Path(".cache/vendor/crforge/env/build/install/env")
DEFAULT_TABLES = Path.home() / ".crforge/tables/16.635.3"
DECK_TAIL = ["Zap", "Knight", "Arrows", "Minions", "Fireball", "Skeletons", "IceSpirits"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_lookup_probe(probe, *, expect):
    """Check actual live state, without equating READY alone with a living unit."""
    require(probe["state"] == 2, "the native champion controller must be READY")
    require(probe["alive_copies"] > 0, "a live champion copy must actually exist")
    if expect == "broken":
        require(not probe["available"], "baseline unexpectedly offers the ability")
        require(not probe["try_given"], "baseline unexpectedly resolves the late unit")
        require(probe["try_refusal"] == "NO_CHAMPION", "unexpected baseline refusal")
    else:
        require(probe["available"], "native mask fails to offer a live READY champion")
        require(probe["try_given"], "tryAbility cannot resolve the live champion")
        require(probe["try_refusal"] == "NONE", "native try unexpectedly refused")


def validate_ability_result(result, cost):
    require(result["kind"] == "ability" and result["success"], "ability did not run")
    require(result["ability_outcome_code"] == 0, "nonzero native ability outcome")
    require(
        result["elixir_before"] - result["elixir_after"] == cost * 10000,
        "actual native elixir payment differs from the quoted cost",
    )


def validate_labelled_replay(replay):
    require(replay.counts.get("refusal") is None, f"replay refused: {replay.counts}")
    ability_rows = replay.actions >= 2305
    require(int(ability_rows.sum()) == 1, "exported replay must contain exactly one ability")
    require((replay.flags[ability_rows] == 0).all(), "ability label is flagged")
    require(
        replay.masks[np.flatnonzero(ability_rows), replay.actions[ability_rows]].all(),
        "exported ability target is masked",
    )
    require(replay.drive is not None, "native replay drive check was not performed")
    require(replay.drive["reproduced"], f"commands not reproduced: {replay.drive}")
    require(replay.drive["state_reproduced"], f"state not reproduced: {replay.drive}")


def _alive(unit):
    hp = unit.getHitPoints()
    return hp is not None and int(hp.getHitPoints()) > 0


def _giving(tried):
    return {
        "given": bool(tried.given()),
        "refusal": str(tried.refusal()),
        "code": int(tried.code()),
        "cost": int(tried.cost()),
        "given_tick": int(tried.givenTick()),
        "run_tick": int(tried.runTick()),
    }


def readonly_try(observer, seat, name):
    before = (
        int(observer.match.side(seat).getElixir()),
        len(observer.standard.getCommands()),
        len(observer.standard.getAbilityUses()),
    )
    tried = _giving(observer.env.commands().tryAbility(seat, name))
    after = (
        int(observer.match.side(seat).getElixir()),
        len(observer.standard.getCommands()),
        len(observer.standard.getAbilityUses()),
    )
    require(before == after, "tryAbility mutated fees or native command/result counts")
    return tried


def _step(vector, observer, action=0):
    vector.actions[:] = 0
    vector.actions[0, 0] = action
    vector.step()
    info = vector.info(0)
    require(not info["fault"], f"native simulation fault at {info['tick']}")
    require(not info["illegal_0"] and not info["illegal_1"], "native submission refused")
    return observer.poll()


def _select_play(vector, card, x=13500, y=3500):
    hand_fields = list(vector.schema.header["hand_card"])
    hand = vector.observations[0, 0, hand_fields]
    card_id = vector.card_id(card)
    positions = np.flatnonzero(hand == card_id)
    require(len(positions) == 1, f"{card} is not in the initial hand")
    slot = int(positions[0])
    # A seat-zero observation mirrors world x; masks are in the seat's grid.
    desired_col, desired_row = (18000 - x) // 1000, y // 1000
    legal = np.flatnonzero(vector.masks[0, 0, 1 + slot * 576 : 1 + (slot + 1) * 576])
    require(len(legal) > 0, f"no legal tile for {card}")
    tile = min(
        legal,
        key=lambda t: (abs(int(t) % 18 - desired_col) + abs(int(t) // 18 - desired_row), int(t)),
    )
    return 1 + slot * 576 + int(tile)


def _reset_with_card(vector, observer_class, card, flag, preferred_seed):
    deck, flags = [card] + DECK_TAIL, [flag] + [0] * 7
    for seed in [preferred_seed] + list(range(47, 247)):
        vector.reset([seed], deck, DECK_TAIL + ["Giant"], flags, [0] * 8)
        observer = observer_class(vector)
        hand = vector.observations[0, 0, list(vector.schema.header["hand_card"])]
        if vector.card_id(card) in hand:
            return observer, seed
    raise AssertionError(f"no seeded opening hand found for {card}")


def _injected_negative_probes(vector, observer, original, deploy_index):
    """Fault injection after export: never part of positive trajectories."""
    from crforge_env import _jvm

    character = _jvm.java_class("org.crforge.core.battle.unit.CharacterEntity")
    world_entity = _jvm.java_class("org.crforge.core.battle.unit.WorldEntity")
    mark_clone = character.class_.getDeclaredMethod("markClone", world_entity.class_)
    mark_clone.setAccessible(True)
    world = observer.standard.getWorld()
    cases = [
        ("clone", original.getData(), deploy_index, 0, True),
        ("wrong_deploy_provenance", original.getData(), 999999, 0, False),
        ("unplayed_provenance", original.getData(), -1, 0, False),
        (
            "nonchampion_source_provenance",
            world.getRecords().unit("ArcherQueen"),
            deploy_index,
            0,
            False,
        ),
        ("no_ability", world.getRecords().unit("Knight"), deploy_index, 0, False),
    ]
    result = []
    for index, (kind, data, provenance, seat, clone) in enumerate(cases):
        name = f"negative_probe_{index}"
        unit = character(world, data, name, seat, 5000 + index * 500, 5000, 11)
        unit.setDeployIndex(provenance)
        if clone:
            mark_clone.invoke(unit, original)
        world.getHolder().add(unit)
        _step(vector, observer)
        tried = readonly_try(observer, seat, name)
        require(not tried["given"], f"invalid injected {kind} unexpectedly passed")
        require(
            tried["refusal"] in {"NO_CHAMPION", "NOT_CHAMPION", "CLONE"},
            f"invalid {kind} was rejected by an unrelated gate: {tried}",
        )
        require(
            readonly_try(observer, 0, str(original.name()))["given"],
            f"negative {kind} changed the legitimate champion's availability",
        )
        result.append(
            {
                "case": kind,
                "synthetic_failure_injection": True,
                "name": name,
                "alive": _alive(unit),
                "clone": bool(unit.isClone()),
                "deploy_index": int(unit.getDeployIndex()),
                "try": tried,
            }
        )
    # A dead original stays addressable by its saved name, but cannot be tapped.
    original.getHitPoints().setHitPoints(0)
    _step(vector, observer)
    tried = readonly_try(observer, 0, str(original.name()))
    require(not _alive(original) and not tried["given"], "dead champion accepted")
    result.append({"case": "dead", "synthetic_failure_injection": True, "try": tried})
    return result


def verify_case(vector, observer_class, args, card, flag, seed):
    observer, seed = _reset_with_card(vector, observer_class, card, flag, seed)
    while int(vector.info(0)["tick"]) < 90:
        _step(vector, observer)
    submission_tick = int(vector.info(0)["tick"])
    outcomes = _step(vector, observer, _select_play(vector, card))["new_results"]
    selected = None
    while int(vector.info(0)["tick"]) < 800:
        for slot in observer.ability_slots(0):
            if slot["card"] != card or slot["state"] != 2:
                continue
            controller = observer.standard.getWorld().kingTower(0).championSlot(slot["slot"] + 1)
            copies = [unit for unit in controller.champions() if _alive(unit)]
            if copies:
                selected = (slot, copies[0], len(copies))
                break
        if selected:
            break
        outcomes.extend(_step(vector, observer)["new_results"])
    require(selected is not None, f"no alive READY {card} champion found")
    slot, unit, copies = selected
    name = str(unit.name())
    tried = readonly_try(observer, 0, name)
    probe = {
        "tick": int(vector.info(0)["tick"]),
        "state": slot["state"],
        "alive_copies": copies,
        "available": slot["available"],
        "try_given": tried["given"],
        "try_refusal": tried["refusal"],
        "native_unit_name": slot["native_unit_name"],
        "actual_unit_name": name,
        "play_units_count": [len(play.units()) for play in observer.standard.getPlays()],
    }
    validate_lookup_probe(probe, expect=args.expect if card == "BarbLog" else "fixed")
    if card == "BarbLog":
        require(sum(probe["play_units_count"]) == 0, "fixture no longer exercises a late spawn")
    record = {
        "card": card,
        "hero_flag": flag,
        "seed": seed,
        "submission_tick": submission_tick,
        "slot": slot,
        "lookup_probe": probe,
        "read_only_try": tried,
    }
    if args.expect == "broken":
        return record
    require(slot["native_unit_name"] == name, "mask resolves a different unit")
    for _ in range(3):
        require(readonly_try(observer, 0, name) == tried, "readonly try changed without a tick")
    enemy = readonly_try(observer, 1, name)
    require(not enemy["given"], "enemy unit accepted")
    record["enemy_probe"] = enemy
    charges_before = slot["charges"]
    polled = _step(vector, observer, slot["action"])
    outcomes.extend(polled["new_results"])
    pending_try = readonly_try(observer, 0, name)
    require(
        bool(vector.masks[0, 0, slot["action"]]) == pending_try["given"],
        "pending command mask disagrees with native tryAbility",
    )
    record["pending_command_probe"] = pending_try
    while not any(row["kind"] == "ability" for row in outcomes):
        require(int(vector.info(0)["tick"]) < 1000, "ability never produced a native result")
        outcomes.extend(_step(vector, observer)["new_results"])
    uses = [row for row in outcomes if row["kind"] == "ability"]
    require(len(uses) == 1, "ability ran more than once")
    validate_ability_result(uses[0], tried["cost"])
    after = observer.ability_slots(0)[slot["slot"]]
    after_try = readonly_try(observer, 0, name)
    require(not after_try["given"], "cast/cooldown ability was immediately reusable")
    require(
        bool(vector.masks[0, 0, slot["action"]]) == after_try["given"],
        "cast/cooldown mask disagrees with native tryAbility",
    )
    record["after_run_probe"] = after_try
    expected_charges = charges_before - 1 if charges_before > 0 else charges_before
    require(after["charges"] == expected_charges, "ability charge count not spent exactly once")
    record.update(
        native_ability_result=uses[0],
        charges_before=charges_before,
        charges_after=after["charges"],
        native_ability_uses=len(observer.standard.getAbilityUses()),
    )
    (args.out / f"{card}.live_receipt.json").write_text(
        json.dumps(
            {"scope": "Native execution only; replay export not yet verified", **record}, indent=2
        ),
        encoding="utf-8",
    )
    replay_path = args.out / f"{card}.replay.json"
    replay_path.write_text(vector.replay_json(0), encoding="utf-8")
    from crforge_env import label_replays

    labelled = label_replays(
        [replay_path], args.install, args.tables, decision_ticks=5, check=True, mask_version=2
    )
    require(not labelled.unreadable and len(labelled.replays) == 1, "exported replay unreadable")
    replay = labelled.replays[0]
    validate_labelled_replay(replay)
    record["export_replay"] = {
        "path": str(replay_path),
        "sha256": sha256(replay_path),
        "counts": replay.counts,
        "drive": replay.drive,
    }
    if card == "BarbLog" and not args.skip_injected_negatives:
        # Keep the successful replay pristine, and test bad provenance while a
        # fresh legitimate champion still has an unused, affordable charge.
        observer, _ = _reset_with_card(vector, observer_class, card, flag, seed)
        while int(vector.info(0)["tick"]) < 90:
            _step(vector, observer)
        _step(vector, observer, _select_play(vector, card))
        negative_unit = None
        while int(vector.info(0)["tick"]) < 800:
            for current in observer.ability_slots(0):
                if current["card"] == card and current["available"]:
                    controller = (
                        observer.standard.getWorld().kingTower(0).championSlot(current["slot"] + 1)
                    )
                    copies = [
                        candidate for candidate in controller.champions() if _alive(candidate)
                    ]
                    if copies:
                        negative_unit = copies[0]
                        break
            if negative_unit is not None:
                break
            _step(vector, observer)
        require(negative_unit is not None, "negative fixture has no legitimate live champion")
        record["negative_probes"] = _injected_negative_probes(
            vector, observer, negative_unit, int(negative_unit.getDeployIndex())
        )
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, default=DEFAULT_INSTALL)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect", choices=("broken", "fixed"), default="fixed")
    parser.add_argument(
        "--cards",
        nargs="+",
        choices=("BarbLog", "ArcherQueen", "Goblinstein"),
        help="Limit diagnostic cases; omit for the complete fixed-build verification",
    )
    parser.add_argument("--skip-injected-negatives", action="store_true")
    args = parser.parse_args(argv)
    args.install, args.tables = (
        args.install.expanduser().resolve(),
        args.tables.expanduser().resolve(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    java_home = Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
    if (java_home / "lib/server/libjvm.dylib").exists():
        os.environ["JAVA_HOME"] = str(java_home)
    from crforge_env import LadderConfig, VectorBattle, _jvm

    from cr_rl.data.native_replay import NativeReplayObserver

    receipt = {
        "expect": args.expect,
        "install": str(args.install),
        "tables": str(args.tables),
        "jars": {jar.name: sha256(jar) for jar in sorted((args.install / "lib").glob("*.jar"))},
        "protocol": (
            "Legal native card/ability actions + exported replay re-drive; "
            "isolated negative failure injection happens after export."
        ),
        "cases": [],
        "passed": False,
    }
    try:
        with VectorBattle(
            1,
            1,
            install=args.install,
            tables=args.tables,
            config=LadderConfig(
                decision_ticks=5, level=11, tower_level=11, mask_version=2, arena="Arena_L"
            ),
        ) as vector:
            vector.set_autoreset(False)
            standard = _jvm.java_class("org.crforge.core.battle.unit.Standard1v1Battle")
            receipt["patch_marker"] = (
                str(standard.commandLookupPatch())
                if hasattr(standard, "commandLookupPatch")
                else None
            )
            if args.expect == "fixed":
                require(receipt["patch_marker"] == PATCH_MARKER, "wrong or unpatched core jar")
            cases = [("BarbLog", 2, 20261209)]
            if args.expect == "fixed":
                cases += [("ArcherQueen", 0, 20261110), ("Goblinstein", 0, 20261110)]
            if args.cards:
                cases = [case for case in cases if case[0] in args.cards]
            require(bool(cases), "no verification cases selected")
            receipt["selected_cards"] = [case[0] for case in cases]
            for card, flag, seed in cases:
                receipt["cases"].append(
                    verify_case(vector, NativeReplayObserver, args, card, flag, seed)
                )
                print(json.dumps({"case": card, "passed": True}), flush=True)
        receipt["passed"] = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "receipt": str(args.out / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
