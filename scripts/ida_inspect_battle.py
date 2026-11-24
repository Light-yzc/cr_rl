"""IDAPython batch script 4: decompile Summoner hand/card/side/clock/entity getters."""

import traceback

import ida_auto
import ida_funcs
import ida_hexrays
import ida_name
import ida_pro
import idautils
import idc

TARGETS = [
    ("PickAllySummoner", 0xD9D6FC),
    ("PickEnemySummoner", 0xD9DC4C),
    ("BattleGetSideSummoner", 0xE14514),
    ("BattleGetSummonerByTeam", 0xE16768),
    ("BattleGetTeamIndex", 0xE169AC),
    ("SummonerHandCount", 0x109A73C),
    ("SummonerGetHandCard", 0x109B28C),
    ("SummonerGetCardCostOrLvl", 0x109AE6C),
    ("SummonerCardsPlayed", 0x109B1EC),
    ("EntityGetTeam", 0x108AF7C),
    ("CharacterGetSpell", 0xF6573C),
    ("HUD_BattleClock_9273f4", 0x9273F4),
    ("HUD_ElixirCardSlot_938488", 0x938488),
    ("HUD_HandSlotRender_938cf8", 0x938CF8),
    ("CardSlotDescriptor_f671e4", 0xF671E4),
    ("BattleIsOver_e10848", 0xE10848),
    ("BattleTowersHP_e1961c", 0xE1961C),
    ("BattleTowersHP_e19754", 0xE19754),
    ("BuildingIsKing_ec1ba8", 0xEC1BA8),
    ("BuildingIsPrincess_ec1bb0", 0xEC1BB0),
    ("LogicDataGetSCID_ec4", 0xEC1858),
    ("TimelineClock_e10aa0", 0xE10AA0),
    ("EntityBuffsCheck", 0xECEA00),
]


def force_func(ea: int):
    f = ida_funcs.get_func(ea)
    if f is None:
        ida_auto.plan_and_wait(ea, ea + 0x1400)
        f = ida_funcs.get_func(ea)
        if f is None:
            ida_funcs.add_func(ea)
            ida_auto.plan_and_wait(ea, ea + 0x1400)
            f = ida_funcs.get_func(ea)
    return f


def main():
    out_path = r"G:\Code\cr_rl\artifacts\nulls_profile_166351\ida_battle_decompile4.txt"
    has_hx = ida_hexrays.init_hexrays_plugin()
    lines = [f"HexRays available: {has_hx}\n"]
    seen = set()

    for label, ea in TARGETS:
        try:
            f = force_func(ea)
            if f is None:
                lines.append(f"\n=== {label} @ 0x{ea:x}: NO FUNCTION ===\n")
                continue
            if f.start_ea in seen:
                lines.append(f"\n=== {label} @ 0x{ea:x}: (already dumped 0x{f.start_ea:x}) ===\n")
                continue
            seen.add(f.start_ea)
            name = ida_name.get_name(f.start_ea) or f"sub_{f.start_ea:x}"
            lines.append(
                f"\n{'=' * 80}\n"
                f"[{label}] {name} @ 0x{f.start_ea:x}..0x{f.end_ea:x} (size=0x{f.end_ea - f.start_ea:x})\n"
            )
            if has_hx and (f.end_ea - f.start_ea) <= 0x6000:
                try:
                    cfunc = ida_hexrays.decompile(f.start_ea)
                    if cfunc:
                        text = str(cfunc)
                        if len(text) > 12000:
                            text = text[:12000] + "\n... [truncated] ..."
                        lines.append("--- DECOMPILED C ---\n" + text + "\n")
                        continue
                except Exception as exc:
                    lines.append(f"[decompile failed: {exc}]\n")
            for head in list(idautils.Heads(f.start_ea, f.end_ea))[:80]:
                lines.append(f"  0x{head:x}: {idc.generate_disasm_line(head, 0)}\n")
        except Exception:
            lines.append(f"\n=== ERROR on {label} @ 0x{ea:x} ===\n{traceback.format_exc()}\n")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(r"G:\Code\cr_rl\artifacts\nulls_profile_166351\ida_battle_error4.txt", "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
    finally:
        ida_pro.qexit(0)
