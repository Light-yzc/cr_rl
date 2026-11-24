"""Offline static ELF/AArch64 survey for Null's Royale libg.so (read-only analysis)."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any


ELF_MAGIC = b"\x7fELF"
PT_LOAD = 1


def parse_elf64_segments_and_sections(data: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if data[:4] != ELF_MAGIC or data[4] != 2 or data[5] != 1:
        raise ValueError("Expected little-endian ELF64 binary")
    e_phoff, e_shoff = struct.unpack_from("<QQ", data, 32)
    e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHHHH", data, 54)

    segments: list[dict[str, Any]] = []
    for idx in range(e_phnum):
        off = e_phoff + idx * e_phentsize
        p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack_from(
            "<IIQQQQQQ", data, off
        )
        if p_type == PT_LOAD:
            segments.append(
                {
                    "index": idx,
                    "flags": p_flags,
                    "offset": p_offset,
                    "vaddr": p_vaddr,
                    "filesz": p_filesz,
                    "memsz": p_memsz,
                }
            )

    sections: list[dict[str, Any]] = []
    if e_shoff and e_shnum:
        shstr_off = e_shoff + e_shstrndx * e_shentsize
        _, _, _, _, shstr_file_off, shstr_size = struct.unpack_from("<IIQQQQ", data, shstr_off)[:6]
        shstr = data[shstr_file_off : shstr_file_off + shstr_size]
        for idx in range(e_shnum):
            off = e_shoff + idx * e_shentsize
            sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = (
                struct.unpack_from("<IIQQQQIIQQ", data, off)
            )
            end = shstr.find(b"\x00", sh_name)
            name = shstr[sh_name : end if end != -1 else sh_name].decode("latin1", errors="replace")
            sections.append(
                {
                    "index": idx,
                    "name": name,
                    "type": sh_type,
                    "flags": sh_flags,
                    "vaddr": sh_addr,
                    "offset": sh_offset,
                    "size": sh_size,
                    "entsize": sh_entsize,
                }
            )
    return segments, sections


def va_to_file_offset(segments: list[dict[str, Any]], va: int) -> int | None:
    for seg in segments:
        start = seg["vaddr"]
        end = start + seg["filesz"]
        if start <= va < end:
            return seg["offset"] + (va - start)
    return None


def file_offset_to_va(segments: list[dict[str, Any]], file_off: int) -> int | None:
    for seg in segments:
        start = seg["offset"]
        end = start + seg["filesz"]
        if start <= file_off < end:
            return seg["vaddr"] + (file_off - start)
    return None


def extract_cstrings_in_range(
    data: bytes, segments: list[dict[str, Any]], start_va: int, end_va: int, min_len: int = 4
) -> list[tuple[int, str]]:
    start_off = va_to_file_offset(segments, start_va)
    end_off = va_to_file_offset(segments, end_va - 1)
    if start_off is None or end_off is None:
        return []
    blob = data[start_off : end_off + 1]
    out: list[tuple[int, str]] = []
    for m in re.finditer(rb"[\x20-\x7e]{" + str(min_len).encode() + rb",}\x00", blob):
        s = m.group(0)[:-1].decode("ascii", errors="ignore")
        va = start_va + m.start()
        out.append((va, s))
    return out


def find_relocations_and_rtti(
    data: bytes, segments: list[dict[str, Any]], sections: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check whether RELA relocations exist to resolve typeinfo & vtable pointers statically."""
    rela_info: dict[int, int] = {}
    dynsym_sec = next((s for s in sections if s["name"] == ".dynsym"), None)
    dynstr_sec = next((s for s in sections if s["name"] == ".dynstr"), None)
    dynstr = b""
    if dynstr_sec:
        dynstr = data[dynstr_sec["offset"] : dynstr_sec["offset"] + dynstr_sec["size"]]

    exports: list[tuple[str, int]] = []
    if dynsym_sec:
        ent = dynsym_sec["entsize"] or 24
        for off in range(dynsym_sec["offset"], dynsym_sec["offset"] + dynsym_sec["size"], ent):
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", data, off)
            if st_value != 0 and dynstr:
                end = dynstr.find(b"\x00", st_name)
                name = dynstr[st_name : end if end != -1 else st_name].decode("latin1", errors="replace")
                if any(
                    kw in name
                    for kw in (
                        "Battle",
                        "Logic",
                        "Elixir",
                        "Hand",
                        "Card",
                        "Summon",
                        "Combat",
                        "Avatar",
                        "Character",
                        "Projectile",
                        "Building",
                        "HUD",
                        "Game",
                        "Manager",
                        "JNI",
                    )
                ):
                    exports.append((name, st_value))

    R_AARCH64_RELATIVE = 1027
    R_AARCH64_ABS64 = 257
    rel_count = 0
    for s in sections:
        if s["name"] in (".rela.dyn", ".rela.plt"):
            ent = s["entsize"] or 24
            for off in range(s["offset"], s["offset"] + s["size"], ent):
                r_offset, r_info, r_addend = struct.unpack_from("<QQq", data, off)
                r_type = r_info & 0xFFFFFFFF
                if r_type == R_AARCH64_RELATIVE:
                    rela_info[r_offset] = r_addend & 0xFFFFFFFFFFFFFFFF
                    rel_count += 1
                elif r_type == R_AARCH64_ABS64 and r_addend != 0:
                    rela_info[r_offset] = r_addend & 0xFFFFFFFFFFFFFFFF
                    rel_count += 1

    return {
        "reloc_map": rela_info,
        "reloc_count": rel_count,
        "interesting_exports": exports[:200],
        "total_interesting_exports": len(exports),
    }


def scan_adrp_xrefs(
    data: bytes, segments: list[dict[str, Any]], code_start: int, code_end: int, target_vas: set[int]
) -> dict[int, list[dict[str, int]]]:
    """Scan AArch64 ADRP + (ADD | LDR/STR offset) instructions for exact target VAs."""
    start_off = va_to_file_offset(segments, code_start)
    end_off = va_to_file_offset(segments, code_end - 4)
    if start_off is None or end_off is None:
        return {}
    code = memoryview(data)[start_off : end_off + 4]
    pages = {va & ~0xFFF for va in target_vas}
    hits: dict[int, list[dict[str, int]]] = {va: [] for va in target_vas}

    n_insns = len(code) // 4
    # Store recent ADRP per register (simple 12-instruction forward window)
    for i in range(n_insns):
        insn = struct.unpack_from("<I", code, i * 4)[0]
        # ADRP: op=1 (bit 31), bits[28:24] == 0b10000 -> mask 0x9F000000 == 0x90000000
        if (insn & 0x9F000000) != 0x90000000:
            continue
        rd = insn & 0x1F
        immlo = (insn >> 29) & 0x3
        immhi = (insn >> 5) & 0x7FFFF
        imm = (immhi << 2) | immlo
        if imm & (1 << 20):
            imm -= 1 << 21
        pc = code_start + i * 4
        page = (pc & ~0xFFF) + (imm << 12)
        if page not in pages:
            continue

        # Look ahead up to 8 instructions for ADD or LDR/STR using Rn == rd
        for j in range(1, min(9, n_insns - i)):
            nxt = struct.unpack_from("<I", code, (i + j) * 4)[0]
            # ADD (immediate, 64-bit): bits[31:22] == 0b1001000100 (0xFFC00000 mask -> 0x91000000)
            if (nxt & 0xFFC00000) == 0x91000000:
                rn = (nxt >> 5) & 0x1F
                if rn == rd:
                    sh = (nxt >> 22) & 1
                    imm12 = (nxt >> 10) & 0xFFF
                    off_val = imm12 << (12 if sh else 0)
                    target = page + off_val
                    if target in hits:
                        hits[target].append({"adrp_va": pc, "use_va": code_start + (i + j) * 4, "kind": 0})
                    break
            # LDR/STR unsigned imm 64-bit / 32-bit: (nxt & 0x3B000000) == 0x39000000
            if (nxt & 0x3B000000) == 0x39000000:
                rn = (nxt >> 5) & 0x1F
                if rn == rd:
                    size_shift = (nxt >> 30) & 0x3
                    imm12 = (nxt >> 10) & 0xFFF
                    target = page + (imm12 << size_shift)
                    if target in hits:
                        hits[target].append({"adrp_va": pc, "use_va": code_start + (i + j) * 4, "kind": 1})
                    break
            # Stop tracking if rd gets overwritten by another non-ADD instruction with Rd==rd
            if (nxt & 0x9F000000) == 0x90000000 and (nxt & 0x1F) == rd:
                break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", default="artifacts/nulls_profile_166351/libg.so")
    parser.add_argument("--out", default="artifacts/nulls_profile_166351/static_elf_survey.json")
    args = parser.parse_args()

    so_path = Path(args.so)
    data = so_path.read_bytes()
    segments, sections = parse_elf64_segments_and_sections(data)
    rel_info = find_relocations_and_rtti(data, segments, sections)
    reloc_map: dict[int, int] = rel_info["reloc_map"]

    # Identify .rodata and .text ranges
    rodata = next((s for s in sections if s["name"] == ".rodata"), None)
    text = next((s for s in sections if s["name"] == ".text"), None)
    ro_start = rodata["vaddr"] if rodata else 0
    ro_end = (rodata["vaddr"] + rodata["size"]) if rodata else segments[0]["vaddr"] + segments[0]["filesz"]
    text_start = text["vaddr"] if text else segments[0]["vaddr"]
    text_end = (text["vaddr"] + text["size"]) if text else segments[0]["vaddr"] + segments[0]["filesz"]

    cstrings = extract_cstrings_in_range(data, segments, ro_start, ro_end, min_len=5)

    # Filter interesting battle / RTTI strings
    rtti_matches: list[tuple[int, str]] = []
    battle_kw_re = re.compile(
        r"(Battle|Logic|Elixir|Summoner|Hand|Card|Deck|Tower|Character|Building|Projectile|AreaEffect|HUD_topRight|BattleState|Hitpoints|Shield|Clock|Overtime|Arena|Avatar|Deploy|Cooldown|Champion|Ability)",
        re.IGNORECASE,
    )
    keyword_matches: list[tuple[int, str]] = []
    for va, s in cstrings:
        if re.match(r"^(N?\d+[A-Z]|P?K?N?\d+Logic)", s) and battle_kw_re.search(s):
            rtti_matches.append((va, s))
        elif len(s) <= 80 and any(
            k in s
            for k in (
                "ElixirFullBarMS",
                "elixir_container",
                "HUD_topRight",
                "BattleState",
                "ElixirNotifyChange",
                "GenerateElixir",
                "SpendElixir",
                "ShieldHitpoints",
                "GainCombatElixirTiming",
                "BattleClock",
                "Overtime",
                "getCard",
                "HandSlot",
                "NextCard",
                "DeployCard",
                "Summon",
                "Combat",
                "Arena",
                "TowerHP",
                "CrownTower",
                "elixirRegen",
                "ElixirBarLeftNumbers",
            )
        ):
            keyword_matches.append((va, s))

    # Build reverse reloc lookup: target_va -> [slot_vas in data]
    rev_reloc: dict[int, list[int]] = {}
    for slot_va, target_va in reloc_map.items():
        rev_reloc.setdefault(target_va, []).append(slot_va)

    # If RTTI typeinfo pointers are present via R_AARCH64_RELATIVE:
    # typeinfo struct = [vtable_ptr, name_ptr, ...] -> name_ptr is at typeinfo + 8
    # vtable struct = [offset_to_top, typeinfo_ptr, func0, func1, ...] -> typeinfo_ptr is at vtable + 8
    rtti_classes: list[dict[str, Any]] = []
    for name_va, name_str in rtti_matches[:500]:
        slots_pointing_to_name = rev_reloc.get(name_va, [])
        for slot in slots_pointing_to_name:
            typeinfo_va = slot - 8
            # Also read base typeinfo if __si_class_type_info (+0x10)
            base_ti = reloc_map.get(typeinfo_va + 16)
            vtables_for_ti: list[dict[str, Any]] = []
            for ti_ref in rev_reloc.get(typeinfo_va, []):
                # Ensure offset_to_top == 0 (or small negative) and vtable starts at ti_ref - 8
                vtable_start = ti_ref - 8
                top_off_file = va_to_file_offset(segments, vtable_start)
                if top_off_file is not None:
                    offset_to_top = struct.unpack_from("<q", data, top_off_file)[0]
                    if -0x1000 <= offset_to_top <= 0:
                        vfuncs: list[int] = []
                        for k in range(8):
                            vf = reloc_map.get(vtable_start + 16 + k * 8)
                            if vf is None or not (segments[0]["vaddr"] <= vf < segments[0]["vaddr"] + segments[0]["filesz"]):
                                break
                            vfuncs.append(vf)
                        if vfuncs:
                            vtables_for_ti.append(
                                {
                                    "vtable_meta_va": vtable_start,
                                    "vtable_entries_va": vtable_start + 16,
                                    "offset_to_top": offset_to_top,
                                    "first_vfuncs": vfuncs,
                                }
                            )
            if vtables_for_ti or base_ti:
                rtti_classes.append(
                    {
                        "name": name_str,
                        "name_va": hex(name_va),
                        "typeinfo_va": hex(typeinfo_va),
                        "base_typeinfo_va": hex(base_ti) if base_ti else None,
                        "vtables": [
                            {
                                "meta_va": hex(v["vtable_meta_va"]),
                                "entries_va": hex(v["vtable_entries_va"]),
                                "offset_to_top": v["offset_to_top"],
                                "first_vfuncs": [hex(f) for f in v["first_vfuncs"]],
                            }
                            for v in vtables_for_ti
                        ],
                    }
                )

    # Scan ADRP+ADD xrefs in .text for keyword_matches
    target_vas = {va for va, _ in keyword_matches}
    xrefs = scan_adrp_xrefs(data, segments, text_start, text_end, target_vas)
    keyword_results = [
        {"va": hex(va), "text": s, "xrefs": [{k: hex(v) if k != "kind" else v for k, v in xr.items()} for xr in xrefs.get(va, [])]}
        for va, s in keyword_matches
    ]

    summary = {
        "segments": [{k: hex(v) if k in ("offset", "vaddr", "filesz", "memsz") else v for k, v in s.items()} for s in segments],
        "sections_subset": [
            {k: hex(v) if k in ("vaddr", "offset", "size", "flags") else v for k, v in s.items()}
            for s in sections
            if s["name"] in (".text", ".rodata", ".data.rel.ro", ".data", ".bss", ".got", ".rela.dyn", ".dynsym")
        ],
        "reloc_count": rel_info["reloc_count"],
        "total_rodata_cstrings": len(cstrings),
        "rtti_string_count": len(rtti_matches),
        "rtti_classes_with_vtables": rtti_classes[:250],
        "rtti_strings_sample": [{"va": hex(va), "text": s} for va, s in rtti_matches[:120]],
        "keyword_matches": keyword_results,
        "interesting_exports": [{"name": n, "va": hex(v)} for n, v in rel_info["interesting_exports"]],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "segments": len(segments),
                "reloc_count": rel_info["reloc_count"],
                "rtti_strings": len(rtti_matches),
                "rtti_classes_resolved": len(rtti_classes),
                "keyword_matches": len(keyword_results),
                "interesting_exports": rel_info["total_interesting_exports"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
