"""Convert local Null's Royale live recordings into CrForge-verified dual-seat ReplayStart shards.

Scans `artifacts/nulls_live` (including nested batch directories) for sessions with
`diagnostics/frames_*.jsonl.gz`, reconstructs both seats' exact opening 8-card
deal (`hand + queue`), decks, Evo/Hero slot flags, and chronological plays (using
controller decisions for own seat when available and Null's memory `played_count`
+ `aim_x/aim_y`/spawn coordinates for the human opponent), replays every match on
CrForge's 8-tick grid (`VectorBattle`, `set_hand_order`), and writes a standard
`crrl-bc-shards-v1` dataset (`live_loss` "真机错题本" + `live_win`) for `ReplayStartPool`.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / ".cache/vendor/crforge/python") not in sys.path:
    sys.path.insert(0, str(ROOT / ".cache/vendor/crforge/python"))


def _ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME") and Path(os.environ["JAVA_HOME"]).exists():
        return
    candidates = [
        Path(r"G:\Android studio\jbr"),
        Path(r"C:\Program Files\Eclipse Adoptium\jdk-21"),
        Path(r"C:\Program Files\Java\jdk-21"),
    ]
    for c in candidates:
        if (c / "bin/java.exe").is_file() or (c / "bin/java").is_file():
            os.environ["JAVA_HOME"] = str(c)
            return


def _discover_live_runs(live_root: Path) -> list[Path]:
    found: list[Path] = []
    for diag in sorted(live_root.rglob("diagnostics")):
        if not diag.is_dir():
            continue
        if any(diag.glob("frames_*.jsonl.gz")):
            found.append(diag.parent)
    return sorted(set(found))


def _infer_outcome(run_dir: Path, frames: list[dict], own_seat: int) -> str:
    res_p = run_dir / "result.json"
    if res_p.is_file():
        try:
            o = json.loads(res_p.read_text(encoding="utf-8")).get("outcome")
            if o in ("win", "loss", "draw"):
                return str(o)
        except Exception:
            pass
    ev_p = run_dir / "events.jsonl"
    if ev_p.is_file():
        for ln in reversed(ev_p.read_text(encoding="utf-8", errors="replace").splitlines()):
            if '"result_recorded"' in ln:
                try:
                    o = json.loads(ln).get("outcome")
                    if o in ("win", "loss", "draw"):
                        return str(o)
                except Exception:
                    pass
    term_p = run_dir / "terminal.json"
    obs_end = None
    if term_p.is_file():
        try:
            obs_end = json.loads(term_p.read_text(encoding="utf-8")).get("obs")
        except Exception:
            pass
    if obs_end is None and frames:
        obs_end = frames[-1].get("obs")
    if isinstance(obs_end, list) and len(obs_end) >= 48:
        from cr_rl.client.schema_v4 import SchemaV4 as S

        c_own, c_ene = int(obs_end[S.H_CROWNS]), int(obs_end[S.H_CROWNS_ENEMY])
        if c_own > c_ene:
            return "win"
        if c_own < c_ene:
            return "loss"
    return "unknown"


def _extract_match_spec(run_dir: Path, adapter, vocab_set: set[str]):
    from cr_rl.client.schema_adapter import VT_AREA_EFFECT, VT_CHARACTER, VT_KING_TOWER, VT_PROJECTILE
    from cr_rl.client.schema_v4 import SchemaV4 as S

    frames: list[dict] = []
    for fp in sorted((run_dir / "diagnostics").glob("frames_*.jsonl.gz")):
        try:
            with gzip.open(fp, "rt", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        frames.append(json.loads(ln))
        except (OSError, EOFError, json.JSONDecodeError):
            break
    if len(frames) < 45:
        return None, "too_few_frames"

    snap0 = frames[0].get("snapshot") or {}
    t0 = int(snap0.get("tick", 999))
    own_seat = snap0.get("own_seat")
    if not (0 <= t0 <= 40) or own_seat not in (0, 1):
        return None, "not_started_at_opening"
    sums0 = snap0.get("summoners") or []
    if len(sums0) != 2:
        return None, "missing_summoners"

    decks: list[list[str]] = []
    forms: list[list[int]] = []
    orders: list[list[int]] = []
    for s in (0, 1):
        sm = sums0[s]
        raw_deck = sm.get("deck") or []
        if len(raw_deck) != 8:
            return None, "incomplete_deck"
        d_names: list[str] = []
        d_forms: list[int] = []
        for c in raw_deck:
            nm = adapter.normalize_name(c.get("name"))
            if not nm or nm not in vocab_set:
                return None, f"unsupported_card:{c.get('name')}"
            d_names.append(nm)
            sf = int(c.get("slot_flags") or 0) & (S.SLOT_EVOLUTION | S.SLOT_HERO)
            if (c.get("evolution_count") or 0) > 0:
                sf |= S.SLOT_EVOLUTION
            if sf & S.SLOT_HERO:
                d_forms.append(S.FORM_HERO)
            elif sf & S.SLOT_EVOLUTION:
                d_forms.append(S.FORM_EVOLVED)
            else:
                d_forms.append(S.FORM_PLAIN)
        if len(set(d_names)) != 8:
            return None, "duplicate_deck_cards"
        hand4 = [int(x) for x in (sm.get("hand") or [])[:4]]
        queue4 = [int(x) for x in (sm.get("queue") or [])[:4]]
        order8 = hand4 + queue4
        if sorted(order8) != list(range(8)):
            return None, "invalid_opening_deal"
        decks.append(d_names)
        forms.append(d_forms)
        orders.append(order8)

    # Pre-index newly spawned entities across the timeline (and resolve tunnel landing points)
    first_seen_idx: dict[int, int] = {}
    ent_records: dict[int, list[tuple[int, dict]]] = {}
    for idx, fr in enumerate(frames):
        sn = fr.get("snapshot") or {}
        tk = int(sn.get("tick", 0))
        for ed in sn.get("entities") or ():
            eid = ed.get("eid")
            if not isinstance(eid, int) or eid <= 0:
                continue
            if eid not in first_seen_idx:
                first_seen_idx[eid] = idx
            ent_records.setdefault(eid, []).append((tk, ed))

    def _ent_kind(ed: dict, nname: str | None) -> int | None:
        vt = ed.get("vt")
        if vt == VT_PROJECTILE:
            return S.KIND_PROJECTILE
        if vt == VT_AREA_EFFECT:
            return S.KIND_AREA
        if vt == VT_KING_TOWER:
            return S.KIND_TOWER
        if vt != VT_CHARACTER:
            return None
        if nname in ("KingTower", "PrincessTower"):
            return S.KIND_TOWER
        unit = adapter.catalog.units.get(nname or "")
        if unit is not None and (unit.king or unit.summoner_tower):
            return S.KIND_TOWER
        return S.KIND_CHARACTER

    spawn_events_by_seat: dict[int, list[tuple[int, str, int, int]]] = {0: [], 1: []}
    for eid, f_idx in first_seen_idx.items():
        hist = ent_records[eid]
        tk0, ed0 = hist[0]
        if f_idx == 0 and tk0 <= t0:
            # Initial towers at frame 0
            n0 = adapter.normalize_name(ed0.get("name"))
            if _ent_kind(ed0, n0) == S.KIND_TOWER:
                continue
        if ed0.get("is_clone"):
            continue
        side = int(ed0.get("side", -1))
        if side not in (0, 1):
            continue
        nname = adapter.normalize_name(ed0.get("name"))
        kind = _ent_kind(ed0, nname)
        if kind is None or kind == S.KIND_TOWER:
            continue
        base_c, cast_c, _form = adapter._resolve_enemy_play_card(None, kind, nname, decks[side], set())
        if not base_c:
            continue
        p_kind = adapter._card_play_kind.get(cast_c or base_c, adapter._card_play_kind.get(base_c, S.PLAY_TROOP))
        aim_x, aim_y = ed0.get("aim_x"), ed0.get("aim_y")
        if p_kind == S.PLAY_LAUNCHED and kind == S.KIND_PROJECTILE and aim_x is not None and aim_y is not None:
            bx, by = int(aim_x), int(aim_y)
        elif p_kind == S.PLAY_TUNNEL:
            bx, by = int(ed0.get("x", 9000)), int(ed0.get("y", 16000))
            for tk_h, ed_h in hist:
                if tk_h - tk0 > 90:
                    break
                bx, by = int(ed_h.get("x", bx)), int(ed_h.get("y", by))
                if not ed_h.get("is_underground") and ed_h.get("state") != 6:
                    break
        else:
            bx, by = int(ed0.get("x", 9000)), int(ed0.get("y", 16000))
        spawn_events_by_seat[side].append((tk0, base_c, bx, by))

    # Index own-seat controller play decisions for exact tap coordinates
    own_decisions: list[tuple[int, str, int, int]] = []
    for fr in frames:
        act = int(fr.get("action") or 0)
        if 1 <= act < S.FIRST_ABILITY:
            dec = S.decode_action(act)
            slot = int(dec["slot"])
            hand_names = fr.get("hand") or []
            if 0 <= slot < len(hand_names):
                cname = adapter.normalize_name(hand_names[slot])
                if cname:
                    own_decisions.append((int(fr.get("tick", 0)), cname, int(dec["col"]), int(dec["row"])))

    # Walk played_count transitions per seat to build clean ordered play queues
    plays: dict[int, list[dict]] = {0: [], 1: []}
    used_own_dec: set[int] = set()
    used_spawns: dict[int, set[int]] = {0: set(), 1: set()}

    prev_pc = [int(sums0[0].get("played_count", 0)), int(sums0[1].get("played_count", 0))]
    prev_hand = [list(sums0[0].get("hand", [])[:4]), list(sums0[1].get("hand", [])[:4])]

    for fr in frames[1:]:
        sn = fr.get("snapshot") or {}
        tk = int(sn.get("tick", 0))
        sums = sn.get("summoners") or []
        if len(sums) != 2:
            continue
        for seat in (0, 1):
            sm = sums[seat]
            cur_pc = int(sm.get("played_count", prev_pc[seat]))
            cur_hand = list(sm.get("hand", [])[:4])
            if cur_pc > prev_pc[seat] and len(cur_hand) == 4 and len(prev_hand[seat]) == 4:
                rotated = [s for s in range(4) if prev_hand[seat][s] != cur_hand[s]]
                for s in rotated[: max(1, cur_pc - prev_pc[seat])]:
                    d_idx = int(prev_hand[seat][s])
                    if not (0 <= d_idx < 8):
                        continue
                    cname = decks[seat][d_idx]
                    col = row = None
                    submit_tick = max(0, tk - 20)

                    # 1) Own seat: match exact controller decision within [tk - 55, tk + 4]
                    if seat == own_seat:
                        for d_i, (dtk, dcard, dcol, drow) in enumerate(own_decisions):
                            if d_i not in used_own_dec and dcard == cname and (tk - 55) <= dtk <= (tk + 4):
                                used_own_dec.add(d_i)
                                col, row = dcol, drow
                                submit_tick = max(0, dtk)
                                break

                    # 2) Spawned entities cluster within [tk - 16, tk + 28]
                    if col is None:
                        matched_xy: list[tuple[int, int]] = []
                        for sp_i, (stk, scard, sbx, sby) in enumerate(spawn_events_by_seat[seat]):
                            if sp_i not in used_spawns[seat] and scard == cname and (tk - 16) <= stk <= (tk + 28):
                                used_spawns[seat].add(sp_i)
                                matched_xy.append((sbx, sby))
                        if matched_xy:
                            bx = int(sum(p[0] for p in matched_xy) / len(matched_xy))
                            by = int(sum(p[1] for p in matched_xy) / len(matched_xy))
                            sx = S.to_seat_x(seat, bx)
                            sy = S.to_seat_y(seat, by)
                            col = max(0, min(17, sx // 1000))
                            row = max(0, min(31, sy // 1000))

                    # 3) Fallback target when an instant spell / 1-tick spirit died between 8-tick reads:
                    #    NEVER drop a confirmed hand rotation, otherwise the seat's 8-card cycle jams!
                    is_fallback = False
                    if col is None:
                        is_fallback = True
                        opp_ents = [
                            ed for ed in (sn.get("entities") or ())
                            if isinstance(ed.get("eid"), int)
                            and ed.get("eid", 0) > 0
                            and int(ed.get("side", -1)) == (1 - seat)
                            and (ed.get("hp") or 0) > 0
                        ]
                        p_kind = adapter._card_play_kind.get(cname, S.PLAY_TROOP)
                        if p_kind != S.PLAY_TROOP and opp_ents:
                            bx = int(sum(int(e.get("x", 9000)) for e in opp_ents) / len(opp_ents))
                            by = int(sum(int(e.get("y", 16000)) for e in opp_ents) / len(opp_ents))
                            col = max(0, min(17, S.to_seat_x(seat, bx) // 1000))
                            row = max(0, min(31, S.to_seat_y(seat, by) // 1000))
                        elif p_kind != S.PLAY_TROOP:
                            col, row = 3, 24  # Enemy left princess tower in seat frame
                        else:
                            col, row = 8, 9

                    plays[seat].append({
                        "submit_tick": int(submit_tick),
                        "card": cname,
                        "col": int(col),
                        "row": int(row),
                        "fallback": is_fallback,
                    })
            prev_pc[seat] = cur_pc
            prev_hand[seat] = cur_hand

    total_plays = len(plays[0]) + len(plays[1])
    if total_plays < 4:
        return None, "too_few_plays"

    end_tick = int((frames[-1].get("snapshot") or {}).get("tick", 0))
    outcome = _infer_outcome(run_dir, frames, int(own_seat))
    return {
        "session_id": run_dir.name,
        "own_seat": int(own_seat),
        "outcome": outcome,
        "end_tick": end_tick,
        "decks": decks,
        "forms": forms,
        "orders": orders,
        "plays": plays,
    }, None


def _replay_in_crforge(vector, spec: dict, *, max_delay_ticks: int = 120):
    from cr_rl.data.replay_repair import nearest_legal_action

    schema = vector.schema
    d0, d1 = spec["decks"]
    f0, f1 = list(spec["forms"][0]), list(spec["forms"][1])
    o0, o1 = spec["orders"]

    # Reset lane 0 with exact Evo/Hero forms (falling back to plain if a form is unsupported)
    try:
        vector.reset([1], [d0], [d1], [f0], [f1])
    except Exception:
        f0 = [0] * 8
        f1 = [0] * 8
        vector.reset([1], [d0], [d1], [f0], [f1])
    vector.set_hand_order(0, o0, o1)

    q0 = deque(spec["plays"][0])
    q1 = deque(spec["plays"][1])
    total_src = len(q0) + len(q1)
    executed = 0
    omitted = 0

    hand_idx = list(schema.header["hand_card"])
    pend_idx = list(schema.header["hand_pending"])
    max_steps = min(750, max(60, spec["end_tick"] // 8 + 4))

    obs_s0: list[np.ndarray] = []
    obs_s1: list[np.ndarray] = []
    acts_s0: list[int] = []
    acts_s1: list[int] = []
    ticks_list: list[int] = []

    for step in range(max_steps):
        if vector.dones[0] or vector.fault(0):
            break
        cur_tick = step * 8
        chosen = [0, 0]
        for seat, q in ((0, q0), (1, q1)):
            while q and cur_tick > q[0]["submit_tick"] + max_delay_ticks:
                q.popleft()
                omitted += 1
            if not q or cur_tick + 4 < q[0]["submit_tick"]:
                continue
            obs_row = vector.observations[0, seat]
            if np.any(obs_row[pend_idx] > 0):
                continue
            ev = q[0]
            cid = vector.card_id(ev["card"])
            hand_cards = [int(x) for x in obs_row[hand_idx]]
            if cid not in hand_cards:
                continue
            slot = hand_cards.index(cid)
            req_act = int(schema.play(slot, ev["col"], ev["row"]))
            mask_row = vector.masks[0, seat]
            act_use = None
            if mask_row[req_act]:
                act_use = req_act
            else:
                snapped = nearest_legal_action(req_act, mask_row, 3)
                if snapped is not None:
                    act_use = int(snapped[0])
                elif ev.get("fallback"):
                    leg_slot = np.flatnonzero(mask_row[1 + slot * 576 : 1 + (slot + 1) * 576])
                    if len(leg_slot) > 0:
                        act_use = int(1 + slot * 576 + int(leg_slot[len(leg_slot) // 2]))
            if act_use is not None:
                chosen[seat] = act_use
                q.popleft()
                executed += 1

        obs_s0.append(vector.observations[0, 0].copy())
        obs_s1.append(vector.observations[0, 1].copy())
        acts_s0.append(chosen[0])
        acts_s1.append(chosen[1])
        ticks_list.append(cur_tick)

        vector.step([[chosen[0], chosen[1]]])

    if vector.fault(0):
        return None, "crforge_fault"
    T = len(ticks_list)
    if T < 60:
        return None, "too_short_in_crforge"
    last_tk = ticks_list[-1]
    due_remaining = sum(1 for q in (q0, q1) for ev in q if ev["submit_tick"] <= last_tk - 24)
    due_total = executed + omitted + due_remaining
    exec_ratio = executed / max(1, due_total)
    if executed < 4 or exec_ratio < 0.78:
        return None, f"low_exec_ratio:{executed}/{due_total}"

    obs_arr = np.concatenate([np.stack(obs_s0), np.stack(obs_s1)], axis=0).astype(np.int32)
    acts_arr = np.asarray(acts_s0 + acts_s1, dtype=np.int32)
    tks_arr = np.asarray(ticks_list + ticks_list, dtype=np.int32)
    seats_arr = np.concatenate([np.zeros(T, dtype=np.int32), np.ones(T, dtype=np.int32)])
    return {
        "observations": obs_arr,
        "actions": acts_arr,
        "ticks": tks_arr,
        "seats": seats_arr,
        "steps": T,
        "executed_plays": executed,
        "source_plays": total_src,
        "omitted_plays": omitted,
        "forms": [f0, f1],
    }, None


def build_live_dataset(
    live_root: Path,
    out_dir: Path,
    *,
    install_dir: Path,
    tables_dir: Path,
    zip_out: Path | None = None,
) -> dict:
    _ensure_java_home()
    from crforge_env import LadderConfig, VectorBattle
    from cr_rl.client.schema_adapter import NullsToSchemaV4Adapter

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)

    adapter = NullsToSchemaV4Adapter(tables_dir)
    runs = _discover_live_runs(live_root)
    rejected = Counter()
    replays_meta: list[dict] = []
    shards_manifest: list[dict] = []
    assignments: dict[str, str] = {}
    seen_sigs: set[str] = set()

    cfg = LadderConfig(decision_ticks=8, level=11, tower_level=11, mask_version=2, arena="Arena_L")
    with VectorBattle(1, 1, install=str(install_dir), tables=str(tables_dir), config=cfg) as vector:
        vector.set_autoreset(False)
        native_meta = dict(vector.metadata)
        vocab_raw = native_meta["vocabulary"]
        vocab_names = vocab_raw["names"] if isinstance(vocab_raw, dict) else list(vocab_raw)
        vocab_set = set(vocab_names)
        schema_dict = {
            "header": {k: list(v) for k, v in vector.schema.header.items()},
            "seat_ints": int(vector.schema.seat_ints),
        }

        for run_dir in runs:
            spec, err = _extract_match_spec(run_dir, adapter, vocab_set)
            if spec is None:
                rejected[err or "extract_failed"] += 1
                continue
            replayed, r_err = _replay_in_crforge(vector, spec)
            if replayed is None:
                key = r_err.split(":")[0] if r_err else "replay_failed"
                rejected[key] += 1
                continue

            sig_src = json.dumps(
                [spec["decks"], spec["orders"], replayed["actions"].tolist()],
                separators=(",", ":"),
            ).encode("utf-8")
            sig = "live_" + hashlib.sha256(sig_src).hexdigest()[:20]
            if sig in seen_sigs:
                rejected["duplicate_signature"] += 1
                continue
            seen_sigs.add(sig)

            idx = len(shards_manifest)
            rel_path = f"shards/live_{idx:05d}.npz"
            np.savez_compressed(
                out_dir / rel_path,
                observations=replayed["observations"],
                actions=replayed["actions"],
                ticks=replayed["ticks"],
                seats=replayed["seats"],
            )

            is_loss = spec["outcome"] == "loss"
            cohort = "live_loss" if is_loss else "live_win"
            selected_seats = [spec["own_seat"]] if is_loss else [0, 1]

            replays_meta.append({
                "canonical_game_signature": sig,
                "session_id": spec["session_id"],
                "outcome": spec["outcome"],
                "own_seat": spec["own_seat"],
                "cohort": cohort,
                "decks": spec["decks"],
                "steps": replayed["steps"],
                "executed_plays": replayed["executed_plays"],
                "source_plays": replayed["source_plays"],
            })
            shards_manifest.append({
                "path": rel_path,
                "replay_offset": idx,
                "cohort": cohort,
                "selected_seats": selected_seats,
            })
            assignments[sig] = "train"

    snapshot_id = f"nulls_live_starts_{len(shards_manifest)}g"
    (out_dir / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "decision_ticks": 8,
                "schema": schema_dict,
                "vocabulary": {"names": vocab_names},
                "replays": replays_meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "dataset.manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "format": "crrl-bc-shards-v1",
                "metadata": "dataset_metadata.json",
                "decision_ticks": 8,
                "shards": shards_manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "split_registry.json").write_text(
        json.dumps({"snapshot_id": snapshot_id, "assignments": assignments}, indent=2) + "\n",
        encoding="utf-8",
    )

    cohort_counts = Counter(s["cohort"] for s in shards_manifest)
    late_count = sum(1 for r in replays_meta if r["steps"] >= 290)
    summary = {
        "snapshot_id": snapshot_id,
        "scanned_runs": len(runs),
        "accepted_games": len(shards_manifest),
        "cohorts": dict(cohort_counts),
        "late_games_ge290_steps": late_count,
        "rejected": dict(rejected),
        "out_dir": str(out_dir),
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if zip_out is not None:
        zip_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_z = zip_out.with_suffix(".zip.partial")
        with zipfile.ZipFile(tmp_z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(out_dir).as_posix())
        tmp_z.replace(zip_out)
        summary["zip_out"] = str(zip_out)
        summary["zip_bytes"] = int(zip_out.stat().st_size)

    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live-root", type=Path, default=ROOT / "artifacts/nulls_live")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/live_replay_starts_v1")
    p.add_argument("--zip-out", type=Path, default=ROOT / "artifacts/live_replay_starts_v1.zip")
    p.add_argument(
        "--install",
        type=Path,
        default=ROOT / ".cache/vendor/crforge/env/build/install/env",
    )
    p.add_argument(
        "--tables",
        type=Path,
        default=Path.home() / ".crforge/tables/16.635.3",
    )
    args = p.parse_args()
    summary = build_live_dataset(
        args.live_root,
        args.out,
        install_dir=args.install,
        tables_dir=args.tables,
        zip_out=args.zip_out,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
