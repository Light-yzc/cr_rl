"""Bounded read-only live pointer-chain & multi-frame probe for nullsroyale.rel.free (<= 4 MiB)."""

from __future__ import annotations

import argparse
import struct
import subprocess
import time

# 0x4000000 max hard ceiling in project guidelines; keep default <= 4 MiB per run.
HARD_CAP_BYTES = 4 * 1024 * 1024


def adb_output(device: str, *args: str) -> bytes:
    return subprocess.check_output(["adb", "-s", device, *args])


def get_pid_and_libg_base(device: str, pkg: str) -> tuple[str, int]:
    pid = adb_output(device, "shell", f"pidof {pkg}").decode().strip().split()[0]
    maps = adb_output(device, "shell", f"grep libg.so /proc/{pid}/maps").decode()
    base = int(maps.splitlines()[0].split("-")[0], 16)
    return pid, base


def is_ptr(val: int) -> bool:
    low = val & 0x00FFFFFFFFFFFFFF
    return 0x10000000 < low < 0x7FFFFFFFFFFF


class BoundedMemReader:
    def __init__(self, device: str, pid: str, max_bytes: int = 3 * 1024 * 1024) -> None:
        if max_bytes > HARD_CAP_BYTES:
            raise ValueError(f"max_bytes {max_bytes} exceeds cap {HARD_CAP_BYTES}")
        self.device = device
        self.pid = pid
        self.max_bytes = max_bytes
        self.total_read = 0
        self.page_cache: dict[int, bytes] = {}

    def clear_cache(self) -> None:
        self.page_cache.clear()

    def prefetch_pages(self, addrs: list[int]) -> None:
        pages = sorted({(a & 0x00FFFFFFFFFFFFFF) & ~0xFFF for a in addrs if is_ptr(a) and ((a & 0x00FFFFFFFFFFFFFF) & ~0xFFF) not in self.page_cache})
        if not pages:
            return
        # Batch up to 32 pages per single adb exec-out call
        for off in range(0, len(pages), 32):
            batch = pages[off : off + 32]
            need = len(batch) * 4096
            if self.total_read + need > self.max_bytes:
                raise RuntimeError(f"Read budget exceeded: {self.total_read + need} > {self.max_bytes}")
            self.total_read += need
            cmds = ";".join(f"dd if=/proc/{self.pid}/mem bs=4096 skip={p // 4096} count=1 2>/dev/null" for p in batch)
            raw = adb_output(self.device, "exec-out", f"sh -c '{cmds}'")
            if len(raw) != len(batch) * 4096:
                continue
            for i, p in enumerate(batch):
                self.page_cache[p] = raw[i * 4096 : (i + 1) * 4096]

    def _read_pages(self, start_page: int, count: int) -> bytes:
        # Check cache first for single-page reads
        if count == 1 and start_page in self.page_cache:
            return self.page_cache[start_page]
        need = count * 4096
        if self.total_read + need > self.max_bytes:
            raise RuntimeError(f"Read budget exceeded: {self.total_read + need} > {self.max_bytes}")
        self.total_read += need
        raw = adb_output(
            self.device,
            "exec-out",
            f"dd if=/proc/{self.pid}/mem bs=4096 skip={start_page // 4096} count={count} 2>/dev/null",
        )
        for i in range(count):
            chunk = raw[i * 4096 : (i + 1) * 4096]
            if len(chunk) == 4096:
                self.page_cache[start_page + i * 4096] = chunk
        return raw

    def read(self, addr: int, size: int) -> bytes:
        addr &= 0x00FFFFFFFFFFFFFF
        if addr < 0x10000 or size <= 0:
            return b""
        page = addr & ~0xFFF
        off = addr - page
        pages = (off + size + 0xFFF) // 0x1000
        if all((page + i * 4096) in self.page_cache for i in range(pages)):
            raw = b"".join(self.page_cache[page + i * 4096] for i in range(pages))
        else:
            raw = self._read_pages(page, pages)
        return raw[off : off + size]

    def u64(self, addr: int) -> int:
        b = self.read(addr, 8)
        return struct.unpack("<Q", b)[0] if len(b) == 8 else 0

    def i32(self, addr: int) -> int:
        b = self.read(addr, 4)
        return struct.unpack("<i", b)[0] if len(b) == 4 else 0

    def read_cstr(self, addr: int, max_len: int = 48) -> str:
        b = self.read(addr, max_len)
        if not b:
            return ""
        end = b.find(b"\x00")
        raw = b[:end] if end != -1 else b
        return raw.decode("ascii", errors="ignore")

    def read_scstr(self, buf: bytes, offset: int) -> str:
        """Read Supercell SSO string or raw char* at struct offset."""
        if offset + 16 > len(buf):
            return ""
        slen, cap = struct.unpack_from("<ii", buf, offset)
        if 0 < slen < 64 and 0 < cap <= 4096:
            if cap >= 8:
                ptr = struct.unpack_from("<Q", buf, offset + 8)[0]
                if is_ptr(ptr):
                    txt = self.read_cstr(ptr, min(64, slen + 1))
                    if txt and all(32 <= ord(c) < 127 for c in txt):
                        return f"SSO({txt})"
            else:
                raw = buf[offset + 8 : offset + 8 + slen]
                txt = raw.decode("ascii", errors="ignore")
                if txt and all(32 <= ord(c) < 127 for c in txt):
                    return f"SSOi({txt})"
        ptr = struct.unpack_from("<Q", buf, offset)[0]
        if is_ptr(ptr):
            txt = self.read_cstr(ptr, 48)
            if len(txt) >= 3 and all(32 <= ord(c) < 127 for c in txt):
                return f"*({txt})"
        return ""


def dump_logic_data(mem: BoundedMemReader, base: int, ptr: int) -> str:
    if not is_ptr(ptr):
        return f"0x{ptr:x}"
    b = mem.read(ptr, 0x190)
    if len(b) < 0x50:
        return f"0x{ptr:x} (short)"
    vt = struct.unpack_from("<Q", b, 0)[0]
    scid = struct.unpack_from("<i", b, 0x40)[0]
    cost = struct.unpack_from("<i", b, 0x148)[0] if len(b) >= 0x14C else -1
    strs = []
    for off in (0x10, 0x18, 0x20, 0x28, 0x30, 0x38):
        s = mem.read_scstr(b, off)
        if s:
            strs.append(f"+0x{off:x}:{s}")
    return f"vt=0x{vt-base:x} scid@0x40={scid} cost@0x148={cost} strs={strs[:2]}"


def inspect_once(mem: BoundedMemReader, base: int, detailed: bool = True) -> dict[str, object]:
    st = mem.u64(base + 0x1B7D210)
    if not is_ptr(st):
        return {"mode": -1, "st": st}
    st_b = mem.read(st, 0x60)
    mode = struct.unpack_from("<i", st_b, 56)[0]
    gm = struct.unpack_from("<Q", st_b, 0x28)[0]
    local_acc_obj = mem.u64(base + 0x1B728B8)
    local_acc = None
    if is_ptr(local_acc_obj):
        ab = mem.read(local_acc_obj + 0x28, 8)
        if len(ab) == 8:
            local_acc = struct.unpack("<ii", ab)

    out: dict[str, object] = {
        "mode": mode,
        "st": st,
        "gm": gm,
        "local_acc": local_acc,
    }
    if mode not in (3, 4) or not is_ptr(gm):
        return out

    ctx = mem.u64(gm + 0x90)
    if not is_ptr(ctx):
        return out
    ctx_b = mem.read(ctx, 0x180)
    v7 = struct.unpack_from("<Q", ctx_b, 64)[0]
    lb = struct.unpack_from("<Q", ctx_b, 168)[0]
    out["ctx"] = ctx
    out["lb"] = lb

    # Timeline & summoners
    if is_ptr(lb):
        lb_b = mem.read(lb, 0x220)
        p190 = struct.unpack_from("<Q", lb_b, 0x190)[0]
        if is_ptr(p190):
            b190 = mem.read(p190, 0x90)
            tick = struct.unpack_from("<i", b190, 0x10)[0]
            time_ms = struct.unpack_from("<i", b190, 0x14)[0]
            sec_el, sec_len = struct.unpack_from("<ii", b190, 0x24)
            rate_el, rate_len = struct.unpack_from("<ii", b190, 0x38)
            out["clock"] = (tick, time_ms, sec_el, sec_len, rate_el, rate_len)
            if detailed:
                t_ints = [(f"0x{o:x}", struct.unpack_from("<i", b190, o)[0]) for o in range(8, 0x90, 4) if struct.unpack_from("<i", b190, o)[0] != 0]
                out["clock_all"] = t_ints

        summoners = []
        s0, s1 = struct.unpack_from("<QQ", lb_b, 0xE0)
        d0, d1 = struct.unpack_from("<QQ", lb_b, 0x88)
        for side, s in enumerate((s0, s1)):
            if not is_ptr(s):
                continue
            sb = mem.read(s, 0x3C0)
            s_acc = struct.unpack_from("<ii", sb, 0x20)
            elix = struct.unpack_from("<i", sb, 760)[0]
            leaked = struct.unpack_from("<i", sb, 836)[0]
            played = struct.unpack_from("<i", sb, 840)[0]
            slots_648 = struct.unpack_from("<8i", sb, 648)
            hp_idx = struct.unpack_from("<Q", sb, 528)[0]
            qp_idx = struct.unpack_from("<Q", sb, 544)[0]
            hand = struct.unpack("<4i", mem.read(hp_idx, 16)) if is_ptr(hp_idx) else ()
            queue = struct.unpack("<8i", mem.read(qp_idx, 32)) if is_ptr(qp_idx) else ()
            summoners.append({
                "side": side,
                "is_local": local_acc == s_acc,
                "acc": s_acc,
                "elixir": elix,
                "leaked": leaked,
                "played": played,
                "hand": hand,
                "queue": queue,
                "s648": slots_648,
            })
        out["summoners"] = summoners

        if detailed:
            for d_idx, dptr in enumerate((d0, d1)):
                if not is_ptr(dptr):
                    continue
                db = mem.read(dptr, 0x80)
                cards_arr = struct.unpack_from("<Q", db, 0x20)[0]
                if is_ptr(cards_arr):
                    cptrs = struct.unpack("<8Q", mem.read(cards_arr, 64))
                    cards_info = []
                    for cidx, cp in enumerate(cptrs):
                        if not is_ptr(cp):
                            continue
                        cb = mem.read(cp, 0x80)
                        cvt = struct.unpack_from("<Q", cb, 0)[0]
                        cand_ptrs = []
                        cand_ints = []
                        for o in range(8, 0x70, 8):
                            val = struct.unpack_from("<Q", cb, o)[0]
                            if is_ptr(val):
                                cand_ptrs.append((f"+0x{o:x}", dump_logic_data(mem, base, val)))
                        for o in range(8, 0x70, 4):
                            iv = struct.unpack_from("<i", cb, o)[0]
                            if 0 <= iv <= 300 or 26000000 <= iv <= 29000000:
                                cand_ints.append((f"+0x{o:x}", iv))
                        cards_info.append((cidx, f"vt=0x{cvt-base:x}", cand_ints[:8], cand_ptrs[:3]))
                    out[f"deck_{d_idx}"] = cards_info

    # Entities
    ents = []
    if is_ptr(v7):
        vb = mem.read(v7, 0x20)
        arr = struct.unpack_from("<Q", vb, 8)[0]
        cap, cnt = struct.unpack_from("<ii", vb, 16)
        out["ent_cap_cnt"] = (cap, cnt)
        if is_ptr(arr) and 0 < cnt < 256:
            ptrs = struct.unpack(f"<{cnt}Q", mem.read(arr, cnt * 8))
            mem.prefetch_pages([p for p in ptrs if is_ptr(p)])
            for idx, p in enumerate(ptrs):
                if not is_ptr(p):
                    continue
                eb = mem.read(p, 0x160)
                if len(eb) < 0x128:
                    continue
                evt = struct.unpack_from("<Q", eb, 0)[0]
                eid = struct.unpack_from("<i", eb, 8)[0]
                sub18 = struct.unpack_from("<Q", eb, 0x18)[0]
                kind_b = eb[0x30]
                data_p = struct.unpack_from("<Q", eb, 0x48)[0]
                side = struct.unpack_from("<i", eb, 0x78)[0]
                x, y = struct.unpack_from("<ii", eb, 0x7C)
                z = struct.unpack_from("<i", eb, 0x8C)[0]
                lvl = struct.unpack_from("<i", eb, 0x120)[0]
                hp, mhp, shield, hp_extra = -1, -1, -1, ()
                if is_ptr(sub18):
                    s18 = mem.read(sub18, 0x20)
                    p10 = struct.unpack_from("<Q", s18, 0x10)[0] if len(s18) >= 0x18 else 0
                    if is_ptr(p10):
                        p10b = mem.read(p10, 0x38)
                        if len(p10b) >= 0x28:
                            hp, mhp = struct.unpack_from("<ii", p10b, 0x10)
                            hp_extra = struct.unpack_from("<6i", p10b, 0x18)
                dinfo = dump_logic_data(mem, base, data_p) if detailed else f"scid={mem.i32(data_p + 0x40) if is_ptr(data_p) else -1}"
                ents.append({
                    "i": idx,
                    "vt": hex(evt - base),
                    "id": eid,
                    "k30": kind_b,
                    "side": side,
                    "pos": (x, y, z),
                    "lvl": lvl,
                    "hp": (hp, mhp),
                    "hp_tail": hp_extra if detailed else (),
                    "data": dinfo,
                })
    out["ents"] = ents
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="127.0.0.1:16416")
    parser.add_argument("--pkg", default="nullsroyale.rel.free")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--interval-ms", type=int, default=400)
    args = parser.parse_args()

    pid, base = get_pid_and_libg_base(args.device, args.pkg)
    mem = BoundedMemReader(args.device, pid, max_bytes=3 * 1024 * 1024)
    for f_idx in range(args.frames):
        mem.clear_cache()
        t0 = time.perf_counter()
        snap = inspect_once(mem, base, detailed=(f_idx == 0))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"=== Frame {f_idx} dt={dt_ms:.1f}ms read={mem.total_read/1024:.1f}KiB mode={snap.get('mode')} acc={snap.get('local_acc')} ===")
        if "clock" in snap:
            print("  clock(tick,ms,sec_el,sec_len,rate_el,rate_len):", snap["clock"])
        if "clock_all" in snap:
            print("  clock_nonzero:", snap["clock_all"])
        for sm in snap.get("summoners", []):
            print(f"  Summoner[{sm['side']}] local={sm['is_local']} acc={sm['acc']} elix={sm['elixir']} leaked={sm['leaked']} played={sm['played']} hand={sm['hand']} q={sm['queue']} s648={sm['s648']}")
        for dk in ("deck_0", "deck_1"):
            if dk in snap:
                print(f"  {dk}:")
                for row in snap[dk]:
                    print("   ", row)
        print(f"  ents {snap.get('ent_cap_cnt')}:")
        for e in snap.get("ents", []):
            print(f"    [{e['i']}] id={e['id']} vt={e['vt']} k30={e['k30']} side={e['side']} pos={e['pos']} lvl={e['lvl']} hp={e['hp']} tail={e['hp_tail']} {e['data']}")
        if f_idx + 1 < args.frames:
            time.sleep(max(0.01, args.interval_ms / 1000.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
