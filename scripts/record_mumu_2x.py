"""Lightweight, ultra-compact 2x-speed MuMu emulator recorder for Clash Royale highlights.

Supports two zero-lag capture backends (`--mode auto` selects `window` on interactive
Windows desktop and falls back seamlessly to `adb` framebuffer recording):
  1. `window`: Win32 `PrintWindow(PW_RENDERFULLCONTENT)` on `MuMuNxDevice` / `MuMuPlayer`
     (captures only the emulator window even if partially covered by other apps),
     downscaled to 360x640 and piped directly to FFmpeg H.264 (`CRF 30`) at 2.0x speed.
  2. `adb`: Direct Android `screenrecord` MP4 segments inside MuMu (preserves exact VFR
     wall-clock timestamps, zero window borders/cursors, works even when minimized or
     launched from a background terminal), finalized via FFmpeg at 2.0x speed (`CRF 30`).

Typical output size: ~2.0 - 3.5 MB for a full 3-minute battle (90s video at 2x speed).

Usage:
    python scripts/record_mumu_2x.py
    python scripts/record_mumu_2x.py --width 360 --height 640 --speed 2.0 --crf 30
    python scripts/record_mumu_2x.py --mode adb --seconds 200
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


DEFAULT_ADB = r"G:/sdk/platform-tools/adb.exe"
DEFAULT_SERIAL = "127.0.0.1:16416"
MUMU_EXES = ("mumunxdevice.exe", "mumuplayer.exe", "nemuplayer.exe")


def _find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise FileNotFoundError("ffmpeg not found on PATH or imageio_ffmpeg")


def _find_mumu_hwnd() -> tuple[int, str, tuple[int, int, int, int]] | None:
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

    # Map PID -> exe name via CreateToolhelp32Snapshot (no extra privileges required)
    TH32CS_SNAPPROCESS = 0x00000002

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    pid_to_exe: dict[int, str] = {}
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap and snap != ctypes.c_void_p(-1).value:
        pe = PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if kernel32.Process32FirstW(snap, ctypes.byref(pe)):
            while True:
                pid_to_exe[int(pe.th32ProcessID)] = pe.szExeFile.lower()
                if not kernel32.Process32NextW(snap, ctypes.byref(pe)):
                    break
        kernel32.CloseHandle(snap)

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    candidates: list[tuple[int, int, str, tuple[int, int, int, int]]] = []

    @WNDENUMPROC
    def _enum_cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w < 240 or h < 400:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        tbuf = ctypes.create_unicode_buffer(length + 1)
        if length > 0:
            user32.GetWindowTextW(hwnd, tbuf, length + 1)
        title = tbuf.value
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = pid_to_exe.get(int(pid.value), "")
        is_device = exe == "mumunxdevice.exe"
        is_mumu = exe in MUMU_EXES or ("mumu" in title.lower()) or ("模拟器" in title)
        if is_mumu and h > w:
            prio = 2 if is_device else 1
            candidates.append((prio, w * h, int(hwnd), title or exe, (rect.left, rect.top, w, h)))
        return True

    user32.EnumWindows(_enum_cb, 0)
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, hwnd, label, rect = candidates[0]
    return hwnd, label, rect


def _capture_hwnd_bgr(hwnd: int):
    import numpy as np

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 16 or h <= 16:
        return None

    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        return None
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    old = gdi32.SelectObject(mem_dc, bmp)
    # PW_RENDERFULLCONTENT (2) captures DirectX/OpenGL/Angle swapchains cleanly on Win10/11
    ok = user32.PrintWindow(hwnd, mem_dc, 2)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    gdi32.SelectObject(mem_dc, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)
    if not ok:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    return arr


def record_window_mode(args, ffmpeg_bin: str, out_path: Path, hwnd: int, title: str) -> bool:
    import cv2

    first = _capture_hwnd_bgr(hwnd)
    if first is None or float(first.std()) < 2.0:
        return False

    out_w = int(args.width) - (int(args.width) % 2)
    out_h = int(args.height) - (int(args.height) % 2)
    out_fps = float(args.fps)
    cap_fps = max(4.0, out_fps / float(args.speed))

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        f"{out_fps:.2f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    stop_flag = False

    def _on_sig(_sig, _frm):
        nonlocal stop_flag
        stop_flag = True

    old_int = signal.signal(signal.SIGINT, _on_sig)
    old_term = signal.signal(signal.SIGTERM, _on_sig)
    print(
        f"[window] Capturing MuMu window '{title}' (HWND={hwnd}) -> {out_w}x{out_h} @ {args.speed:.1f}x speed (CRF={args.crf})",
        flush=True,
    )
    print("[window] Recording... Press Ctrl+C anytime to stop and save MP4!", flush=True)

    t0 = time.perf_counter()
    frames = 0
    last_frame = cv2.resize(first, (out_w, out_h), interpolation=cv2.INTER_AREA)
    try:
        while not stop_flag:
            elapsed = time.perf_counter() - t0
            if args.seconds > 0 and elapsed >= args.seconds:
                break
            raw = _capture_hwnd_bgr(hwnd)
            if raw is not None and float(raw.std()) >= 1.5:
                last_frame = cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_AREA)
            assert proc.stdin is not None
            proc.stdin.write(last_frame.tobytes())
            frames += 1
            if frames % int(max(1, round(cap_fps * 5))) == 0:
                print(
                    f"  recorded {elapsed:5.1f}s live -> {frames / out_fps:5.1f}s video ({args.speed:.1f}x)",
                    flush=True,
                )
            target_t = t0 + frames / cap_fps
            sleep_s = target_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.wait(timeout=30)
    return out_path.is_file() and out_path.stat().st_size > 1024


def record_adb_mode(args, ffmpeg_bin: str, out_path: Path) -> bool:
    out_w = int(args.width) - (int(args.width) % 2)
    out_h = int(args.height) - (int(args.height) % 2)
    adb = str(args.adb)
    serial = str(args.serial)

    subprocess.run([adb, "connect", serial], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    subprocess.run(
        [adb, "-s", serial, "shell", "rm -f /sdcard/crrl_rec_seg_*.mp4"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )

    stop_flag = False

    def _on_sig(_sig, _frm):
        nonlocal stop_flag
        stop_flag = True

    old_int = signal.signal(signal.SIGINT, _on_sig)
    old_term = signal.signal(signal.SIGTERM, _on_sig)

    print(
        f"[adb] Capturing MuMu ({serial}) framebuffer -> {out_w}x{out_h} @ {args.speed:.1f}x speed (CRF={args.crf})",
        flush=True,
    )
    print("[adb] Works even if MuMu is minimized or covered! Press Ctrl+C anytime to stop & save MP4.", flush=True)

    remote_segs: list[str] = []
    t0 = time.perf_counter()
    seg_idx = 0
    bitrate = max(400_000, int(args.adb_bitrate))

    try:
        while not stop_flag:
            rem_total = (args.seconds - (time.perf_counter() - t0)) if args.seconds > 0 else 175.0
            if args.seconds > 0 and rem_total <= 0.4:
                break
            seg_limit = max(1, min(175, int(round(rem_total))))
            rpath = f"/sdcard/crrl_rec_seg_{seg_idx:03d}.mp4"
            seg_idx += 1
            proc = subprocess.Popen(
                [
                    adb,
                    "-s",
                    serial,
                    "shell",
                    f"screenrecord --size {out_w}x{out_h} --bit-rate {bitrate} --time-limit {seg_limit} {rpath}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            remote_segs.append(rpath)
            last_log = time.perf_counter()
            while proc.poll() is None:
                if stop_flag or (args.seconds > 0 and (time.perf_counter() - t0) >= args.seconds):
                    stop_flag = True
                    subprocess.run(
                        [adb, "-s", serial, "shell", "pkill -INT screenrecord"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                    break
                if time.perf_counter() - last_log >= 5.0:
                    el = time.perf_counter() - t0
                    print(f"  recorded {el:5.1f}s live -> ~{el / args.speed:5.1f}s @ {args.speed:.1f}x...", flush=True)
                    last_log = time.perf_counter()
                time.sleep(0.2)
            # Give Android MPEG4Writer a brief moment to flush moov atom
            time.sleep(0.35)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)

    if not remote_segs:
        return False

    print("[adb] Finalizing 2x-speed H.264 MP4...", flush=True)
    with tempfile.TemporaryDirectory(prefix="crrl_mumu_vid_") as tmpd:
        tmp_dir = Path(tmpd)
        local_segs: list[Path] = []
        for i, rpath in enumerate(remote_segs):
            lpath = tmp_dir / f"seg_{i:03d}.mp4"
            subprocess.run(
                [adb, "-s", serial, "pull", rpath, str(lpath)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
            if lpath.is_file() and lpath.stat().st_size > 1024:
                local_segs.append(lpath)
        subprocess.run(
            [adb, "-s", serial, "shell", "rm -f /sdcard/crrl_rec_seg_*.mp4"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if not local_segs:
            return False

        vf = f"setpts=PTS/{float(args.speed):.4f},fps={int(args.fps)},scale={out_w}:{out_h}:flags=area"
        if len(local_segs) == 1:
            ff_cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(local_segs[0]),
                "-an",
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(args.crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        else:
            lst = tmp_dir / "concat.txt"
            lst.write_text(
                "\n".join(f"file '{p.as_posix()}'" for p in local_segs) + "\n",
                encoding="utf-8",
            )
            ff_cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-an",
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(args.crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        subprocess.run(ff_cmd, check=True, timeout=120)

    return out_path.is_file() and out_path.stat().st_size > 1024


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["auto", "window", "adb"], default="auto", help="Capture backend (default: auto)")
    p.add_argument("--width", type=int, default=360, help="Output width in pixels (default: 360)")
    p.add_argument("--height", type=int, default=640, help="Output height in pixels (default: 640)")
    p.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier (default: 2.0x)")
    p.add_argument("--fps", type=int, default=30, help="Output video FPS after speedup (default: 30)")
    p.add_argument("--crf", type=int, default=30, help="H.264 CRF quality/size factor; higher=smaller (28..33, default: 30)")
    p.add_argument("--seconds", type=float, default=0.0, help="Max live seconds to record (0 = until Ctrl+C)")
    p.add_argument("--out", type=Path, default=None, help="Output MP4 path (default: artifacts/videos/mumu_2x_<ts>.mp4)")
    p.add_argument("--adb", default=DEFAULT_ADB, help="Path to adb executable")
    p.add_argument("--serial", default=DEFAULT_SERIAL, help="MuMu ADB serial (default: 127.0.0.1:16416)")
    p.add_argument("--adb-bitrate", type=int, default=1_200_000, help="Intermediate ADB screenrecord bitrate")
    return p.parse_args(argv)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args(argv)
    ffmpeg_bin = _find_ffmpeg()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cr_mumu_2x_{ts}.mp4"
    else:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = False
    if args.mode in ("auto", "window"):
        win_info = _find_mumu_hwnd()
        if win_info is not None:
            hwnd, title, _rect = win_info
            ok = record_window_mode(args, ffmpeg_bin, out_path, hwnd, title)
        elif args.mode == "window":
            print("ERROR: Could not find a visible MuMu window on the current desktop.", file=sys.stderr)
            return 1
        else:
            print("[auto] Desktop HWND not accessible in this session; switching to direct MuMu ADB capture.", flush=True)

    if not ok and args.mode in ("auto", "adb"):
        ok = record_adb_mode(args, ffmpeg_bin, out_path)

    if not ok or not out_path.is_file():
        print("ERROR: Recording failed to produce a valid MP4.", file=sys.stderr)
        return 1

    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSAVED_2X_VIDEO: {out_path} ({mb:.2f} MiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
