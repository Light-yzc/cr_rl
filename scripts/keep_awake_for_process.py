"""Temporary wake guard for an already-running Windows collector process."""
import argparse
import ctypes
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cr_rl.client.runtime import keep_system_awake


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',type=int,required=True)
    args=parser.parse_args()
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=(ctypes.c_void_p,ctypes.c_ulong)
    kernel.CloseHandle.argtypes=(ctypes.c_void_p,)
    handle=kernel.OpenProcess(0x00100000,False,args.pid)
    if not handle:return
    try:
        with keep_system_awake() as enabled:
            print(json.dumps(dict(wake_guard=enabled,collector_pid=args.pid)),flush=True)
            while kernel.WaitForSingleObject(handle,30000)==0x102:pass
    finally:kernel.CloseHandle(handle)


if __name__=='__main__':main()
