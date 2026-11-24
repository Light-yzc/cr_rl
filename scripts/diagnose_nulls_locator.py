"""Bounded read-only test of an existing locator against the active client.

This is a compatibility diagnostic, not a verified game observation source.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adb',type=Path,default=Path('G:/sdk/platform-tools/adb.exe'))
    p.add_argument('--serial',required=True)
    p.add_argument('--max-mib',type=int,default=32)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not 1 <= a.max_mib <= 64:
        p.error('Use a bounded 1..64 MiB diagnostic')
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'.cache/direct_state_research/cr-memory-reader/x86_64/deploy'))
    from adb_runtime import AdbConfig,AdbRuntime
    from proc_memory import RootProcessMemory,BattleStateLocator,AdbError
    adb=[str(a.adb),'-s',a.serial]
    pid=int(subprocess.check_output(adb+['shell','pidof','nullsroyale.rel.free'],timeout=10).strip())
    class BoundedMemory(RootProcessMemory):
        read_bytes=0
        def read_many(self,requests,*,timeout=20):
            size=sum(n for _,n in requests)
            if self.read_bytes+size > a.max_mib*1024**2:
                raise AdbError('Read budget exceeded; refusing full-process memory dump')
            self.read_bytes+=size
            return super().read_many(requests,timeout=min(timeout,20))
    memory=BoundedMemory(AdbRuntime(AdbConfig(a.adb.resolve(),a.serial)),pid)
    locator=BattleStateLocator(memory)
    result={'pid':pid,'profile':'upstream-x86_64-heuristic','verified':False,'unix':time.time()}
    try:
        pointers=locator.locate()
        result['locator_timing']=locator.last_timing
        result['candidate_found']=pointers is not None
        if pointers:
            result['pointers']=dataclasses.asdict(pointers)
            result['samples']=[]
            for _ in range(3):
                result['samples'].append(dataclasses.asdict(locator.poll(pointers)))
                time.sleep(.4)
    except Exception as error:
        result['error']=str(error)
    result['read_bytes']=memory.read_bytes
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
