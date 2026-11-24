"""Measure read-only snapshot latency and optionally harmless corner tap pairs."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from cr_rl.client.nulls_reader import NullsMemoryReader,NotInBattleError,resolve_pid_and_libg_base
from cr_rl.client.persistent_memory import PersistentAdbMemReader,UnmappedGamePageError
from cr_rl.client.persistent_touch import PersistentAdbTouch


def summary(rows):
    return {'samples':len(rows),'p50':float(np.median(rows)),
            'p95':float(np.quantile(rows,.95)),'max':max(rows)} if rows else {'samples':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial',default='127.0.0.1:16416')
    p.add_argument('--adb',default='G:/sdk/platform-tools/adb.exe')
    p.add_argument('--samples',type=int,default=40)
    p.add_argument('--corner-taps',action='store_true',help='Compare 8 tap pairs at screen corner (0,0)')
    a=p.parse_args();report={}
    if a.corner_taps:
        persistent=PersistentAdbTouch(a.serial,adb=a.adb)
        try:
            fast=[];shell=[]
            for _ in range(8):
                start=time.perf_counter();persistent(((0,0),(0,0)))
                fast.append(1000*(time.perf_counter()-start))
                start=time.perf_counter()
                subprocess.run([a.adb,'-s',a.serial,'shell','input tap 0 0 && input tap 0 0'],check=True,timeout=5)
                shell.append(1000*(time.perf_counter()-start))
            report['persistent_taps_ms']=summary(fast);report['shell_taps_ms']=summary(shell)
        finally:persistent.close()
    pid,base=resolve_pid_and_libg_base(a.serial,adb=a.adb)
    memory=PersistentAdbMemReader(a.serial,pid,adb=a.adb)
    readers={mode:NullsMemoryReader(memory,base,cache_metadata=enabled)
             for mode,enabled in [('uncached',False),('cached',True)]}
    rows={mode:[] for mode in readers};failures=0;lobby=0
    try:
        for i in range(a.samples):
            # Reverse the order on alternating iterations to limit ordering bias.
            for mode in list(readers)[::1 if i%2==0 else -1]:
                start=time.perf_counter()
                try:snap=readers[mode].read_snapshot()
                except NotInBattleError:lobby+=1;continue
                except UnmappedGamePageError:failures+=1;continue
                if i>1:rows[mode].append({'ms':1000*(time.perf_counter()-start),
                                        'bytes':snap.bytes_read,'exchanges':memory.poll_exchanges})
            time.sleep(.05)
    finally:memory.close()
    report['memory']={mode:{field:summary([r[field] for r in values]) for field in ('ms','bytes','exchanges')}
                      for mode,values in rows.items()}
    report.update(page_failures=failures,nonbattle_polls=lobby)
    out=Path('artifacts/nulls_live/io_benchmark.json')
    out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
