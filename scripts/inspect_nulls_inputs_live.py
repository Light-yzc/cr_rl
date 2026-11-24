"""Bounded read-only live input audit. No model loading or touch commands."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

from cr_rl.client.diagnostics import model_input_quality
from cr_rl.client.nulls_reader import (NullsMemoryReader,NotInBattleError,resolve_pid_and_libg_base,
                                      VT_CHARACTER,VT_KING_TOWER,VT_PROJECTILE,VT_AREA_EFFECT)
from cr_rl.client.persistent_memory import PersistentAdbMemReader,UnmappedGamePageError
from cr_rl.client.schema_adapter import NullsToSchemaV4Adapter
from cr_rl.client.schema_v4 import SchemaV4 as S
from cr_rl.client.runtime import prepare_private_device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial',default='127.0.0.1:16416')
    p.add_argument('--adb',default='G:/sdk/platform-tools/adb.exe')
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not 0<a.seconds<=60:p.error('Read-only audit is bounded to 60 seconds')
    a.out.mkdir(parents=True,exist_ok=True)
    prepare_private_device(a.serial,adb=a.adb)
    pid,base=resolve_pid_and_libg_base(a.serial,adb=a.adb)
    mem=PersistentAdbMemReader(a.serial,pid,adb=a.adb)
    reader=NullsMemoryReader(mem,base)
    adapter=NullsToSchemaV4Adapter(Path.home()/'.crforge/tables/16.635.3')
    report={'pid':pid,'frames':0,'nonbattle_polls':0,'page_errors':0,
            'unknown_vtables':{},'unknown_unit_fields':Counter(),
            'multi_buff_count_mismatches':0,'raw_entity_kinds':Counter()}
    start=time.monotonic();last_tick=None;latencies=[]
    try:
        with gzip.open(a.out/'snapshots.jsonl.gz','wt',encoding='utf-8') as stream:
            while time.monotonic()-start<a.seconds:
                tick_start=time.monotonic()
                try:snap=reader.read_snapshot()
                except NotInBattleError:report['nonbattle_polls']+=1;time.sleep(.4);continue
                except UnmappedGamePageError:report['page_errors']+=1;time.sleep(.1);continue
                latency=1000*(time.monotonic()-tick_start)
                if snap.tick==last_tick:time.sleep(.05);continue
                last_tick=snap.tick;latencies.append(latency)
                adapted=adapter.adapt(snap)
                quality=model_input_quality(adapted.obs)
                report['frames']+=1
                report['unknown_unit_fields'].update(quality['unknown_unit_fields'])
                for ent in snap.entities:
                    report['raw_entity_kinds'][hex(ent.vt)]+=1
                    if ent.vt not in (VT_CHARACTER,VT_KING_TOWER,VT_PROJECTILE,VT_AREA_EFFECT):
                        report['unknown_vtables'].setdefault(hex(ent.vt),{
                            'name':ent.name,'eid':ent.eid,'scid':ent.scid,
                            'assigned_kind':adapter._entity_kind(ent,ent.name)})
                    if ent.buffs is not None and len(ent.buffs)>3:
                        rows=adapted.obs[S.HEADER_INTS:].reshape(S.MAX_ENTITIES,S.ENTITY_INTS)[:adapted.obs[S.H_ENTITY_COUNT]]
                        own_id=adapter._seat_id_of(ent.eid)
                        matched=[r for r in rows if r[S.E_ID]==own_id]
                        if matched and matched[0][S.E_BUFF_COUNT]!=len(ent.buffs):
                            report['multi_buff_count_mismatches']+=1
                stream.write(json.dumps({'snapshot':snap.to_dict(),'obs':adapted.obs.tolist(),
                                         'legal_actions':adapted.legal_mask.nonzero()[0].tolist(),
                                         'input_quality':quality,'read_ms':latency})+'\n')
                stream.flush()
                time.sleep(max(0.,.4-(time.monotonic()-tick_start)))
    finally:mem.close()
    report['read_ms_mean']=sum(latencies)/len(latencies) if latencies else None
    report['read_ms_max']=max(latencies,default=None)
    (a.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
