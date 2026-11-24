"""Bounded skill-interface probe on a private-server training battle, not policy data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from cr_rl.client.adb_actions import ActionObservation,AdbActionBridge,mumu_900x1600_calibration
from cr_rl.client.live_control import PendingPlay,PendingAbility,reconcile_play,reconcile_ability
from cr_rl.client.nulls_reader import NullsMemoryReader,resolve_pid_and_libg_base,NotInBattleError
from cr_rl.client.persistent_memory import PersistentAdbMemReader
from cr_rl.client.runtime import prepare_private_device
from cr_rl.client.schema_adapter import NullsToSchemaV4Adapter
from cr_rl.client.schema_v4 import SchemaV4 as S


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    serial='127.0.0.1:16416';adb='G:/sdk/platform-tools/adb.exe'
    prepare_private_device(serial,adb=adb)
    pid,base=resolve_pid_and_libg_base(serial,adb=adb)
    mem=PersistentAdbMemReader(serial,pid,adb=adb)
    reader=NullsMemoryReader(mem,base)
    adapter=NullsToSchemaV4Adapter(Path.home()/'.crforge/tables/16.635.3',allow_abilities=True)
    current=None;pending=None;sent=0;before=None
    bridge=AdbActionBridge(mumu_900x1600_calibration(serial,validated=True,
        ability_points=((820,1215),None)),adb=adb,state_source=lambda:current)
    log=(a.out/'events.jsonl').open('w',encoding='utf-8',buffering=1)
    def emit(kind,**data):
        row={'event':kind,'unix':time.time(),**data};log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    deadline=time.monotonic()+50
    try:
        while time.monotonic()<deadline:
            start=time.monotonic()
            try:snap=reader.read_snapshot()
            except NotInBattleError:time.sleep(.2);continue
            own=snap.summoners[snap.own_seat]
            adapted=adapter.adapt(snap)
            abilities=own.abilities or ()
            ab=next((b for b in abilities if b.slot==0),None)
            if ab is not None and ab.deck_idx<0:
                raise RuntimeError('Current deck has no slot-0 active skill; probe will not cycle ordinary cards')
            current=ActionObservation('probe',start,adapted.hand_card_ids,
                tuple(map(bool,adapted.legal_mask)),900,1600,True,
                (ab.unit_eid if ab else None,None))
            emit('observed',tick=snap.tick,own=own.to_dict())
            if isinstance(pending,PendingAbility):
                result=reconcile_ability(pending,match_id='probe',abilities=abilities,now=time.monotonic())
                if result:
                    emit('receipt',receipt=bridge.resolve(result),abilities=[b.to_dict() for b in abilities])
                    if result!='confirmed':raise RuntimeError('Skill probe was not confirmed')
                    profile_path=Path('configs/nulls_abilities_900x1600.json')
                    profile=json.loads(profile_path.read_text())
                    profile.update(validated=True,validated_slots=[0],evidence=str(a.out),
                                   before=before,after=ab.to_dict())
                    profile_path.write_text(json.dumps(profile,indent=2),encoding='utf-8')
                    return
            elif isinstance(pending,PendingPlay):
                result=reconcile_play(pending,match_id='probe',played_count=own.played_count,
                                      hand=own.hand,now=time.monotonic())
                if result:
                    emit('receipt',receipt=bridge.resolve(result))
                    if result!='confirmed':raise RuntimeError('Probe card was not confirmed')
                    pending=None
            elif ab is not None and adapted.legal_mask[2305]:
                before=ab.to_dict()
                receipt=bridge.submit(2305,match_id='probe',expected_ability_unit_id=ab.unit_eid,execute=True)
                pending=PendingAbility('probe',0,ab.deck_idx,ab.unit_eid,ab.charges,ab.cooldown_ms,time.monotonic())
                emit('skill_sent',receipt=receipt,before=before)
            elif ab is not None and ab.unit_eid is None and sent<5 and snap.tick>=90:
                candidates=[]
                for slot,index in enumerate(own.hand):
                    if 0<=index<len(own.deck):
                        tiles=np.flatnonzero(adapted.legal_mask[1+slot*576:1+(slot+1)*576])
                        if len(tiles):candidates.append((index!=ab.deck_idx,own.deck[index].cost,slot,tiles))
                if candidates:
                    _,_,slot,tiles=min(candidates,key=lambda x:x[:2])
                    tile=int(min(tiles,key=lambda t:(t%18-9)**2+(t//18-4)**2))
                    action=S.play_tile(slot,tile)
                    receipt=bridge.submit(action,match_id='probe',expected_card_id=adapted.hand_card_ids[slot],execute=True)
                    pending=PendingPlay('probe',own.played_count,own.hand,slot,time.monotonic())
                    sent+=1;emit('probe_card_sent',card=adapted.hand_cards[slot],receipt=receipt)
            time.sleep(max(.02,.2-(time.monotonic()-start)))
        raise RuntimeError('No confirmed skill within bounded probe')
    finally:mem.close();log.close()


if __name__=='__main__':main()
