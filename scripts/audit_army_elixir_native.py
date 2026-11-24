"""P0 native provenance audit, with no model/GPU and no per-step disk writes."""
import json
from pathlib import Path
import time

import numpy as np
from crforge_env import VectorBattle, LadderConfig

ROOT=Path(__file__).resolve().parents[1]

def play_record(play):
    item=play.mirror() or play.variant() or play.evolution()
    result=play.result()
    return dict(name=str(play.name()),side=int(play.side()),tick=int(play.tick()),
        placed=bool(result is not None and result.placed()),
        cost=int(item.cost()) if item is not None else None,
        mirror=play.mirror() is not None,variant=play.variant() is not None,
        form=int(play.evolution().field()) if play.evolution() is not None else None,
        units=[dict(id=int(u.getId()),row=str(u.getData().name()),
                    cast_id=int(u.castId()),deploy_index=int(u.getDeployIndex())) for u in play.units()])

def main():
    install=ROOT/'.cache/vendor/crforge/env/build/install/env'
    tables=Path.home()/'.crforge/tables/16.635.3'
    deck=['Knight','Archer','MinionHorde','Cannon','Golem','Tombstone','Graveyard','Mirror']
    report=dict(backend_commit='7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6',
                decision_ticks=8,cases=[],table_rows={})
    for table in ('spells_characters','spells_buildings','spells_other','characters','buildings'):
        data=json.loads((tables/(table+'.json')).read_text(encoding='utf8'))
        report['table_rows'][table]={name:row['columns'] for name,row in data['rows'].items()
            if name in deck or name in ('Golemite','Skeleton')}
    with VectorBattle(1,1,install=install,tables=tables,
                      config=LadderConfig(decision_ticks=8),jvm_args=['-Xmx512m']) as env:
        report['id_version']=env.id_version
        for target in ('Knight','Archer','MinionHorde','Cannon','Golem','Tombstone','Graveyard'):
            for seed in range(10,40):
                env.reset([seed],deck,deck)
                hand=env.observations[0,0,list(env.schema.header['hand_card'])]
                if env.vocabulary.index(target) in hand: break
            std=env._env.env(0).battle()
            evidence=[];chosen=False;stable={};identity_changes=0;refused_plays=0
            started=time.perf_counter()
            for step in range(150):
                actions=np.zeros((1,2),dtype=np.int32)
                if not chosen:
                    hand=env.observations[0,0,list(env.schema.header['hand_card'])]
                    for slot in np.flatnonzero(hand==env.vocabulary.index(target)):
                        legal=np.flatnonzero(env.masks[0,0,1+slot*576:1+(slot+1)*576])
                        if len(legal):
                            actions[0,0]=int(1+slot*576+legal[len(legal)//2]);chosen=True
                env.step(actions)
                if env.fault(0): raise RuntimeError(str(env.fault(0)))
                for unit in std.getBattle().getHolder().entities():
                    identity=int(unit.getId());name=str(unit.name())
                    if identity in stable and stable[identity]!=name: identity_changes+=1
                    stable[identity]=name
                if step in (15,30,60,100,149):
                    evidence.append(dict(step=step,plays=[play_record(p) for p in std.getPlays()],
                        entities=[dict(id=int(u.getId()),name=str(u.name()))
                            for u in std.getBattle().getHolder().entities()]))
                if env.dones[0]: break
            report['cases'].append(dict(card=target,seed=seed,steps=step+1,
                seconds=time.perf_counter()-started,id_reuse_observed=identity_changes,evidence=evidence))
            print(json.dumps(dict(card=target,plays=[play_record(p) for p in std.getPlays()],
                identity_changes=identity_changes)),flush=True)
    out=ROOT/'artifacts/army_elixir/p0_native_audit.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('REPORT',str(out))

if __name__=='__main__': main()
