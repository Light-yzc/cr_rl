"""Same native actions/seeds, bounded CPU timing and coverage; no policy/model."""
import json
from pathlib import Path
import time

import numpy as np
from crforge_env import _jvm
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.training.army_elixir import NativeArmyElixir, load_values

root=Path(__file__).resolve().parents[1]
_jvm.start(root/'.cache/vendor/crforge/env/build/install/env',['-Xmx512m'])
deck=['Knight','Archer','MinionHorde','Cannon','Giant','MiniPekka','Musketeer','ZapMachine']
results=[]
with CrForgeVector(1,1,independent=True,decision_ticks=8) as env:
    for repeat in range(4):
        for mode in ('off','elixir','fallback'):
            env.vector.reset([193+repeat],[deck],[deck])
            tracker=(NativeArmyElixir(env,load_values(root/'configs'/
                ('army_elixir_stat_fallback.yaml' if mode=='fallback' else 'army_elixir_values.yaml')))
                if mode!='off' else None)
            rng=np.random.default_rng(222+repeat)
            steps=0;read_seconds=0.;mapped=unknown=0;violations=0;started=time.perf_counter()
            for step in range(650):
                actions=np.zeros((1,2),dtype=np.int32)
                if step%8==0:
                    for seat in (0,1):
                        legal=np.flatnonzero(env.masks[0,seat,1:2305])+1
                        if len(legal): actions[0,seat]=rng.choice(legal)
                env.step(actions);steps+=1
                if env.vector.fault(0): raise RuntimeError(str(env.vector.fault(0)))
                if tracker:
                    began=time.perf_counter();_,stats=tracker.poll();read_seconds+=time.perf_counter()-began
                    mapped+=stats[0]['mapped_member_count'];unknown+=stats[0]['unknown_lineage_count']
                    violations=max(violations,stats[0]['budget_violation_count'])
                if env.dones[0]: break
            results.append(dict(repeat=repeat,mode=mode,steps=steps,seconds=time.perf_counter()-started,
                shaping_seconds=read_seconds,mapped_members=mapped,unknown_members=unknown,
                budget_violations=violations))
report=dict(cases=results,checksum_scans=False)
out=root/'artifacts/army_elixir/native_benchmark.json'
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
print(json.dumps(report))
