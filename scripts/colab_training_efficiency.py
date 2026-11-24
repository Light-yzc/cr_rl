"""Read completed update counters, excluding inherited historical counters."""
import json
import time
from pathlib import Path
out=Path('/content/crrl_training/runs/longrun')
rows=[]
with (out/'console.log').open('rb') as stream:
    stream.seek(max(0,(out/'console.log').stat().st_size-8*1024**2))
    for line in stream:
        try: row=json.loads(line)
        except ValueError: continue
        if row.get('phase')=='state_ppo': rows.append(row)
timings=('seconds','rollout_seconds','rollout_model_seconds','rollout_env_seconds',
         'actor_update_seconds','critic_update_seconds')
report=dict(updates=len(rows),first_update=rows[0]['update'] if rows else None,
 last_update=rows[-1]['update'] if rows else None,
 physical_games=sum(r.get('terminal_games',0) for r in rows),
 learner_frames=sum(r.get('valid_learner_samples',0) for r in rows),
 timing_totals={k:sum(r.get(k,0) for r in rows) for k in timings},
 launch_elapsed_seconds=time.time()-(out/'launch.json').stat().st_mtime)
report['recent']=[{k:r.get(k) for k in ('update','terminal_games','valid_learner_samples',
 'seconds','samples_per_second','rollout_model_seconds','rollout_env_seconds',
 'actor_update_seconds','critic_update_seconds','awbc_playbuffer','expert_bc')} for r in rows[-3:]]
print(json.dumps(report))
