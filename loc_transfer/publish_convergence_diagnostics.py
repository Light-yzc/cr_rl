"""Expose sample utilization and independently audited MC value diagnostics to TB."""
import json
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

root=Path(__file__).resolve().parents[1]
run=(root/'artifacts/local_dual_v100').resolve()
writer=EventFileWriter(str(run/'tensorboard/convergence_diagnostics'))
rows={}
for line in (run/'metrics.jsonl').open():
    r=json.loads(line)
    if 'sample_utilization' in r: rows[r['update']]=r
for update,r in sorted(rows.items()):
    usage=r['sample_utilization']
    values={'utilization/actor_kl_stopped':float(r['actor_kl_stopped'])}
    for suffix,denom,num in [
        ('planned_presentations_committed_fraction','actor_planned_presentations','actor_samples_committed'),
        ('unique_choices_committed_fraction','actor_eligible_unique','actor_unique_samples_committed'),
    ]:
        if usage.get(denom,0):values['utilization/'+suffix]=usage[num]/usage[denom]
    writer.add_event(Event(wall_time=time.time(),step=update,summary=Summary(value=[Summary.Value(tag=k,simple_value=v) for k,v in values.items()])))
mc=json.loads((root/'artifacts/local_dual_v100_checks/critic_returns_181_182.json').read_text())
values={f'critic_complete_MC/{k}':mc[k] for k in ['explained_variance','mse','mean_error','target_std','value_std','eligible_games','target_count']}
writer.add_event(Event(wall_time=time.time(),step=182,summary=Summary(value=[Summary.Value(tag=k,simple_value=v) for k,v in values.items()])))
writer.flush();writer.close()
print('Published utilization history and audited complete-episode MC diagnostics')
