"""Publish completed BC evaluation aggregates to TensorBoard without importing Torch."""
import json
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

root=Path(__file__).resolve().parents[1]
run=(root/'artifacts/local_dual_v100').resolve()
p=root/'artifacts/local_dual_v100_checks/bc_suite_179/summary.json'
if (p.parent/'final_report.json').exists():
    p=p.parent/'final_report.json'
r=json.loads(p.read_text())
values={
    'bc_anchor/wins':r['wins'],
    'bc_anchor/losses':r['losses'],
    'bc_anchor/draws':r['draws'],
    'bc_anchor/completed_games':r['completed_games'],
    'bc_anchor/invalid_games':r['invalid_games'],
    'bc_anchor/score':r['mean_score'],
    'bc_anchor/planned_games':r['planned_games'],
}
interval=r.get('deck_pair_cluster_bootstrap_95_interval', r.get('paired_bootstrap_95_interval'))
if interval:
    values['bc_anchor/score_ci_low'],values['bc_anchor/score_ci_high']=interval
base=p.parent
strategy_path=base/'strategy.json'
if not strategy_path.exists():
    strategy_path=base/'seed0/chunk0/strategy_first16.json'
if strategy_path.exists():
    strategy=json.loads(strategy_path.read_text())
    for side in ('learner','opponent'):
        for key in ('games','paid_elixir_per_successful_card','cost_at_least5_fraction',
                    'mean_full_elixir_wait_legal_seconds'):
            values[f'bc_anchor/executed/{side}/{key}']=strategy[side][key]
writer=EventFileWriter(str(run/'tensorboard/bc_evaluation_179'))
writer.add_event(Event(wall_time=time.time(),step=r['completed_games'],summary=Summary(value=[Summary.Value(tag=k,simple_value=v) for k,v in values.items() if v is not None])))
writer.flush();writer.close()
print(json.dumps(values))
