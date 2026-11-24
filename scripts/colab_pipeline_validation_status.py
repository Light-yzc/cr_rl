"""Read status without exposing command environments or credentials."""
import json
from pathlib import Path
import subprocess

work=Path('/content/crrl_pipeline_validation')
paths=[work/'queue.log',work/'result.json',work/'validation.log',work/'throughput.log',
       work/'finish.log',work/'combined_passed.json',
       work/'longrun-launch.log',
       Path('/content/crrl_training/runs/throughput/results.json'),
       Path('/content/crrl_training/runs/awbc-smoke/expert_bc_latest_report.json'),
       Path('/content/crrl_training/runs/awbc-smoke/awbc_smoke_report.json'),
       Path('/content/crrl_training/runs/longrun/exit.json'),
       Path('/content/crrl_army_elixir_1117/experiment.log'),
       Path('/content/crrl_army_elixir_1117/artifacts/army_elixir/complete.json')]
for p in paths:
    if p.exists():
        with p.open('rb') as file:
            file.seek(max(0,p.stat().st_size-2000))
            print(p.name,file.read().decode(errors='replace'))
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        matches=[Path(a.decode()).name for a in p.read_bytes().split(b'\0')
                 if a.endswith((b'/train_selfplay.py',b'/evaluate_state_policy.py',
                                b'/run_army_elixir_experiments.py',b'/crrl_queue_pipeline.py',
                                b'/colab_longrun_job.py',b'/crrl_queue_throughput.py',
                                b'/benchmark_colab_pipeline.py',b'/cr_train_pipeline.py',
                                b'/crrl_finish_pipeline.py',b'/crrl_queue_longrun.py'))]
        if matches:
            stats=subprocess.run(['ps','-p',p.parent.name,'-o','stat=,etime=,time=,rss='],
                                 capture_output=True,text=True).stdout.strip()
            print(json.dumps(dict(pid=p.parent.name,scripts=matches,process_stats=stats)))
    except (OSError,UnicodeError): pass
reports=Path('/content/crrl_army_elixir_1117/artifacts/army_elixir')
for p in sorted(reports.glob('*/evaluation/result.json')):
    data=json.loads(p.read_text())
    print('REWARD_EVAL',p.parents[1].name,{k:data.get(k) for k in ('games','wins','losses','draws')})

run=Path('/content/crrl_training/runs/awbc-smoke')
progress=run/'progress.jsonl'
if progress.exists():
    with progress.open('rb') as file:
        file.seek(max(0,progress.stat().st_size-32768))
        lines=file.read().splitlines()
    for line in lines[-8:]:
        try: row=json.loads(line)
        except ValueError: continue
        print('VALIDATION_PROGRESS',{k:row[k] for k in ('phase','update','ran','reason',
              'actor_attempted','accepted','value_gate_passed','batch_games') if k in row})
phase=run/'awbc_latest_report.json'
if phase.exists():
    row=json.loads(phase.read_text())
    print('AWBC_GATE',{k:row.get(k) for k in ('accepted','actor_attempted','reason',
          'min_new_games')},'eligible_fresh',len(row.get('eligible_new_train_indices',[])))
work_features=run/'awbc_value_features'
if work_features.is_dir():
    print('AWBC_FEATURE_FILES',len(list(work_features.glob('*.npy'))))
for p in sorted(Path('/content/crrl_training/runs/throughput').glob('*/summary.json')):
    row=json.loads(p.read_text())
    print('PROFILE_UPDATE',p.parent.name,{k:row.get(k) for k in
          ('update','samples_per_second','peak_cuda_allocated_gib','optimizer_steps','faulted_samples')})
