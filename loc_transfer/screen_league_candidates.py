"""Paired screening of immutable old policies; not automatic league admission."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=(ROOT/'artifacts/local_dual_v100').resolve()
OUT=ROOT/'artifacts/ppo_convergence_trials_20261108/opponent_screen'
GENS=[0,6,119,130,133,136,139,142]
GPU_QUEUE=Queue()
V100_GPUS = (
    'GPU-95f9c3e4-242f-9098-3c63-dc4dc1557d69',
    'GPU-cb00ba20-7367-bd99-c5d6-0824adaaa5b2',
)
for device in V100_GPUS:GPU_QUEUE.put(device)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run_one(generation):
    policy_id=0 if generation==0 else (1<<30)+generation
    opponent=ROOT/'artifacts/v100_handoff/opponents'/f'opponent_{policy_id:06d}.pt'
    target=OUT/f'gen_{generation:03d}'
    target.mkdir(exist_ok=True)
    report_path=target/'evaluation.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    reusable=(report.get('games')==8 and report.get('invalid_games')==0
              and report.get('policy_sha256')==digest(BASE/'actor_export.pt')
              and report.get('opponent_sha256')==digest(opponent)
              and report.get('pair_start')==0 and report.get('pair_count')==4
              and report.get('sampling_seed')==20261201
              and report.get('pairing_seed')==20261201
              and report.get('simulator_seed_base')==11000000)
    if not reusable and (report_path.exists() or list(target.glob('match_*.json'))):
        target.rename(OUT/f'gen_{generation:03d}.interrupted_{time.time_ns()}')
        target.mkdir()
    gpu=GPU_QUEUE.get()
    try:
        cmd=[sys.executable,str(ROOT/'scripts/evaluate_policy.py'),
             '--policy',str(BASE/'actor_export.pt'),'--opponent',str(opponent),
             '--decks',str(ROOT/'configs/decks_expansion12.yaml'),'--out',str(target),
             '--pairs','4','--pair-start','0','--sampling-seed','20261201',
             '--pairing-seed','20261201','--simulator-seed-base','11000000',
             '--device','cuda:0','--frame-cache','--forced-wait-fastpath']
        exit_code=0
        if not reusable:
            with (target/'evaluation.log').open('w') as log:
                result=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),
                                      stdout=log,stderr=subprocess.STDOUT)
                exit_code=result.returncode
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        valid=exit_code==0 and report.get('games')==8 and report.get('invalid_games')==0
        row=dict(generation=generation,policy_id=policy_id,checkpoint=str(opponent),
                 exit_code=exit_code,valid=valid,reused=reusable,
                 reference='BC_anchor' if generation==0 else 'historical_PPO',
                 learner_wins=report.get('wins'),learner_losses=report.get('losses'),
                 games=report.get('games',0),evaluation=str(report_path),
                 opponent_score=(report.get('losses',0)+.5*report.get('draws',0))/8 if valid else None,
                 role='screening_only_not_confirmed_expert')
        print(json.dumps(row),flush=True)
        return row
    finally:GPU_QUEUE.put(gpu)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    summary=json.loads((BASE/'summary.json').read_text())
    assert summary['update']==182
    protocol=dict(candidate_update=182,candidate_sha256=digest(BASE/'actor_export.pt'),
                  generations=GENS,games_per_candidate=8,
                  confirmation='Rank non-anchor candidates; confirm top3 with independent seeds before expert admission',
                  pair_start=0,pair_count=4,pairing_seed=20261201,
                  simulator_seed_base=11000000,sampling_seed=20261201,
                  memory_max_gib=4,max_concurrent_evaluators=2,
                  new_evaluation_gpu_uuids=V100_GPUS,started_unix=time.time())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(run_one,g) for g in GENS]
        for future in as_completed(futures):
            rows.append(future.result())
            temporary=OUT/'screening.tmp.json'
            temporary.write_text(json.dumps(sorted(rows,key=lambda r:r['generation']),indent=2)+'\n')
            temporary.replace(OUT/'screening.json')
    assert all(r['valid'] for r in rows), 'At least one screening failed; no automatic admission'
