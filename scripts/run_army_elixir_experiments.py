"""Isolated U929 reward comparisons. Never modifies the original training tree."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/content/crrl_training')
OUT=ROOT/'artifacts/army_elixir'


def set_arg(command,option,value):
    command=list(command)
    for i,item in enumerate(command):
        if item==option:
            command[i+1]=str(value)
            return command
    return command+[option,str(value)]


def evaluate(name):
    import yaml
    import torch
    from types import SimpleNamespace
    import evaluate_state_policy as evaluation
    from cr_rl.evaluation.cohort import paired_match_specs
    root=OUT/name
    target=root/'evaluation'
    target.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    actor,info=evaluation.load_frozen_policy(root/'actor_export.pt','cuda:0')
    opponent,opp_info=evaluation.load_frozen_policy(SOURCE/'prepared/anchor_actor.pt','cuda:0')
    league=yaml.safe_load((SOURCE/'prepared/source_league.yaml').read_text())
    matchups=paired_match_specs(list(league['decks']),2,0,simulator_seed_base=1930000,pairing_seed=20261117)
    args=SimpleNamespace(envs=2,device='cuda:0',precision='fp16',out=target,
        max_decisions=1500,sampling_seed=20261117,no_file_hashes=True,native_audit=False)
    latency=evaluation.evaluate_games(actor,opponent,[info,opp_info],league,matchups,args)
    valid=[row for row in matchups if row.get('valid')]
    result=dict(games=len(valid),wins=sum(row['reward']>0 for row in valid),
        losses=sum(row['reward']<0 for row in valid),draws=sum(row['reward']==0 for row in valid),
        matchups=matchups,latency=latency,checksum_scans=False,
        limitation='Four paired games: integration/scale check, not evidence of improved win rate')
    (target/'result.json').write_text(json.dumps(result,indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'.cache/vendor/crforge/python'),
        OMP_NUM_THREADS='2')
    tests=[sys.executable,'-m','pytest','-q','tests/test_reward_shaping.py',
        'tests/test_army_elixir_shaping.py','tests/test_ppo_auxiliary_integration.py',
        'tests/test_state_ppo_v3.py','tests/test_expert_bc.py','tests/test_fast_publication.py']
    with (OUT/'tests.log').open('w') as log:
        result=subprocess.run(tests,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('CPU regression failed; see tests.log')
    baseline=json.loads((SOURCE/'runs/bc-smoke/exit.json').read_text())['argv']
    baseline[1]=str(ROOT/'scripts/train_selfplay.py')
    # Remove BC hook from reward experiment; keep the same U929/PPO config otherwise.
    for option in ('--expert-bc-dataset','--expert-bc-interval','--expert-bc-sequences'):
        while option in baseline:
            index=baseline.index(option);del baseline[index:index+2]
    baseline=[x for x in baseline if x not in ('--expert-bc-experimental','--full-episode-bptt')]
    baseline=set_arg(baseline,'--steps',128)
    baseline=set_arg(baseline,'--checkpoint-interval',2)
    baseline += ['--allow-reward-migration','--army-reward-mode','uniform']
    variants=[('A_none',0,None),('B_uniform',.015,None),
              ('C_elixir004',0,.004),('C_elixir002',0,.002),('C_elixir008',0,.008)]
    for name,weight,beta in variants:
        folder=OUT/name
        folder.mkdir(exist_ok=True)
        command=set_arg(baseline,'--out',folder)
        command=set_arg(command,'--army-reward',weight)
        if beta is not None:
            command=set_arg(command,'--army-reward-mode','elixir_potential')
            command+=['--army-elixir-beta',str(beta),'--army-elixir-config',str(ROOT/'configs/army_elixir_values.yaml')]
        (folder/'command.json').write_text(json.dumps(command,indent=2))
        started=time.time()
        with (folder/'train.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        (folder/'exit.json').write_text(json.dumps(dict(exit_code=result.returncode,seconds=time.time()-started)))
        print(json.dumps(dict(stage='trained',variant=name,exit_code=result.returncode)),flush=True)
        if result.returncode: raise RuntimeError('Reward variant failed: '+name)
    # Restore a v2 reward checkpoint and execute PPO again from a fresh world.
    folder=OUT/'resume';folder.mkdir(exist_ok=True)
    command=json.loads((OUT/'C_elixir004/command.json').read_text())
    command=set_arg(command,'--resume',OUT/'C_elixir004/ppo_latest.pt')
    command=set_arg(command,'--out',folder);command=set_arg(command,'--updates',1)
    with (folder/'train.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Elixir resume failed')
    # Same collector implementation also exercises complete-episode BPTT.
    folder=OUT/'full_episode';folder.mkdir(exist_ok=True)
    command=set_arg(command,'--out',folder)
    command+=['--full-episode-bptt']
    with (folder/'train.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Elixir full-episode collector failed')
    for name,_,_ in variants:
        result=subprocess.run([sys.executable,str(__file__),'--evaluate',name],cwd=ROOT,env=env)
        if result.returncode: raise RuntimeError('Frozen paired evaluation failed: '+name)
    (OUT/'complete.json').write_text(json.dumps(dict(completed=True,checksum_scans=False,
        variants=[v[0] for v in variants],long_training_recommended=False)))

if __name__=='__main__':
    if '--evaluate' in sys.argv: evaluate(sys.argv[-1])
    else: main()
