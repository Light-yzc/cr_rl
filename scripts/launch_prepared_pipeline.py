"""Print or launch explicitly selected prepared PPO/AWBC stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def configure_awbc_smoke(command, sessions):
    """A smoke run must hit the hook and run PPO again; do not inherit its interval."""
    command = [item for item in command if item != '--require-awbc-smoke']
    for option, value in (('--updates', '2'), ('--awbc-interval', '1'),
                          ('--awbc-sessions', str(sessions))):
        clean = []
        index = 0
        while index < len(command):
            item = command[index]
            if item == option:
                if index + 1 >= len(command):
                    raise ValueError('Missing value for ' + option)
                index += 2
            elif item.startswith(option + '='):
                index += 1
            else:
                clean.append(item)
                index += 1
        command = clean + [option, value]
    command.append('--require-awbc-smoke')
    return command


def command_for(stage, report, root, sessions, wandb=False):
    if stage == 'ppo-smoke':
        command = list(report['smoke_argv'])
        command[0] = sys.executable
        command[1] = str(root/'scripts/train_selfplay.py')
    else:
        command = [sys.executable,str(root/'scripts/train_live_awbc.py'),
                   '--checkpoint',str(root/'prepared/input/ppo_latest.pt'),
                   '--sessions',str(sessions),'--out',str(root/'runs/awbc_v1'),
                   '--device','cuda','--epochs','8']
    if wandb:
        command.append('--wandb')
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/content/crrl_training'))
    p.add_argument('--stage',choices=('ppo-smoke','awbc'),required=True)
    p.add_argument('--sessions',type=Path,default=Path('/content/drive/MyDrive/cr_rl_ppo/sessions'))
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--wandb',action='store_true')
    p.add_argument('--awbc-playbuffer',action='store_true',help='Enable real replay at PPO update boundaries')
    a = p.parse_args()
    report = json.loads((a.root/'prepared/single_checkpoint_preparation.json').read_text())
    command = command_for(a.stage,report,a.root,a.sessions,a.wandb)
    if a.awbc_playbuffer:
        if a.stage!='ppo-smoke':p.error('--awbc-playbuffer applies to the PPO stage')
        command = configure_awbc_smoke(command,a.sessions)
    print(json.dumps(dict(stage=a.stage,argv=command,plan_only=a.plan_only,
                         awbc_smoke_required=a.awbc_playbuffer)),flush=True)
    if a.plan_only:
        return
    import torch
    if not torch.cuda.is_available():
        p.error('This CPU runtime is for preparation; switch to GPU before training')
    if a.stage=='awbc' and (not a.sessions.is_dir() or len(list(a.sessions.rglob('result.json')))<12):
        p.error('AWBC requires at least 12 result-labeled real sessions; weights alone are insufficient')
    env = dict(os.environ,PYTHONPATH=str(a.root/'src')+':'+str(a.root/'.cache/vendor/crforge/python'))
    raise SystemExit(subprocess.call(command,cwd=a.root,env=env))


if __name__=='__main__':
    main()
