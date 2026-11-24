"""Launch bounded GPU checks; all hot paths and logs stay on local SSD."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/content/crrl_training')
mode = os.environ.get('CRRL_JOB', 'ppo-smoke')
out = root/'runs'/mode
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=str(root/'src')+':'+str(root/'.cache/vendor/crforge/python'),
           OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
if mode == 'tests':
    command = [sys.executable,'-m','pytest','-q','tests/test_ppo_auxiliary_integration.py',
               'tests/test_quality_bc_split.py','tests/test_awbc_playbuffer.py',
               'tests/test_live_awbc.py','tests/test_live_awbc_cli.py']
else:
    metadata = json.loads((root/'prepared/source_metadata.json').read_text())
    saved = metadata['training_args']
    league = root/'prepared/source_league.yaml'
    if not league.exists():
        import torch
        import yaml
        checkpoint = torch.load(root/'prepared/input/ppo_latest.pt',map_location='cpu',
                                mmap=True,weights_only=False)
        league.write_text(yaml.safe_dump(checkpoint['league_config']))
        del checkpoint
    command = [sys.executable,str(root/'scripts/train_selfplay.py'),'--v3',
        '--resume',str(root/'prepared/input/ppo_latest.pt'),
        '--bc-anchor',str(root/'prepared/anchor_actor.pt'),'--out',str(out),
        '--role-league','--device','cuda:0','--critic-device','cuda:0',
        '--opponent-device','cuda:0','--precision','fp16','--envs','1',
        '--steps','32','--updates','2','--epochs','1','--critic-epochs','1',
        '--batch-size','1','--actor-batch-size','1','--critic-batch-size','1',
        '--grad-accumulation','1','--frame-microbatch','16','--pool-capacity','2',
        '--opponent-cache-size','1','--training-opponent-cache-size','1',
        '--checkpoint-interval','1','--validation-interval','0',
        '--keep-checkpoints','1','--keep-periodic-checkpoints','0',
        '--torch-threads','2','--env-threads','2',
        '--decks',str(league)]
    for key in ('gamma','gae_lambda','critic_gae_lambda','army_reward','princess_reward',
                'king_reward','crown_reward','overflow_penalty','entropy_coef',
                'card_entropy_coef','tile_entropy_coef','anchor_kl_coef','reference_kl_coef'):
        if saved.get(key) is not None:
            command += ['--'+key.replace('_','-'),str(saved[key])]
    for key in ('ppo_actor_fp32','gradient_checkpointing','cache_state_frames','reserve_to_wait',
                'full_episode_bptt','tower_shaping'):
        if saved.get(key): command.append('--'+key.replace('_','-'))
    if mode == 'bc-smoke':
        dataset = json.loads((root/'prepared/smoke_dataset.json').read_text())['local_path']
        command += ['--expert-bc-dataset',dataset,'--expert-bc-experimental',
                    '--expert-bc-interval','1','--expert-bc-sequences','1']
    if mode == 'resume-smoke':
        command[command.index('--resume')+1] = str(root/'runs/bc-smoke/ppo_latest.pt')
        dataset = json.loads((root/'prepared/smoke_dataset.json').read_text())['local_path']
        command += ['--expert-bc-dataset',dataset,'--expert-bc-experimental',
                    '--expert-bc-interval','100','--expert-bc-sequences','1']
        command[command.index('--updates')+1] = '1'
    if mode == 'awbc-smoke':
        dataset = json.loads((root/'prepared/full_dataset.json').read_text())['local_path']
        command += ['--expert-bc-dataset',dataset,'--expert-bc-experimental',
                    '--expert-bc-interval','1','--expert-bc-sequences','1',
                    '--awbc-sessions',str(root/'sessions'),'--awbc-interval','1',
                    '--require-awbc-smoke']
        command[command.index('--updates')+1] = '3'
if '--worker' in sys.argv:
    result = subprocess.run(command,cwd=root,env=env)
    (out/'exit.json').write_text(json.dumps(dict(exit_code=result.returncode,argv=command)))
    raise SystemExit(result.returncode)
else:
    pid_file = out/'supervisor.pid'
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()),0)
        except ProcessLookupError: pass
        else: raise RuntimeError('Existing job is still running')
    (out/'exit.json').unlink(missing_ok=True)
    with (out/'console.log').open('w') as log:
        p = subprocess.Popen([sys.executable,'-u',str(root/'scripts/colab_longrun_job.py'),'--worker'],
             cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid_file.write_text(str(p.pid))
    print(json.dumps(dict(job=mode,pid=p.pid,log=str(out/'console.log'))))
