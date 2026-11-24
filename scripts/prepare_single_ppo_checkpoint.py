"""Prepare a fresh league branch from one full PPO checkpoint, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys

import torch


def copy_checkpoint(source, target):
    h = hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.partial')
    with source.open('rb') as src, temporary.open('wb') as dst:
        for block in iter(lambda: src.read(4*1024**2), b''):
            dst.write(block)
            h.update(block)
    temporary.replace(target)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report_path = a.out/'single_checkpoint_preparation.json'
    if report_path.exists():
        raise FileExistsError('Already prepared; use report or a new directory')
    local = a.out/'input/ppo_latest.pt'
    checksum = copy_checkpoint(a.checkpoint, local)
    ckpt = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    if ckpt.get('model_architecture_version') != 3 or ckpt.get('decision_ticks') != 8:
        raise ValueError('Expected architecture-v3 0.4s checkpoint')
    for key in ('model', 'critic', 'critic_config', 'critic_optimizer', 'training_args', 'league_config'):
        if not ckpt.get(key):
            raise ValueError('Full PPO resume field is missing: '+key)
    fields = ('model','config','temporal_config','learned_placement_definition',
              'model_architecture_version','decision_ticks','updates','backend_commit',
              'data_version','content_sha','level_profile')
    actor = {key:ckpt[key] for key in fields if key in ckpt}
    anchor = a.out/'anchor_actor.pt'
    temporary = anchor.with_suffix('.partial')
    torch.save(actor, temporary)
    temporary.replace(anchor)
    from cr_rl.client.rl_recording import actor_fingerprint
    fingerprint = actor_fingerprint(actor)
    training_args = ckpt['training_args']
    reward = ckpt.get('reward_config', {})
    # These are smoke argv only; never claim an automatically tuned production configuration.
    argv = [sys.executable, 'scripts/train_selfplay.py', '--v3', '--resume', str(local),
            '--bc-anchor',str(anchor),'--out',str(a.out/'ppo'), '--role-league',
            '--device','cuda:0','--critic-device','cuda:0','--opponent-device','cuda:0',
            '--precision','fp16','--envs','1','--steps','32','--updates','1',
            '--epochs','1','--critic-epochs','1','--batch-size','1',
            '--frame-microbatch','16','--pool-capacity','4',
            '--current-opponent-fraction','.5','--historical-opponent-fraction','.25',
            '--anchor-opponent-fraction','.25','--history-snapshot-interval','20000',
            '--checkpoint-interval','1','--validation-interval','0',
            '--keep-checkpoints','1','--keep-periodic-checkpoints','0']
    if training_args.get('full_episode_bptt'):
        argv.append('--full-episode-bptt')
    if training_args.get('ppo_actor_fp32'):
        argv.append('--ppo-actor-fp32')
    if reward.get('army_potential_weight') is not None:
        argv += ['--army-reward',str(reward['army_potential_weight'])]
    if training_args.get('level_mixture'):
        argv += ['--level-mixture', *map(str,training_args['level_mixture'])]
    report = dict(format='crrl-single-ppo-preparation-v1', checkpoint_sha256=checksum,
                  checkpoint_bytes=local.stat().st_size, source_checkpoint=str(a.checkpoint),
                  actor_fingerprint=fingerprint, generation=ckpt.get('updates'),
                  anchor=str(anchor), anchor_kind='frozen_starting_ppo_actor_not_expert_bc',
                  historical_policy_count=0, history_empty_fallback='anchor',
                  pool_capacity=4, requested_role_fractions=dict(current=.5,historical=.25,anchor=.25),
                  provenance={key:ckpt.get(key) for key in ('backend_commit','data_version','content_sha')},
                  saved_training_args=training_args, saved_reward=reward,
                  smoke_argv=argv, gpu_smoke_verified=False, training_started=False,
                  smoke_scope='PPO only unless launch_prepared_pipeline --awbc-playbuffer is requested',
                  awbc_smoke_requirements=dict(updates=2,interval=1,actor_attempt_required=True,
                                              subsequent_ppo_update_required=True),
                  variant_opponents_implemented=False, auxiliary_bc_hook_implemented=False)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf8')
    print(json.dumps({k:report[k] for k in ('checkpoint_sha256','generation','actor_fingerprint',
                                          'historical_policy_count','training_started')}), flush=True)


if __name__ == '__main__':
    main()
