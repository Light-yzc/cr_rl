"""Shared PPO command construction and bounded progress for Colab notebooks/jobs."""
import json
from pathlib import Path
import sys


def ppo_command(root, config, *, out=None, resume=None):
    root=Path(root)
    saved=json.loads((root/'prepared/source_metadata.json').read_text())['training_args']
    output=Path(out or root/'runs'/config['run_name'])
    entry=config.get('trainer_entry') or root/'scripts/train_selfplay.py'
    command=[sys.executable,str(entry),'--v3','--role-league',
        '--resume',str(resume or (output/'ppo_latest.pt' if (output/'ppo_latest.pt').exists()
                                 else root/'prepared/input/ppo_latest.pt')),
        '--bc-anchor',str(root/'prepared/anchor_actor.pt'),'--out',str(output),
        '--device','cuda:0','--critic-device','cuda:0','--opponent-device','cuda:0',
        '--precision','fp16','--decks',str(root/'prepared/source_league.yaml'),
        '--pool-capacity',str(config.get('pool_capacity',4)),
        '--history-snapshot-interval',str(config.get('history_snapshot_interval',20000)),
        '--opponent-cache-size',str(config.get('opponent_cache_size',4)),
        '--training-opponent-cache-size',str(config.get('training_opponent_cache_size',4)),
        '--current-opponent-fraction',str(config.get('current_opponent_fraction','.5')),
        '--historical-opponent-fraction',str(config.get('historical_opponent_fraction','.25')),
        '--anchor-opponent-fraction',str(config.get('anchor_opponent_fraction','.25')),
        '--validation-interval',str(config.get('validation_interval',0)),
        '--validation-pairs',str(config.get('validation_pairs',4)),
        '--keep-checkpoints','1','--keep-periodic-checkpoints',str(config.get('keep_periodic_checkpoints',0)),
        '--rollout-save-interval','0','--keep-rollouts','0']
    for key,value in dict(updates=config.get('updates',10000),envs=config.get('envs',8),
            steps=config.get('steps',128),epochs=2,critic_epochs=2,batch_size=1,
            actor_batch_size=config.get('actor_batch_size',1),
            critic_batch_size=config.get('critic_batch_size',1),grad_accumulation=8,
            frame_microbatch=config.get('frame_microbatch',32),torch_threads=config.get('torch_threads',4),
            env_threads=config.get('env_threads',8),
            checkpoint_interval=config.get('checkpoint_interval',5)).items():
        command+=['--'+key.replace('_','-'),str(value)]
    if config.get('async_checkpoint', True):
        command.append('--async-checkpoint')
    if config.get('cache_reference', False):
        command.append('--cache-reference')
    if config.get('validation_argmax_lanes', 0):
        command += ['--validation-argmax-lanes', str(config['validation_argmax_lanes'])]
    for key in ('gamma','gae_lambda','critic_gae_lambda','army_reward','princess_reward',
                'king_reward','crown_reward','overflow_penalty','entropy_coef',
                'card_entropy_coef','tile_entropy_coef','anchor_kl_coef','reference_kl_coef',
                'temporal_checkpoint_mode','tbptt_chunk','target_kl','target_card_kl','target_tile_kl',
                'learning_rate','resume_actor_learning_rate','resume_critic_learning_rate',
                'actor_effective_batch_size','critic_effective_batch_size'):
        if key == 'army_reward' and config.get('army_reward_mode') == 'elixir_potential':
            value = config.get(key, 0.0)
        else:
            value = config.get(key, saved.get(key))
        if value is not None: command+=['--'+key.replace('_','-'),str(value)]
    for key in ('ppo_actor_fp32','cache_state_frames','reserve_to_wait','tower_shaping','continue_after_kl_reject'):
        if config.get(key, saved.get(key)): command.append('--'+key.replace('_','-'))
    if config.get('adaptive_actor_lr', saved.get('adaptive_actor_lr')):
        command.append('--adaptive-actor-lr')
        for key in ('adaptive_actor_lr_min', 'adaptive_actor_lr_max'):
            val = config.get(key, saved.get(key))
            if val is not None:
                command += ['--' + key.replace('_', '-'), str(val)]
        if config.get('override_adaptive_actor_lr', saved.get('override_adaptive_actor_lr')):
            command.append('--override-adaptive-actor-lr')
    if config.get('gradient_checkpointing', saved.get('gradient_checkpointing')):
        command.append('--gradient-checkpointing')
    if config.get('full_episode_bptt',True): command.append('--full-episode-bptt')
    # New elixir mode stays opt-in until coverage and independent evaluation pass.
    command+=['--army-reward-mode','uniform']
    if config.get('army_reward_mode') == 'elixir_potential':
        command[-1] = 'elixir_potential'
        elixir_cfg = config.get('army_elixir_config') or (root/'configs/army_elixir_values.yaml')
        command += ['--army-elixir-beta', str(config.get('army_elixir_beta', .004)),
                    '--army-elixir-config', str(elixir_cfg)]
        if config.get('army_elixir_bar_factor') is not None:
            command += ['--army-elixir-bar-factor', str(config['army_elixir_bar_factor'])]
        if config.get('allow_reward_migration'):
            command.append('--allow-reward-migration')
    ppo_only = config.get('ppo_only', False)
    if ppo_only:
        command.append('--ppo-only')
    if not ppo_only and config.get('expert_bc',True):
        dataset=config.get('expert_bc_dataset')
        if dataset is None:
            dataset=json.loads((root/'prepared/full_dataset.json').read_text())['local_path']
        command+=['--expert-bc-dataset',str(dataset),'--expert-bc-experimental',
            '--expert-bc-interval',str(config.get('expert_bc_interval',100)),
            '--expert-bc-sequences',str(config.get('expert_bc_sequences',4))]
        for key in ('expert_bc_learning_rate', 'expert_bc_max_kl'):
            if config.get(key) is not None:
                command += ['--' + key.replace('_', '-'), str(config[key])]
        if config.get('expert_bc_plan_b'):
            command.append('--expert-bc-plan-b')
    if not ppo_only and config.get('macro_search', False):
        command.append('--macro-search')
        for key in ('macro_search_interval', 'macro_search_offset', 'macro_search_candidates', 'macro_search_win_margin',
                    'macro_search_max_buffer', 'macro_search_batch_episodes',
                    'macro_search_learning_rate', 'macro_search_max_kl'):
            if config.get(key) is not None:
                command += ['--' + key.replace('_', '-'), str(config[key])]
    if not ppo_only and config.get('lookahead_distill', False):
        command.append('--lookahead-distill')
        for key in ('lookahead_distill_interval', 'lookahead_distill_offset', 'lookahead_distill_games',
                    'lookahead_distill_parallel_games', 'lookahead_distill_parallel_searches',
                    'lookahead_distill_threads', 'lookahead_distill_max_buffer',
                    'lookahead_distill_candidates', 'lookahead_distill_rollouts',
                    'lookahead_distill_horizon', 'lookahead_distill_trigger_elixir',
                    'lookahead_distill_max_searches', 'lookahead_distill_batch_episodes',
                    'lookahead_distill_steps', 'lookahead_distill_margin',
                    'lookahead_distill_learning_rate', 'lookahead_distill_max_kl',
                    'lookahead_distill_temporal_blocks'):
            if config.get(key) is not None:
                command += ['--' + key.replace('_', '-'), str(config[key])]
        if config.get('lookahead_distill_placement') is not None:
            command.append(
                '--lookahead-distill-placement'
                if config['lookahead_distill_placement']
                else '--no-lookahead-distill-placement'
            )
    if config.get('focal_decks') and '--focal-decks' not in command:
        command += ['--focal-decks', str(config['focal_decks']), '--focal-deck-prob', str(config.get('focal_deck_prob', 1.0))]
    if config.get('expand_decks') and '--expand-decks' not in command:
        command += ['--expand-decks', str(config['expand_decks'])]
    if config.get('tower_troop_mixture') and '--tower-troop-mixture' not in command:
        command += ['--tower-troop-mixture', str(config['tower_troop_mixture'])]
    if config.get('validation_decks') and '--validation-decks' not in command:
        command += ['--validation-decks', str(config['validation_decks'])]
    if not ppo_only and config.get('awbc',True):
        command+=['--awbc-sessions',str(root/'sessions'),'--awbc-interval',str(config.get('awbc_interval',50)),
            '--awbc-min-new-games','12','--awbc-batch-games','24','--awbc-max-reuse','2',
            '--awbc-epochs','2','--awbc-value-epochs','20','--awbc-learning-rate','1e-6','--awbc-max-kl','.01']
    if config.get('backup',True):
        command+=['--backup-dir','/content/drive/MyDrive/cr_rl_ppo/training/'+config['run_name'],
            '--async-publication','--no-backup-hash','--publication-interval',str(config.get('publication_interval',1))]
        command.append('--checkpoint-first-update')
    if config.get('wandb',True): command.append('--wandb')
    if config.get('critic_fp16_resume_scale_cap') is not None:
        command+=['--critic-fp16-resume-scale-cap',str(config['critic_fp16_resume_scale_cap'])]
    return command


def progress(root, run='longrun'):
    path=Path(root)/'runs'/run/'summary.json'
    if not path.exists(): return dict(stage='initializing')
    data=json.loads(path.read_text())
    return {k:data.get(k) for k in ('update','learner_samples_total','completed_training_games_total',
        'samples_per_second','seconds','peak_cuda_allocated_gib','metrics','awbc_playbuffer','expert_bc','macro_search','lookahead_distill')}
