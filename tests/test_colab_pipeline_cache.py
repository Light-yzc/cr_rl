import json

from cr_rl.training.colab_pipeline import ppo_command


def test_resident_cache_default_and_explicit_configuration(tmp_path):
    (tmp_path/'prepared').mkdir()
    (tmp_path/'prepared/source_metadata.json').write_text(json.dumps({'training_args':{}}))
    config=dict(run_name='run',awbc=False,expert_bc=False,backup=False,wandb=False)
    command=ppo_command(tmp_path,config)
    assert '--async-checkpoint' in command
    assert command[command.index('--checkpoint-interval')+1]=='5'
    for option in ('--opponent-cache-size','--training-opponent-cache-size'):
        assert command[command.index(option)+1]=='4'
    command=ppo_command(tmp_path,dict(config,opponent_cache_size=3,training_opponent_cache_size=2))
    assert command[command.index('--opponent-cache-size')+1]=='3'
    assert command[command.index('--training-opponent-cache-size')+1]=='2'
    command=ppo_command(tmp_path,dict(config,async_checkpoint=False,checkpoint_interval=3))
    assert '--async-checkpoint' not in command
    assert command[command.index('--checkpoint-interval')+1]=='3'


def test_explicit_expert_bc_dataset_uses_specialist_snapshot(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'source_metadata.json').write_text(json.dumps({'training_args': {}}))
    general = str(tmp_path / 'quality_bc')
    (prepared / 'full_dataset.json').write_text(json.dumps({'local_path': general}))
    config = dict(run_name='sparky_pilot', awbc=False, expert_bc=True, backup=False, wandb=False)
    command = ppo_command(tmp_path, config)
    assert command[command.index('--expert-bc-dataset') + 1] == general
    specialist = tmp_path / 'sparky_bc'
    command = ppo_command(tmp_path, dict(config, expert_bc_dataset=specialist))
    assert command[command.index('--expert-bc-dataset') + 1] == str(specialist)
    assert '--expert-bc-experimental' in command


def test_runtime_gradient_checkpointing_override_and_parallel_validation(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'source_metadata.json').write_text(
        json.dumps({'training_args': {'gradient_checkpointing': True}})
    )
    config = dict(run_name='run', awbc=False, expert_bc=False, backup=False, wandb=False)
    assert '--gradient-checkpointing' in ppo_command(tmp_path, config)
    command = ppo_command(tmp_path, dict(config, gradient_checkpointing=False, validation_argmax_lanes=16))
    assert '--gradient-checkpointing' not in command
    assert command[command.index('--validation-argmax-lanes') + 1] == '16'


def test_ppo_only_omits_auxiliary_inputs(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'source_metadata.json').write_text(json.dumps({'training_args': {}}))
    command = ppo_command(tmp_path, dict(run_name='rollback', ppo_only=True, backup=False, wandb=False))
    assert '--ppo-only' in command
    assert not any(arg.startswith(('--expert-bc', '--awbc')) for arg in command)


def test_plan_b_passes_reward_and_bc_overrides(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'source_metadata.json').write_text(json.dumps({'training_args': {'gae_lambda': .985}}))
    command = ppo_command(tmp_path, dict(
        run_name='plan_b', awbc=False, backup=False, wandb=False,
        expert_bc_plan_b=True, expert_bc_dataset='snapshot', expert_bc_interval=5,
        expert_bc_sequences=12, expert_bc_learning_rate=1e-6, expert_bc_max_kl=.005,
        gae_lambda=.990, keep_periodic_checkpoints=2, army_reward_mode='elixir_potential',
        army_elixir_bar_factor=.75, allow_reward_migration=True,
    ))
    assert '--expert-bc-plan-b' in command
    assert '--allow-reward-migration' in command
    for flag, value in {
        '--gae-lambda': '0.99', '--army-elixir-bar-factor': '0.75',
        '--keep-periodic-checkpoints': '2', '--expert-bc-interval': '5',
        '--expert-bc-sequences': '12', '--expert-bc-max-kl': '0.005',
    }.items():
        assert command[command.index(flag)+1] == value


def test_reserve_to_wait_disable_and_adaptive_lr_forwarding(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'source_metadata.json').write_text(
        json.dumps({'training_args': {'reserve_to_wait': True, 'target_card_kl': 0.015, 'target_tile_kl': 0.06}})
    )
    cmd_off = ppo_command(tmp_path, dict(
        run_name='rollback_clean', awbc=False, expert_bc=False, backup=False, wandb=False,
        reserve_to_wait=False, adaptive_actor_lr=True, adaptive_actor_lr_min=2.5e-6,
        adaptive_actor_lr_max=6e-6, override_adaptive_actor_lr=True,
        actor_effective_batch_size=2048, critic_effective_batch_size=8192,
        continue_after_kl_reject=True,
    ))
    assert '--reserve-to-wait' not in cmd_off
    assert '--adaptive-actor-lr' in cmd_off
    assert '--override-adaptive-actor-lr' in cmd_off
    assert '--continue-after-kl-reject' in cmd_off
    assert cmd_off[cmd_off.index('--actor-effective-batch-size') + 1] == '2048'
    assert cmd_off[cmd_off.index('--critic-effective-batch-size') + 1] == '8192'

    cmd_ld = ppo_command(tmp_path, dict(
        run_name='rollback_clean', awbc=False, expert_bc=False, backup=False, wandb=False,
        lookahead_distill=True, lookahead_distill_games=12,
        lookahead_distill_parallel_games=12, lookahead_distill_parallel_searches=4,
        lookahead_distill_batch_episodes=12,
    ))
    assert cmd_ld[cmd_ld.index('--lookahead-distill-games') + 1] == '12'
    assert cmd_ld[cmd_ld.index('--lookahead-distill-parallel-games') + 1] == '12'
    assert cmd_ld[cmd_ld.index('--lookahead-distill-parallel-searches') + 1] == '4'
    assert cmd_ld[cmd_ld.index('--lookahead-distill-batch-episodes') + 1] == '12'
