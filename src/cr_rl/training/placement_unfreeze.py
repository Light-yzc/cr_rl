"""Explicit, resumable transition from placement warmup to joint BC."""
from copy import deepcopy
import math

from .placement_stage import set_placement_training_mode


def arm_unfreeze(args, definition, state=None):
    stage = definition.get('placement_stage')
    requested = getattr(args, 'unfreeze_after_checkpoints', None)
    if requested is None:
        return
    if stage is None:
        raise ValueError('Scheduled unfreeze requires a learned placement stage')
    if args.freeze_frame or args.lr_schedule not in (None, 'constant') or (state and state.get('lr_schedule')):
        raise ValueError('Scheduled unfreeze requires unfrozen frame setting and constant LR')
    if args.checkpoint_every_episodes < 1:
        raise ValueError('Scheduled unfreeze requires a positive checkpoint interval')
    config = {'checkpoints': requested, 'checkpoint_interval': args.checkpoint_every_episodes,
              'after_training_seats': requested * args.checkpoint_every_episodes,
              'learning_rate': args.unfreeze_learning_rate}
    existing = stage.get('unfreeze')
    if existing is not None and any(existing.get(key) != value for key, value in config.items()):
        raise ValueError('Resume must preserve the saved unfreeze schedule')
    if existing is None:
        stage['unfreeze'] = config
        if state:
            # This opt-in is the only permitted resume definition change.
            state['training_config']['placement_stage'] = deepcopy(stage)


def configure_training_mode(model, stage):
    if stage.get('mode') == 'placement_only':
        set_placement_training_mode(model)
    elif stage.get('mode') == 'full_finetune' and stage.get('unfreeze', {}).get('transition'):
        model.requires_grad_(True)
        for name, parameter in model.named_parameters():
            if name.endswith('.anchor'):
                parameter.requires_grad_(False)
    else:
        raise ValueError('Unsupported placement freeze mode')


def optimizer_parameters(model, stage):
    if not stage or stage.get('mode') != 'full_finetune':
        return [p for p in model.parameters() if p.requires_grad]
    # Match the group order established by add_param_group at transition.
    heads, trunk = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (heads if name.startswith('learned_placement.') else trunk).append(parameter)
    return [{'params': heads}, {'params': trunk}]


def maybe_unfreeze(model, optimizer, definition, *, epoch, next_episode, train_seats, optimizer_steps):
    stage = definition.get('placement_stage', {})
    config = stage.get('unfreeze')
    if not config or stage.get('mode') != 'placement_only':
        return None
    completed = epoch * train_seats + next_episode
    if completed < config['after_training_seats']:
        return None
    lr = config['learning_rate']
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid full-finetune learning rate')
    active = {id(p) for group in optimizer.param_groups for p in group['params']}
    model.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if name.endswith('.anchor'):
            parameter.requires_grad_(False)
    new_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in active]
    if not new_parameters:
        raise ValueError('Unfreeze found no new trunk parameters')
    optimizer.add_param_group({'params': new_parameters, 'lr': lr})
    for group in optimizer.param_groups:
        group['lr'] = lr
    record = {'epoch': epoch, 'next_episode': next_episode, 'training_seats': completed,
              'optimizer_steps': optimizer_steps, 'learning_rate': lr,
              'head_adam_preserved': True, 'new_trunk_adam_initialized_lazily': True}
    config['transition'] = record
    stage['mode'] = 'full_finetune'
    return record
