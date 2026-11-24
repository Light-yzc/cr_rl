import json
from dataclasses import replace

import pytest
import torch

from cr_rl.training.adaptive_kl import (
    AdaptiveActorLearningRate,
    AdaptiveActorLRConfig,
    restore_actor_lr_controller,
)


def optimizer(lr=1e-5):
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def observe(controller, card=0.003, tile=0.003, samples=64, rejected=False):
    controller.observe(
        card_kl=card,
        tile_kl=tile,
        target_card_kl=0.015,
        target_tile_kl=0.06,
        samples=samples,
        rejected=rejected,
    )


def complete(controller, opt, *, card=0.003, tile=0.003, rejected=False, overflows=0):
    controller.begin_update(opt)
    observe(controller, card=card, tile=tile)
    observe(controller, card=card, tile=tile, rejected=rejected)
    return controller.finish_update(
        opt, actor_enabled=True, committed_steps=1 if rejected else 2, overflow_steps=overflows
    )


def test_rejected_group_is_observed_and_only_changes_next_update_lr():
    opt = optimizer()
    controller = AdaptiveActorLearningRate()
    controller.begin_update(opt)
    observe(controller)
    observe(controller, card=0.03, rejected=True)
    assert opt.param_groups[0]['lr'] == 1e-5
    event = controller.finish_update(opt, actor_enabled=True, committed_steps=1)
    assert event['reason'] == 'kl_rejected'
    assert event['max_normalized_kl'] == pytest.approx(2)
    assert event['rejected_groups'] == 1 and event['rejected_samples'] == 64
    assert event['samples_observed'] == 128 and event['groups_observed'] == 2
    assert event['lr_used'] == 1e-5 and event['next_lr'] == 5e-6
    assert opt.param_groups[0]['lr'] == 5e-6
    assert not opt.state


def test_both_heads_drive_controller_without_changing_their_limits():
    controller = AdaptiveActorLearningRate()
    opt = optimizer()
    event = complete(controller, opt, card=0.001, tile=0.061, rejected=True)
    assert event['reason'] == 'kl_rejected' and event['next_lr'] == 5e-6
    event = complete(controller, opt, card=0.014, tile=0.001)
    assert event['reason'] == 'kl_near_limit' and event['next_lr'] == 2.5e-6
    assert complete(controller, opt, card=0.03, rejected=True)['next_lr'] == 2.5e-6


def test_increase_requires_two_clean_updates_and_obeys_original_lr_ceiling():
    controller = AdaptiveActorLearningRate()
    opt = optimizer(5e-6)
    first = complete(controller, opt)
    assert first['reason'] == 'low_kl_wait' and first['next_lr'] == 5e-6
    second = complete(controller, opt)
    assert second['reason'] == 'low_kl_increase' and second['next_lr'] == 6.25e-6
    for _ in range(20):
        event = complete(controller, opt)
    assert event['next_lr'] == 1e-5
    assert controller.config.max_lr == 1e-5


def test_too_little_evidence_never_changes_lr_or_accumulates_low_streak():
    controller = AdaptiveActorLearningRate()
    opt = optimizer()
    controller.begin_update(opt)
    observe(controller, card=0.15, samples=64, rejected=True)
    event = controller.finish_update(opt, actor_enabled=True, committed_steps=0)
    assert event['reason'] == 'insufficient_evidence' and event['next_lr'] == 1e-5
    controller.begin_update(opt)
    observe(controller, samples=4)
    observe(controller, samples=4)
    event = controller.finish_update(opt, actor_enabled=True, committed_steps=2)
    assert event['reason'] == 'insufficient_evidence' and event['low_kl_updates'] == 0


def test_overflow_and_disabled_actor_do_not_push_learning_rate_up():
    controller = AdaptiveActorLearningRate()
    opt = optimizer(5e-6)
    complete(controller, opt)
    event = complete(controller, opt, overflows=1)
    assert event['reason'] == 'amp_overflow' and event['next_lr'] == 5e-6
    assert event['low_kl_updates'] == 0
    controller.begin_update(opt)
    event = controller.finish_update(opt, actor_enabled=False, committed_steps=0)
    assert event['reason'] == 'actor_disabled' and event['next_lr'] == 5e-6
    assert event['samples_observed'] == 0


def test_json_state_resume_keeps_rate_and_low_kl_history():
    original = AdaptiveActorLearningRate()
    opt = optimizer(5e-6)
    complete(original, opt)
    saved = json.loads(json.dumps(original.state_dict()))
    restored = AdaptiveActorLearningRate()
    restored.load_state_dict(saved)
    restored_opt = optimizer(5e-6)
    assert complete(original, opt) == complete(restored, restored_opt)
    assert original.state_dict() == restored.state_dict()
    with pytest.raises(ValueError, match='disagree'):
        restored.begin_update(optimizer(1e-5))


def test_invalid_bounds_and_corrupt_resume_are_rejected():
    config = AdaptiveActorLRConfig()
    for changes in ({'min_lr': 0}, {'max_lr': 1e-7}, {'low_ratio': 2},
                    {'min_groups': 0}, {'min_samples': True}, {'decrease_factor': 1}):
        with pytest.raises(ValueError):
            replace(config, **changes)
    controller = AdaptiveActorLearningRate()
    state = controller.state_dict()
    state['config']['max_lr'] = 2e-5
    with pytest.raises(ValueError, match='settings differ'):
        controller.load_state_dict(state)
    controller.begin_update(optimizer())
    with pytest.raises(RuntimeError, match='unfinished'):
        controller.begin_update(optimizer())
    with pytest.raises(RuntimeError, match='unfinished'):
        controller.state_dict()
    with pytest.raises(FloatingPointError, match='Non-finite'):
        observe(controller, card=float('nan'))


def test_restore_defaults_are_opt_in_and_saved_ceiling_survives_a_lower_current_rate():
    assert restore_actor_lr_controller(optimizer()) is None
    original = restore_actor_lr_controller(optimizer(), enabled=True)
    opt = optimizer()
    complete(original, opt, card=0.03, rejected=True)
    saved = original.state_dict()
    restored = restore_actor_lr_controller(opt, saved)
    assert restored.state_dict() == saved
    assert restored.config.max_lr == 1e-5
    assert restored.current_lr == 5e-6
    assert complete(original, opt) == complete(restored, optimizer(5e-6))


def test_resume_conflicting_rate_or_bounds_fails_before_feedback():
    controller = restore_actor_lr_controller(optimizer(), enabled=True)
    saved = controller.state_dict()
    with pytest.raises(ValueError, match='disagree'):
        restore_actor_lr_controller(optimizer(5e-6), saved)
    with pytest.raises(ValueError, match='bounds differ'):
        restore_actor_lr_controller(optimizer(), saved, enabled=True, max_lr=2e-5)
    with pytest.raises(ValueError, match='outside'):
        restore_actor_lr_controller(optimizer(), enabled=True, max_lr=5e-6)


def test_minor_amp_overflow_does_not_block_low_kl_increase():
    controller = AdaptiveActorLearningRate()
    opt = optimizer(5e-6)
    for step_idx in range(2):
        controller.begin_update(opt)
        observe(controller)
        observe(controller)
        event = controller.finish_update(
            opt, actor_enabled=True, committed_steps=32, overflow_steps=3
        )
        expected_reason = 'low_kl_wait' if step_idx == 0 else 'low_kl_increase'
        assert event['reason'] == expected_reason
    assert event['next_lr'] == pytest.approx(6.25e-6)

    controller.begin_update(opt)
    observe(controller)
    observe(controller)
    severe = controller.finish_update(
        opt, actor_enabled=True, committed_steps=16, overflow_steps=5
    )
    assert severe['reason'] == 'amp_overflow'
    assert severe['low_kl_updates'] == 0


def test_restore_with_bound_override_clamps_lr_into_new_bounds():
    saved_ctrl = restore_actor_lr_controller(
        optimizer(1.875e-6), enabled=True, min_lr=1.5e-6, max_lr=5e-6
    )
    saved = saved_ctrl.state_dict()
    opt = optimizer(1.875e-6)
    restored = restore_actor_lr_controller(
        opt,
        saved,
        enabled=True,
        min_lr=2.5e-6,
        max_lr=5e-6,
        allow_bound_override=True,
    )
    assert restored.config.min_lr == pytest.approx(2.5e-6)
    assert restored.current_lr == pytest.approx(2.5e-6)
    assert opt.param_groups[0]['lr'] == pytest.approx(2.5e-6)
