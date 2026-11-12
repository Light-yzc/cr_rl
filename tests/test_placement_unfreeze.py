"""Tiny CPU correctness checks for GT-only BC and the four-save handoff."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.training.bc import hierarchical_bc_loss
from cr_rl.training.state_bc import prepare_state_chunk
from cr_rl.training.placement_stage import initialize_placement_stage
from cr_rl.training.placement_unfreeze import (
    arm_unfreeze, maybe_unfreeze, configure_training_mode, optimizer_parameters,
)
from test_placement_stage import fixture
from test_placement_stage import cli_module


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_gt_only_training_keeps_loss_and_every_parameter_gradient(tmp_path):
    _, dataset, split, args = fixture(tmp_path)
    model, _ = initialize_placement_stage(args, dataset, split)
    model.requires_grad_(True)
    torch.manual_seed(43)
    with torch.no_grad():
        model.learned_placement.decoder.original.original.tile_actor[-1].weight.normal_(0, .05)
        model.learned_placement.decoder.spatial.output.weight.normal_(0, .05)
    fast = deepcopy(model)
    model.train(); fast.train()
    iterator = dataset.episodes(split, 'train')
    try:
        episode = next(iterator)
        inputs = prepare_state_chunk(episode, episode.rows, CrForgeObservationAdapter(dataset.schema, 16), 'cpu')
        labels = torch.as_tensor(episode.labels.copy())
        selected = torch.as_tensor(episode.supervised.copy())
        placement = selected & (labels > 0) & (labels < 2305)
        options = dict(compute_value=False, placement_mask=placement)
        queries = []
        handle = fast.learned_placement.decoder.original.original.tile_cross_attention.register_forward_pre_hook(
            lambda module, values: queries.append(values[0].shape[1]))
        normal = model.forward_state(**inputs, **options)[0]
        optimized = fast.forward_state(**inputs, **options,
            placement_slots=((labels - 1) // 576).clamp(0, 3))[0]
        handle.remove()
        assert queries and set(queries) == {576}
        loss_options = dict(conditional_card_loss=True, position_coef=1., tile_label_sigma=1.,
                            timing_prior=.2, sampled_action_prior=6/11)
        a = hierarchical_bc_loss(normal[selected], labels[selected], inputs['legal_action_mask'][selected], **loss_options)[0]
        b = hierarchical_bc_loss(optimized[selected], labels[selected], inputs['legal_action_mask'][selected], **loss_options)[0]
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        a.backward(); b.backward()
        for (name, left), (_, right) in zip(model.named_parameters(), fast.named_parameters(), strict=True):
            ga = torch.zeros_like(left) if left.grad is None else left.grad
            gb = torch.zeros_like(right) if right.grad is None else right.grad
            torch.testing.assert_close(ga, gb, atol=1e-5, rtol=3e-4, msg=name)
    finally:
        iterator.close()


def test_four_checkpoint_handoff_preserves_head_adam_and_resumes_groups(tmp_path):
    _, dataset, split, args = fixture(tmp_path)
    model, stage = initialize_placement_stage(args, dataset, split)
    definition = {'placement_stage': stage}
    args.unfreeze_after_checkpoints = 4
    args.unfreeze_learning_rate = 1e-5
    args.checkpoint_every_episodes = 512
    args.freeze_frame = False
    args.lr_schedule = 'constant'
    arm_unfreeze(args, definition)
    optimizer = torch.optim.AdamW(optimizer_parameters(model, stage), lr=3e-4)
    parameter = next(p for p in model.learned_placement.parameters() if p.requires_grad)
    parameter.sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    before = deepcopy(optimizer.state[parameter])
    options = dict(epoch=0, train_seats=20446, optimizer_steps=256)
    assert maybe_unfreeze(model, optimizer, definition, next_episode=2040, **options) is None
    record = maybe_unfreeze(model, optimizer, definition, next_episode=2048, **options)
    assert record['head_adam_preserved'] and len(optimizer.param_groups) == 2
    assert model.frame.entity_in.weight.requires_grad
    assert model.wait_actor.weight.requires_grad
    assert all(group['lr'] == 1e-5 for group in optimizer.param_groups)
    for key, value in before.items():
        torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
    assert maybe_unfreeze(model, optimizer, definition, next_episode=2560, **options) is None
    # Constructing the resumed optimizer must preserve exact group order/sizes.
    restored = deepcopy(model)
    configure_training_mode(restored, stage)
    resumed = torch.optim.AdamW(optimizer_parameters(restored, stage), lr=3e-4)
    resumed.load_state_dict(optimizer.state_dict())
    assert [len(g['params']) for g in resumed.param_groups] == [len(g['params']) for g in optimizer.param_groups]
    assert [g['lr'] for g in resumed.param_groups] == [1e-5, 1e-5]


def test_unfreeze_schedule_can_be_added_explicitly_but_not_changed():
    args = SimpleNamespace(unfreeze_after_checkpoints=4, unfreeze_learning_rate=1e-5,
        checkpoint_every_episodes=512, freeze_frame=False, lr_schedule='constant')
    definition = {'placement_stage': {'mode':'placement_only'}}
    state = {'training_config':deepcopy(definition)}
    arm_unfreeze(args, definition, state)
    assert state['training_config'] == definition
    args.unfreeze_after_checkpoints = 3
    with pytest.raises(ValueError, match='preserve'):
        arm_unfreeze(args, definition, state)


def test_real_cli_joint_transition_and_resume_preserve_cursor_and_optimizer(tmp_path):
    import json
    original, dataset, split, args = fixture(tmp_path)
    module = cli_module()
    registry = tmp_path/'registry.json'
    registry.write_text(json.dumps({'assignments':{'game-0':'train','game-1':'validation','game-2':'heldout'}}))
    out = tmp_path/'joint'
    base = [str(dataset.path),'--out',str(out),'--device','cpu','--precision','fp32',
        '--epochs','4','--max-episodes','5','--max-validation-episodes','1','--cpu-threads','1',
        '--chunk-size','2','--frame-microbatch','2','--checkpoint-every-episodes','1',
        '--validation-every-episodes','0','--no-initial-validation','--split-registry',str(registry),
        '--learning-rate','0.0003','--lr-schedule','constant','--unfreeze-after-checkpoints','4']
    first = module.run(module.parse_args(base+['--upgrade-placement-from',str(args.upgrade_placement_from)]))
    assert first['optimizer_steps'] == 5 and first['epoch'] == 2 and first['next_episode'] == 1
    saved = torch.load(out/'latest.pt', weights_only=False, map_location='cpu')
    stage = saved['training_config']['placement_stage']
    assert stage['mode'] == 'full_finetune'
    assert stage['unfreeze']['transition']['training_seats'] == 4
    assert len(saved['optimizer']['param_groups']) == 2
    assert not torch.equal(saved['model']['wait_actor.weight'], original.wait_actor.weight)
    base[base.index('--max-episodes')+1] = '1'
    second = module.run(module.parse_args(base+['--resume',str(out/'latest.pt')]))
    assert second['optimizer_steps'] == 6 and second['epoch'] == 3 and second['next_episode'] == 0
    resumed = torch.load(out/'latest.pt', weights_only=False, map_location='cpu')
    assert resumed['training_config']['placement_stage'] == stage
    assert [g['lr'] for g in resumed['optimizer']['param_groups']] == [1e-5,1e-5]
