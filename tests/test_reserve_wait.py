from dataclasses import asdict, replace

import numpy as np
import torch

from cr_rl.models.policy import ActorCritic
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.ppo import factorized_entropy, factorized_reverse_kl
from cr_rl.client.action_selection import select_live_action, spell_has_target
from cr_rl.client.schema_v4 import SchemaV4 as S
from test_state_policy import model_pair, frames


def test_reserved_mass_normalized_legal_and_differentiable():
    cards = torch.tensor([[0., 8., -2., 0., -2., 0., 0.]], requires_grad=True)
    tiles = torch.zeros(1,4,576,requires_grad=True)
    legal = torch.zeros(1,2307,dtype=torch.bool)
    legal[:,0] = True
    legal[:,1+2*576:1+3*576] = True
    present = torch.tensor([[True,False,True,False]])
    logp = ActorCritic._joint_log_probs(cards,tiles,legal,hand_present_mask=present)
    p = logp.exp()
    assert p[0,0] > .99
    assert p[:,1+2*576:1+3*576].sum() < .05
    torch.testing.assert_close(p.sum(-1),torch.ones(1))
    assert (logp[~legal] == -1e9).all()
    entropy = sum(factorized_entropy(logp))
    kl = sum(factorized_reverse_kl(logp,logp.detach()))
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(kl,torch.zeros_like(kl),atol=1e-6,rtol=0)
    (-logp[0,0]-.001*entropy.sum()).backward()
    assert torch.isfinite(cards.grad).all() and torch.isfinite(tiles.grad).all()
    assert cards.grad[0,1] < 0


def test_legacy_gate_migration_is_exact_and_checkpointing_matches_gradients():
    _,model = model_pair()
    data = frames(batch=1,length=3)
    h,e,m,legal,hands = data
    ckpt = {**model.checkpoint_config(), 'model': {k:v for k,v in model.state_dict().items() if not k.startswith('wait_hand_gate.')}}
    restored = state_model_from_checkpoint(ckpt)
    with torch.no_grad():
        expected = model.forward_state(h,e,m,legal,hand_card_ids=hands)[0]
        actual = restored.eval().forward_state(h,e,m,legal,hand_card_ids=hands)[0]
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        assert restored.wait_hand_gate(torch.randn(4,12)).eq(0).all()
    reference_grads = None
    for checkpointing in (False,True):
        restored.train(); restored.gradient_checkpointing = checkpointing
        restored.zero_grad(set_to_none=True)
        out = restored.forward_state(h,e,m,legal,hand_card_ids=hands)[0]
        loss = -out[...,100].mean()
        loss.backward()
        grads = {k:p.grad.clone() for k,p in restored.named_parameters() if p.grad is not None}
        if reference_grads is None:
            reference_grads = grads
        else:
            assert grads.keys() == reference_grads.keys()
            for k in grads:
                torch.testing.assert_close(grads[k],reference_grads[k],atol=1e-6,rtol=1e-5)


def test_live_confidence_and_spell_guard_in_seat_frame():
    obs = np.zeros(S.SEAT_INTS,dtype=np.int32)
    obs[S.H_ELIXIR] = 35000
    legal = np.zeros(S.ACTIONS,dtype=bool); legal[0] = True
    action = S.play(0,5,5); legal[action] = True
    p = np.zeros(S.ACTIONS);p[0] = .8;p[action] = .2
    assert select_live_action(p,legal,obs,('Giant',None,None,None))[0] == 0
    obs[S.H_ELIXIR] = 95000
    assert select_live_action(p,legal,obs,('Giant',None,None,None))[0] == action
    for spell in ('Zap','Arrows'):
        result,info = select_live_action(p,legal,obs,(spell,None,None,None))
        assert result == 0 and info['spell_vetoed']
    obs[S.H_ENTITY_COUNT] = 1
    row = obs[S.HEADER_INTS:S.HEADER_INTS+S.ENTITY_INTS]
    row[S.E_KIND] = S.KIND_TOWER;row[S.E_SIDE] = 1;row[S.E_HP] = 1000
    row[S.E_X] = 5500;row[S.E_Y] = 5500
    assert not spell_has_target(action,'Zap',obs)
    row[S.E_HP] = 450
    assert spell_has_target(action,'Zap',obs)
    row[S.E_HP] = 1000;row[S.E_KIND] = S.KIND_CHARACTER
    assert spell_has_target(action,'Zap',obs)
    row[S.E_SIDE] = 0
    assert not spell_has_target(action,'Zap',obs)


def test_optimizer_migration_preserves_decoder_moments_by_name():
    from cr_rl.training.state_selfplay import restore_actor_optimizer
    _,old=model_pair()
    old.add_module('learned_placement',torch.nn.Linear(32,7))
    del old.wait_hand_gate
    optimizer=torch.optim.AdamW(old.parameters(),lr=1e-4)
    for i,p in enumerate(old.parameters()):
        p.grad=torch.full_like(p,float(i+1)/100)
    optimizer.step()
    moments={name:optimizer.state[p]['exp_avg'].clone() for name,p in old.named_parameters()}
    _,new=model_pair()
    new.add_module('learned_placement',torch.nn.Linear(32,7))
    new.load_state_dict(old.state_dict())
    restored=torch.optim.AdamW(new.parameters(),lr=1e-4)
    restore_actor_optimizer(restored,optimizer.state_dict(),new)
    for name,p in new.named_parameters():
        if name.startswith('wait_hand_gate.'):
            assert not restored.state[p]
        else:
            torch.testing.assert_close(restored.state[p]['exp_avg'],moments[name])
        p.grad=torch.ones_like(p)
    restored.step()


def test_wait_gate_receives_detached_scores_and_normalized_cost_deficits():
    from test_representation_v2 import configuration
    cfg=replace(configuration(),header_fields=('elixir','elixir_reserved','hand_cost','hand_cost','hand_cost','hand_cost','hand_card','hand_card'))
    model=ActorCritic(cfg)
    captured=[]
    handle=model.wait_hand_gate.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    slots=torch.tensor([[8.,2.,1.,-1.]],requires_grad=True)
    hands=torch.tensor([[1,2,-1,3]])
    header=torch.tensor([[.35,.05,.6,.3,-.1,.9,0.,0.]])
    out=model.wait_hand_residual(header,slots,hands)
    features=captured[0]
    torch.testing.assert_close(features,torch.tensor([[8.,2.,0.,-1.,.3,0.,0.,.6,0.,1.,0.,0.]]),atol=1e-6,rtol=0)
    assert not features.requires_grad
    assert out.eq(0).all()
    out.sum().backward()
    assert slots.grad is None
    assert model.wait_hand_gate[-1].weight.grad is not None
    handle.remove()
