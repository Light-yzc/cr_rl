"""Small CPU compatibility/strict-loading checks; no data/checkpoint downloads."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cr_rl.models.learned_placement import (DECODER_PAYLOAD_FORMAT, PlacementAdapter,
    attach_learned_placement, definition_for, placement_from_decoder_payload)
from cr_rl.models.state_policy import StateActorCritic, state_model_from_checkpoint
from cr_rl.models.state_temporal import StateTemporalConfig
from test_representation_v2 import configuration


SHA = 'a'*64


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def source_model():
    torch.manual_seed(332)
    cfg = replace(configuration(), readout_version=2, tile_query_residual=True,
        spatial_fourier=True, tile_head_fp32=True, vocab_size=16)
    temporal = StateTemporalConfig(width=16, heads=2, kv_heads=1, head_dim=8,
        pattern=('kda','gqa'), ffn_multiplier=1., chunk_size=2)
    return StateActorCritic(cfg, temporal).eval()


def raw_inputs():
    torch.manual_seed(42)
    return dict(context=torch.randn(2,16), entities=torch.randn(2,4,16),
        entity_mask=torch.tensor([[True,True,False,False],[True,True,True,False]]),
        card_features=torch.randn(2,4,16), entity_xy=torch.rand(2,4,2),
        card_ids=torch.tensor([[0,1,2,3],[4,5,6,-1]]))


def payload_for(source, head, variant):
    return {'format':DECODER_PAYLOAD_FORMAT,'source_checkpoint_sha256':SHA,
        'contains_trunk':False,'contains_optimizer':False,'heldout_used_for_selection':False,
        'model_config':source.checkpoint_config(),'variant':variant,
        'architecture_config':{'entity_dim':16,'temporal_dim':16,'readout_version':2,
            'width':32,'cnn_channels':32,'alpha':.01 if variant in ('card_prior_001','card_prior_cnn_001') else
                1. if variant=='card_prior_1' else None},
        'parameter_count':sum(p.numel() for p in head.parameters()),
        'decoder':{key:value.detach().clone() for key,value in head.state_dict().items()}}


def test_state_policy_dispatch_and_strict_checkpoint_roundtrip():
    model = source_model()
    adapter = PlacementAdapter(model, definition_for(model, 'card_prior_cnn_001', cnn_channels=32))
    # A nonconstant trained table makes accidental dispatch to the old head visible.
    with torch.no_grad():
        adapter.decoder.original.card_location_bias.weight.normal_(0., .2)
        adapter.decoder.spatial.output.weight.normal_(0., .01)
    model.add_module('learned_placement', adapter)
    model.learned_placement_definition = adapter.definition
    b, t, n = 1, 3, 4
    header = torch.zeros(b, t, model.cfg.header_dim)
    entities = torch.zeros(b, t, n, model.cfg.entity_feature_dim)
    entities[..., 4:6] = .3
    mask = torch.ones(b, t, n, dtype=torch.bool)
    legal = torch.ones(b, t, 2307, dtype=torch.bool)
    hands = torch.tensor([[[0, 1, 2, 3]]]).expand(b, t, 4)
    with torch.inference_mode():
        expected = model.forward_state(header, entities, mask, legal, hand_card_ids=hands)[0]
    checkpoint = {**model.checkpoint_config(), 'model': model.state_dict()}
    restored = state_model_from_checkpoint(checkpoint, 'cpu').eval()
    with torch.inference_mode():
        actual = restored.forward_state(header, entities, mask, legal, hand_card_ids=hands)[0]
    assert restored.checkpoint_config() == model.checkpoint_config()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    # Omitting the versioned definition must fail rather than discard learned tensors.
    incomplete = {key: value for key, value in checkpoint.items() if key != 'learned_placement_definition'}
    with pytest.raises(RuntimeError, match='Unexpected key'):
        state_model_from_checkpoint(incomplete, 'cpu')


def diagnostic_head(source, variant, monkeypatch):
    # Import scripts only in compatibility tests; production module has no
    # dependency on them. These are the exact local diagnostic checkpoint keys.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from probe_state_bc_head_capacity import TrainablePlacementHead, WiderResidualReadout, CardPriorSpatialHead
    from state_bc_card_location_probe import CardLocationPriorResidual
    from state_bc_spatial_probe import EntitySpatialResidual
    original = TrainablePlacementHead(source)
    mass = [0.]*576
    mass[8],mass[32] = 2.,1.
    prior = {'mass':{'2':mass},'card_counts':{'2':3},'fit_games':1,'sigma':1.}
    if variant == 'wide_residual':
        original.tile_actor = WiderResidualReadout(original.tile_actor,16,32)
        return original
    if variant == 'spatial_cnn':
        return EntitySpatialResidual(original,entity_dim=16,context_dim=16,card_dim=16,channels=32)
    if variant == 'card_prior_cnn_001':
        return CardPriorSpatialHead(original,prior,16,32)
    if variant.startswith('card_prior'):
        return CardLocationPriorResidual(original,prior,vocab_size=16,alpha=.01 if variant=='card_prior_001' else 1.)
    return original


@pytest.mark.parametrize('variant',['original','wide_residual','spatial_cnn','card_prior_001','card_prior_1','card_prior_cnn_001'])
def test_exact_diagnostic_state_keys_and_gt_output_match(variant,monkeypatch):
    source=source_model()
    head=diagnostic_head(source,variant,monkeypatch).eval()
    # Exercise nonzero trained-style residuals, rather than only zero-output
    # initialization where a wrong internal graph could go unnoticed.
    with torch.no_grad():
        for name,p in head.named_parameters():
            if name.endswith(('output.weight','output.bias')) or '.tile_actor.3.' in name:
                p.add_(torch.randn_like(p)*.01)
    payload=payload_for(source,head,variant)
    adapter=placement_from_decoder_payload(source,payload,source_checkpoint_sha256=SHA)
    assert set(adapter.decoder.state_dict())==set(head.state_dict())
    inputs=raw_inputs()
    slots=torch.tensor([2,1])
    gt_ids=inputs['card_ids'].gather(1,slots[:,None]).squeeze(1)
    expected_inputs={key:inputs[key] for key in ('context','entities','entity_mask','card_features')}
    expected_inputs['slots']=slots
    if variant in ('spatial_cnn','card_prior_cnn_001'):
        expected_inputs['entity_xy']=inputs['entity_xy']
    if variant.startswith('card_prior'):
        expected_inputs['card_ids']=gt_ids
    with torch.no_grad():
        expected=head(**expected_inputs)
        actual=adapter.forward_gt(**inputs,slots=slots)
        all_slots=adapter(**inputs)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(all_slots[torch.arange(2),slots],expected,atol=2e-6,rtol=2e-6)
    assert all_slots.shape==(2,4,576) and torch.isfinite(all_slots).all()
    if variant.startswith('card_prior'):
        # GT scalar identity path and all-hand path must mean the same thing.
        gt_inputs={**inputs,'card_ids':gt_ids}
        torch.testing.assert_close(adapter.forward_gt(**gt_inputs,slots=slots),actual,atol=0,rtol=0)


def test_all_slot_inference_preserves_card_wait_mass_and_legality(monkeypatch):
    source=source_model()
    head=diagnostic_head(source,'card_prior_cnn_001',monkeypatch)
    adapter=placement_from_decoder_payload(source,payload_for(source,head,'card_prior_cnn_001'),source_checkpoint_sha256=SHA)
    inputs=raw_inputs()
    card_logits=torch.randn(2,7)
    legal=torch.ones(2,2307,dtype=torch.bool)
    legal[:,10:30]=False
    legal[0,1:]=False
    before=source._joint_log_probs(card_logits,torch.randn(2,4,576),legal)
    after=source._joint_log_probs(card_logits,adapter(**inputs),legal)
    def masses(lp):
        return torch.cat((lp[:,:1].exp(),lp[:,1:2305].exp().reshape(2,4,576).sum(-1),lp[:,2305:].exp()),-1)
    torch.testing.assert_close(masses(before),masses(after),atol=2e-7,rtol=1e-6)
    assert after.exp()[~legal].sum()==0 and after[0,0].exp()==1


@pytest.mark.parametrize('corruption',['source','config','shape','missing','extra','nonfinite','version','count','heldout'])
def test_strict_loading_rejects_mismatch_without_mutating_model(corruption,monkeypatch):
    source=source_model()
    head=diagnostic_head(source,'card_prior_cnn_001',monkeypatch)
    payload=payload_for(source,head,'card_prior_cnn_001')
    if corruption=='source':payload['source_checkpoint_sha256']='b'*64
    elif corruption=='config':payload['model_config']['config']['entity_heads']=1
    elif corruption=='shape':payload['decoder']['original.card_location_bias.weight']=torch.zeros(18,576)
    elif corruption=='missing':payload['decoder'].pop('spatial.output.bias')
    elif corruption=='extra':payload['decoder']['accidental.tensor']=torch.tensor(1.)
    elif corruption=='nonfinite':payload['decoder']['spatial.output.bias'].fill_(float('inf'))
    elif corruption=='version':payload['format']='unknown-v2'
    elif corruption=='count':payload['parameter_count']+=1
    elif corruption=='heldout':payload['heldout_used_for_selection']=True
    before={key:value.clone() for key,value in source.state_dict().items()}
    generation=source.memory_generation
    with pytest.raises((ValueError,RuntimeError)):
        attach_learned_placement(source,payload,source_checkpoint_sha256=SHA)
    assert not hasattr(source,'learned_placement') and generation==source.memory_generation
    for key,value in source.state_dict().items():
        assert torch.equal(value,before[key])


def test_attach_versioned_full_state_roundtrip_and_reject_wrong_extra_state(monkeypatch):
    source=source_model()
    head=diagnostic_head(source,'card_prior_cnn_001',monkeypatch)
    definition=definition_for(source,'card_prior_cnn_001',width=32,cnn_channels=32)
    adapter=attach_learned_placement(source,payload_for(source,head,'card_prior_cnn_001'),source_checkpoint_sha256=SHA)
    assert source.memory_generation==1 and source.learned_placement_definition==definition
    state=deepcopy(adapter.state_dict())
    restored=PlacementAdapter(source_model(),definition)
    restored.load_state_dict(state,strict=True)
    inputs=raw_inputs()
    torch.testing.assert_close(restored(**inputs),adapter(**inputs),atol=0,rtol=0)
    state['_extra_state']['format']='wrong'
    with pytest.raises(ValueError,match='definition'):
        restored.load_state_dict(state,strict=True)
    with pytest.raises(ValueError,match='already attached'):
        attach_learned_placement(source,payload_for(source,head,'card_prior_cnn_001'),source_checkpoint_sha256=SHA)


def test_missing_xy_or_raw_hands_rejected(monkeypatch):
    source=source_model()
    head=diagnostic_head(source,'card_prior_cnn_001',monkeypatch)
    adapter=placement_from_decoder_payload(source,payload_for(source,head,'card_prior_cnn_001'),source_checkpoint_sha256=SHA)
    inputs=raw_inputs()
    with pytest.raises(ValueError,match='entity_xy'):
        adapter(**{**inputs,'entity_xy':None})
    with pytest.raises(ValueError,match='hand card IDs'):
        adapter(**{**inputs,'card_ids':None})
    with pytest.raises(ValueError,match='outside'):
        adapter(**{**inputs,'card_ids':torch.full((2,4),16)})


def test_optional_positive_temperature_scales_only_placement_logits(monkeypatch):
    source=source_model()
    head=diagnostic_head(source,'card_prior_cnn_001',monkeypatch)
    payload=payload_for(source,head,'card_prior_cnn_001')
    baseline=placement_from_decoder_payload(source,payload,source_checkpoint_sha256=SHA)
    payload['temperature']=2.
    calibrated=placement_from_decoder_payload(source,payload,source_checkpoint_sha256=SHA)
    inputs=raw_inputs()
    before,after=baseline(**inputs),calibrated(**inputs)
    torch.testing.assert_close(after,before/2,atol=0,rtol=0)
    assert torch.equal(before.argmax(-1),after.argmax(-1))
    for invalid in (0.,-1.,float('nan'),float('inf'),True):
        payload['temperature']=invalid
        with pytest.raises(ValueError,match='temperature'):
            placement_from_decoder_payload(source,payload,source_checkpoint_sha256=SHA)
