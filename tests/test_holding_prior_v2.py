from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_elixir_holding_prior import tiny_config

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorPolicy, ModelConfig
from cr_rl.models.holding_prior import ElixirHoldingPrior
from cr_rl.models.incremental_frames import IncrementalFrameHistory
from cr_rl.training.inference import FrozenInference, history_tensors
from cr_rl.training.separate import forward_inputs


def fixture(count=1):
    cfg = replace(tiny_config(), elixir_wait_bias=5, holding_prior_version=2,
                  core_card_ids=(10, 11), tower_card_ids=(2, 3), core_slot_bias=.7)
    prior = ElixirHoldingPrior(cfg)
    header = torch.zeros(count, cfg.header_dim)
    header[:, prior.header['elixir'][0]] = .3
    header[:, prior.header['hand_card']] = torch.tensor([1, 4, 10, 11]) / cfg.vocab_size
    header[:, prior.header['hand_cost']] = torch.tensor([.2, .4, .5, .6])
    entities = torch.zeros(count, 4, cfg.entity_feature_dim)
    mask = torch.zeros(count, 4, dtype=torch.bool)
    mask[:, 0] = True
    entities[:, 0, prior.entity['row']] = 2 / cfg.vocab_size
    entities[:, 0, prior.entity['hp']] = 1
    entities[:, 0, prior.entity['kind']] = 2 / 8
    entities[:, 0, prior.entity['x']] = .5
    entities[:, 0, prior.entity['y']] = .1
    legal = torch.ones(count, 2307, dtype=torch.bool)
    return cfg, prior, header, entities, mask, legal


def test_disable_prior_preserves_weights_and_unbiased_distribution(tmp_path):
    torch.set_num_threads(1)
    cfg, _, h, e, m, legal = fixture()
    source = ActorPolicy(cfg).eval()
    clean_cfg = cfg.without_holding_prior()
    assert clean_cfg.elixir_wait_bias == clean_cfg.core_slot_bias == 0
    assert not clean_cfg.core_card_ids and not clean_cfg.tower_card_ids
    clean = ActorPolicy(clean_cfg).eval()
    clean.load_state_dict(source.state_dict(), strict=True)
    assert clean.holding_prior is None
    # A legacy checkpoint with a zero WAIT strength is the established unbiased path.
    baseline = ActorPolicy(replace(cfg, holding_prior_version=1, elixir_wait_bias=0)).eval()
    baseline.load_state_dict(source.state_dict(), strict=True)
    h = h[:, None].expand(-1, cfg.context_len, -1)
    e = e[:, None].expand(-1, cfg.context_len, -1, -1)
    m = m[:, None].expand(-1, cfg.context_len, -1)
    times = torch.ones(1, cfg.context_len, dtype=torch.bool)
    inputs = h, e, m, legal, times
    with torch.no_grad():
        actual, _ = clean(*inputs)
        expected, _ = baseline(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    path = tmp_path / 'clean.pt'
    torch.save(dict(config=clean_cfg.__dict__, model=clean.state_dict()), path)
    saved = torch.load(path, weights_only=False)
    exported = ActorPolicy(ModelConfig(**saved['config'])).eval()
    exported.load_state_dict(saved['model'], strict=True)
    assert exported.holding_prior is None
    with torch.no_grad():
        restored, _ = exported(*inputs)
    torch.testing.assert_close(restored, actual, atol=0, rtol=0)


def test_pressure_uses_alive_tower_distance_motion_and_attack_target():
    _, prior, _, e, m, _ = fixture(7)
    m[1:, 1] = True
    for name, value in [('side', .5), ('kind', 1/8), ('hp', .0001), ('x', .5), ('y', .5)]:
        e[1:, 1, prior.entity[name]] = value
    e[2, 1, prior.entity['y']] = .23  # Low HP at tower is still urgent.
    e[3:5, 1, prior.entity['y']] = .3
    e[3:5, 1, prior.entity['state']] = 1/16
    e[3:5, 1, prior.entity['target_x']] = .5
    e[3, 1, prior.entity['target_y']] = .1
    e[4, 1, prior.entity['target_y']] = .6  # Retreating, outside immediate range.
    e[5, 1, prior.entity['y']] = .35
    e[5, 1, prior.entity['state']] = 2/16
    e[5, 1, prior.entity['target_x']] = .5
    e[5, 1, prior.entity['target_y']] = .1
    e[6, 0, prior.entity['hp']] = 0  # Missing live towers fails conservatively.
    torch.testing.assert_close(
        prior.pressure_scale(e, m), torch.tensor([1., .4, 0., 0., .4, 0., 0.])
    )


def test_ready_core_releases_wait_and_respects_pending_masks_full_elixir():
    _, p, h, e, m, legal = fixture(6)
    h[1:, p.header['elixir'][0]] = .5
    h[2, p.header['hand_pending'][2]] = 1
    legal[3, 1 + 2*576:1 + 3*576] = False
    h[4, p.header['elixir'][0]] = 1
    h[5, p.header['hand_card']] = 1 / p.vocab_size  # No listed core.
    bias = p.action_bias(h, e, m, legal)
    torch.testing.assert_close(bias[:, 0], torch.tensor([5., 0., 5., 5., 0., 0.]))
    torch.testing.assert_close(bias[:, 3], torch.tensor([0., .7, 0., 0., .7, 0.]))
    assert not bias[:, 5:].any()  # No ability bonus.
    h[0, p.header['hand_cost'][1]] = .3
    assert p.action_bias(h, e, m, legal)[0, 0] == 5  # Cheap support unlocking does not dilute.


def test_new_policy_export_and_training_use_same_bias(tmp_path):
    torch.set_num_threads(1)
    cfg, p, h, e, m, legal = fixture()
    model = ActorPolicy(cfg).eval()
    legacy = ActorPolicy(replace(cfg, holding_prior_version=1, elixir_wait_bias=0)).eval()
    legacy.load_state_dict(model.state_dict(), strict=True)
    h[:, p.header['elixir'][0]] = .5
    hand = torch.tensor([[1, 4, 10, 11]])
    history_h = h[:, None].expand(-1, cfg.context_len, -1)
    history_e = e[:, None].expand(-1, cfg.context_len, -1, -1)
    history_m = m[:, None].expand(-1, cfg.context_len, -1)
    times = torch.ones(1, cfg.context_len, dtype=torch.bool)
    args = history_h, history_e, history_m, legal, times
    with torch.no_grad():
        actual, _ = model(*args, hand_card_ids=hand)
        baseline, _ = legacy(*args, hand_card_ids=hand)
    # Conditional slot odds receive precisely +.7, leaving the tile conditional intact.
    new_cards = actual[:, 1:2305].exp().reshape(1, 4, 576).sum(-1)
    old_cards = baseline[:, 1:2305].exp().reshape(1, 4, 576).sum(-1)
    torch.testing.assert_close((new_cards[:, 2]/new_cards[:, 0])/(old_cards[:, 2]/old_cards[:, 0]),
                               torch.tensor([.7]).exp(), atol=1e-5, rtol=1e-5)
    path = tmp_path/'policy.pt'
    torch.save(dict(config=cfg.__dict__, model=model.state_dict()), path)
    saved = torch.load(path, weights_only=False)
    restored = ActorPolicy(ModelConfig(**saved['config']))
    restored.load_state_dict(saved['model'])
    restored.train()
    trained, _ = restored(*args, hand_card_ids=hand)
    torch.testing.assert_close(trained, actual)


def test_v2_cached_and_incremental_inference_matches_full_history():
    torch.set_num_threads(1)
    cfg, prior, _, _, _, _ = fixture()
    schema = SimpleNamespace(header_fields=cfg.header_fields, entity_fields=cfg.entity_fields,
        header=prior.header | {'entity_count': [cfg.header_fields.index('entity_count')]},
        header_ints=cfg.header_dim, entity_ints=cfg.entity_feature_dim, max_entities=4,
        seat_ints=cfg.header_dim+4*cfg.entity_feature_dim)
    # Inference needs deck/seen fields as well as the prior's fields.
    schema.header = {n: tuple(i for i, f in enumerate(cfg.header_fields) if f == n)
                     for n in set(cfg.header_fields)}
    adapter = CrForgeObservationAdapter(schema, cfg.vocab_size)
    model = ActorPolicy(cfg).eval()
    engine = FrozenInference(model, adapter, 'cpu')
    ring = IncrementalFrameHistory(1, cfg.context_len, schema.seat_ints)
    raw = np.zeros((1, schema.seat_ints), dtype=np.int32)
    raw[:, schema.header['entity_count']] = 1
    raw[:, schema.header['hand_card']] = [1, 4, 10, 11]
    raw[:, schema.header['hand_cost']] = [2, 4, 5, 6]
    tower = raw[:, cfg.header_dim:].reshape(1, 4, cfg.entity_feature_dim)
    for name, value in [('row', 2), ('kind', 2), ('hp', 1000), ('x', 9000), ('y', 3200)]:
        tower[:, 0, cfg.entity_fields.index(name)] = value
    legal = np.ones((1, 2307), dtype=bool)
    frames = []
    for step in range(6):
        if step == 3:
            frames.clear()
        raw[:, schema.header['elixir']] = 30000 if step % 2 else 50000
        ring.append(raw, np.array([int(step >= 3)]))
        frames.append(raw[0].copy())
        windows = np.zeros((1, cfg.context_len, schema.seat_ints), dtype=np.int32)
        windows[0, -len(frames):] = np.stack(frames)
        times = np.arange(cfg.context_len)[None] >= cfg.context_len-len(frames)
        with torch.no_grad():
            full, _ = forward_inputs(model, history_tensors(windows, times, legal, adapter, 'cpu'))
        torch.testing.assert_close(
            engine.evaluate(windows, times, legal), full, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            engine.evaluate_incremental(ring.request(), legal), full, atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize('field,value', [('core_slot_bias', float('nan')), ('core_slot_bias', 3),
                                         ('holding_prior_version', 3), ('core_card_ids', (-1,))])
def test_invalid_configuration_fails(field, value):
    with pytest.raises(ValueError):
        replace(tiny_config(), **{field: value})
