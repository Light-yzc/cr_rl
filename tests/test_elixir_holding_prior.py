from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorPolicy, ModelConfig
from cr_rl.models.holding_prior import ElixirHoldingPrior
from cr_rl.models.incremental_frames import IncrementalFrameHistory
from cr_rl.training.inference import FrozenInference, history_tensors
from cr_rl.training.separate import forward_inputs

ENTITY_FIELDS = (
    "id",
    "kind",
    "side",
    "row",
    "x",
    "y",
    "hp",
    "max_hp",
    "shield",
    "state",
    "flags",
    "level",
    "dx",
    "dy",
    "target",
    "target_x",
    "target_y",
    "deploy_ms",
    "attack_ms",
    "load_ms",
    "speed_percent",
    "hit_speed_percent",
    "buff_count",
    "buff_0",
    "buff_ms_0",
    "buff_1",
    "buff_ms_1",
    "buff_2",
    "buff_ms_2",
    "aim_x",
    "aim_y",
    "time_left_ms",
    "radius",
    "ability_ticks",
)
HEADER_FIELDS = (
    "tick",
    "result",
    "elixir",
    "elixir_reserved",
    "crowns",
    "crowns_enemy",
    "hand_refill_ms",
    "hand_card",
    "hand_card",
    "hand_card",
    "hand_card",
    "hand_cost",
    "hand_cost",
    "hand_cost",
    "hand_cost",
    "hand_pending",
    "hand_pending",
    "hand_pending",
    "hand_pending",
    "next_card",
    *(["deck"] * 8),
    *(["enemy_seen"] * 8),
    "entity_count",
)


def tiny_config():
    return ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=16,
        temporal_heads=2,
        temporal_layers=1,
        context_len=4,
        header_dim=len(HEADER_FIELDS),
        entity_feature_dim=len(ENTITY_FIELDS),
        entity_fields=ENTITY_FIELDS,
        header_fields=HEADER_FIELDS,
    ).for_v2(
        SimpleNamespace(
            entity_fields=ENTITY_FIELDS,
            header_fields=HEADER_FIELDS,
        )
    )


def state(cfg, count=1):
    header = torch.zeros(count, cfg.header_dim)
    entities = torch.zeros(count, 2, cfg.entity_feature_dim)
    mask = torch.ones(count, 2, dtype=torch.bool)
    prior = ElixirHoldingPrior(cfg)
    header[:, prior.header["elixir"][0]] = 0.25
    header[:, prior.header["hand_cost"]] = torch.tensor([0.2, 0.3, 0.5, 0.6])
    return prior, header, entities, mask


def test_holding_bias_requires_unaffordable_core_and_disables_for_threat_or_full_elixir():
    cfg = replace(tiny_config(), elixir_wait_bias=3)
    prior, h, e, m = state(cfg, 4)
    h[1, prior.header["elixir"][0]] = 0.6
    h[2, prior.header["elixir"][0]] = 1
    e[3, 0, prior.entity["side"]] = 0.5
    e[3, 0, prior.entity["kind"]] = 1 / 8
    e[3, 0, prior.entity["y"]] = 0.4
    e[3, 0, prior.entity["hp"]] = 0.1
    torch.testing.assert_close(prior(h, e, m), torch.tensor([3.0, 0.0, 0.0, 0.0]))
    h[0, prior.header["elixir"][0]] = 0.6
    h[0, prior.header["elixir_reserved"][0]] = 0.2
    assert float(prior(h, e, m)[0]) == 3
    h[0, prior.header["hand_pending"][2:]] = 1
    assert float(prior(h, e, m)[0]) == 0


def test_prior_is_consistent_across_rollout_training_cache_and_saved_policy(tmp_path):
    torch.set_num_threads(1)
    cfg = replace(tiny_config(), elixir_wait_bias=3)
    model = ActorPolicy(cfg).eval()
    baseline = ActorPolicy(replace(cfg, elixir_wait_bias=0)).eval()
    baseline.load_state_dict(model.state_dict(), strict=True)
    assert model.parameter_count() == baseline.parameter_count()
    header = torch.zeros(1, cfg.context_len, cfg.header_dim)
    header[..., cfg.header_fields.index("elixir")] = 0.25
    header[
        ...,
        list(range(cfg.header_fields.index("hand_card"), cfg.header_fields.index("hand_card") + 4)),
    ] = 1
    header[
        ...,
        list(range(cfg.header_fields.index("hand_cost"), cfg.header_fields.index("hand_cost") + 4)),
    ] = torch.tensor([0.2, 0.3, 0.5, 0.6])
    entities = torch.zeros(1, cfg.context_len, 2, cfg.entity_feature_dim)
    masks = torch.zeros(1, 2307, dtype=torch.bool)
    masks[:, 0] = True
    masks[:, 1:3] = True
    entity_mask = torch.ones(1, cfg.context_len, 2, dtype=torch.bool)
    time_mask = torch.ones(1, cfg.context_len, dtype=torch.bool)
    hand = torch.zeros(1, 4, dtype=torch.long)
    cards = torch.zeros(1, 8, dtype=torch.long)
    args = (header, entities, entity_mask, masks, time_mask)
    kwargs = dict(hand_card_ids=hand, deck_card_ids=cards, enemy_seen_card_ids=cards)
    with torch.no_grad():
        actual, _ = model(*args, **kwargs)
        original, _ = baseline(*args, **kwargs)
    p, q = original.exp()[0, 0], actual.exp()[0, 0]
    torch.testing.assert_close(
        q / (1 - q) / (p / (1 - p)), torch.exp(torch.tensor(3.0)), rtol=1e-4, atol=1e-4
    )
    path = tmp_path / "actor.pt"
    torch.save(dict(config=model.cfg.__dict__, model=model.state_dict()), path)
    saved = torch.load(path, weights_only=False)
    restored = ActorPolicy(ModelConfig(**saved["config"])).eval()
    restored.load_state_dict(saved["model"], strict=True)
    with torch.no_grad():
        restored_logits, _ = restored(*args, **kwargs)
    torch.testing.assert_close(restored_logits, actual, atol=1e-5, rtol=1e-5)

    schema = SimpleNamespace(
        header_fields=HEADER_FIELDS,
        entity_fields=ENTITY_FIELDS,
        header_ints=len(HEADER_FIELDS),
        entity_ints=len(ENTITY_FIELDS),
        max_entities=2,
        seat_ints=len(HEADER_FIELDS) + 2 * len(ENTITY_FIELDS),
        header={
            name: tuple(i for i, field in enumerate(HEADER_FIELDS) if field == name)
            for name in set(HEADER_FIELDS)
        },
    )
    adapter = CrForgeObservationAdapter(schema, entity_bucket=1)
    raw = np.zeros((1, schema.seat_ints), dtype=np.int32)
    raw[:, list(schema.header["elixir"])] = 25000
    raw[:, list(schema.header["hand_cost"])] = [2, 3, 5, 6]
    raw[:, list(schema.header["hand_card"])] = [0, 0, 0, 0]
    legal = masks.numpy()
    ring = IncrementalFrameHistory(1, cfg.context_len, schema.seat_ints)
    engine = FrozenInference(restored, adapter, "cpu")
    window = deque(maxlen=cfg.context_len)
    for step in range(6):
        if step == 3:
            window.clear()
        ring.append(raw, np.array([int(step >= 3)]))
        window.append(raw[0].copy())
        windows = np.zeros((1, cfg.context_len, schema.seat_ints), dtype=np.int32)
        windows[0, -len(window) :] = np.stack(window)
        times = np.arange(cfg.context_len)[None] >= cfg.context_len - len(window)
        tensors = history_tensors(windows, times, legal, adapter, "cpu")
        with torch.no_grad():
            full, _ = forward_inputs(restored, tensors)
        cached = engine.evaluate(windows, times, legal)
        incremental = engine.evaluate_incremental(ring.request(), legal)
        torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
        restored.train()
        trained, _ = forward_inputs(restored, tensors)
        torch.testing.assert_close(trained, full, atol=1e-5, rtol=1e-5)
        restored.eval()
