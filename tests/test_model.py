import torch

from cr_rl.models import ActorCritic, ModelConfig


def test_tiny_forward_and_mask():
    cfg = ModelConfig.preset("tiny")
    model = ActorCritic(cfg)
    header = torch.zeros(2, 4, cfg.header_dim)
    entities = torch.zeros(2, 4, cfg.max_entities, cfg.entity_feature_dim)
    mask = torch.zeros(2, 4, cfg.max_entities, dtype=torch.bool)
    mask[..., 0] = True
    logits, values = model(header, entities, mask)
    assert logits.shape == (2, cfg.action_count)
    assert values.shape == (2,)
    dist = model.masked_categorical(logits)
    assert torch.isfinite(dist.log_prob(torch.zeros(2, dtype=torch.long))).all()

    hand_ids = torch.tensor([[3, 8, 21, 55], [7, 12, -1, 99]])
    logits_with_cards, _ = model(header, entities, mask, hand_card_ids=hand_ids)
    assert logits_with_cards.shape == (2, cfg.action_count)
    assert torch.isfinite(logits_with_cards).all()


def test_causal_temporal_attention():
    cfg = ModelConfig.preset("tiny")
    model = ActorCritic(cfg).eval()
    h = torch.zeros(1, 4, cfg.header_dim)
    e = torch.zeros(1, 4, cfg.max_entities, cfg.entity_feature_dim)
    m = torch.zeros(1, 4, cfg.max_entities, dtype=torch.bool)
    m[..., 0] = True
    with torch.no_grad():
        prefix_logits, _ = model(h[:, :3], e[:, :3], m[:, :3])
        full_logits, _ = model(h, e, m, target_index=2)
        h[:, 3, 0] = 1.0
        changed_future_logits, _ = model(h, e, m, target_index=2)
    torch.testing.assert_close(prefix_logits, full_logits)
    torch.testing.assert_close(full_logits, changed_future_logits)


def test_hierarchical_action_distribution_normalizes_with_legal_mask():
    card = torch.randn(2, 7)
    tile = torch.randn(2, 4, 576)
    legal = torch.zeros(2, 2307, dtype=torch.bool)
    legal[..., 0] = True  # WAIT
    legal[..., 1 + 2 * 576 + 17] = True  # one tile for hand slot 2
    joint = ActorCritic._joint_log_probs(card, tile, legal)
    probs = joint.exp()
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2))
    assert torch.isfinite(joint[..., 0]).all()
    assert torch.isfinite(joint[..., 1 + 2 * 576 + 17]).all()
    assert torch.all(joint[..., 1] < -1e8)


def test_deck_and_public_seen_card_embeddings_are_optional_and_finite():
    cfg = ModelConfig.preset("tiny")
    model = ActorCritic(cfg).eval()
    header = torch.zeros(2, 4, cfg.header_dim)
    entities = torch.zeros(2, 4, cfg.max_entities, cfg.entity_feature_dim)
    mask = torch.zeros(2, 4, cfg.max_entities, dtype=torch.bool)
    mask[..., 0] = True
    hand = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    deck = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23, 24, 25, 26, 27]])
    seen = torch.tensor([[30, 31, -1, -1, -1, -1, -1, -1], [-1] * 8])
    with torch.no_grad():
        baseline, _ = model(header, entities, mask, hand_card_ids=hand)
        conditioned, _ = model(
            header,
            entities,
            mask,
            hand_card_ids=hand,
            deck_card_ids=deck,
            enemy_seen_card_ids=seen,
        )
    assert torch.isfinite(conditioned).all()
    assert not torch.allclose(baseline, conditioned)
