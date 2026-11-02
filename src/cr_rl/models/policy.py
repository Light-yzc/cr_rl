from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    entity_layers: int = 1
    entity_dim: int = 64
    entity_heads: int = 4
    temporal_layers: int = 2
    temporal_dim: int = 128
    temporal_heads: int = 4
    context_len: int = 32
    max_entities: int = 200
    header_dim: int = 224
    entity_feature_dim: int = 34
    action_count: int = 2307
    vocab_size: int = 1466

    @classmethod
    def preset(cls, name: str) -> ModelConfig:
        import yaml

        from cr_rl.config import ROOT

        raw = yaml.safe_load((ROOT / "configs/model.yaml").read_text())["presets"][name]
        return cls(**raw)


class EntityFrameEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.entity_in = nn.Linear(cfg.entity_feature_dim, cfg.entity_dim)
        layer = nn.TransformerEncoderLayer(
            cfg.entity_dim,
            cfg.entity_heads,
            4 * cfg.entity_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.entity_layers, enable_nested_tensor=False)
        self.header = nn.Sequential(
            nn.Linear(cfg.header_dim, cfg.entity_dim),
            nn.GELU(),
            nn.Linear(cfg.entity_dim, cfg.entity_dim),
        )
        self.norm = nn.LayerNorm(cfg.entity_dim)

    def forward(self, header: torch.Tensor, entities: torch.Tensor, entity_mask: torch.Tensor):
        x = self.encoder(self.entity_in(entities), src_key_padding_mask=~entity_mask.bool())
        denom = entity_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (x * entity_mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.norm(pooled + self.header(header)), x


class ActorCritic(nn.Module):
    """Structured frame encoder and causal temporal Transformer actor-critic."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.frame = EntityFrameEncoder(cfg)
        self.to_temporal = nn.Linear(cfg.entity_dim, cfg.temporal_dim)
        layer = nn.TransformerEncoderLayer(
            cfg.temporal_dim,
            cfg.temporal_heads,
            4 * cfg.temporal_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.temporal = nn.TransformerEncoder(
            layer, cfg.temporal_layers, enable_nested_tensor=False
        )
        self.position = nn.Parameter(torch.zeros(1, cfg.context_len, cfg.temporal_dim))
        # Hierarchical policy: select WAIT/card slot/ability, then a legal tile.
        self.card_embedding = nn.Embedding(cfg.vocab_size + 1, cfg.entity_dim)
        self.deck_context = nn.Linear(cfg.entity_dim, cfg.temporal_dim, bias=False)
        self.seen_context = nn.Linear(cfg.entity_dim, cfg.temporal_dim, bias=False)
        self.card_slot_embedding = nn.Parameter(torch.randn(1, 4, cfg.entity_dim) * 0.02)
        self.wait_actor = nn.Linear(cfg.temporal_dim, 1)
        self.slot_actor = nn.Sequential(
            nn.LayerNorm(cfg.temporal_dim + cfg.entity_dim),
            nn.Linear(cfg.temporal_dim + cfg.entity_dim, cfg.temporal_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.temporal_dim // 2, 1),
        )
        self.ability_actor = nn.Linear(cfg.temporal_dim, 2)
        self.tile_queries = nn.Parameter(torch.randn(1, 4, 18 * 32, cfg.entity_dim) * 0.02)
        self.slot_embedding = nn.Parameter(torch.randn(1, 4, 1, cfg.entity_dim) * 0.02)
        self.tile_context = nn.Linear(cfg.temporal_dim, cfg.entity_dim)
        self.tile_cross_attention = nn.MultiheadAttention(
            cfg.entity_dim, cfg.entity_heads, batch_first=True
        )
        self.tile_actor = nn.Sequential(
            nn.LayerNorm(cfg.entity_dim),
            nn.Linear(cfg.entity_dim, cfg.entity_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.entity_dim // 2, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(cfg.temporal_dim, cfg.temporal_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.temporal_dim // 2, 1),
        )
        nn.init.normal_(self.wait_actor.weight, std=0.01)
        nn.init.zeros_(self.wait_actor.bias)

    def forward(
        self,
        header,
        entities,
        entity_mask,
        legal_action_mask=None,
        time_mask=None,
        target_index=-1,
        hand_card_ids=None,
        deck_card_ids=None,
        enemy_seen_card_ids=None,
    ):
        """Encode a history window and emit one action distribution for its final frame."""
        batch, steps, count, features = entities.shape
        flat_entity_mask = entity_mask.reshape(batch * steps, count)
        frame, entity_tokens = self.frame(
            header.reshape(batch * steps, -1),
            entities.reshape(batch * steps, count, features),
            flat_entity_mask,
        )
        x = self.to_temporal(frame.reshape(batch, steps, -1)) + self.position[:, :steps]
        causal = torch.triu(torch.ones(steps, steps, device=x.device, dtype=torch.bool), diagonal=1)
        padding = ~time_mask.bool() if time_mask is not None else None
        x = self.temporal(x, mask=causal, src_key_padding_mask=padding)
        # History informs the temporal state; only the current frame needs spatial decoding.
        target_index = target_index % steps
        context = x[:, target_index]
        if deck_card_ids is not None:
            deck_ids = deck_card_ids.long()
            deck_valid = deck_ids >= 0
            deck_embeddings = self.card_embedding((deck_ids + 1).clamp(0, self.cfg.vocab_size))
            deck_pooled = (deck_embeddings * deck_valid.unsqueeze(-1)).sum(dim=1)
            deck_pooled = deck_pooled / deck_valid.sum(dim=1, keepdim=True).clamp_min(1)
            context = context + self.deck_context(deck_pooled)
        if enemy_seen_card_ids is not None:
            seen_ids = enemy_seen_card_ids.long()
            seen_valid = seen_ids >= 0
            seen_embeddings = self.card_embedding((seen_ids + 1).clamp(0, self.cfg.vocab_size))
            seen_pooled = (seen_embeddings * seen_valid.unsqueeze(-1)).sum(dim=1)
            seen_pooled = seen_pooled / seen_valid.sum(dim=1, keepdim=True).clamp_min(1)
            context = context + self.seen_context(seen_pooled)
        current_entities = entity_tokens.reshape(batch, steps, count, -1)[:, target_index]
        current_entity_mask = entity_mask[:, target_index].bool()
        if hand_card_ids is None:
            hand_card_ids = torch.full((batch, 4), -1, dtype=torch.long, device=x.device)
        card_indices = (hand_card_ids.long() + 1).clamp(0, self.cfg.vocab_size)
        card_features = self.card_embedding(card_indices) + self.card_slot_embedding
        context_slots = context.unsqueeze(1).expand(-1, 4, -1)
        slot_logits = self.slot_actor(torch.cat((context_slots, card_features), dim=-1)).squeeze(-1)
        card_logits = torch.cat(
            (self.wait_actor(context), slot_logits, self.ability_actor(context)), dim=-1
        )
        queries = self.tile_queries.expand(batch, -1, -1, -1)
        queries = (
            queries
            + self.slot_embedding
            + card_features[:, :, None, :]
            + self.tile_context(context)[:, None, None, :]
        )
        queries = queries.reshape(batch, 4 * 576, -1)
        tile_features, _ = self.tile_cross_attention(
            queries,
            current_entities,
            current_entities,
            key_padding_mask=~current_entity_mask,
            need_weights=False,
        )
        tile_logits = self.tile_actor(tile_features).squeeze(-1).reshape(batch, 4, 576)
        if legal_action_mask is None:
            legal_action_mask = torch.ones(
                batch, self.cfg.action_count, dtype=torch.bool, device=x.device
            )
        joint_log_probs = self._joint_log_probs(card_logits, tile_logits, legal_action_mask)
        return joint_log_probs, self.critic(context).squeeze(-1)

    @staticmethod
    def _joint_log_probs(
        card_logits: torch.Tensor, tile_logits: torch.Tensor, legal: torch.Tensor
    ) -> torch.Tensor:
        """Compose log p(slot) + log p(tile|slot) in the simulator's flat action codec."""
        if legal.shape[-1] < 2307:
            raise ValueError("Action mask is shorter than WAIT + four tile grids + two abilities")
        tile_legal = legal[..., 1:2305].reshape(*legal.shape[:-1], 4, 576)
        card_legal = torch.cat(
            (legal[..., :1], tile_legal.any(dim=-1), legal[..., 2305:2307]), dim=-1
        )
        if not torch.all(card_legal.any(dim=-1)):
            raise ValueError("Every state must have at least one legal action (WAIT)")
        card_logp = torch.log_softmax(card_logits.masked_fill(~card_legal, -1e9), dim=-1)
        tile_logp = torch.log_softmax(tile_logits.masked_fill(~tile_legal, -1e9), dim=-1)
        tile_logp = tile_logp.masked_fill(~tile_legal, -1e9)
        placement_logp = (card_logp[..., 1:5].unsqueeze(-1) + tile_logp).flatten(-2)
        return torch.cat((card_logp[..., :1], placement_logp, card_logp[..., 5:7]), dim=-1)

    @staticmethod
    def masked_categorical(log_probs: torch.Tensor):
        """Build a distribution from already-normalised joint action log probabilities."""
        return torch.distributions.Categorical(logits=log_probs)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
