from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn


def spatial_features(coordinates, dimensions):
    """Shared nonperiodic coordinates and Fourier basis for units and grid cells."""
    xy = coordinates.float()
    frequencies = torch.tensor([0.5, 1.0, 2.0, 4.0], device=xy.device)
    angles = xy[..., :, None] * frequencies * (2 * torch.pi)
    features = torch.cat((xy, angles.sin().flatten(-2), angles.cos().flatten(-2)), -1)
    repeats = (dimensions + features.shape[-1] - 1) // features.shape[-1]
    return features.repeat(*([1] * (features.ndim - 1)), repeats)[..., :dimensions]


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
    tile_query_residual: bool = False
    tile_head_fp32: bool = False
    spatial_fourier: bool = False
    representation_version: int = 1
    entity_pooling: str = "mean"
    entity_fields: tuple[str, ...] = ()
    header_fields: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "entity_fields", tuple(self.entity_fields))
        object.__setattr__(self, "header_fields", tuple(self.header_fields))
        if self.representation_version not in (1, 2):
            raise ValueError("Unsupported representation version")
        if self.entity_pooling not in ("mean", "static1", "static4", "dynamic4"):
            raise ValueError("Unknown entity pooling ablation")
        if self.representation_version == 2:
            if len(self.entity_fields) != self.entity_feature_dim:
                raise ValueError("v2 requires the entity schema in its model config")
            if len(self.header_fields) != self.header_dim:
                raise ValueError("v2 requires the header schema in its model config")

    def for_v2(self, schema, pooling="dynamic4"):
        return replace(
            self,
            representation_version=2,
            entity_pooling=pooling,
            entity_fields=tuple(schema.entity_fields),
            header_fields=tuple(schema.header_fields),
            tile_query_residual=True,
            tile_head_fp32=True,
            spatial_fourier=True,
        )

    def inherit_representation(self, source):
        return replace(
            self,
            **{
                name: getattr(source, name)
                for name in (
                    "representation_version",
                    "entity_pooling",
                    "entity_fields",
                    "header_fields",
                    "tile_query_residual",
                    "tile_head_fp32",
                    "spatial_fourier",
                )
            },
        )

    @classmethod
    def preset(cls, name: str) -> ModelConfig:
        import yaml

        from cr_rl.config import ROOT

        raw = yaml.safe_load((ROOT / "configs/model.yaml").read_text())["presets"][name]
        return cls(**raw)


class EntityFrameEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
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
        if cfg.representation_version == 2:
            self.category_fields = ("row", "kind", "state", "buff_0", "buff_1", "buff_2")
            self.category_scales = (cfg.vocab_size, 8, 16, 100, 100, 100)
            self.category_embeddings = nn.ModuleDict(
                {
                    "row": nn.Embedding(cfg.vocab_size + 1, cfg.entity_dim),
                    "kind": nn.Embedding(33, cfg.entity_dim),
                    "state": nn.Embedding(257, cfg.entity_dim),
                    "buff": nn.Embedding(1025, cfg.entity_dim),
                }
            )
            self.entity_indices = {
                name: cfg.entity_fields.index(name) for name in (*self.category_fields, "flags")
            }
            self.flag_projection = nn.Linear(16, cfg.entity_dim, bias=False)
            self.register_buffer("flag_bits", torch.arange(16), persistent=False)
            if cfg.entity_pooling != "mean":
                queries = 1 if cfg.entity_pooling == "static1" else 4
                self.pool_queries = nn.Parameter(torch.randn(1, queries, cfg.entity_dim) * 0.02)
                self.pool_attention = nn.MultiheadAttention(
                    cfg.entity_dim, cfg.entity_heads, batch_first=True
                )
                self.pool_projection = nn.Linear(queries * cfg.entity_dim, cfg.entity_dim)
                if cfg.entity_pooling == "dynamic4":
                    self.dynamic_queries = nn.Sequential(
                        nn.Linear(cfg.entity_dim, cfg.entity_dim),
                        nn.GELU(),
                        nn.Linear(cfg.entity_dim, queries * cfg.entity_dim),
                    )

    def forward(
        self,
        header: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        card_context=None,
    ):
        continuous = entities
        if self.cfg.representation_version == 2:
            continuous = entities.clone()
            continuous[..., list(self.entity_indices.values())] = 0
        x = self.entity_in(continuous)
        if self.cfg.representation_version == 2:
            for name, scale in zip(self.category_fields, self.category_scales, strict=True):
                raw = entities[..., self.entity_indices[name]].float()
                ids = (raw * scale).round().long() + 1
                embedding = self.category_embeddings["buff" if name.startswith("buff_") else name]
                ids = torch.where((raw >= 0) & (ids < embedding.num_embeddings), ids, 0)
                x = x + embedding(ids).to(x.dtype)
            flags = (entities[..., self.entity_indices["flags"]].float() * 65535).round().long()
            bits = ((flags.unsqueeze(-1) >> self.flag_bits) & 1).to(entities.dtype)
            x = x + self.flag_projection(bits).to(x.dtype)
        if self.cfg.spatial_fourier:
            if entities.shape[-1] < 6:
                raise ValueError("Spatial encoding requires pinned entity x/y fields 4 and 5")
            x = x + 0.02 * spatial_features(entities[..., 4:6], self.cfg.entity_dim).to(x.dtype)
        x = self.encoder(x, src_key_padding_mask=~entity_mask.bool())
        header_state = self.header(header)
        if card_context is not None:
            header_state = header_state + card_context
        if self.cfg.representation_version == 2 and self.cfg.entity_pooling != "mean":
            queries = self.pool_queries.expand(x.shape[0], -1, -1)
            if self.cfg.entity_pooling == "dynamic4":
                queries = queries + self.dynamic_queries(header_state).reshape_as(queries)
            pooled, _ = self.pool_attention(
                queries, x, x, key_padding_mask=~entity_mask.bool(), need_weights=False
            )
            pooled = self.pool_projection((pooled + queries).flatten(1))
        else:
            denom = entity_mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (x * entity_mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.norm(pooled + header_state), x


class HistoryEncoder(nn.Module):
    """History representation; inherited keys preserve architecture-v2 BC weights."""

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
        if cfg.representation_version == 1:
            self.deck_context = nn.Linear(cfg.entity_dim, cfg.temporal_dim, bias=False)
            self.seen_context = nn.Linear(cfg.entity_dim, cfg.temporal_dim, bias=False)
        if cfg.representation_version == 2:
            self.card_groups = tuple(
                tuple(i for i, field in enumerate(cfg.header_fields) if field == name)
                for name in (
                    "hand_card",
                    "next_card",
                    "deck",
                    "enemy_seen",
                    "pending_card",
                    "ability_champion",
                    "enemy_play_card",
                )
            )
            self.early_card_context = nn.Linear(
                len(self.card_groups) * cfg.entity_dim, cfg.entity_dim, bias=False
            )

    def encode_frames(self, header, entities, entity_mask):
        card_context = None
        if self.cfg.representation_version == 2:
            summaries = []
            for indices in self.card_groups:
                raw = header[..., list(indices)].float()
                ids = (raw * self.cfg.vocab_size).round().long() + 1
                valid = (raw >= 0) & (ids <= self.cfg.vocab_size)
                ids = torch.where(valid, ids, 0)
                embedded = self.card_embedding(ids)
                pooled = (embedded * valid.unsqueeze(-1)).sum(-2)
                pooled = pooled / valid.sum(-1, keepdim=True).clamp_min(1)
                summaries.append(pooled)
            card_context = self.early_card_context(torch.cat(summaries, dim=-1))
            header = header.clone()
            header[..., [i for group in self.card_groups for i in group]] = 0
        return self.frame(header, entities, entity_mask, card_context)

    def encode_history(
        self,
        header,
        entities,
        entity_mask,
        time_mask=None,
        target_index=-1,
        deck_card_ids=None,
        enemy_seen_card_ids=None,
    ):
        batch, steps, count, features = entities.shape
        frame, entity_tokens = self.encode_frames(
            header.reshape(batch * steps, -1),
            entities.reshape(batch * steps, count, features),
            entity_mask.reshape(batch * steps, count),
        )
        context = self.encode_temporal(
            frame.reshape(batch, steps, -1),
            time_mask,
            target_index,
            deck_card_ids,
            enemy_seen_card_ids,
        )
        target_index %= steps
        return (
            context,
            entity_tokens.reshape(batch, steps, count, -1)[:, target_index],
            entity_mask[:, target_index].bool(),
        )

    def encode_temporal(
        self, frames, time_mask=None, target_index=-1, deck_card_ids=None, enemy_seen_card_ids=None
    ):
        steps = frames.shape[1]
        x = self.to_temporal(frames) + self.position[:, :steps]
        causal = torch.triu(torch.ones(steps, steps, device=x.device, dtype=torch.bool), diagonal=1)
        padding = ~time_mask.bool() if time_mask is not None else None
        if padding is not None:
            # Left-padded queries have no visible key under causal + padding masks.
            # Some inference kernels emit NaNs there, poisoning later layers even
            # though valid queries cannot attend to padding. Give only these dummy
            # queries one dummy key; valid queries retain the exact original mask.
            blocked = causal.unsqueeze(0) | padding.unsqueeze(1)
            blocked = blocked.clone()
            blocked[:, :, 0] &= ~blocked.all(dim=-1)
            causal = blocked.repeat_interleave(self.cfg.temporal_heads, dim=0)
        x = self.temporal(x, mask=causal)
        target_index %= steps
        context = x[:, target_index]
        late_context = (
            ()
            if self.cfg.representation_version == 2
            else (
                (deck_card_ids, self.deck_context),
                (enemy_seen_card_ids, self.seen_context),
            )
        )
        for ids, projection in late_context:
            if ids is not None:
                ids = ids.long()
                valid = ids >= 0
                embeddings = self.card_embedding((ids + 1).clamp(0, self.cfg.vocab_size))
                pooled = (embeddings * valid.unsqueeze(-1)).sum(dim=1)
                pooled = pooled / valid.sum(dim=1, keepdim=True).clamp_min(1)
                context = context + projection(pooled)
        return context


class ActorCritic(HistoryEncoder):
    """Architecture-v2 shared model, retained for BC and legacy PPO checkpoints."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
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
        grid = torch.arange(576)
        centers = torch.stack(((grid % 18 + 0.5) / 18, (grid // 18 + 0.5) / 32), -1)
        self.register_buffer(
            "tile_grid_features", spatial_features(centers, cfg.entity_dim), persistent=False
        )
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
        compute_value=True,
        encoded_history=None,
    ):
        """Encode a history window and emit one action distribution for its final frame."""
        batch = entities.shape[0]
        if encoded_history is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Cached history is only valid for frozen inference")
            context, current_entities, current_entity_mask = encoded_history
        else:
            context, current_entities, current_entity_mask = self.encode_history(
                header,
                entities,
                entity_mask,
                time_mask,
                target_index,
                deck_card_ids,
                enemy_seen_card_ids,
            )
        if hand_card_ids is None:
            hand_card_ids = torch.full((batch, 4), -1, dtype=torch.long, device=context.device)
        card_indices = (hand_card_ids.long() + 1).clamp(0, self.cfg.vocab_size)
        card_features = self.card_embedding(card_indices) + self.card_slot_embedding
        context_slots = context.unsqueeze(1).expand(-1, 4, -1)
        slot_logits = self.slot_actor(torch.cat((context_slots, card_features), dim=-1)).squeeze(-1)
        card_logits = torch.cat(
            (self.wait_actor(context), slot_logits, self.ability_actor(context)), dim=-1
        )
        tile_logits = self.placement_logits(
            context, current_entities, current_entity_mask, card_features
        )
        if legal_action_mask is None:
            legal_action_mask = torch.ones(
                batch, self.cfg.action_count, dtype=torch.bool, device=context.device
            )
        joint_log_probs = self._joint_log_probs(card_logits, tile_logits, legal_action_mask)
        value = self.critic(context).squeeze(-1) if compute_value else context.new_zeros(batch)
        return joint_log_probs, value

    def placement_logits(self, context, entities, entity_mask, card_features):
        if self.cfg.tile_head_fp32:
            with torch.autocast(device_type=context.device.type, enabled=False):
                return self._placement_logits(
                    context.float(), entities.float(), entity_mask, card_features.float()
                )
        return self._placement_logits(context, entities, entity_mask, card_features)

    def _placement_logits(self, context, entities, entity_mask, card_features):
        batch = context.shape[0]
        queries = self.tile_queries.expand(batch, -1, -1, -1)
        queries = (
            queries
            + self.slot_embedding
            + card_features[:, :, None, :]
            + self.tile_context(context)[:, None, None, :]
        )
        if self.cfg.spatial_fourier:
            queries = queries + 0.02 * self.tile_grid_features[None, None].to(queries.dtype)
        queries = queries.reshape(batch, 4 * 576, -1)
        tile_features, _ = self.tile_cross_attention(
            queries,
            entities,
            entities,
            key_padding_mask=~entity_mask,
            need_weights=False,
        )
        if self.cfg.tile_query_residual:
            tile_features = tile_features + queries
        return self.tile_actor(tile_features).squeeze(-1).reshape(batch, 4, 576)

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


class ActorPolicy(ActorCritic):
    """Deployable policy with no value parameters or value computation.

    The second tuple item is a zero placeholder for legacy sampling callers;
    training must obtain real values from HistoryValueNetwork.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        del self.critic

    def forward(self, *args, **kwargs):
        kwargs["compute_value"] = False
        return super().forward(*args, **kwargs)

    @classmethod
    def from_shared(cls, model: ActorCritic):
        actor = cls(model.cfg).to(next(model.parameters()).device)
        actor.load_state_dict(
            {k: v for k, v in model.state_dict().items() if not k.startswith("critic.")},
            strict=True,
        )
        return actor


class HistoryValueNetwork(HistoryEncoder):
    """Independent Transformer critic; no parameters are shared with the actor."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.critic = nn.Sequential(
            nn.Linear(cfg.temporal_dim, cfg.temporal_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.temporal_dim // 2, 1),
        )

    def initialize_encoder(self, actor: ActorCritic) -> dict:
        source = actor.state_dict()
        own = self.state_dict()
        copied = {
            k: source[k]
            for k in own
            if not k.startswith("critic.") and k in source and source[k].shape == own[k].shape
        }
        self.load_state_dict(copied, strict=False)
        return {
            "encoder_tensors_copied": len(copied),
            "encoder_parameters_copied": sum(v.numel() for v in copied.values()),
            "value_head_initialized_from_bc": False,
        }

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
        context, _, _ = self.encode_history(
            header,
            entities,
            entity_mask,
            time_mask,
            target_index,
            deck_card_ids,
            enemy_seen_card_ids,
        )
        return self.critic(context).squeeze(-1)
