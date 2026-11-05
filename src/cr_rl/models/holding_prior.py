"""Observation-only WAIT bias shared by rollout, PPO and exported inference."""

from torch import nn


class ElixirHoldingPrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.strength = cfg.elixir_wait_bias
        self.minimum_cost = cfg.elixir_hold_min_cost
        self.header = {
            name: [i for i, field in enumerate(cfg.header_fields) if field == name]
            for name in ("elixir", "elixir_reserved", "hand_card", "hand_cost", "hand_pending")
        }
        self.entity = {name: cfg.entity_fields.index(name) for name in ("side", "kind", "y", "hp")}
        if any(not indices for indices in self.header.values()):
            raise ValueError("Elixir prior requires the pinned hand/elixir observation schema")

    def forward(self, header, entities, entity_mask):
        # Adapter units: elixir /100000, hand cost /10, side /2, kind /8.
        elixir = header[..., self.header["elixir"][0]].float()
        reserved = header[..., self.header["elixir_reserved"][0]].float()
        available = 10 * (elixir - reserved).clamp_min(0)
        costs = 10 * header[..., self.header["hand_cost"]].float()
        known = header[..., self.header["hand_card"]] >= 0
        pending = header[..., self.header["hand_pending"]] > 0
        expensive = known & ~pending & (costs >= self.minimum_cost)
        locked = expensive & (costs > available.unsqueeze(-1) + 1e-5)
        fraction = locked.float().sum(-1) / expensive.sum(-1).clamp_min(1)
        # Pinned Schema: side1 is enemy; kind1 is a character; own back line y=0.
        enemy = (entities[..., self.entity["side"]].float() * 2).round() == 1
        character = (entities[..., self.entity["kind"]].float() * 8).round() == 1
        near_own_half = entities[..., self.entity["y"]].float() <= 0.55
        alive = entities[..., self.entity["hp"]] > 0
        threatened = (entity_mask.bool() & enemy & character & near_own_half & alive).any(-1)
        eligible = ~threatened & (elixir < 0.99)
        return self.strength * fraction * eligible.float()
