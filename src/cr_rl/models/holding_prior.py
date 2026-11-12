"""Observation-only WAIT bias shared by rollout, PPO and exported inference."""

import torch
from torch import nn


class ElixirHoldingPrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.strength = cfg.elixir_wait_bias
        self.minimum_cost = cfg.elixir_hold_min_cost
        self.version = cfg.holding_prior_version
        self.vocab_size = cfg.vocab_size
        self.core_ids = cfg.core_card_ids
        self.tower_ids = cfg.tower_card_ids
        self.slot_strength = cfg.core_slot_bias
        self.header = {
            name: [i for i, field in enumerate(cfg.header_fields) if field == name]
            for name in ("elixir", "elixir_reserved", "hand_card", "hand_cost", "hand_pending")
        }
        self.entity = {name: cfg.entity_fields.index(name) for name in ("side", "kind", "y", "hp")}
        if any(not indices for indices in self.header.values()):
            raise ValueError("Elixir prior requires the pinned hand/elixir observation schema")
        if self.version == 2:
            self.entity = {name: cfg.entity_fields.index(name) for name in (
                "side", "kind", "x", "y", "hp", "row", "state", "target_x", "target_y"
            )}

    @staticmethod
    def _member(values, ids):
        return (values[..., None] == values.new_tensor(ids)).any(-1)

    def pressure_scale(self, entities, entity_mask):
        """Pinned coordinates in tiles; emergency overrides ignore enemy HP magnitude."""
        e = self.entity
        alive = entity_mask.bool() & (entities[..., e["hp"]] > 0)
        side = (entities[..., e["side"]] * 2).round().long()
        kind = (entities[..., e["kind"]] * 8).round().long()
        rows = (entities[..., e["row"]] * self.vocab_size).round().long()
        enemy = alive & (side == 1) & (kind == 1)
        towers = alive & (side == 0) & self._member(rows, self.tower_ids)
        xy = torch.stack((entities[..., e["x"]] * 18, entities[..., e["y"]] * 32), -1)
        # Only three own towers are possible. Mask padding and destroyed towers explicitly.
        indices = towers.long().topk(min(3, towers.shape[-1]), dim=-1).indices
        tower_xy = xy.gather(-2, indices[..., None].expand(*indices.shape, 2))
        tower_valid = towers.gather(-1, indices)
        distances = (xy[..., :, None, :] - tower_xy[..., None, :, :]).square().sum(-1)
        distances = distances.masked_fill(~tower_valid[..., None, :], float("inf"))
        nearest = distances.amin(-1)
        targets = torch.stack((entities[..., e["target_x"]] * 18,
                               entities[..., e["target_y"]] * 32), -1)
        target_distance = (targets[..., :, None, :] - tower_xy[..., None, :, :]).square().sum(-1)
        targets_tower = (target_distance.masked_fill(~tower_valid[..., None, :], float("inf"))
                         .amin(-1) <= 1.5**2)
        state = (entities[..., e["state"]] * 16).round().long()
        nearest_xy = tower_xy.gather(-2, distances.argmin(-1)[..., None].expand(*nearest.shape, 2))
        approaching = ((targets - xy) * (nearest_xy - xy)).sum(-1) > 0
        # Moving/dashing near any tower receives extra reaction room for command delay.
        mobile = ((state == 1) | (state == 3) | (state == 5)) & approaching
        urgent_unit = ((nearest <= 5**2) | (mobile & (nearest <= 7**2))
                       | ((state == 2) & targets_tower & (nearest <= 9**2))
                       | (entities[..., e["y"]] <= .28))
        urgent = (enemy & urgent_unit).any(-1) | ~tower_valid.any(-1)
        pressure = (enemy & (entities[..., e["y"]] <= .55)).any(-1)
        return torch.where(urgent, 0.0, torch.where(pressure, 0.4, 1.0))

    def action_bias(self, header, entities, entity_mask, legal_action_mask):
        """Versioned observation-only bias; never forces or unmasks an action."""
        e = header[..., self.header["elixir"][0]].float()
        reserved = header[..., self.header["elixir_reserved"][0]].float()
        available = 10 * (e - reserved).clamp_min(0)
        costs = 10 * header[..., self.header["hand_cost"]].float()
        cards = (header[..., self.header["hand_card"]] * self.vocab_size).round().long()
        known = header[..., self.header["hand_card"]] >= 0
        pending = header[..., self.header["hand_pending"]] > 0
        core = known & ~pending & self._member(cards, self.core_ids) & (costs >= self.minimum_cost)
        affordable = costs <= available[..., None] + 1e-5
        legal = (torch.ones_like(core) if legal_action_mask is None else
                 legal_action_mask[..., 1:2305].reshape(*core.shape, 576).any(-1))
        ready = core & affordable & legal
        locked = core & ~affordable
        scale = self.pressure_scale(entities, entity_mask)
        wait = self.strength * scale * locked.any(-1) * ~ready.any(-1) * (e < .99)
        slots = self.slot_strength * scale[..., None] * ready
        return torch.cat((wait[..., None], slots, torch.zeros_like(slots[..., :2])), -1)

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
