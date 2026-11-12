"""Small diagnostic entity-to-grid residual for an existing placement head.

The wrapped head owns the unchanged five-field call and returns GT-slot logits
of shape [B,576]. The extra branch consumes encoded entity tokens plus their
aligned PRE-encoder adapter coordinates. It does not infer geometry from tokens,
change the old attention query shape, load a checkpoint, or save any weights.

Run ``python scripts/state_bc_spatial_probe.py --self-test`` for CPU-only checks.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F


RAW_FIELDS = ('context', 'entities', 'entity_mask', 'card_features', 'slots')
SPATIAL_FIELDS = RAW_FIELDS + ('entity_xy',)
GRID_HEIGHT, GRID_WIDTH = 32, 18


def bilinear_scatter(projected, entity_xy, entity_mask, *, height=GRID_HEIGHT, width=GRID_WIDTH):
    """Return feature sums and occupancy, preserving duplicate-position units.

    Coordinates use the adapter's normalized x/y arena axes. Pixel centers are
    ((x+.5)/18,(y+.5)/32), matching the existing placement grid. Positions outside
    the arena clip to the closest border center; invalid entities contribute
    exactly zero, including when their padded coordinates/features are NaN.
    Accumulation stays FP32. No full [B,N,576,C] tensor is constructed.
    """
    if projected.ndim != 3 or entity_xy.shape != (*projected.shape[:2], 2):
        raise ValueError('Expected projected [B,N,C] and aligned entity_xy [B,N,2]')
    if entity_mask.shape != projected.shape[:2] or min(height, width) < 1:
        raise ValueError('Entity mask shape or scatter grid is invalid')
    if not (projected.device == entity_xy.device == entity_mask.device):
        raise ValueError('Scatter inputs must be on the same device')
    mask = entity_mask.bool()
    # torch.where, rather than multiply-by-zero, prevents masked NaNs leaking.
    features = torch.where(mask[..., None], projected.float(), 0.)
    xy = torch.where(mask[..., None], entity_xy.float(), 0.)
    px = (xy[..., 0] * width - .5).clamp(0., width - 1.)
    py = (xy[..., 1] * height - .5).clamp(0., height - 1.)
    left, top = px.floor().long(), py.floor().long()
    right, bottom = (left + 1).clamp_max(width - 1), (top + 1).clamp_max(height - 1)
    dx, dy = px - left, py - top
    weights = torch.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1)
    weights = weights * mask[..., None]
    indices = torch.stack((top*width+left, top*width+right,
        bottom*width+left, bottom*width+right), -1)
    batch, count, channels = projected.shape
    flat_index = indices.reshape(batch, 1, count * 4)
    values = (features[:, :, None] * weights[..., None]).reshape(batch, count * 4, channels).transpose(1, 2)
    spatial = features.new_zeros(batch, channels, height * width).scatter_add(
        2, flat_index.expand(-1, channels, -1), values)
    occupancy = features.new_zeros(batch, 1, height * width).scatter_add(
        2, flat_index, weights.reshape(batch, 1, count * 4))
    return spatial.reshape(batch, channels, height, width), occupancy.reshape(batch, 1, height, width)


class _ConditionedSpatialBlock(nn.Module):
    def __init__(self, channels, condition_dim, *, device):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels, device=device)
        self.norm2 = nn.GroupNorm(8, channels, device=device)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, device=device)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, device=device)
        self.film = nn.Linear(condition_dim, 2 * channels, device=device)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, spatial, condition):
        hidden = self.conv1(F.gelu(self.norm1(spatial)))
        scale, shift = self.film(condition).chunk(2, -1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return spatial + self.conv2(F.gelu(hidden))


class EntitySpatialResidual(nn.Module):
    """Original GT-slot placement logits plus a learned spatial residual.

    ``original_head`` is an arbitrary nn.Module accepting RAW_FIELDS and returning
    [B,576]; no import of the head probe/trainer is required. Ownership is retained
    directly, not deep-copied: construct independent original heads for A/B arms.
    Original parameters remain trainable by default. All new parameters are FP32,
    matching the current placement safety boundary; outer FP16 autocast is fine.

    At initialization the final convolution is exactly zero, so output equals
    the unchanged old call. The first update learns that output projection; on
    subsequent updates gradients reach the projection/CNN/FiLM. No second zero
    gate is placed in front of the zero-output branch.
    """
    def __init__(self, original_head, *, entity_dim=256, context_dim=1280,
                 card_dim=256, channels=64, train_original=True):
        super().__init__()
        if channels not in (32, 64) or min(entity_dim, context_dim, card_dim) < 1:
            raise ValueError('Use positive feature dimensions and 32 or 64 spatial channels')
        self.original = original_head
        if not train_original:
            self.original.requires_grad_(False)
        self.entity_dim, self.context_dim, self.card_dim = entity_dim, context_dim, card_dim
        self.channels = channels
        reference = next(original_head.parameters(), None)
        device = None if reference is None else reference.device
        self.entity_projection = nn.Linear(entity_dim, channels, device=device)
        self.spatial_input = nn.Conv2d(channels + 3, channels, 1, device=device)
        condition_dim = context_dim + card_dim
        self.condition_norm = nn.LayerNorm(condition_dim, device=device)
        self.blocks = nn.ModuleList([_ConditionedSpatialBlock(channels, condition_dim, device=device)
            for _ in range(2)])
        self.output = nn.Conv2d(channels, 1, 1, device=device)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        yy, xx = torch.meshgrid((torch.arange(GRID_HEIGHT, device=device).float() + .5) / GRID_HEIGHT,
            (torch.arange(GRID_WIDTH, device=device).float() + .5) / GRID_WIDTH, indexing='ij')
        self.register_buffer('grid_xy', torch.stack((xx, yy))[None], persistent=False)
        if self.added_parameter_count() >= 1_000_000:
            raise ValueError('Diagnostic spatial branch exceeds its one-million-parameter budget')

    def added_parameter_count(self):
        return sum(parameter.numel() for name, parameter in self.named_parameters()
            if not name.startswith('original.'))

    def complexity_estimate(self, *, entities=200, batch=1):
        """Approximate MACs for ONE GT slot, excluding norms/scatter/activation/base.

        Full four-card inference would need four conditioned branches, though
        entity projection/scatter can eventually be shared. This is arithmetic,
        not a GPU latency or end-to-end training speed measurement.
        """
        channels, grid = self.channels, GRID_HEIGHT * GRID_WIDTH
        projection = entities * self.entity_dim * channels
        input_projection = grid * (channels + 3) * channels
        convolutions = 4 * grid * channels * channels * 9
        conditioning = 2 * (self.context_dim + self.card_dim) * (2 * channels)
        readout = grid * channels
        return {'added_parameters': self.added_parameter_count(), 'entities': entities, 'batch': batch,
            'gt_slot_macs_approx': batch * (projection + input_projection + convolutions + conditioning + readout),
            'excludes': 'Original head, norms, scatter, nonlinearities, memory movement; not measured latency'}

    def residual_logits(self, context, entities, entity_mask, card_features, slots, entity_xy):
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError('Context dimensions differ from the declared spatial branch')
        batch = len(context)
        if entities.ndim != 3 or entities.shape[0] != batch or entities.shape[-1] != self.entity_dim:
            raise ValueError('Encoded entities must be aligned [B,N,entity_dim]')
        if entity_mask.shape != entities.shape[:2] or entity_xy.shape != (*entities.shape[:2], 2):
            raise ValueError('Raw coordinates, encoded tokens and mask must use identical entity rows')
        if card_features.shape != (batch, 4, self.card_dim) or slots.shape != (batch,):
            raise ValueError('Expected four card features and one true slot per action row')
        if slots.dtype not in (torch.int32, torch.int64):
            raise ValueError('Slots must be integral indices')
        if self.entity_projection.weight.dtype != torch.float32:
            raise ValueError('This diagnostic branch uses FP32 parameters; use .to(device), not .half()')
        if len({value.device for value in (context, entities, entity_mask, card_features, slots, entity_xy)}) != 1:
            raise ValueError('All aligned cached fields must be on one device')
        with torch.autocast(device_type=context.device.type, enabled=False):
            mask = entity_mask.bool()
            safe_entities = torch.where(mask[..., None], entities.float(), 0.)
            projected = self.entity_projection(safe_entities)
            feature_sum, occupancy = bilinear_scatter(projected, entity_xy, mask)
            # Average coincident entities but keep their density in a separate
            # log-count plane. Empty cells stay zero; no entity is overwritten.
            spatial = feature_sum / occupancy.clamp_min(1e-6)
            spatial = torch.cat((spatial, occupancy.log1p(), self.grid_xy.expand(batch, -1, -1, -1)), 1)
            spatial = self.spatial_input(spatial)
            # gather rejects out-of-range and negative slots rather than Python
            # advanced indexing silently accepting a negative card index.
            index = slots.long()[:, None, None].expand(-1, 1, self.card_dim)
            card = card_features.float().gather(1, index).squeeze(1)
            condition = self.condition_norm(torch.cat((context.float(), card), -1))
            for block in self.blocks:
                spatial = block(spatial, condition)
            return self.output(spatial).flatten(1)

    def forward(self, context, entities, entity_mask, card_features, slots, entity_xy):
        # Keep the original call and full-query numeric path untouched.
        base = self.original(context=context, entities=entities, entity_mask=entity_mask,
            card_features=card_features, slots=slots)
        if base.shape != (len(context), GRID_HEIGHT * GRID_WIDTH):
            raise ValueError('Original placement head must return [B,576] GT-slot logits')
        residual = self.residual_logits(context, entities, entity_mask, card_features, slots, entity_xy)
        return base + residual.to(base.dtype)


def cpu_self_test():
    """CPU-only functional/numerical checks; synthetic tensors, no model files."""
    class FixtureHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1280, 576)

        def forward(self, context, entities, entity_mask, card_features, slots):
            masked = torch.where(entity_mask[..., None], entities, 0.)
            card = card_features.gather(1, slots[:, None, None].expand(-1, 1, 256)).squeeze(1)
            return self.linear(context) + masked.mean((1, 2))[:, None] + card.mean(-1)[:, None]

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261111)
            batch, count = 2, 7
            raw = {'context': torch.randn(batch, 1280), 'entities': torch.randn(batch, count, 256),
                'entity_mask': torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool),
                'card_features': torch.randn(batch, 4, 256), 'slots': torch.tensor([0, 3]),
                'entity_xy': torch.rand(batch, count, 2)}
            model = EntitySpatialResidual(FixtureHead(), channels=32).eval()
            base = model.original(**{key: raw[key] for key in RAW_FIELDS})
            initial = model(**raw)
            assert initial.shape == (batch, 576) and torch.equal(initial, base)
            assert torch.count_nonzero(model.residual_logits(**raw)) == 0

            # Known duplicate location: both contributions survive; bilinear
            # occupancy mass is the number of real units, including borders.
            projected = torch.tensor([[[2., 4.], [2., 4.], [float('nan'), float('nan')]]])
            xy = torch.tensor([[[.5/18, .5/32], [.5/18, .5/32], [float('nan'), float('inf')]]])
            mask = torch.tensor([[True, True, False]])
            sums, occupancy = bilinear_scatter(projected, xy, mask)
            assert torch.isfinite(sums).all() and torch.isfinite(occupancy).all()
            assert occupancy.sum() == 2 and occupancy[0, 0, 0, 0] == 2
            assert torch.equal(sums[0, :, 0, 0], torch.tensor([4., 8.]))
            borders = torch.tensor([[[-1., -1.], [2., 2.]]])
            _, edge_counts = bilinear_scatter(torch.ones(1, 2, 1), borders, torch.ones(1, 2, dtype=torch.bool))
            assert edge_counts.sum() == 2 and edge_counts[0, 0, 0, 0] == 1 and edge_counts[0, 0, -1, -1] == 1

            # First update activates only the zero-output projection; gradients
            # throughout the new branch become nonzero on the second pass.
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            initial.square().mean().backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
            assert model.output.weight.grad.abs().sum() > 0
            assert model.entity_projection.weight.grad.abs().sum() == 0
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            trained = model(**raw)
            trained.square().mean().backward()
            assert model.entity_projection.weight.grad.abs().sum() > 0
            assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

            altered = {key: value.clone() for key, value in raw.items()}
            invalid = ~altered['entity_mask']
            altered['entities'][invalid] = float('nan')
            altered['entity_xy'][invalid] = float('nan')
            assert torch.equal(model(**raw), model(**altered))
            order = torch.tensor([6, 2, 5, 0, 4, 1, 3])
            reordered = {**raw, 'entities': raw['entities'][:, order],
                'entity_mask': raw['entity_mask'][:, order], 'entity_xy': raw['entity_xy'][:, order]}
            assert torch.allclose(model.residual_logits(**raw), model.residual_logits(**reordered), atol=1e-6, rtol=1e-6)
            empty_sum, empty_count = bilinear_scatter(torch.empty(2, 0, 3), torch.empty(2, 0, 2), torch.empty(2, 0, dtype=torch.bool))
            assert empty_sum.shape == (2, 3, 32, 18) and torch.count_nonzero(empty_count) == 0
            assert model.added_parameter_count() < 1_000_000
            wider = EntitySpatialResidual(FixtureHead(), channels=64)
            assert wider.added_parameter_count() < 1_000_000
            return {'passed': True, 'device': 'cpu', 'zero_output_exact': True,
                'masked_nan_invariance': True, 'duplicate_mass_preserved': True,
                'second_step_projection_gradient_nonzero': True, 'finite_gradients': True,
                'entity_permutation_tolerance': 1e-6, 'empty_scatter_supported': True,
                'channels32': model.complexity_estimate(), 'channels64': wider.complexity_estimate(),
                'limitation': 'CPU synthetic checks only; no full-head GPU parity, real-data quality or L4 latency claim'}
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('This module only exposes --self-test; no training or weight export entry point')
    print(json.dumps(cpu_self_test(), allow_nan=False))
