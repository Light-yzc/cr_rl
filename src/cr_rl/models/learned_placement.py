"""Versioned learned placement decoders, independent of diagnostic scripts.

The all-slot path consumes real hand identities and pre-encoder coordinates;
it never consumes a ground-truth action. ``forward_gt`` is a supervised-only
convenience. Legal masking/card-WAIT normalization remain the policy's concern.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import math
import re

import torch
from torch import nn
from torch.nn import functional as F


DECODER_PAYLOAD_FORMAT = 'crrl-placement-decoder-probe-v1'
PLACEMENT_DEFINITION_FORMAT = 'crrl-learned-placement-v1'
TILES, HEIGHT, WIDTH = 576, 32, 18
VARIANTS = frozenset(('original', 'wide_residual', 'spatial_cnn',
    'card_prior_001', 'card_prior_1', 'card_prior_cnn_001'))


class OriginalPlacementHead(nn.Module):
    """Copy the complete original head with the diagnostic state-dict names."""
    def __init__(self, source):
        super().__init__()
        self.cfg = source.cfg
        for name in ('tile_queries', 'slot_embedding'):
            setattr(self, name, nn.Parameter(getattr(source, name).detach().clone()))
        for name in ('tile_context', 'tile_cross_attention', 'tile_actor'):
            setattr(self, name, deepcopy(getattr(source, name)))
        self.register_buffer('tile_grid_features', source.tile_grid_features.detach().clone())
        if self.cfg.readout_version == 2:
            for name in ('tile_spatial_projection', 'tile_spatial_norm', 'tile_film'):
                setattr(self, name, deepcopy(getattr(source, name)))
            self.tile_spatial_gate = nn.Parameter(source.tile_spatial_gate.detach().clone())

    def features(self, context, entities, entity_mask, card_features, slots=None):
        with torch.autocast(device_type=context.device.type, enabled=False):
            context, entities, card_features = context.float(), entities.float(), card_features.float()
            n = len(context)
            queries = (self.tile_queries.expand(n, -1, -1, -1) + self.slot_embedding +
                card_features[:, :, None] + self.tile_context(context)[:, None, None])
            if self.cfg.spatial_fourier:
                queries = queries + .02 * self.tile_grid_features[None, None]
            if self.cfg.readout_version == 2:
                spatial = self.tile_queries + .02 * self.tile_grid_features[None, None]
                normalized = self.tile_spatial_norm(self.tile_queries +
                    self.tile_spatial_projection(self.tile_grid_features)[None])
                gamma, beta = self.tile_film(card_features + self.tile_context(context)[:, None]).chunk(2, -1)
                modulated = normalized * (1 + gamma[:, :, None]) + beta[:, :, None]
                queries = queries + self.tile_spatial_gate.tanh() * (modulated - spatial)
            if slots is not None:
                queries = queries[torch.arange(n, device=context.device), slots]
            else:
                queries = queries.reshape(n, 4*TILES, -1)
            features, _ = self.tile_cross_attention(queries, entities, entities,
                key_padding_mask=~entity_mask.bool(), need_weights=False)
            if self.cfg.tile_query_residual:
                features = features + queries
            return features if slots is not None else features.reshape(n, 4, TILES, -1)

    def forward_all(self, context, entities, entity_mask, card_features):
        with torch.autocast(device_type=context.device.type, enabled=False):
            return self.tile_actor(self.features(context, entities, entity_mask, card_features)).squeeze(-1)

    def forward(self, context, entities, entity_mask, card_features, slots):
        with torch.autocast(device_type=context.device.type, enabled=False):
            features = self.features(context, entities, entity_mask, card_features, slots=slots)
            return self.tile_actor(features).squeeze(-1)


class _WideReadout(nn.Module):
    def __init__(self, original, dimensions, width):
        super().__init__()
        self.original = original
        self.branch = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, width),
            nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, features):
        return self.original(features) + self.branch(features)


def bilinear_scatter(projected, entity_xy, entity_mask):
    mask = entity_mask.bool()
    features = torch.where(mask[..., None], projected.float(), 0.)
    xy = torch.where(mask[..., None], entity_xy.float(), 0.)
    px, py = (xy[..., 0]*WIDTH-.5).clamp(0., WIDTH-1.), (xy[..., 1]*HEIGHT-.5).clamp(0., HEIGHT-1.)
    left, top = px.floor().long(), py.floor().long()
    right, bottom = (left+1).clamp_max(WIDTH-1), (top+1).clamp_max(HEIGHT-1)
    dx, dy = px-left, py-top
    weights = torch.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1)*mask[..., None]
    indices = torch.stack((top*WIDTH+left, top*WIDTH+right, bottom*WIDTH+left, bottom*WIDTH+right), -1)
    batch, count, channels = projected.shape
    index = indices.reshape(batch, 1, count*4)
    values = (features[:, :, None]*weights[..., None]).reshape(batch, count*4, channels).transpose(1, 2)
    spatial = features.new_zeros(batch, channels, TILES).scatter_add(2, index.expand(-1, channels, -1), values)
    occupancy = features.new_zeros(batch, 1, TILES).scatter_add(2, index, weights.reshape(batch, 1, count*4))
    return spatial.reshape(batch, channels, HEIGHT, WIDTH), occupancy.reshape(batch, 1, HEIGHT, WIDTH)


class _SpatialBlock(nn.Module):
    def __init__(self, channels, condition_dim, *, device):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels, device=device)
        self.norm2 = nn.GroupNorm(8, channels, device=device)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, device=device)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, device=device)
        self.film = nn.Linear(condition_dim, 2*channels, device=device)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, spatial, condition):
        hidden = self.conv1(F.gelu(self.norm1(spatial)))
        scale, shift = self.film(condition).chunk(2, -1)
        hidden = self.norm2(hidden)*(1+scale[:, :, None, None])+shift[:, :, None, None]
        return spatial + self.conv2(F.gelu(hidden))


class _ZeroBase(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), device=device), requires_grad=False)


class SpatialResidual(nn.Module):
    """Exact diagnostic spatial keys/math; all new parameters remain FP32."""
    def __init__(self, original, *, entity_dim, context_dim, card_dim, channels):
        super().__init__()
        if channels not in (32, 64):
            raise ValueError('Supported production CNN channels are 32/64')
        self.original = original
        device = next(original.parameters()).device
        self.card_dim = card_dim
        self.entity_projection = nn.Linear(entity_dim, channels, device=device)
        self.spatial_input = nn.Conv2d(channels+3, channels, 1, device=device)
        condition_dim = context_dim+card_dim
        self.condition_norm = nn.LayerNorm(condition_dim, device=device)
        self.blocks = nn.ModuleList([_SpatialBlock(channels, condition_dim, device=device) for _ in range(2)])
        self.output = nn.Conv2d(channels, 1, 1, device=device)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        yy, xx = torch.meshgrid((torch.arange(HEIGHT, device=device).float()+.5)/HEIGHT,
            (torch.arange(WIDTH, device=device).float()+.5)/WIDTH, indexing='ij')
        self.register_buffer('grid_xy', torch.stack((xx, yy))[None], persistent=False)

    def _project_spatial(self, entities, entity_mask, entity_xy):
        safe = torch.where(entity_mask[..., None].bool(), entities.float(), 0.)
        sums, count = bilinear_scatter(self.entity_projection(safe), entity_xy, entity_mask)
        grid = self.grid_xy.expand(len(entities), -1, -1, -1)
        return self.spatial_input(torch.cat((sums/count.clamp_min(1e-6), count.log1p(), grid), 1))

    def _conditioned_logits(self, spatial, context, card):
        condition = self.condition_norm(torch.cat((context.float(), card.float()), -1))
        for block in self.blocks:
            spatial = block(spatial, condition)
        return self.output(spatial).flatten(1)

    def residual_logits(self, context, entities, entity_mask, card_features, slots, entity_xy):
        with torch.autocast(device_type=context.device.type, enabled=False):
            spatial = self._project_spatial(entities, entity_mask, entity_xy)
            card = card_features.gather(1, slots[:, None, None].expand(-1, 1, self.card_dim)).squeeze(1)
            return self._conditioned_logits(spatial, context, card)

    def legal_residual_logits(self, context, entities, entity_mask, card_features, entity_xy, legal_slots):
        with torch.autocast(device_type=context.device.type, enabled=False):
            frames = len(context)
            slots, rows = legal_slots.transpose(0, 1).nonzero(as_tuple=True)
            logits = context.new_zeros(frames * 4, TILES, dtype=torch.float32)
            if len(rows):
                spatial = self._project_spatial(entities, entity_mask, entity_xy)
                residuals = []
                # Keep each convolution batch at most as large as one dense slot.
                for chunk_rows, chunk_slots in zip(rows.split(frames), slots.split(frames), strict=True):
                    residuals.append(self._conditioned_logits(
                        spatial[chunk_rows], context[chunk_rows], card_features[chunk_rows, chunk_slots]))
                logits = logits.index_copy(0, rows * 4 + slots, torch.cat(residuals))
            return logits.reshape(frames, 4, TILES)


class CardLocationResidual(nn.Module):
    """State-dict compatible learned card logits; no fixed tactical rules."""
    def __init__(self, original, vocab_size):
        super().__init__()
        self.original = original
        self.card_location_bias = nn.Embedding(vocab_size+1, TILES,
            device=next(original.parameters()).device, dtype=torch.float32)
        nn.init.zeros_(self.card_location_bias.weight)

    def prior_logits(self, card_ids):
        return self.card_location_bias(card_ids.long()+1)


class CardPriorSpatialHead(nn.Module):
    def __init__(self, source, vocab_size, channels):
        super().__init__()
        self.original = CardLocationResidual(source, vocab_size)
        self.spatial = SpatialResidual(_ZeroBase(source.tile_queries.device), entity_dim=source.cfg.entity_dim,
            context_dim=source.cfg.temporal_dim, card_dim=source.cfg.entity_dim, channels=channels)


def definition_for(source, variant, *, width=1024, cnn_channels=64, alpha=None, temperature=1.):
    if variant not in VARIANTS:
        raise ValueError('Unsupported learned placement variant')
    expected_alpha = .01 if variant in ('card_prior_001', 'card_prior_cnn_001') else 1. if variant == 'card_prior_1' else None
    if alpha is not None and alpha != expected_alpha:
        raise ValueError('Decoder alpha does not match its declared variant')
    if isinstance(width, bool) or not isinstance(width, int) or width < 1 or cnn_channels not in (32, 64):
        raise ValueError('Invalid learned placement width or CNN channels')
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Placement temperature must be finite and positive')
    return {'format': PLACEMENT_DEFINITION_FORMAT, 'variant': variant, 'width': width,
        'cnn_channels': cnn_channels, 'alpha': expected_alpha,
        'entity_dim': source.cfg.entity_dim, 'temporal_dim': source.cfg.temporal_dim,
        'entity_heads': source.cfg.entity_heads, 'vocab_size': source.cfg.vocab_size,
        'readout_version': source.cfg.readout_version, 'grid': [HEIGHT, WIDTH], 'temperature': float(temperature),
        'input_contract': 'raw_adapter_xy_and_raw_hand_ids; all_four_slots; fp32_placement',
        'card_wait_mass_preserved': True}


class PlacementAdapter(nn.Module):
    """Complete all-four-slot inference plus an explicitly selected GT path.

    ``decoder`` preserves the exact diagnostic state-dict keys. External callers
    load it with strict=True; adapter extra state requires a versioned definition
    when reconstructing a complete model. All computations use FP32 placement.
    """
    def __init__(self, source, definition):
        super().__init__()
        expected = definition_for(source, definition.get('variant'), width=definition.get('width', 1024),
            cnn_channels=definition.get('cnn_channels', 64), alpha=definition.get('alpha'),
            temperature=definition.get('temperature', 1.))
        if dict(definition) != expected:
            raise ValueError('Learned placement definition/version/input contract differs')
        self.definition = deepcopy(expected)
        self.vocab_size = source.cfg.vocab_size
        head = OriginalPlacementHead(source)
        variant = expected['variant']
        if variant == 'wide_residual':
            head.tile_actor = _WideReadout(head.tile_actor, source.cfg.entity_dim, expected['width']).to(head.tile_queries.device)
        if variant in ('card_prior_001', 'card_prior_1'):
            head = CardLocationResidual(head, self.vocab_size)
        elif variant == 'card_prior_cnn_001':
            head = CardPriorSpatialHead(head, self.vocab_size, expected['cnn_channels'])
        elif variant == 'spatial_cnn':
            head = SpatialResidual(head, entity_dim=source.cfg.entity_dim, context_dim=source.cfg.temporal_dim,
                card_dim=source.cfg.entity_dim, channels=expected['cnn_channels'])
        self.decoder = head.float()

    def get_extra_state(self):
        return deepcopy(self.definition)

    def set_extra_state(self, state):
        if state != self.definition:
            raise ValueError('Loaded placement extra-state definition differs')

    def _validate(self, context, entities, entity_mask, card_features, entity_xy, card_ids):
        b, d = len(context), self.definition['entity_dim']
        if context.ndim != 2 or context.shape[1] != self.definition['temporal_dim']:
            raise ValueError('Expected context [B,temporal_dim]')
        if entities.ndim != 3 or entities.shape[0] != b or entities.shape[2] != d or entity_mask.shape != entities.shape[:2]:
            raise ValueError('Encoded entities and entity mask differ')
        if card_features.shape != (b, 4, d) or not entity_mask.bool().any(-1).all():
            raise ValueError('Four card features and at least one valid entity token are required')
        variant = self.definition['variant']
        if variant in ('spatial_cnn', 'card_prior_cnn_001') and (entity_xy is None or entity_xy.shape != (*entities.shape[:2], 2)):
            raise ValueError('Spatial decoder requires aligned raw adapter entity_xy')
        if variant.startswith('card_prior'):
            if card_ids is None or card_ids.shape != (b, 4) or card_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError('Card decoder requires raw current hand card IDs [B,4]')
            if ((card_ids < -1) | (card_ids >= self.vocab_size)).any():
                raise ValueError('Raw hand card ID is outside [-1,vocab_size)')
        values = [context, entities, entity_mask, card_features]
        values += [value for value in (entity_xy, card_ids) if value is not None]
        if any(value.device != context.device for value in values):
            raise ValueError('Placement input fields must share one device')
        if next(self.decoder.parameters()).dtype != torch.float32:
            raise ValueError('Learned placement parameters require FP32; use autocast for the trunk')

    def forward(self, context, entities, entity_mask, card_features, entity_xy=None, card_ids=None, *,
                legal_slots=None):
        self._validate(context, entities, entity_mask, card_features, entity_xy, card_ids)
        variant, decoder = self.definition['variant'], self.decoder
        base = decoder
        if variant == 'card_prior_cnn_001':
            base = decoder.original.original
        elif variant.startswith('card_prior') or variant == 'spatial_cnn':
            base = decoder.original
        output = base.forward_all(context, entities, entity_mask, card_features)
        if variant.startswith('card_prior'):
            prior = decoder.original if variant == 'card_prior_cnn_001' else decoder
            output = output + prior.prior_logits(card_ids)
        if variant in ('spatial_cnn', 'card_prior_cnn_001'):
            spatial = decoder.spatial if variant == 'card_prior_cnn_001' else decoder
            if legal_slots is None:
                residuals = [spatial.residual_logits(context, entities, entity_mask, card_features,
                    torch.full((len(context),), slot, dtype=torch.long, device=context.device), entity_xy)
                    for slot in range(4)]
                output = output + torch.stack(residuals, 1)
            else:
                # These callers mask unavailable slots from the joint distribution.
                if (legal_slots.shape != (len(context), 4) or legal_slots.dtype != torch.bool
                        or legal_slots.device != context.device):
                    raise ValueError('Expected a boolean legal slot mask [B,4] on the placement device')
                output = output + spatial.legal_residual_logits(
                    context, entities, entity_mask, card_features, entity_xy, legal_slots)
        return output / self.definition['temperature']

    def forward_gt(self, context, entities, entity_mask, card_features, slots, entity_xy=None, card_ids=None):
        if slots.shape != (len(context),) or slots.dtype not in (torch.int32, torch.int64) or ((slots < 0) | (slots > 3)).any():
            raise ValueError('GT slots must be integral [B] in [0,3]')
        if card_ids is not None and card_ids.shape == (len(context),):
            hands = card_ids[:, None].expand(-1, 4)
        else:
            hands = card_ids
        self._validate(context, entities, entity_mask, card_features, entity_xy, hands)
        variant, decoder = self.definition['variant'], self.decoder
        base = decoder
        if variant == 'card_prior_cnn_001':
            base = decoder.original.original
        elif variant.startswith('card_prior') or variant == 'spatial_cnn':
            base = decoder.original
        output = base(context, entities, entity_mask, card_features, slots)
        if variant.startswith('card_prior'):
            prior = decoder.original if variant == 'card_prior_cnn_001' else decoder
            ids = hands.gather(1, slots.long()[:, None]).squeeze(1)
            output = output + prior.prior_logits(ids)
        if variant in ('spatial_cnn', 'card_prior_cnn_001'):
            spatial = decoder.spatial if variant == 'card_prior_cnn_001' else decoder
            output = output + spatial.residual_logits(context, entities, entity_mask, card_features, slots, entity_xy)
        return output / self.definition['temperature']


def _canonical(value):
    if isinstance(value, Mapping):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def placement_from_decoder_payload(model, payload, *, source_checkpoint_sha256):
    """Validate immutable source binding and strict-load the small decoder.

    The caller must provide a SHA256 verified when loading the original trunk.
    This function cannot infer a file hash from live tensors. The payload's
    versioned probe format is deliberately accepted as an explicit conversion;
    the returned adapter has its own versioned production definition.
    """
    if not isinstance(payload, Mapping) or payload.get('format') != DECODER_PAYLOAD_FORMAT:
        raise ValueError('Unsupported decoder payload format/version')
    if not re.fullmatch(r'[0-9a-f]{64}', str(source_checkpoint_sha256)) or payload.get('source_checkpoint_sha256') != source_checkpoint_sha256:
        raise ValueError('Decoder source checkpoint SHA256 differs from the verified trunk')
    if payload.get('contains_trunk') is not False or payload.get('contains_optimizer') is not False:
        raise ValueError('Expected a decoder-only payload without trunk/optimizer')
    if payload.get('heldout_used_for_selection') is not False:
        raise ValueError('Decoder payload lacks its heldout selection audit')
    model_config = payload.get('model_config', {})
    if model_config.get('model_architecture_version') != getattr(model, 'architecture_version', None):
        raise ValueError('Decoder source model architecture version differs')
    if _canonical(model_config.get('config')) != _canonical(asdict(model.cfg)):
        raise ValueError('Decoder source model config differs from the current trunk')
    if hasattr(model, 'temporal_cfg') and _canonical(model_config.get('temporal_config')) != _canonical(asdict(model.temporal_cfg)):
        raise ValueError('Decoder source temporal architecture differs')
    if hasattr(model, 'decision_ticks') and model_config.get('decision_ticks') != model.decision_ticks:
        raise ValueError('Decoder source decision interval differs')
    architecture = payload.get('architecture_config', {})
    for field in ('entity_dim', 'temporal_dim', 'readout_version'):
        if architecture.get(field) != getattr(model.cfg, field):
            raise ValueError('Decoder architecture shape/config differs: '+field)
    definition = definition_for(model, payload.get('variant'), width=architecture.get('width', 1024),
        cnn_channels=architecture.get('cnn_channels', 64), alpha=architecture.get('alpha'),
        temperature=payload.get('temperature', 1.))
    weights = payload.get('decoder')
    if not isinstance(weights, Mapping) or not weights or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Decoder state must contain only finite tensors')
    adapter = PlacementAdapter(model, definition)
    adapter.decoder.load_state_dict(weights, strict=True)
    if payload.get('parameter_count') != sum(value.numel() for value in adapter.decoder.parameters()):
        raise ValueError('Decoder parameter count differs from the declared structure')
    adapter.train(model.training)
    return adapter


def attach_learned_placement(model, decoder_payload, *, source_checkpoint_sha256=None):
    """Register only after complete validation; policy dispatch is explicit.

    Integration calls ``model.learned_placement(context,tokens,mask,cards,xy,hands)``.
    This function does not monkeypatch placement_logits or silently change the
    state-policy checkpoint factory. Full model restoration must first build the
    saved ``learned_placement_definition`` then strictly load every model tensor.
    """
    if hasattr(model, 'learned_placement'):
        raise ValueError('A learned placement decoder is already attached')
    verified = source_checkpoint_sha256 or getattr(model, 'source_checkpoint_sha256', None)
    adapter = placement_from_decoder_payload(model, decoder_payload, source_checkpoint_sha256=verified)
    model.add_module('learned_placement', adapter)
    model.learned_placement_definition = deepcopy(adapter.definition)
    if hasattr(model, 'mark_weights_updated'):
        model.mark_weights_updated()
    return adapter
