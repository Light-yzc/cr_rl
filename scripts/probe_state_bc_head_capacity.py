"""Placement capacity/learning-rate probes on a frozen chronological BC model.

This does not change cloud training, save fitted weights, or install anything.
Readout-only mode freezes placement query/cross-attention as well; full-head
mode fits the whole placement decoder. Every original chronological frame
advances memory. No fitted weights are saved or published.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from diagnose_state_bc_generalization import atomic_json, digest, soft_targets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True, help='Prepared diagnostic cohort manifest')
    parser.add_argument('--parent-manifest', type=Path, required=True, help='Checkpoint-bound original full manifest')
    parser.add_argument('--cohort', type=Path, help='Frozen cohort JSON; alternative to explicit per-partition IDs')
    parser.add_argument('--train-scope', choices=('readout-only', 'full-head'), default='readout-only')
    parser.add_argument('--float32-matmul-precision', choices=('high', 'highest'),
                        help='Default high for cached MLP, highest for full-head kernel comparison')
    parser.add_argument('--include-spatial-film', action='store_true',
                        help='Full-head only: add zero-gated readout-v2 spatial normalization/FiLM arm')
    parser.add_argument('--variants', help='Explicit comma-separated original,wide_residual,spatial_film,spatial_cnn arms')
    parser.add_argument('--cnn-channels', type=int, choices=(32, 64), default=64)
    parser.add_argument('--xy-cache', type=Path, help='Small aligned pre-encoder entity-coordinate NPZ sidecar')
    parser.add_argument('--train-card-prior', type=Path, help='Full-training-only mass/count JSON for data-initialization arms')
    parser.add_argument('--prepare-xy-cache', action='store_true', help='CPU only: verify/prepare coordinate sidecar, then exit')
    parser.add_argument('--prepare-card-cache', action='store_true', help='CPU only: add aligned GT hand-card IDs to existing small sidecar')
    parser.add_argument('--feature-cache-dir', type=Path,
                        help='Reuse the single raw feature cache from a matching completed readout probe')
    parser.add_argument('--delete-feature-cache-after', action='store_true',
                        help='Remove only the verified reused block_*.pt cache after successful completion')
    parser.add_argument('--out', type=Path, required=True)
    for part in ('train', 'validation', 'heldout'):
        parser.add_argument('--' + part + '-games', help='Comma-separated original replay IDs')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=0, help='Override steps: one epoch exposes each training play once')
    parser.add_argument('--extend-winner-to', type=int, default=0,
                        help='Continue the best final validation-CE arm with its live Adam state; no checkpoint saved')
    parser.add_argument('--extend-winner-epochs', type=int, default=0, help='Continuation target on the same action-epoch clock')
    parser.add_argument('--extend-top-k', type=int, choices=(1, 2), default=1)
    parser.add_argument('--selection-policy', choices=('ce', 'ce-and-within2'), default='ce')
    parser.add_argument('--geometry-ce-margin', type=float, default=.1,
                        help='Geometric finalist must have CE <= original baseline validation CE + this margin')
    parser.add_argument('--heldout-final-only', action='store_true',
                        help='Do not score heldout during screening; only final continued finalists and frozen baseline')
    parser.add_argument('--restore-best-validation', action='store_true',
                        help='Keep only small decoder GPU weights at the best validation epoch; test that epoch after continuation')
    parser.add_argument('--validation-patience', type=int, default=0,
                        help='Continuation only: stop after this many validation nodes without a new lower CE; zero disables')
    parser.add_argument('--save-best-decoder', type=Path,
                        help='Explicitly retain ONE final validation-ranked decoder, no trunk/Adam, for the authorized implementation follow-up')
    parser.add_argument('--learning-rates', default='3e-5,3e-4')
    parser.add_argument('--batch-size', type=int, default=16, help='GT ordinary-play rows, not frames')
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.)
    parser.add_argument('--ram-limit-gib', type=float, default=6.)
    parser.add_argument('--vram-cache-limit-gib', type=float, default=4.)
    parser.add_argument('--gpu-memory-limit-gib', type=float, default=0.,
                        help='Optional PyTorch per-process allocator cap for bounded shared-GPU probes')
    parser.add_argument('--cache-loader-workers', type=int, choices=(1, 2), default=2,
                        help='Bounded CPU deserialization prefetch; never keep more than two small blocks')
    parser.add_argument('--verify-rows', type=int, default=32)
    parser.add_argument('--max-initial-logp-error', type=float, default=1e-3)
    parser.add_argument('--save-feature-cache', action='store_true',
                        help='Optional bounded per-block raw/features cache under out; never model weights')
    args = parser.parse_args(argv)
    args.games = {}
    cohort = json.loads(args.cohort.read_text()) if args.cohort else None
    for part in ('train', 'validation', 'heldout'):
        explicit = getattr(args, part + '_games')
        if explicit and cohort:
            parser.error('Use --cohort or explicit game IDs, not both')
        if not cohort and not explicit:
            parser.error('Provide --cohort or all three per-partition game ID arguments')
        games = set(cohort['partitions'][part]['game_ids']) if cohort else set(map(int, explicit.split(',')))
        if not games:
            parser.error('Each partition needs at least one game')
        args.games[part] = games
    if any(args.games[left] & args.games[right] for left, right in
           [('train', 'validation'), ('train', 'heldout'), ('validation', 'heldout')]):
        parser.error('Cohorts must be disjoint complete games')
    args.learning_rates = [float(value) for value in args.learning_rates.split(',')]
    if (min(args.steps, args.batch_size, args.eval_every, args.eval_batch_size,
            args.width, args.verify_rows) < 1 or min(args.learning_rates) <= 0 or
            args.max_grad_norm <= 0 or args.ram_limit_gib <= 0 or args.vram_cache_limit_gib <= 0):
        parser.error('Positive steps, sizes, rates, guards, and tolerance required')
    if args.weight_decay < 0 or args.max_initial_logp_error <= 0:
        parser.error('Invalid weight decay or verification tolerance')
    if min(args.epochs, args.extend_winner_epochs, args.extend_winner_to, args.gpu_memory_limit_gib, args.geometry_ce_margin, args.validation_patience) < 0:
        parser.error('Epoch counts, continuation steps, memory cap and geometry margin cannot be negative')
    if args.delete_feature_cache_after and not args.feature_cache_dir:
        parser.error('Deleting cache requires an explicitly named --feature-cache-dir')
    if args.include_spatial_film and args.train_scope != 'full-head':
        parser.error('Spatial FiLM arm belongs only to the full-head experiment')
    args.variants = (args.variants.split(',') if args.variants else
        (['original', 'wide_residual', 'spatial_film'] if args.include_spatial_film else ['original', 'wide_residual']))
    if set(args.variants) - {'original', 'wide_residual', 'spatial_film', 'spatial_cnn', 'card_prior_001', 'card_prior_1', 'card_prior_cnn_001'}:
        parser.error('Unknown experimental variant')
    if set(args.variants) & {'spatial_film', 'spatial_cnn', 'card_prior_001', 'card_prior_1', 'card_prior_cnn_001'} and args.train_scope != 'full-head':
        parser.error('Spatial variants require full-head mode')
    if (args.prepare_xy_cache or args.prepare_card_cache) and (not args.xy_cache or not args.feature_cache_dir):
        parser.error('Spatial CNN/coordinate preparation requires --xy-cache and --feature-cache-dir')
    if any(value.startswith('card_prior_') for value in args.variants) and not args.train_card_prior:
        parser.error('Card-prior initialization arms require --train-card-prior and cached GT IDs')
    args.float32_matmul_precision = args.float32_matmul_precision or ('highest' if args.train_scope == 'full-head' else 'high')
    return args


def gt_tile_features(model, context, entities, entity_mask, card_features, slots):
    """Exactly the existing placement decoder, selecting one true slot per row.

    Attention has no query-to-query interaction: reducing 4*576 to 576 queries
    changes no mathematical attention distribution. Extraction verifies the
    actual mixed-precision result against the full four-slot production head.
    """
    if not model.cfg.tile_head_fp32 or model.tile_context.weight.dtype != torch.float32:
        raise ValueError('This probe pins the current checkpoint FP32 placement boundary')
    with torch.autocast(device_type=context.device.type, enabled=False):
        context, entities, card_features = context.float(), entities.float(), card_features.float()
        batch = len(context)
        index = torch.arange(batch, device=context.device)
        card = card_features[index, slots]
        base = model.tile_queries[0, slots]
        projected_context = model.tile_context(context)
        queries = base + model.slot_embedding[0, slots] + card[:, None] + projected_context[:, None]
        if model.cfg.spatial_fourier:
            queries = queries + .02 * model.tile_grid_features[None].to(queries.dtype)
        if model.cfg.readout_version == 2:
            spatial = base + .02 * model.tile_grid_features[None]
            normalized = model.tile_spatial_norm(
                base + model.tile_spatial_projection(model.tile_grid_features)[None])
            gamma, beta = model.tile_film(card + projected_context).chunk(2, -1)
            modulated = normalized * (1 + gamma[:, None]) + beta[:, None]
            queries = queries + model.tile_spatial_gate.tanh() * (modulated - spatial)
        features, _ = model.tile_cross_attention(queries, entities, entities,
            key_padding_mask=~entity_mask.bool(), need_weights=False)
        if model.cfg.tile_query_residual:
            features = features + queries
        return features


class WiderResidualReadout(nn.Module):
    """Same initial function, with a zero-output wider MLP residual branch."""
    def __init__(self, original, dimensions, width):
        super().__init__()
        self.original = deepcopy(original)
        self.branch = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, width),
            nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, features):
        return self.original(features) + self.branch(features)


RAW_FIELDS = ('context', 'entities', 'entity_mask', 'card_features', 'slots')


class TrainablePlacementHead(nn.Module):
    """All placement parameters, preserving the original four-slot attention path."""
    probe_input_fields = RAW_FIELDS
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

    def add_zero_gated_spatial_film(self):
        if self.cfg.readout_version != 1:
            raise ValueError('Spatial FiLM migration arm expects original readout version 1')
        self.cfg = replace(self.cfg, readout_version=2)
        dimensions, device = self.cfg.entity_dim, self.tile_queries.device
        self.tile_spatial_projection = nn.Linear(dimensions, dimensions, bias=False, device=device)
        nn.init.eye_(self.tile_spatial_projection.weight)
        self.tile_spatial_norm = nn.LayerNorm(dimensions, device=device)
        self.tile_film = nn.Linear(dimensions, 2 * dimensions, device=device)
        nn.init.zeros_(self.tile_film.weight)
        nn.init.zeros_(self.tile_film.bias)
        self.tile_spatial_gate = nn.Parameter(torch.zeros((), device=device))

    def forward(self, context, entities, entity_mask, card_features, slots):
        # Keep all 4*576 attention queries to avoid the query-count TF32 kernel
        # drift found by the extraction smoke test. Only the GT slot's MLP is
        # evaluated; no unselected tile head contributes to this tile-only loss.
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
                    self.tile_spatial_projection(self.tile_grid_features)[None, None])
                gamma, beta = self.tile_film(card_features + self.tile_context(context)[:, None]).chunk(2, -1)
                modulated = normalized * (1 + gamma[:, :, None]) + beta[:, :, None]
                queries = queries + self.tile_spatial_gate.tanh() * (modulated - spatial)
            queries = queries.reshape(n, 4 * 576, -1)
            features, _ = self.tile_cross_attention(queries, entities, entities,
                key_padding_mask=~entity_mask.bool(), need_weights=False)
            if self.cfg.tile_query_residual:
                features = features + queries
            features = features.reshape(n, 4, 576, -1)[torch.arange(n, device=context.device), slots]
            return self.tile_actor(features).squeeze(-1)


class CardPriorSpatialHead(nn.Module):
    """Training-only card initialization + original decoder + zero-output CNN."""
    probe_input_fields = RAW_FIELDS + ('entity_xy', 'card_ids')

    def __init__(self, source, prior, vocab_size, channels):
        super().__init__()
        from state_bc_card_location_probe import CardLocationPriorResidual
        from state_bc_spatial_probe import EntitySpatialResidual
        self.original = CardLocationPriorResidual(source, prior, vocab_size=vocab_size, alpha=.01)
        class ZeroBase(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros((), device=device), requires_grad=False)
            def forward(self, context, **_kwargs):
                return context.new_zeros(len(context), 576)
        self.spatial = EntitySpatialResidual(ZeroBase(source.tile_queries.device),
            entity_dim=source.cfg.entity_dim, context_dim=source.cfg.temporal_dim,
            card_dim=source.cfg.entity_dim, channels=channels)

    def forward(self, context, entities, entity_mask, card_features, slots, entity_xy, card_ids):
        inputs = dict(context=context, entities=entities, entity_mask=entity_mask,
            card_features=card_features, slots=slots)
        return self.original(**inputs, card_ids=card_ids) + self.spatial.residual_logits(**inputs, entity_xy=entity_xy)

    def prior_logits(self, card_ids):
        return self.original.prior_logits(card_ids)

    def initialization_info(self):
        return {**self.original.initialization_info(),
            'spatial_zero_output_branch': self.spatial.complexity_estimate(),
            'same_initial_function_as_card_prior_001': True}


def fixed_batches(size, batch_size, steps, seed):
    """Same shuffled action exposure, including equal partial epoch batches."""
    if size < 1:
        raise ValueError('No training plays')
    rng = np.random.default_rng(seed)
    batches = []
    while len(batches) < steps:
        order = rng.permutation(size)
        for start in range(0, size, batch_size):
            batches.append(order[start:start + batch_size])
            if len(batches) == steps:
                break
    return batches


def gather_rows(blocks, lookup, indices, full_head=False, spatial=False, card_ids=False):
    """Gather one small batch without ever concatenating the whole GPU cache."""
    grouped = defaultdict(list)
    for index in indices:
        block, row = lookup[int(index)]
        grouped[block].append(row)
    values = defaultdict(list)
    maximum_entities = max(blocks[index]['entities'].shape[1] for index in grouped) if full_head else 0
    fields = ('legal', 'targets', 'entropy', 'tiles') + (RAW_FIELDS if full_head else ('features',))
    if spatial:
        fields += ('entity_xy',)
    if card_ids:
        fields += ('card_ids',)
    for block_index, rows in grouped.items():
        block = blocks[block_index]
        chosen = torch.as_tensor(rows, device=block['tiles'].device)
        for key in fields:
            selected = block[key].index_select(0, chosen)
            # Entity buckets differ across chronological chunks. Added tokens
            # are masked; no game/seat history or real token is discarded.
            if full_head and key in ('entities', 'entity_mask', 'entity_xy'):
                padding = maximum_entities - selected.shape[1]
                if padding:
                    selected = torch.nn.functional.pad(selected,
                        (0, 0, 0, padding) if key in ('entities', 'entity_xy') else (0, padding))
            values[key].append(selected)
    return {key: torch.cat(value) for key, value in values.items()}


def conditional_logp(readout, features, legal):
    # Match the production probability boundary/mask exactly.
    logits = (readout(**features) if isinstance(features, dict) else readout(features).squeeze(-1)).float()
    return torch.log_softmax(logits.masked_fill(~legal.bool(), -1e9), -1).masked_fill(~legal.bool(), -1e9)


@torch.no_grad()
def evaluate_unbatched(readout, blocks, batch_size, include_heldout=True):
    readout.eval()
    seats = defaultdict(lambda: defaultdict(float))
    for block in blocks:
        if not include_heldout and block['partition'] == 'heldout':
            continue
        key = (block['partition'], block['replay_id'], block['seat'])
        for start in range(0, len(block['tiles']), batch_size):
            selection = slice(start, start + batch_size)
            fields = getattr(readout, 'probe_input_fields', None)
            features = ({key: block[key][selection] for key in fields}
                if fields else block['features'][selection])
            lp = conditional_logp(readout, features, block['legal'][selection])
            ce = -(block['targets'][selection] * lp).sum(-1)
            target, predicted = block['tiles'][selection], lp.argmax(-1)
            distance = (predicted % 18 - target % 18).abs() + (predicted // 18 - target // 18).abs()
            packets = torch.stack((ce.sum(), block['entropy'][selection].sum(),
                (distance <= 2).float().sum(), (predicted == target).float().sum(), distance.float().sum())).cpu().tolist()
            for name, value in zip(('ce', 'entropy', 'within2', 'top1', 'distance'), packets):
                seats[key][name] += value
            seats[key]['plays'] += len(ce)
    per_seat = []
    for (part, replay, seat), totals in seats.items():
        n = totals['plays']
        row = {'partition': part, 'replay_id': replay, 'seat': seat, 'plays': int(n),
               **{key: value / n for key, value in totals.items() if key != 'plays'}}
        row['excess_ce'] = row['ce'] - row['entropy']
        per_seat.append(row)
    summary = {}
    for part in ('train', 'validation', 'heldout'):
        selected = [row for row in per_seat if row['partition'] == part]
        if not selected:
            continue
        n = sum(row['plays'] for row in selected)
        metrics = ('ce', 'entropy', 'excess_ce', 'within2', 'top1', 'distance')
        summary[part] = {'games': len({row['replay_id'] for row in selected}),
            'seats': len(selected), 'plays': n,
            'action_weighted': {key: sum(row[key] * row['plays'] for row in selected) / n for key in metrics},
            'equal_seat': {key: float(np.mean([row[key] for row in selected])) for key in metrics}}
    return {'summary': summary, 'seats': per_seat}


@torch.no_grad()
def evaluate(readout, blocks, batch_size, include_heldout=True):
    """Batch independent decoder rows with equal entity bucket sizes.

    Whole-game representations are already frozen. No temporal state is moved,
    no entity padding is increased, and every action retains its own game/seat
    attribution. This avoids 8k tiny decoder launches on the full corpus.
    """
    readout.eval()
    grouped, lookup = defaultdict(list), []
    for block_index, block in enumerate(blocks):
        if not include_heldout and block['partition'] == 'heldout':
            continue
        bucket = block['entities'].shape[1]
        for row in range(len(block['tiles'])):
            grouped[bucket].append(len(lookup))
            lookup.append((block_index, row))
    totals = defaultdict(lambda: defaultdict(float))
    fields = getattr(readout, 'probe_input_fields', None)
    for indices in grouped.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            # One gather per decoder input field, preserve its actual row order.
            batch = gather_rows(blocks, lookup, selected, full_head=bool(fields),
                spatial=bool(fields and 'entity_xy' in fields), card_ids=bool(fields and 'card_ids' in fields))
            features = {key: batch[key] for key in fields} if fields else batch['features']
            lp = conditional_logp(readout, features, batch['legal'])
            target, predicted = batch['tiles'], lp.argmax(-1)
            ce = -(batch['targets'] * lp).sum(-1)
            distance = (predicted % 18 - target % 18).abs() + (predicted // 18 - target // 18).abs()
            packets = torch.stack((ce, batch['entropy'], (distance <= 2).float(),
                (predicted == target).float(), distance.float()), -1).cpu().tolist()
            # gather_rows groups by block, so its output ordering is grouped
            # likewise rather than the caller's original shuffled order.
            by_block = defaultdict(list)
            for index in selected:
                block_index, row = lookup[index]
                by_block[block_index].append(row)
            order = [(block_index, row) for block_index, rows in by_block.items() for row in rows]
            for (block_index, _), packet in zip(order, packets):
                block = blocks[block_index]
                key = (block['partition'], block['replay_id'], block['seat'])
                totals[key]['plays'] += 1
                for name, value in zip(('ce', 'entropy', 'within2', 'top1', 'distance'), packet):
                    totals[key][name] += value
            del batch, features, lp, packets
    seats = []
    for (part, replay, seat), values in totals.items():
        n = values['plays']
        row = {'partition': part, 'replay_id': replay, 'seat': seat, 'plays': int(n),
            **{key: value / n for key, value in values.items() if key != 'plays'}}
        row['excess_ce'] = row['ce'] - row['entropy']
        seats.append(row)
    summary = {}
    for part in ('train', 'validation', 'heldout'):
        chosen = [row for row in seats if row['partition'] == part]
        if not chosen:
            continue
        n = sum(row['plays'] for row in chosen)
        metrics = ('ce', 'entropy', 'excess_ce', 'within2', 'top1', 'distance')
        summary[part] = {'games': len({row['replay_id'] for row in chosen}), 'seats': len(chosen), 'plays': n,
            'action_weighted': {key: sum(row[key] * row['plays'] for row in chosen) / n for key in metrics},
            'equal_seat': {key: float(np.mean([row[key] for row in chosen])) for key in metrics}}
    return {'summary': summary, 'seats': seats}


def extract(args, model, metadata, settings, assignments):
    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.training.state_bc import ReplaySequenceDataset, check_ram_budget, make_state_schema, prepare_state_chunk
    schema = make_state_schema(metadata)
    adapter = CrForgeObservationAdapter(schema, model.cfg.vocab_size, entity_bucket=16)
    needed = set().union(*args.games.values())
    blocks, cache_bytes, seen, verified = [], 0, set(), []
    original_placement = model.placement_logits
    active = {}
    cache_root = args.out / 'feature_cache'
    if args.save_feature_cache:
        cache_root.mkdir(exist_ok=True)

    def capture(context, entities, entity_mask, card_features):
        nonlocal cache_bytes
        start, stop = active['cursor'], active['cursor'] + len(context)
        active['cursor'] = stop
        chosen = active['play'][start:stop]
        # Placement output never feeds temporal state or card logits. Unused
        # WAIT/ability rows may therefore have zero tile logits in this probe.
        output = context.new_zeros((len(context), 4, 576), dtype=torch.float32)
        if not chosen.any():
            return output
        chosen_t = torch.as_tensor(chosen, device='cuda')
        labels = active['labels'][start:stop][chosen]
        slots = torch.as_tensor((labels - 1) // 576, device='cuda')
        tiles = torch.as_tensor((labels - 1) % 576, device='cuda')
        raw = {'context': context[chosen_t].detach().clone(),
            'entities': entities[chosen_t].detach().clone(),
            'entity_mask': entity_mask[chosen_t].detach().clone(),
            'card_features': card_features[chosen_t].detach().clone(), 'slots': slots}
        # Query-count changes can select different TF32 attention kernels.
        # Cache the ACTUAL existing four-slot head's pre-MLP features, then
        # keep only the GT slot. The repeated MLP probes use 576 tiles/row.
        captured = []
        def keep_gt_features(_module, inputs):
            full_features = inputs[0].reshape(len(slots), 4, 576, -1)
            captured.append(full_features[torch.arange(len(slots), device='cuda'), slots].detach().clone())
        hook = model.tile_actor.register_forward_pre_hook(keep_gt_features)
        try:
            full_logits = original_placement(raw['context'], raw['entities'], raw['entity_mask'], raw['card_features'])
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError('Original placement path did not call the existing tile readout once')
        features = captured[0]
        legal = active['legal'][start:stop][chosen, 1:2305].reshape(-1, 4, 576)[
            np.arange(len(labels)), (labels - 1) // 576]
        legal_t = torch.as_tensor(legal, dtype=torch.bool, device='cuda')
        targets = torch.as_tensor(soft_targets(tiles.cpu().numpy(), legal, settings['tile_label_sigma']),
            dtype=torch.float32, device='cuda')
        block = {**raw, 'features': features.detach(), 'legal': legal_t, 'targets': targets,
            'tiles': tiles, 'entropy': -(targets * targets.clamp_min(1e-30).log()).sum(-1),
            'partition': active['partition'], 'replay_id': active['replay_id'], 'seat': active['seat']}
        with torch.autocast('cuda', enabled=False):
            local_logits = model.tile_actor(features).squeeze(-1).float()
        if sum(row['rows'] for row in verified) < args.verify_rows:
            truth = full_logits[torch.arange(len(slots), device='cuda'), slots].float()
            full_lp = torch.log_softmax(truth.masked_fill(~legal_t, -1e9), -1)
            local_lp = torch.log_softmax(local_logits.masked_fill(~legal_t, -1e9), -1)
            error = float((full_lp - local_lp).abs()[legal_t].max())
            verified.append({'replay_id': active['replay_id'], 'seat': active['seat'],
                'rows': len(slots), 'legal_conditional_logp_max_error': error,
                'raw_logit_max_error': float((truth - local_logits).abs().max())})
            if error > args.max_initial_logp_error:
                raise ValueError('GT-slot-only head differs from full production head: ' + str(error))
        block_bytes = sum(t.numel() * t.element_size() for t in block.values() if isinstance(t, torch.Tensor))
        if cache_bytes + block_bytes > args.vram_cache_limit_gib * 1024**3:
            raise MemoryError('Feature cache exceeds configured VRAM cap; reduce cohort games')
        blocks.append(block)
        cache_bytes += block_bytes
        if args.save_feature_cache:
            # Transfer at most one small selected-play block to CPU at a time.
            payload = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in block.items()}
            torch.save(payload, cache_root / f'block_{len(blocks):06d}.pt')
            del payload
        selected_indices = chosen_t.nonzero().flatten()
        output[selected_indices, slots] = local_logits
        return output

    dataset = ReplaySequenceDataset(args.dataset, args.out / '.replay-cache', ram_limit_gib=args.ram_limit_gib)
    if not needed <= dataset.replay_ids or (args.cohort and dataset.replay_ids != needed):
        raise ValueError('Prepared manifest does not contain the requested frozen cohort')
    dataset_assignments = {rid: 'probe' for rid in needed}
    episodes = dataset.episodes(dataset_assignments, 'probe')
    model.placement_logits = capture
    try:
        with torch.no_grad():
            for episode in episodes:
                rid, seat = episode.replay_id, episode.seat
                seen.add(rid)
                state = model.initial_state(1, device='cuda', dtype=torch.float16)
                for start, rows in episode.chunks(32):
                    check_ram_budget(args.ram_limit_gib)
                    inputs = prepare_state_chunk(episode, rows, adapter, 'cuda')
                    labels = episode.labels[start:start + len(rows)]
                    active.update(cursor=0, labels=labels, legal=episode.arrays['masks'][rows],
                        play=episode.supervised[start:start + len(rows)] & (labels > 0) & (labels < 2305),
                        partition=assignments[rid], replay_id=rid, seat=seat)
                    with torch.autocast('cuda', dtype=torch.float16):
                        lp, _, state = model.forward_state(**inputs, state=state, compute_value=False)
                    if active['cursor'] != len(rows) or not torch.isfinite(lp).all():
                        raise ValueError('Chronological placement capture lost frame alignment')
                    state = model.detach_state(state)
                del state
                print(json.dumps({'phase': 'extract', 'game': rid, 'seat': seat, 'games_done': len(seen),
                    'blocks': len(blocks), 'cache_gib': cache_bytes / 1024**3,
                    'rss_gib': check_ram_budget(args.ram_limit_gib) / 1024**3}), flush=True)
    finally:
        episodes.close()
        model.placement_logits = original_placement
    if seen != needed or not verified:
        raise ValueError('Missing requested games or no production-head verification rows')
    return blocks, {'bytes': cache_bytes, 'games': sorted(seen), 'blocks': len(blocks),
        'raw_capture_retained': ['context', 'entities', 'entity_mask', 'card_features', 'slots'],
        'full_head_parity': verified, 'all_frames_from_tick_zero': True,
        'features': 'Original full four-slot production attention, GT slot pre-MLP features only',
        'gt_slot_attention_recompute_used': False,
        'recompute_reason': '4090 TF32 query-count kernel change produced 0.01069 logp error in smoke; not accepted'}


def prepare_entity_xy(args):
    """CPU-only join of cached GT-play rows to original adapter coordinates."""
    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.training.state_bc import ReplaySequenceDataset, check_ram_budget
    if args.xy_cache.resolve().drive.upper() != 'G:':
        raise ValueError('Coordinate sidecar must stay on G:')
    if args.xy_cache.exists():
        raise ValueError('Coordinate sidecar already exists; do not duplicate cache')
    needed = set().union(*args.games.values())
    dataset = ReplaySequenceDataset(args.dataset, args.xy_cache.parent / '.xy-replay-cache', ram_limit_gib=args.ram_limit_gib)
    adapter = CrForgeObservationAdapter(dataset.schema, len(dataset.metadata['vocabulary']['names']), entity_bucket=16)
    files = sorted(args.feature_cache_dir.glob('block_*.pt'))
    episodes = dataset.episodes({rid: 'probe' for rid in needed}, 'probe')
    coordinates, index, seats = {}, 0, set()
    try:
        for episode in episodes:
            for start, rows in episode.chunks(32):
                labels = episode.labels[start:start + len(rows)]
                selected = episode.supervised[start:start + len(rows)] & (labels > 0) & (labels < 2305)
                if not selected.any():
                    continue
                payload = torch.load(files[index], map_location='cpu', weights_only=False)
                index += 1
                if payload['replay_id'] != episode.replay_id or payload['seat'] != episode.seat:
                    raise ValueError('Coordinate sidecar game/seat ordering mismatch')
                truth = labels[selected]
                if (not np.array_equal(payload['slots'].numpy(), (truth - 1) // 576) or
                        not np.array_equal(payload['tiles'].numpy(), (truth - 1) % 576)):
                    raise ValueError('Coordinate sidecar GT action labels mismatch')
                observations = np.array(episode.arrays['observations'][rows], copy=True)
                _, entities, mask = adapter(observations)
                if not np.array_equal(payload['entity_mask'].numpy(), mask[selected]):
                    raise ValueError('Coordinate sidecar entity row/mask alignment mismatch')
                legal = episode.arrays['masks'][rows][selected, 1:2305].reshape(-1, 4, 576)[
                    np.arange(len(truth)), (truth - 1) // 576]
                if not np.array_equal(payload['legal'].numpy(), legal):
                    raise ValueError('Coordinate sidecar legal tile mask mismatch')
                coordinates[files[index - 1].stem] = np.array(entities[selected, :, 4:6], dtype=np.float32, copy=True)
                hand = np.asarray(dataset.schema.header['hand_card'])
                coordinates[files[index - 1].stem + '_card_ids'] = np.array(
                    observations[selected, hand[(truth - 1) // 576]], dtype=np.int64, copy=True)
                seats.add((episode.replay_id, episode.seat))
                del payload, observations, entities, mask
                check_ram_budget(args.ram_limit_gib)
    finally:
        episodes.close()
    if index != len(files) or len(seats) != 2 * len(needed):
        raise ValueError('Coordinate sidecar did not cover the exact cached complete cohort')
    np.savez_compressed(args.xy_cache, **coordinates)
    report = {'cpu_only': True, 'blocks': index, 'seats': len(seats),
        'sidecar_bytes': args.xy_cache.stat().st_size, 'sha256': digest(args.xy_cache),
        'coordinate_fields': 'Original pre-encoder adapter entities[...,4:6], FP32, bucket16',
        'alignment_checks': ['replay_id', 'seat', 'GT slots', 'GT tiles', 'entity mask', 'GT legal mask']}
    atomic_json(args.xy_cache.with_suffix('.json'), report)
    return report


def add_card_ids_to_coordinate_sidecar(args):
    """CPU-only bounded join; do not reload/copy the large raw feature cache."""
    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.training.state_bc import ReplaySequenceDataset, check_ram_budget
    report_path = args.xy_cache.with_suffix('.json')
    report = json.loads(report_path.read_text())
    if digest(args.xy_cache) != report['sha256']:
        raise ValueError('Existing coordinate sidecar checksum changed')
    with np.load(args.xy_cache, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    needed = set().union(*args.games.values())
    dataset = ReplaySequenceDataset(args.dataset, args.xy_cache.parent / '.xy-replay-cache', ram_limit_gib=args.ram_limit_gib)
    adapter = CrForgeObservationAdapter(dataset.schema, len(dataset.metadata['vocabulary']['names']), entity_bucket=16)
    hand, index = np.asarray(dataset.schema.header['hand_card']), 0
    episodes = dataset.episodes({rid: 'probe' for rid in needed}, 'probe')
    try:
        for episode in episodes:
            for start, rows in episode.chunks(32):
                labels = episode.labels[start:start + len(rows)]
                selected = episode.supervised[start:start + len(rows)] & (labels > 0) & (labels < 2305)
                if not selected.any():
                    continue
                index += 1
                key = f'block_{index:06d}'
                obs = np.array(episode.arrays['observations'][rows], copy=True)
                _, entities, _ = adapter(obs)
                if not np.array_equal(arrays[key], entities[selected, :, 4:6]):
                    raise ValueError('Card-ID join no longer aligns with verified coordinate rows')
                slots = (labels[selected] - 1) // 576
                arrays[key + '_card_ids'] = np.array(obs[selected, hand[slots]], dtype=np.int64, copy=True)
                check_ram_budget(args.ram_limit_gib)
    finally:
        episodes.close()
    if index != report['blocks']:
        raise ValueError('Card-ID join did not cover the verified coordinate sidecar')
    temporary = args.xy_cache.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(args.xy_cache)
    report.update(card_ids_added=True, card_identity='Original pre-action hand_card[GT slot]',
        sidecar_bytes=args.xy_cache.stat().st_size, sha256=digest(args.xy_cache))
    atomic_json(report_path, report)
    return report


def bounded_cache_reads(files, workers=2):
    if workers == 1:
        for path in files:
            yield path, torch.load(path, map_location='cpu', weights_only=False)
        return
    iterator, pending = iter(files), deque()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        def enqueue():
            path = next(iterator, None)
            if path is not None:
                pending.append((path, executor.submit(torch.load, path, map_location='cpu', weights_only=False)))
        for _ in range(workers):
            enqueue()
        while pending:
            path, future = pending.popleft()
            value = future.result()
            yield path, value
            del value, future
            enqueue()


def packed_cache_reads(root, index):
    """Sequential mmap -> one small writable CPU block -> GPU; never load all RAM."""
    binary = (root / index['bin']['path']).resolve()
    if binary.parent != root or binary.stat().st_size != index['bin']['bytes']:
        raise ValueError('Packed raw cache binary path/size mismatch')
    if digest(binary) != index['bin']['sha256']:
        raise ValueError('Packed raw cache binary checksum mismatch')
    memory = np.memmap(binary, mode='r', dtype=np.uint8)
    try:
        for record in index['blocks']:
            values = dict(record['metadata'])
            for key, spec in record['tensors'].items():
                dtype, shape = np.dtype(spec['dtype']), tuple(spec['shape'])
                if int(np.prod(shape)) * dtype.itemsize != spec['nbytes'] or spec['offset'] + spec['nbytes'] > len(memory):
                    raise ValueError('Packed raw tensor shape/offset mismatch')
                view = np.ndarray(shape, dtype=dtype, buffer=memory, offset=spec['offset'])
                values[key] = torch.from_numpy(np.array(view, copy=True))
                del view
            yield root / (record['name'] + '.pt'), values
            del values
    finally:
        memory._mmap.close()


def load_feature_cache(args, checkpoint_sha, parent_sha, assignments):
    """Load at most one small CPU block at a time, retain the whole cache on GPU."""
    from cr_rl.training.state_bc import check_ram_budget
    root = args.feature_cache_dir.resolve()
    source_protocol = json.loads((root.parent / 'protocol.json').read_text())
    if source_protocol.get('completed') is False:
        raise ValueError('Frozen raw cache extraction has not completed')
    if (root.drive.upper() != 'G:' or root.name != 'feature_cache' or
            source_protocol['checkpoint_sha256'] != checkpoint_sha or
            source_protocol['parent_manifest_sha256'] != parent_sha or
            source_protocol['games'] != {part: sorted(games) for part, games in args.games.items()}):
        raise ValueError('Feature cache does not belong to this exact checkpoint/cohort')
    packed_index = json.loads((root / 'index.json').read_text()) if (root / 'index.json').exists() else None
    if packed_index:
        if (packed_index['format'] != 'crrl-raw-cache-pack-v1' or packed_index['source_protocol_sha256'] != digest(root.parent / 'protocol.json') or
                packed_index['checkpoint_sha256'] != checkpoint_sha):
            raise ValueError('Packed raw cache belongs to a different immutable protocol')
        files = [root / (record['name'] + '.pt') for record in packed_index['blocks']]
        reader = packed_cache_reads(root, packed_index)
    else:
        files = sorted(root.glob('block_*.pt'))
        reader = bounded_cache_reads(files, args.cache_loader_workers)
    if len(files) != source_protocol['extraction']['blocks']:
        raise ValueError('Feature cache is incomplete')
    blocks, cache_bytes, seats = [], 0, set()
    if args.xy_cache:
        sidecar_report = json.loads(args.xy_cache.with_suffix('.json').read_text())
        if digest(args.xy_cache) != sidecar_report['sha256']:
            raise ValueError('Verified coordinate/card sidecar checksum changed')
    xy_source = np.load(args.xy_cache, allow_pickle=False) if args.xy_cache else None
    for path, cpu in reader:
        part, rid, seat = cpu['partition'], cpu['replay_id'], cpu['seat']
        if rid not in args.games[part] or assignments[rid] != part:
            raise ValueError('Feature cache contains a game from a different partition')
        block = {key: value.to('cuda') if isinstance(value, torch.Tensor) else value for key, value in cpu.items()
                 if not (args.train_scope == 'full-head' and key == 'features')}
        if xy_source is not None:
            xy = xy_source[path.stem]
            if xy.shape != (*block['entities'].shape[:2], 2):
                raise ValueError('Coordinate sidecar shape differs from cached entity tokens')
            block['entity_xy'] = torch.as_tensor(xy, device='cuda')
            if path.stem + '_card_ids' in xy_source.files:
                block['card_ids'] = torch.as_tensor(xy_source[path.stem + '_card_ids'], device='cuda')
        if ('spatial_cnn' in args.variants or 'card_prior_cnn_001' in args.variants) and 'entity_xy' not in block:
            raise ValueError('Spatial architecture requires cached original entity coordinates')
        if any(value.startswith('card_prior_') for value in args.variants) and 'card_ids' not in block:
            raise ValueError('Data initialization requires cached GT hand-card identities')
        block_bytes = sum(value.numel() * value.element_size() for value in block.values() if isinstance(value, torch.Tensor))
        cache_bytes += block_bytes
        if cache_bytes > args.vram_cache_limit_gib * 1024**3:
            raise MemoryError('Feature cache exceeds configured VRAM cap')
        blocks.append(block)
        seats.add((rid, seat))
        del cpu
        check_ram_budget(args.ram_limit_gib)
        if len(blocks) % 500 == 0:
            print(json.dumps({'phase': 'load_raw_cache', 'blocks': len(blocks), 'total': len(files),
                'gpu_cache_gib': cache_bytes / 1024**3}), flush=True)
    if xy_source is not None:
        xy_source.close()
    expected = {(rid, seat) for games in args.games.values() for rid in games for seat in (0, 1)}
    if seats != expected:
        raise ValueError('Feature cache is missing one or more complete seats')
    disk_bytes = (packed_index['bin']['bytes'] + (root / 'index.json').stat().st_size
        if packed_index else sum(path.stat().st_size for path in files))
    return blocks, {**source_protocol['extraction'], 'reused_from': str(root),
        'source_protocol_sha256': digest(root.parent / 'protocol.json'),
        'disk_bytes': disk_bytes, 'packed_sequential_mmap': bool(packed_index),
        'loaded_vram_bytes': cache_bytes, 'unused_tile_feature_tensors_omitted': args.train_scope == 'full-head'}


@torch.no_grad()
def check_initial_full_head(head, frozen_model, blocks, limit, tolerance):
    """Compare full-query heads at the SAME selected matmul precision.

    Cached MLP input features were generated with TF32-high; full-head uses
    highest precision by default and must recompute both its original baseline
    and every experimental arm from the same raw frozen representations.
    """
    head.eval()
    readout_rows, maximum = 0, 0.
    for block in blocks:
        raw = {key: block[key] for key in getattr(head, 'probe_input_fields', RAW_FIELDS)}
        actual = conditional_logp(head, raw, block['legal'])
        full = frozen_model.placement_logits(raw['context'], raw['entities'], raw['entity_mask'], raw['card_features'])
        logits = full[torch.arange(len(raw['slots']), device=full.device), raw['slots']]
        expected = torch.log_softmax(logits.float().masked_fill(~block['legal'], -1e9), -1).masked_fill(~block['legal'], -1e9)
        error = float((actual - expected).abs()[block['legal']].max())
        maximum = max(maximum, error)
        if error > tolerance:
            raise ValueError('Full placement reconstruction differs from original cached head: ' + str(error))
        readout_rows += len(actual)
        if readout_rows >= limit:
            break
    return {'rows': readout_rows, 'max_conditional_logp_error': maximum}


@torch.no_grad()
def check_initial_card_prior(head, blocks, limit=32):
    head.eval()
    count, maximum = 0, 0.
    for block in blocks:
        raw = {key: block[key] for key in head.probe_input_fields}
        actual = conditional_logp(head, raw, block['legal'])
        logits = head.prior_logits(block['card_ids'])
        expected = torch.log_softmax(logits.masked_fill(~block['legal'], -1e9), -1).masked_fill(~block['legal'], -1e9)
        error = float((actual - expected).abs()[block['legal']].max())
        maximum = max(maximum, error)
        if error > 1e-6:
            raise ValueError('Data-initialized head is not the exact training-only card-location prior')
        count += len(actual)
        if count >= limit:
            break
    return {'rows': count, 'prior_logp_max_error': maximum,
        'initial_function': 'Training-only card-location prior; old final projection zero'}


@torch.no_grad()
def check_card_mass_unchanged(model, decoder, blocks, limit=48):
    """Only conditional locations change; audit exact joint card/WAIT marginals."""
    decoder.eval()
    maximum, sampled, same_classes, max_probability = 0., 0, True, 0.
    generator = torch.Generator(device='cuda').manual_seed(20261111)
    for block in blocks:
        n = len(block['tiles'])
        context, card_features = block['context'], block['card_features']
        with torch.autocast('cuda', dtype=torch.float16):
            slots = model.slot_actor(torch.cat((context[:, None].expand(-1, 4, -1), card_features), -1)).squeeze(-1)
            card_logits = torch.cat((model.wait_actor(context), slots, model.ability_actor(context)), -1)
        tiles = model.placement_logits(context, block['entities'], block['entity_mask'], card_features).float()
        fields = getattr(decoder, 'probe_input_fields', RAW_FIELDS)
        candidate = decoder(**{key: block[key] for key in fields}).float()
        candidate_tiles = tiles.clone()
        candidate_tiles[torch.arange(n, device='cuda'), block['slots']] = candidate
        # Only the cached GT slot has a factual legal mask in this probe.
        # Audit that slot plus WAIT; other slots are deliberately marked illegal
        # to avoid inventing legal states. Marginal independence also holds for
        # any available card, but no complete real mask is cached here.
        legal = torch.zeros(n, 2307, dtype=torch.bool, device='cuda')
        legal[:, 0] = True
        grid = legal[:, 1:2305].reshape(n, 4, 576)
        grid[torch.arange(n, device='cuda'), block['slots']] = block['legal']
        before = model._joint_log_probs(card_logits, tiles, legal)
        after = model._joint_log_probs(card_logits, candidate_tiles, legal)
        def masses(lp):
            return torch.cat((lp[:, :1], torch.logsumexp(lp[:, 1:2305].reshape(n, 4, 576), -1), lp[:, 2305:]), -1)
        a, b = masses(before), masses(after)
        maximum = max(maximum, float((a - b).abs().max()))
        pa = -torch.expm1(before[:, 0])
        pb = -torch.expm1(after[:, 0])
        max_probability = max(max_probability, float((pa - pb).abs().max()))
        uniforms = torch.rand(n, device='cuda', generator=generator)
        same_classes &= bool(torch.equal(uniforms < pa, uniforms < pb))
        sampled += n
        if sampled >= limit:
            break
    if maximum > 1e-5 or not same_classes:
        raise ValueError('Candidate changed joint WAIT/card marginals unexpectedly')
    return {'rows': sampled, 'max_slot_wait_ability_logmass_error': maximum,
        'max_action_probability_error': max_probability,
        'same_uniform_nonwait_classifications_identical': same_classes,
        'scope': 'Actual frozen contexts/card features and factual GT-slot legal tile mask plus WAIT; modified conditional location only'}


def initial_weight_audit(original, head, variant, full_head):
    """Verify each original parameter/buffer, independently of GPU kernel noise."""
    expected, actual = original.state_dict(), head.state_dict()
    details, different = {}, []
    for name, reference in expected.items():
        target = name
        if variant == 'wide_residual':
            target = name.replace('tile_actor.', 'tile_actor.original.', 1) if full_head else 'original.' + name
        elif variant == 'card_prior_cnn_001':
            target = 'original.original.' + name
        elif variant == 'spatial_cnn' or variant.startswith('card_prior_'):
            target = 'original.' + name
        source_sha = hashlib.sha256(reference.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        target_sha = hashlib.sha256(actual[target].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        details[name] = {'reference_sha256': source_sha, 'arm_sha256': target_sha}
        if source_sha != target_sha and not (variant.startswith('card_prior_') and name in ('tile_actor.3.weight', 'tile_actor.3.bias')):
            different.append(name)
    if different:
        raise ValueError('Initial arm weights differ from baseline: ' + str(different))
    if variant == 'wide_residual':
        branch = head.tile_actor.branch if full_head else head.branch
        if torch.count_nonzero(branch[-1].weight) or torch.count_nonzero(branch[-1].bias):
            raise ValueError('Wider residual branch is not initially zero-output')
    elif variant == 'spatial_film' and torch.count_nonzero(head.tile_spatial_gate):
        raise ValueError('Spatial FiLM branch is not initially zero-gated')
    elif variant == 'spatial_cnn' and (torch.count_nonzero(head.output.weight) or torch.count_nonzero(head.output.bias)):
        raise ValueError('Spatial CNN branch is not initially zero-output')
    return {'original_tensors_identical': not variant.startswith('card_prior_'),
        'intentional_final_projection_zero_initialization': variant.startswith('card_prior_'), 'per_tensor_sha256': details,
        'added_branch_zero_output': variant in ('wide_residual', 'spatial_film', 'spatial_cnn')}


@torch.no_grad()
def spatial_readout_statistics(head, blocks, limit=32):
    """Read-only query contribution, independent of tile losses or optimization."""
    original = head
    while not isinstance(original, TrainablePlacementHead) and hasattr(original, 'original'):
        original = original.original
    if not isinstance(original, TrainablePlacementHead) or original.cfg.readout_version != 2:
        return {'applicable': False}
    totals = defaultdict(float)
    count = 0
    for block in blocks:
        context, card = block['context'].float(), block['card_features'].float()
        projected = original.tile_context(context)
        base = (original.tile_queries + original.slot_embedding + card[:, :, None] + projected[:, None, None])
        if original.cfg.spatial_fourier:
            base = base + .02 * original.tile_grid_features[None, None]
        spatial = original.tile_queries + .02 * original.tile_grid_features[None, None]
        normalized = original.tile_spatial_norm(original.tile_queries +
            original.tile_spatial_projection(original.tile_grid_features)[None, None])
        gamma, beta = original.tile_film(card + projected[:, None]).chunk(2, -1)
        addition = original.tile_spatial_gate.tanh() * (normalized * (1 + gamma[:, :, None]) + beta[:, :, None] - spatial)
        for key, tensor in [('base', base), ('spatial_addition', addition), ('total', base + addition), ('gamma', gamma), ('beta', beta)]:
            totals[key + '_squared_sum'] += float(tensor.square().sum())
            totals[key + '_elements'] += tensor.numel()
        count += len(context)
        if count >= limit:
            break
    return {'applicable': True, 'sampled_play_frames': count,
        'gate_raw': float(original.tile_spatial_gate), 'gate_tanh': float(original.tile_spatial_gate.tanh()),
        'film_weight_l2': float(original.tile_film.weight.norm()), 'film_bias_l2': float(original.tile_film.bias.norm()),
        'rms': {key: (totals[key + '_squared_sum'] / totals[key + '_elements'])**.5
            for key in ('base', 'spatial_addition', 'total', 'gamma', 'beta')},
        'spatial_addition_l2_over_total_query_l2': (totals['spatial_addition_squared_sum'] / max(1e-30, totals['total_squared_sum']))**.5,
            'scope': 'Chronological representations, all four hand queries on first named GT-play frames; no gradient update'}


def train_updates(args, head, optimizer, blocks, lookup, batches, start, stop, row, log, started,
                  phase='comparison', best_snapshot=None):
    from cr_rl.training.state_bc import check_ram_budget
    for index in range(start, stop):
        step, indices = index + 1, batches[index]
        check_ram_budget(args.ram_limit_gib)
        head.train()
        spatial = 'entity_xy' in getattr(head, 'probe_input_fields', ())
        card_ids = 'card_ids' in getattr(head, 'probe_input_fields', ())
        batch = gather_rows(blocks, lookup, indices, full_head=args.train_scope == 'full-head', spatial=spatial, card_ids=card_ids)
        optimizer.zero_grad(set_to_none=True)
        fields = getattr(head, 'probe_input_fields', None)
        features = {key: batch[key] for key in fields} if fields else batch['features']
        lp = conditional_logp(head, features, batch['legal'])
        loss = -(batch['targets'] * lp).sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite probe loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_every == 0 or step == stop:
            final_heldout = (phase == 'validation_selected_extension' and step == stop) or (
                args.extend_winner_to <= args.steps and step == stop)
            if args.restore_best_validation and phase == 'validation_selected_extension':
                final_heldout = False
            metrics = {'step': step, 'last_batch_loss': float(loss.detach()),
                'last_gradient_norm': float(norm), **evaluate(head, blocks, args.eval_batch_size,
                    include_heldout=not args.heldout_final_only or final_heldout)}
            row['metrics'].append(metrics)
            if best_snapshot is not None:
                score = metrics['summary']['validation']['equal_seat']['ce']
                if score < best_snapshot['validation_ce']:
                    weights = {name: value.detach().clone() for name, value in head.state_dict().items()}
                    best_snapshot.clear()
                    best_snapshot.update(validation_ce=score, step=step, weights=weights, consecutive_nonimproving=0)
                else:
                    best_snapshot['consecutive_nonimproving'] = best_snapshot.get('consecutive_nonimproving', 0) + 1
            event = {'phase': phase, 'variant': row['variant'], 'learning_rate': row['learning_rate'], 'step': step,
                'summary': metrics['summary'], 'elapsed_seconds': time.perf_counter() - started}
            log.write(json.dumps(event, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(event), flush=True)
            if (phase == 'validation_selected_extension' and best_snapshot is not None and args.validation_patience and
                    best_snapshot.get('consecutive_nonimproving', 0) >= args.validation_patience):
                row['early_stop'] = {'step': step, 'rule': 'Consecutive monitored full-validation equal-seat CE nodes without a new lower value',
                    'patience': args.validation_patience, 'best_validation_step': best_snapshot['step'],
                    'heldout_used': False}
                return
        del batch, features, lp, loss


def arm_key(row):
    return row['variant'] + ':' + str(row['learning_rate'])


def rank_finalists(rows, baseline_ce, *, policy='ce', top_k=1, margin=.1):
    """Predeclared selection; uses validation only, never heldout scores."""
    by_ce = sorted(rows, key=lambda row: (row['metrics'][-1]['summary']['validation']['equal_seat']['ce'], arm_key(row)))
    if policy == 'ce' or top_k == 1:
        return [arm_key(row) for row in by_ce[:top_k]], [arm_key(row) for row in by_ce[:top_k]]
    eligible = [row for row in rows if row['metrics'][-1]['summary']['validation']['equal_seat']['ce'] <= baseline_ce + margin]
    geometric = sorted(eligible, key=lambda row: (-row['metrics'][-1]['summary']['validation']['equal_seat']['within2'],
        row['metrics'][-1]['summary']['validation']['equal_seat']['ce'], arm_key(row)))
    selected = [arm_key(by_ce[0])]
    for row in geometric + by_ce:
        if arm_key(row) not in selected:
            selected.append(arm_key(row))
        if len(selected) >= top_k:
            break
    # At most four tiny live heads/moment sets suffice while screening eight
    # candidates; a discarded rank cannot re-enter either constant ordering.
    retain = set(arm_key(row) for row in by_ce[:top_k]) | set(arm_key(row) for row in geometric[:top_k])
    return selected, sorted(retain)


def save_decoder(args, head, model, row, metrics, protocol, *, selected_by):
    """One bounded deployable experiment result, with complete immutable binding."""
    target = args.save_best_decoder.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + '.tmp')
    payload = {'format': 'crrl-placement-decoder-probe-v1',
        'source_checkpoint_sha256': protocol['checkpoint_sha256'],
        'source_checkpoint_progress': protocol['checkpoint_progress'],
        'full_training_manifest_sha256': protocol['dataset_sha256'],
        'checkpoint_parent_manifest_sha256': protocol['parent_manifest_sha256'],
        'cohort_sha256': protocol['cohort_sha256'],
        'variant': row['variant'], 'learning_rate': row['learning_rate'],
        'architecture_config': {'entity_dim': model.cfg.entity_dim, 'temporal_dim': model.cfg.temporal_dim,
            'readout_version': model.cfg.readout_version, 'width': args.width, 'cnn_channels': args.cnn_channels,
            'alpha': .01 if row['variant'] in ('card_prior_001', 'card_prior_cnn_001') else (1. if row['variant'] == 'card_prior_1' else None)},
        'step': metrics['step'], 'action_epoch': metrics['step'] / protocol['action_steps_per_epoch'],
        'validation_equal_seat_ce': metrics['summary']['validation']['equal_seat']['ce'],
        'validation_equal_seat_within2': metrics['summary']['validation']['equal_seat']['within2'],
        'parameter_count': sum(value.numel() for value in head.parameters()),
        'selection': selected_by, 'heldout_used_for_selection': False,
        'training_objective': protocol['optimization'],
        'model_config': model.checkpoint_config(),
        'decoder': {name: value.detach().cpu().clone() for name, value in head.state_dict().items()},
        'contains_trunk': False, 'contains_optimizer': False,
        'plan_change': protocol['retention_plan_change'],
        'deployment_status': 'Offline BC decoder candidate; requires explicit model integration and native/L4 evaluation before deployment'}
    if hasattr(args, '_probe_blocks'):
        payload['card_wait_invariance'] = check_card_mass_unchanged(model, head, args._probe_blocks)
    torch.save(payload, pending)
    pending.replace(target)
    atomic_json(target.with_suffix('.json'), {key: value for key, value in payload.items() if key not in ('decoder', 'model_config')})


def main(argv=None):
    args = parse_args(argv)
    from cr_rl.models.state_policy import state_model_from_checkpoint
    from cr_rl.training.state_bc import check_ram_budget
    if args.prepare_xy_cache:
        print(json.dumps(prepare_entity_xy(args)), flush=True)
        return
    if args.prepare_card_cache:
        print(json.dumps(add_card_ids_to_coordinate_sidecar(args)), flush=True)
        return
    if args.save_best_decoder and args.save_best_decoder.resolve().drive.upper() != 'G:':
        raise ValueError('Retained decoder must stay on G:')
    for part in args.games:
        args.games[part] = set(map(int, args.games[part]))
    if not torch.cuda.is_available():
        raise RuntimeError('Existing CUDA environment required; no installation attempted')
    if args.gpu_memory_limit_gib:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.95, args.gpu_memory_limit_gib * 1024**3 / total_memory))
    if args.out.resolve().drive.upper() != 'G:':
        raise ValueError('All local experiment outputs must stay on G:')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a fresh diagnostic output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    torch.manual_seed(args.seed)
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    manifest = json.loads(args.dataset.read_text())
    parent_manifest = json.loads(args.parent_manifest.read_text())
    parent_sha = digest(args.parent_manifest)
    if (manifest.get('parent_manifest_sha256') != parent_sha or
            manifest['metadata_sha256'] != parent_manifest['metadata_sha256']):
        raise ValueError('Diagnostic cohort is not derived from the immutable parent dataset')
    parent_records = {int(record['replay_offset']): record for record in parent_manifest['shards']}
    for record in manifest['shards']:
        original = parent_records[int(record['replay_offset'])]
        if any(record[key] != original[key] for key in ('rows', 'replays', 'replay_offset')):
            raise ValueError('Diagnostic shard differs from parent row/index contract')
        reference_path = args.parent_manifest.parent / original['path']
        if reference_path.exists():
            if digest(reference_path) != original['sha256']:
                raise ValueError('Parent reference shard checksum changed')
            with np.load(reference_path, allow_pickle=False) as reference, np.load(args.dataset.parent / record['path'], allow_pickle=False) as subset:
                if set(reference.files) != set(subset.files) or any(not np.array_equal(reference[key], subset[key]) for key in reference.files):
                    raise ValueError('Prepared cohort changed a downloaded reference shard')
    metadata_path = args.dataset.parent / manifest['metadata']
    if digest(metadata_path) != manifest['metadata_sha256']:
        raise ValueError('Dataset metadata checksum mismatch')
    metadata = json.loads(metadata_path.read_text())
    checkpoint_sha = digest(args.checkpoint)
    source_sha = manifest['source_sha256']
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if checkpoint['dataset_fingerprint'] != parent_sha:
        raise ValueError('Checkpoint-bound dataset fingerprint mismatch')
    if checkpoint.get('model_architecture_version') != 3 or checkpoint.get('decision_ticks') != 8:
        raise ValueError('Expected current architecture-v3 0.4s checkpoint')
    assignments = {int(key): value for key, value in checkpoint['split'].items()}
    if args.cohort:
        cohort = json.loads(args.cohort.read_text())
        if (cohort['reference_checkpoint_sha256'] != checkpoint_sha or
                cohort['source_dataset_sha256'] != source_sha or
                cohort['checkpoint_bound_dataset_manifest_sha256'] != parent_sha):
            raise ValueError('Frozen cohort belongs to a different immutable checkpoint/source')
    for part, games in args.games.items():
        if any(assignments.get(game) != part for game in games):
            raise ValueError('Requested games do not belong to original ' + part + ' split')
    settings, progress = dict(checkpoint['training_config']), dict(checkpoint['progress'])
    if args.train_card_prior:
        prior = json.loads(args.train_card_prior.read_text())
        if prior['fit_games'] != sum(part == 'train' for part in assignments.values()) or prior['sigma'] != settings['tile_label_sigma']:
            raise ValueError('Card-location prior does not match original training games/target sigma')
    model = state_model_from_checkpoint(checkpoint, 'cuda')
    del checkpoint
    gc.collect()
    model.eval().requires_grad_(False)
    model.frame_microbatch = 32
    check_ram_budget(args.ram_limit_gib)
    blocks, extraction = (load_feature_cache(args, checkpoint_sha, parent_sha, assignments)
        if args.feature_cache_dir else extract(args, model, metadata, settings, assignments))
    args._probe_blocks = blocks
    original_head = (TrainablePlacementHead(model) if args.train_scope == 'full-head'
        else deepcopy(model.tile_actor))
    if args.train_scope == 'full-head':
        extraction['initial_full_head_parity'] = check_initial_full_head(original_head,
            model, blocks, args.verify_rows, args.max_initial_logp_error)
    dimensions = model.cfg.entity_dim
    # The main model remains frozen; raw caches remain available for a future
    # separate full-placement-head experiment. Never train or save this model.
    lookup = [(index, row) for index, block in enumerate(blocks)
              if block['partition'] == 'train' for row in range(len(block['tiles']))]
    action_steps_per_epoch = (len(lookup) + args.batch_size - 1) // args.batch_size
    if args.epochs:
        args.steps = action_steps_per_epoch * args.epochs
        args.eval_every = action_steps_per_epoch
    if args.extend_winner_epochs:
        args.extend_winner_to = action_steps_per_epoch * args.extend_winner_epochs
    batches = fixed_batches(len(lookup), args.batch_size, max(args.steps, args.extend_winner_to), args.seed)
    protocol = {'checkpoint_sha256': checkpoint_sha, 'checkpoint_progress': progress,
        'source_sha256': source_sha, 'dataset_sha256': digest(args.dataset),
        'parent_manifest_sha256': parent_sha, 'cohort_sha256': digest(args.cohort) if args.cohort else None,
        'scope': ('full-placement-head; entity and temporal frozen; query/context/cross-attention/MLP trainable'
            if args.train_scope == 'full-head' else
            'readout-only; entity, temporal, query, context projection and cross-attention frozen'),
        'games': {part: sorted(games) for part, games in args.games.items()},
        'training_rows': len(lookup), 'steps': args.steps, 'batch_size': args.batch_size,
        'exposures': sum(map(len, batches[:args.steps])), 'equivalent_action_epochs': sum(map(len, batches[:args.steps])) / len(lookup),
        'winner_extension_target_steps': args.extend_winner_to,
        'winner_selection': ('Minimum final full-validation equal-seat CE, plus highest within2 alternative whose CE <= original baseline + margin'
            if args.selection_policy == 'ce-and-within2' else 'Minimum final comparison-node validation equal-seat soft tile CE'),
        'heldout_selection': 'Never used for candidate selection',
        'selection_policy': args.selection_policy, 'geometry_ce_margin': args.geometry_ce_margin,
        'heldout_final_only': args.heldout_final_only, 'extend_top_k': args.extend_top_k,
        'restore_best_validation_epoch': args.restore_best_validation,
        'continuation_validation_patience': args.validation_patience,
        'action_steps_per_epoch': action_steps_per_epoch, 'screening_action_epochs': args.epochs or None,
        'learning_rates': args.learning_rates, 'width': args.width, 'weight_decay': args.weight_decay,
        'source_readout_version': model.cfg.readout_version,
        'float32_matmul_precision': args.float32_matmul_precision,
        'allocator_limit_gib': args.gpu_memory_limit_gib,
        'include_zero_gated_spatial_film_arm': args.include_spatial_film,
        'variants': args.variants, 'cnn_channels': args.cnn_channels if 'spatial_cnn' in args.variants else None,
        'xy_sidecar_sha256': digest(args.xy_cache) if args.xy_cache else None,
        'training_card_prior_sha256': digest(args.train_card_prior) if args.train_card_prior else None,
        'card_prior_interpretation': 'Data-initialization candidate has intentionally different initial function; not a capacity-only comparison',
        'optimization': 'Fresh AdamW per arm; mean soft tile CE over identical shuffled GT-play minibatches; FP32 readout',
        'comparison': 'Original head remains trainable; wider/CNN add zero-output branches; card-prior arms intentionally change initialization',
        'same_initial_function_capacity_arms': [value for value in args.variants if not value.startswith('card_prior_')],
        'different_initial_function_training_arms': [value for value in args.variants if value.startswith('card_prior_')],
        'distance_metric': 'Manhattan grid-cell distance; within2 is distance <=2',
        'evaluation': 'Only GT ordinary-play rows and GT slot; both seats per named game; offline teacher-forced',
        'evaluation_batching': 'Independent decoder rows grouped by identical entity bucket; original game/seat attribution retained',
        'extraction': extraction, 'gpu': torch.cuda.get_device_name(0),
        'save_weights': bool(args.save_best_decoder), 'feature_cache_saved': args.save_feature_cache,
        'retained_result_scope': 'At most one explicitly requested usable placement decoder; no trunk or Adam state',
        'retention_plan_change': ('User authorized architecture/test improvements with prompt cleanup; retain only one useful final decoder rather than temporary fitted arms'
            if args.save_best_decoder else None)}
    atomic_json(args.out / 'protocol.json', protocol)
    original_head.requires_grad_(False)
    baseline = evaluate(original_head, blocks, args.eval_batch_size, include_heldout=not args.heldout_final_only)
    atomic_json(args.out / 'baseline.json', baseline)
    results = []
    live_states = {}
    started = time.perf_counter()
    with (args.out / 'progress.jsonl').open('w', encoding='utf-8') as log:
        for rate in args.learning_rates:
            for variant in args.variants:
                torch.manual_seed(args.seed)
                if args.train_scope == 'full-head':
                    head = deepcopy(original_head)
                    if variant == 'wide_residual':
                        head.tile_actor = WiderResidualReadout(head.tile_actor, dimensions, args.width).cuda()
                    elif variant == 'spatial_film':
                        head.add_zero_gated_spatial_film()
                    elif variant == 'spatial_cnn':
                        from state_bc_spatial_probe import EntitySpatialResidual, SPATIAL_FIELDS
                        head = EntitySpatialResidual(head, entity_dim=dimensions,
                            context_dim=model.cfg.temporal_dim, card_dim=dimensions,
                            channels=args.cnn_channels)
                        head.probe_input_fields = SPATIAL_FIELDS
                    elif variant.startswith('card_prior_'):
                        if variant == 'card_prior_cnn_001':
                            head = CardPriorSpatialHead(head, args.train_card_prior, model.cfg.vocab_size, args.cnn_channels)
                        else:
                            from state_bc_card_location_probe import CardLocationPriorResidual
                            head = CardLocationPriorResidual(head, args.train_card_prior,
                                vocab_size=model.cfg.vocab_size, alpha=.01 if variant == 'card_prior_001' else 1.)
                else:
                    head = deepcopy(original_head) if variant == 'original' else WiderResidualReadout(original_head, dimensions, args.width).cuda()
                head.requires_grad_(True)
                optimizer = torch.optim.AdamW(head.parameters(), lr=rate, weight_decay=args.weight_decay)
                row = {'variant': variant, 'learning_rate': rate,
                    'trainable_parameters': sum(p.numel() for p in head.parameters()), 'metrics': []}
                row['initial_weight_audit'] = initial_weight_audit(original_head, head, variant,
                    args.train_scope == 'full-head')
                if variant == 'spatial_cnn':
                    row['added_spatial_complexity'] = head.complexity_estimate()
                if variant.startswith('card_prior_'):
                    row['data_initialization'] = head.initialization_info()
                row['spatial_readout_initial'] = spatial_readout_statistics(head, blocks)
                # Same initial policy is checked before either fresh optimizer.
                initial = evaluate(head, blocks, args.eval_batch_size, include_heldout=not args.heldout_final_only)
                initial_deltas = {}
                for part in initial['summary']:
                    delta = initial['summary'][part]['action_weighted']['ce'] - baseline['summary'][part]['action_weighted']['ce']
                    initial_deltas[part] = delta
                    # The per-row production parity uses the explicit logp
                    # tolerance. A stricter 1e-6 aggregate assertion is not a
                    # valid TF32 cross-allocation equivalence test.
                    if abs(delta) > args.max_initial_logp_error and not variant.startswith('card_prior_'):
                        raise ValueError('Experimental arm initial CE differs: ' + str(initial_deltas))
                if args.train_scope == 'full-head' and not variant.startswith('card_prior_'):
                    row['initial_production_parity'] = check_initial_full_head(head,
                        model, blocks, args.verify_rows, args.max_initial_logp_error)
                elif variant.startswith('card_prior_'):
                    row['initial_data_prior_parity'] = check_initial_card_prior(head, blocks, args.verify_rows)
                row['metrics'].append({'step': 0, 'initial_mean_ce_deltas': initial_deltas, **initial})
                train_updates(args, head, optimizer, blocks, lookup, batches, 0, args.steps, row, log, started)
                row['spatial_readout_final'] = spatial_readout_statistics(head, blocks)
                results.append(row)
                if args.save_best_decoder and args.extend_winner_to <= args.steps and len(args.variants) == 1 and len(args.learning_rates) == 1:
                    save_decoder(args, head, model, row, row['metrics'][-1], protocol,
                        selected_by='Single validation-selected replay of a previously ranked candidate')
                if args.extend_winner_to > args.steps:
                    optimizer.zero_grad(set_to_none=True)
                    live_states[arm_key(row)] = {'head': head, 'optimizer': optimizer, 'row': row}
                    _, retain = rank_finalists(results, baseline['summary']['validation']['equal_seat']['ce'],
                        policy=args.selection_policy, top_k=args.extend_top_k, margin=args.geometry_ce_margin)
                    live_states = {key: value for key, value in live_states.items() if key in retain}
                atomic_json(args.out / 'results.json', {'protocol': protocol, 'arms': results})
                del head, optimizer
                gc.collect()
        if live_states:
            selected_keys, _ = rank_finalists(results, baseline['summary']['validation']['equal_seat']['ce'],
                policy=args.selection_policy, top_k=args.extend_top_k, margin=args.geometry_ce_margin)
            extensions = []
            for rank, key in enumerate(selected_keys, 1):
                winner = live_states.pop(key)
                selected = winner['row']
                extension = {'variant': selected['variant'], 'learning_rate': selected['learning_rate'],
                    'selection': protocol['winner_selection'], 'selected_at_step': args.steps, 'selection_rank': rank,
                    'selected_validation_ce': selected['metrics'][-1]['summary']['validation']['equal_seat']['ce'],
                    'optimizer_continued': True, 'metrics': [selected['metrics'][-1]]}
                best_snapshot = None
                if args.restore_best_validation:
                    best_snapshot = {'validation_ce': extension['selected_validation_ce'], 'step': args.steps,
                        'weights': {name: value.detach().clone() for name, value in winner['head'].state_dict().items()}}
                train_updates(args, winner['head'], winner['optimizer'], blocks, lookup, batches,
                    args.steps, args.extend_winner_to, extension, log, started,
                    phase='validation_selected_extension', best_snapshot=best_snapshot)
                extension['spatial_readout_final'] = spatial_readout_statistics(winner['head'], blocks)
                if best_snapshot is not None:
                    extension['last_epoch_metrics'] = extension['metrics'][-1]
                    winner['head'].load_state_dict(best_snapshot['weights'], strict=True)
                    extension['best_validation_epoch'] = best_snapshot['step'] / action_steps_per_epoch
                    extension['best_validation_metrics'] = {'step': best_snapshot['step'],
                        'selection': 'Minimum full-validation equal-seat soft tile CE among monitored epochs',
                        **evaluate(winner['head'], blocks, args.eval_batch_size, include_heldout=True)}
                    extension['best_validation_spatial_readout'] = spatial_readout_statistics(winner['head'], blocks)
                    extension['restore_note'] = 'Only the small placement head is restored for final evaluation; no optimizer resume or weight export follows'
                    del best_snapshot
                if args.save_best_decoder and rank == 1:
                    chosen = extension.get('best_validation_metrics', extension['metrics'][-1])
                    save_decoder(args, winner['head'], model, extension, chosen, protocol,
                        selected_by='Rank1 full-validation CE candidate, restored minimum validation-CE epoch')
                filename = 'winner_extension.json' if args.extend_top_k == 1 else f'winner_extension_rank{rank}.json'
                atomic_json(args.out / filename, extension)
                extensions.append({'rank': rank, 'variant': selected['variant'], 'learning_rate': selected['learning_rate'], 'file': filename})
                del winner
            live_states.clear()
            atomic_json(args.out / 'winner_extensions.json', {'selection': protocol['winner_selection'], 'extensions': extensions})
    if args.heldout_final_only:
        atomic_json(args.out / 'baseline_final_full.json', evaluate(original_head, blocks, args.eval_batch_size, include_heldout=True))
        if args.train_card_prior:
            # Score fixed training-only histogram baselines only after all
            # candidate/epoch selection is finished. No heldout value can
            # influence screening or continuation.
            from state_bc_card_location_probe import CardLocationPriorResidual
            prior_baselines = []
            class PriorOnly(nn.Module):
                probe_input_fields = RAW_FIELDS + ('card_ids',)
                def __init__(self, head):
                    super().__init__()
                    self.bias = head.card_location_bias
                def forward(self, card_ids, **_kwargs):
                    return self.bias(card_ids.long() + 1)
            for alpha in (.01, 1.):
                prior_head = CardLocationPriorResidual(deepcopy(original_head), args.train_card_prior,
                    vocab_size=model.cfg.vocab_size, alpha=alpha)
                prior_head.requires_grad_(False)
                prior_baselines.append({'alpha': alpha, 'training_data_only': True,
                    'initializer': prior_head.initialization_info(),
                    **evaluate(PriorOnly(prior_head), blocks, args.eval_batch_size, include_heldout=True)})
                del prior_head
            atomic_json(args.out / 'prior_baselines_final_full.json', {'variants': prior_baselines,
                'scope': 'No optimization; fixed training-only priors, both predeclared smoothing strengths'})
    atomic_json(args.out / 'completion.json', {'complete': True, 'model_weights_saved': bool(args.save_best_decoder),
        'checkpoint_sha256_after': digest(args.checkpoint), 'seconds': time.perf_counter() - started,
        'rss_gib': check_ram_budget(args.ram_limit_gib) / 1024**3,
        'peak_vram_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'scope': args.train_scope,
        'limitation': ('Whole placement-head diagnostic on frozen entity/temporal representations; not end-to-end training or battle win rate.'
            if args.train_scope == 'full-head' else
            'Final MLP-only diagnostic. A failure to fit cannot exclude frozen cross-attention/representation limitations.')})
    if args.delete_feature_cache_after:
        root = args.feature_cache_dir.resolve()
        # Only verified named block files in the explicitly requested G: cache;
        # do not recurse or remove any reports, source, checkpoint, or dataset.
        paths = sorted(root.glob('block_*.pt'))
        if root.drive.upper() != 'G:' or root.name != 'feature_cache':
            raise ValueError('Refusing feature cleanup outside the verified G: cache')
        if (root / 'index.json').exists():
            packed = json.loads((root / 'index.json').read_text())
            binary = (root / packed['bin']['path']).resolve()
            if binary.parent != root or binary.name != 'raw_cache.bin':
                raise ValueError('Refusing packed cache cleanup outside the verified directory')
            paths += [binary, root / 'index.json']
        reclaimed = sum(path.stat().st_size for path in paths)
        for path in paths:
            path.unlink()
        root.rmdir()
        atomic_json(args.out / 'cache_cleanup.json', {'removed_verified_cache': str(root),
            'blocks': len(paths), 'bytes_reclaimed': reclaimed})


if __name__ == '__main__':
    main()
