"""Diagnostic-only helpers: bounded cohort materialization and separable BC losses."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from diagnose_state_bc_generalization import atomic_json, digest, stream_games
from cr_rl.data.event_preserving_resample import resample_game


def position_loss_from_logits(logits, labels, legal, *, play_denominator, sigma):
    """Placement-only graph; avoids numerical cancellation through card logits."""
    slots = (labels - 1) // 576
    tiles = (labels - 1) % 576
    index = torch.arange(len(labels), device=labels.device)
    chosen_legal = legal[:, 1:2305].reshape(-1, 4, 576)[index, slots]
    chosen_logits = logits.float()[index, slots].masked_fill(~chosen_legal, -1e9)
    tile_logp = chosen_logits.log_softmax(-1)
    grid = torch.arange(576, device=labels.device)
    if sigma:
        distance = (grid[None] % 18 - tiles[:, None] % 18).float().square()
        distance += (grid[None] // 18 - tiles[:, None] // 18).float().square()
        targets = torch.exp(-distance / (2 * sigma**2)) * chosen_legal
    else:
        targets = (grid[None] == tiles[:, None]).float()
    targets = targets / targets.sum(-1, keepdim=True).clamp_min(1e-12)
    return -(targets * tile_logp).sum() / max(1, play_denominator)


def differentiable_components(logp, labels, legal, normalization, *, sigma, target_prior, sampled_prior):
    """Same additive seat-normalized objective as hierarchical_bc_loss, with graph."""
    cards = torch.cat((logp[:, :1], torch.logsumexp(logp[:, 1:2305].reshape(-1, 4, 576), -1),
                       logp[:, 2305:2307]), -1)
    wait = labels == 0
    play = (labels > 0) & (labels < 2305)
    target_card = torch.where(play, 1 + (labels - 1) // 576, torch.zeros_like(labels))
    target_card = torch.where(labels >= 2305, labels - 2305 + 5, target_card)
    act_logp = torch.logsumexp(cards[:, 1:], -1)
    weights = torch.where(wait, (1 - target_prior) / (1 - sampled_prior), target_prior / sampled_prior)
    timing = -(torch.where(wait, cards[:, 0], act_logp) * weights).sum() / normalization.rows
    conditional = torch.where(~wait, -cards.gather(1, target_card[:, None]).squeeze(1) + act_logp, 0.)
    card = conditional.sum() / max(1, normalization.act)
    if play.any():
        slots = (labels[play] - 1) // 576
        tiles = (labels[play] - 1) % 576
        tile = logp[play, 1:2305].reshape(-1, 4, 576)[torch.arange(len(slots), device=labels.device), slots]
        tile = tile - cards[play].gather(1, (slots + 1)[:, None])
        tile_legal = legal[play, 1:2305].reshape(-1, 4, 576)[torch.arange(len(slots), device=labels.device), slots]
        grid = torch.arange(576, device=labels.device)
        if sigma:
            distance = (grid[None] % 18 - tiles[:, None] % 18).float().square()
            distance += (grid[None] // 18 - tiles[:, None] // 18).float().square()
            targets = torch.exp(-distance / (2 * sigma**2)) * tile_legal
        else:
            targets = (grid[None] == tiles[:, None]).float()
        targets = targets / targets.sum(-1, keepdim=True).clamp_min(1e-12)
        position = -(targets * tile.float()).sum() / max(1, normalization.play)
    else:
        position = logp.sum() * 0.
    return {'timing': timing, 'card': card, 'tile': position}


def ensure_cohort_dataset(source, parent_manifest, cohort, output):
    """Store only named games; preserve original metadata by a same-volume hardlink."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'dataset.manifest.json'
    ids = set(int(game) for part in ('train', 'validation', 'heldout') for game in cohort[part])
    parent_manifest = Path(parent_manifest)
    parent = json.loads(parent_manifest.read_text())
    metadata_path = parent_manifest.parent / parent['metadata']
    if digest(metadata_path) != parent['metadata_sha256']:
        raise ValueError('Original immutable metadata SHA mismatch')
    source_sha = digest(source)
    original_audit = Path(source).parent.parent.parent / 'bc_diagnostic_20261110/supervision/bc100/audit.json'
    if source_sha != json.loads(original_audit.read_text())['source_npz_sha256']:
        raise ValueError('Original source SHA mismatch')
    if target.exists():
        existing = json.loads(target.read_text())
        if existing['diagnostic_game_ids'] != sorted(ids) or existing['source_sha256'] != source_sha:
            raise ValueError('Diagnostic dataset belongs to a different cohort/source')
        for record in existing['shards']:
            if digest(output / record['path']) != record['sha256']:
                raise ValueError('Diagnostic shard changed')
        return target
    metadata = json.loads(metadata_path.read_text())
    name = 'dataset_metadata.json'
    if not (output / name).exists():
        os.link(metadata_path.resolve(), output / name)
    (output / 'shards').mkdir(exist_ok=True)
    columns = [index for index, field in enumerate(metadata['schema']['header_fields']) if field == 'decision_ticks']
    records = []
    for rid, offset, arrays in stream_games(source):
        if rid not in ids:
            continue
        mapped, audit = resample_game(arrays, decision_ticks=8, decision_header_columns=columns,
                                     source_row_ids=np.arange(offset, offset + len(arrays['actions'])))
        mapped['replay_ids'] = np.zeros(len(mapped['actions']), dtype=np.int32)
        relative = f'shards/replay_{rid:06d}.npz'
        np.savez_compressed(output / relative, **mapped)
        records.append({'path': relative, 'sha256': digest(output / relative),
                        'rows': len(mapped['actions']), 'replays': 1, 'replay_offset': rid})
        print(json.dumps({'phase': 'prepare_cohort', 'game': rid, 'done': len(records),
                          'nonwait_preserved': audit['all_nonwait_structurally_preserved']}), flush=True)
        if len(records) == len(ids):
            break
    if len(records) != len(ids):
        raise ValueError('Missing named cohort game')
    atomic_json(target, {**parent, 'metadata': name, 'shards': records, 'replays': len(records),
        'rows': sum(record['rows'] for record in records), 'diagnostic_game_ids': sorted(ids),
        'source_sha256': source_sha, 'parent_manifest_sha256': digest(parent_manifest),
        'diagnostic_only': True, 'native_trajectory': False})
    return target
