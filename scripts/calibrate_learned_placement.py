"""Choose a placement temperature on full validation only, then evaluate heldout.

No optimizer, new model weights, or production deployment. Optionally update
the single retained decoder payload with its auditable scalar calibration.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from diagnose_state_bc_generalization import atomic_json, digest
from probe_state_bc_head_capacity import evaluate, load_feature_cache
from cr_rl.models.learned_placement import attach_learned_placement
from cr_rl.models.state_policy import state_model_from_checkpoint


class GTTemperature(nn.Module):
    probe_input_fields = ('context', 'entities', 'entity_mask', 'card_features', 'slots', 'entity_xy', 'card_ids')

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.temperature = 1.

    def forward(self, **inputs):
        return self.adapter.forward_gt(**inputs) / self.temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--temperatures', default='.5,1,1.5,2,3')
    parser.add_argument('--update-decoder', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires existing CUDA; no installations')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    payload = torch.load(args.decoder, map_location='cpu', mmap=True, weights_only=False)
    if payload.get('temperature', 1.) != 1.:
        raise ValueError('This calibration starts from uncalibrated decoder logits')
    source_sha = digest(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model = state_model_from_checkpoint(state, 'cuda').eval()
    parent_sha = state['dataset_fingerprint']
    assignments = {int(game): part for game, part in state['split'].items()}
    del state
    gc.collect()
    adapter = attach_learned_placement(model, payload, source_checkpoint_sha256=source_sha)
    model.requires_grad_(False)
    cohort = json.loads(args.cohort.read_text())
    loader = SimpleNamespace(feature_cache_dir=args.cache, games={part: set(value['game_ids'])
        for part, value in cohort['partitions'].items()}, xy_cache=None, train_scope='full-head',
        variants=['card_prior_cnn_001'], vram_cache_limit_gib=2., ram_limit_gib=6., cache_loader_workers=1)
    blocks, extraction = load_feature_cache(loader, source_sha, parent_sha, assignments)
    head = GTTemperature(adapter)
    grid = [float(value) for value in args.temperatures.split(',')]
    if 1. not in grid or not all(np.isfinite(grid)) or min(grid) <= 0:
        raise ValueError('Positive fixed grid must include uncalibrated temperature 1')
    scores = []
    validation = [block for block in blocks if block['partition'] == 'validation']
    for temperature in grid:
        head.temperature = temperature
        metrics = evaluate(head, validation, 16, include_heldout=False)
        scores.append({'temperature': temperature, **metrics})
        print(json.dumps({'phase': 'validation_calibration', 'temperature': temperature,
            'equal_seat': metrics['summary']['validation']['equal_seat']}), flush=True)
    selected = min(scores, key=lambda row: (row['summary']['validation']['equal_seat']['ce'], abs(row['temperature'] - 1)))
    head.temperature = selected['temperature']
    final = evaluate(head, blocks, 16, include_heldout=True)
    report = {'source_checkpoint_sha256': source_sha, 'decoder_sha256_before': digest(args.decoder),
        'grid_predeclared': grid, 'selection': 'Minimum full53 validation equal-seat soft tile CE',
        'heldout_used_for_selection': False, 'optimizer_created': False, 'argmax_unchanged': True,
        'grid_validation_results': scores, 'selected_temperature': head.temperature,
        'final': final, 'cache_extraction': extraction,
        'limitation': 'Scalar calibration changes confidence, not ranking or spatial expression; diagnostic development split is reused'}
    if args.update_decoder:
        payload['temperature'] = head.temperature
        payload['temperature_selection'] = {'grid': grid, 'partition': 'validation',
            'games': 53, 'metric': 'equal-seat soft tile CE', 'heldout_used_for_selection': False}
        # The payload is small; avoid writing any full trunk or optimizer.
        pending = args.decoder.with_suffix(args.decoder.suffix + '.tmp')
        torch.save(dict(payload), pending)
        del payload
        gc.collect()
        pending.replace(args.decoder)
        report['decoder_sha256_after'] = digest(args.decoder)
    atomic_json(args.out, report)


if __name__ == '__main__':
    main()
