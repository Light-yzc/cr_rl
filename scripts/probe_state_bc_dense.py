"""Synthetic dense-entity FP16 backward admission check for a frozen BC model.

Only the first real supervised chunk is read. Its entity tokens are repeated
to 200 active entries in owned tensors; no dataset or checkpoint is rewritten.
The full saved Adam state is restored, uses fused updates with lr=0, and model
weight SHA is checked before/after. This is capacity/numerical pressure, never
a native game trajectory or convergence/quality evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_temporal import GQAState, KDAState
from cr_rl.training.bc import BCBatchNormalization, hierarchical_bc_loss
from cr_rl.training.state_bc import ReplaySequenceDataset, prepare_state_chunk
try:
    from profile_state_bc import build_case, weights_digest
except ModuleNotFoundError as error:
    if error.name != 'profile_state_bc':
        raise
    from scripts.profile_state_bc import build_case, weights_digest


def dense_inputs(inputs, entity_count_index, count=200):
    """Repeat actual occupied tokens, never padding, without mutating inputs."""
    if not isinstance(count, int) or not 1 <= count <= 200:
        raise ValueError('Dense pressure count must be within 1..200')
    source, mask = inputs['entities'], inputs['entity_mask'].bool()
    occupied = mask.sum(-1).clamp_min(1)
    indices = torch.arange(count, device=source.device)[None] % occupied[:, None]
    repeated = source.gather(1, indices[..., None].expand(-1, -1, source.shape[-1]))
    header = inputs['header'].clone()
    header[:, entity_count_index] = count
    return {**inputs, 'header': header, 'entities': repeated,
            'entity_mask': torch.ones(repeated.shape[:-1], dtype=torch.bool, device=source.device)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--entities', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--ram-limit-gib', type=float, default=40)
    args = parser.parse_args(argv)
    if not 1 <= args.chunk_size <= 32 or not 1 <= args.entities <= 200:
        parser.error('Use 1..32 real chunk rows and 1..200 dense tokens')
    args.precision = 'fp16'
    return args


def main(argv=None):
    args = parse_args(argv)
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Dense FP16 pressure requires an existing CUDA runtime; no package installation is attempted')
    torch.set_num_threads(2)
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    dataset = ReplaySequenceDataset(args.dataset, args.out.parent / '.dense-probe-cache',
        expected_decision_ticks=8, ram_limit_gib=args.ram_limit_gib)
    if (checkpoint.get('model_architecture_version') != 3 or checkpoint.get('decision_ticks') != 8
            or dataset.fingerprint != checkpoint.get('dataset_fingerprint')):
        raise ValueError('Dense probe must use the admitted architecture-v3 0.4-second dataset/checkpoint pair')
    settings = checkpoint['training_config']
    split = {int(key): value for key, value in checkpoint['split'].items()}
    counts = dataset.partition_counts(split, 'train')
    # Explicit candidate settings; optimizer moments are resident before the peak
    # is measured, so this is not a misleading gradient-only memory estimate.
    case = {'defer_metrics': True, 'frame_microbatch': 32, 'matmul': 'high', 'fused': True}
    original_precision = torch.get_float32_matmul_precision()
    report = {'synthetic_pressure_only': True, 'native_trajectory': False,
              'decision_ticks': 8, 'candidate': case, 'gpu': torch.cuda.get_device_name(args.device),
              'checkpoint_progress': checkpoint['progress'], 'dataset_fingerprint': dataset.fingerprint,
              'admission_passed': False,
              'limitation': 'One dense chunk from an empty temporal state; repeated entities are artificial. '
                            'This checks FP16 backward/Adam capacity and finiteness, not tactics, '
                            'whole-game peak memory, or convergence.'}
    episodes = dataset.episodes(split, 'train', epoch=0, seed=args.seed)
    try:
        episode = next(episodes)
        if len(episode.rows) < args.chunk_size:
            raise ValueError('Real seat is shorter than requested chunk; do not fabricate extra frames')
        model, optimizer, scaler = build_case(checkpoint, settings, case, args)
        model.train()
        adapter = CrForgeObservationAdapter(dataset.schema, model.cfg.vocab_size, entity_bucket=16)
        selected_np = episode.supervised[:args.chunk_size]
        if not selected_np.any():
            raise ValueError('First real chunk has no eligible labels')
        rows = episode.rows[:args.chunk_size]
        original = prepare_state_chunk(episode, rows, adapter, args.device)
        inputs = dense_inputs(original, dataset.schema.header['entity_count'][0], args.entities)
        labels_np = episode.labels[:args.chunk_size]
        labels = torch.as_tensor(labels_np, device=args.device)
        selected = torch.as_tensor(selected_np, device=args.device)
        normalization = BCBatchNormalization.from_actions(torch.as_tensor(labels_np[selected_np].copy()))
        state = model.initial_state(1, device=args.device, dtype=torch.float16)
        before = weights_digest(model)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            # No placement_mask: computing all 32 tile distributions deliberately
            # exceeds the sparse training tile-head path's activation pressure.
            logp, _, state = model.forward_state(**inputs, state=state, compute_value=False)
            if not torch.isfinite(logp).all():
                raise FloatingPointError('Dense policy output is nonfinite')
            loss, _ = hierarchical_bc_loss(logp[selected], labels[selected], inputs['legal_action_mask'][selected],
                position_coef=settings['position_coef'], tile_label_sigma=settings['tile_label_sigma'],
                conditional_card_loss=True, normalization=normalization,
                timing_prior=settings['timing_target_action_prior'] if settings['timing_objective'] == 'natural' else None,
                sampled_action_prior=counts['action_prior'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Dense BC loss is nonfinite')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings['max_grad_norm'], error_if_nonfinite=False))
        if not math.isfinite(norm):
            raise FloatingPointError('Dense unscaled gradient norm is nonfinite')
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - started
        memory = torch.cuda.max_memory_allocated(args.device)
        reserved = torch.cuda.max_memory_reserved(args.device)
        state_dtypes = []
        for layer in state.temporal.layers:
            tensors = [layer.matrix] if isinstance(layer, KDAState) else [layer.key, layer.value]
            if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
                raise FloatingPointError('Dense temporal state is nonfinite')
            if isinstance(layer, KDAState) and layer.matrix.dtype != torch.float32:
                raise FloatingPointError('KDA safety accumulator lost FP32')
            if isinstance(layer, GQAState) and any(tensor.dtype != torch.float16 for tensor in tensors):
                raise FloatingPointError('GQA cache is not FP16')
            state_dtypes.append([str(tensor.dtype) for tensor in tensors])
        unchanged = weights_digest(model) == before
        report.update(admission_passed=unchanged, weights_unchanged=unchanged, output_finite=True, loss_finite=True,
            unscaled_gradient_finite=True, temporal_state_finite=True, frame_microbatch=model.frame_microbatch,
            loss=float(loss.detach()), grad_norm_before_clip=norm, amp_scale=scaler.get_scale(),
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            history_frames=args.chunk_size, entities_per_frame=args.entities,
            source_replay_id=episode.replay_id, source_seat=episode.seat,
            source_actual_entity_max=int(original['entity_mask'].sum(-1).max()),
            supervised_rows=int(selected_np.sum()), ordinary_play_labels=int(((labels_np > 0) & (labels_np < 2305) & selected_np).sum()),
            peak_cuda_allocated_bytes=memory, peak_cuda_reserved_bytes=reserved,
            forward_backward_adam_seconds=elapsed, temporal_state_dtypes=state_dtypes,
            optimizer_lr=0.0, full_saved_adam_restored=True)
        if not unchanged:
            raise FloatingPointError('Frozen dense probe changed model weights')
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        episodes.close()
        torch.set_float32_matmul_precision(original_precision)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
