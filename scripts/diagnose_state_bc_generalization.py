"""Frozen local cross-game BC evaluation; stream one original game at a time.

This is an offline diagnostic, never an optimizer or a production policy change.
Outputs retain per-seat scores so uncertainty can be clustered by whole game.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def stream_games(path):
    """Never expand the entire observations array in RAM or on disk."""
    with np.load(path, allow_pickle=False) as source:
        small = {key: source[key] for key in ('actions', 'flags', 'ticks', 'seats', 'replay_ids')}
    counts = Counter(map(int, small['replay_ids']))
    with zipfile.ZipFile(path) as archive:
        streams = {}
        for field in ('observations', 'masks'):
            stream = archive.open(field + '.npy')
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            if fortran:
                raise ValueError('Row streaming requires C-contiguous NPY arrays')
            streams[field] = (stream, shape, np.dtype(dtype))
        offset = 0
        for replay_id, count in sorted(counts.items()):
            if not np.all(small['replay_ids'][offset:offset + count] == replay_id):
                raise ValueError('Original source games are not contiguous')
            arrays = {key: value[offset:offset + count] for key, value in small.items()}
            for field, (stream, shape, dtype) in streams.items():
                size = count * int(np.prod(shape[1:])) * dtype.itemsize
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError('Truncated original NPY stream')
                arrays[field] = np.frombuffer(raw, dtype=dtype).reshape(count, *shape[1:])
            yield int(replay_id), offset, arrays
            offset += count


def soft_targets(tiles, legal, sigma):
    grid = np.arange(576)
    distance = (grid[None] % 18 - tiles[:, None] % 18)**2
    distance += (grid[None] // 18 - tiles[:, None] // 18)**2
    if sigma:
        targets = np.exp(-distance / (2 * sigma**2)) * legal
    else:
        targets = (grid[None] == tiles[:, None]).astype(float) * legal
    return targets / targets.sum(-1, keepdims=True)


def build_prior(source, assignments, metadata, sigma):
    header = metadata['schema']['header_fields']
    hand = [i for i, name in enumerate(header) if name == 'hand_card']
    mass = defaultdict(lambda: np.zeros(576, dtype=np.float64))
    card_counts = Counter()
    original_counts = defaultdict(Counter)
    for rid, _, arrays in stream_games(source):
        labels = arrays['actions']
        valid = (arrays['flags'] == 0) & arrays['masks'][np.arange(len(labels)), labels].astype(bool)
        part = assignments[rid]
        original_counts[part].update(act=int(((labels != 0) & valid).sum()),
            play=int(((labels > 0) & (labels < 2305) & valid).sum()))
        if part != 'train':
            continue
        rows = np.flatnonzero(valid & (labels > 0) & (labels < 2305))
        slots = (labels[rows] - 1) // 576
        tiles = (labels[rows] - 1) % 576
        cards = arrays['observations'][rows, np.asarray(hand)[slots]]
        legal = arrays['masks'][rows, 1:2305].reshape(-1, 4, 576)[np.arange(len(rows)), slots].astype(bool)
        targets = soft_targets(tiles, legal, sigma)
        for card in np.unique(cards):
            selected = cards == card
            mass[int(card)] += targets[selected].sum(0)
            card_counts[int(card)] += int(selected.sum())
    return dict(mass), card_counts, original_counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True, help='Checkpoint-bound original full manifest')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ram-limit-gib', type=float, default=6.)
    parser.add_argument('--max-games', type=int, default=0, help='Smoke only; zero means all games')
    parser.add_argument('--prior-cache', type=Path, help='Reuse a prior from an identical immutable diagnostic protocol')
    args = parser.parse_args(argv)
    import torch
    from cr_rl.data.event_preserving_resample import FIELDS, resample_game
    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.models.state_policy import state_model_from_checkpoint
    from cr_rl.training.bc import hierarchical_bc_loss
    from cr_rl.training.state_bc import StateEpisode, check_ram_budget, make_state_schema
    if not torch.cuda.is_available():
        raise RuntimeError('Requires the existing CUDA environment; install nothing')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'seats').mkdir(exist_ok=True)
    manifest = json.loads(args.dataset.read_text(encoding='utf-8'))
    metadata_path = args.dataset.parent / manifest['metadata']
    if digest(metadata_path) != manifest['metadata_sha256']:
        raise ValueError('Checkpoint-bound metadata checksum mismatch')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if checkpoint['dataset_fingerprint'] != digest(args.dataset):
        raise ValueError('Checkpoint is not bound to this exact dataset manifest')
    if checkpoint.get('model_architecture_version') != 3 or checkpoint.get('decision_ticks') != 8:
        raise ValueError('Expected architecture-v3 event-preserving 0.4s checkpoint')
    assignments = {int(key): value for key, value in checkpoint['split'].items()}
    settings, progress = dict(checkpoint['training_config']), dict(checkpoint['progress'])
    checkpoint_sha = digest(args.checkpoint)
    protocol = {'checkpoint_sha256': checkpoint_sha, 'checkpoint_progress': progress,
        'manifest_sha256': checkpoint['dataset_fingerprint'], 'source_sha256': digest(args.source),
        'training_config': settings, 'split_games': dict(Counter(assignments.values())),
        'max_games': args.max_games, 'optimizer_created': False,
        'scope': 'Frozen offline teacher-forced chronological inference; not native battle win rate',
        'prior': 'Train-only per-card sigma-target location histogram; fixed alpha .01 and 1 per tile',
        'aggregation': 'Equal episode-seat means; whole-game clustered uncertainty',
        'gpu': torch.cuda.get_device_name(0)}
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Refuse to mix diagnostic results from different immutable protocols')
    atomic_json(protocol_path, protocol)
    print(json.dumps({'phase': 'load_model', 'checkpoint_progress': progress,
        'rss_gib': check_ram_budget(args.ram_limit_gib) / 1024**3}), flush=True)
    model = state_model_from_checkpoint(checkpoint, 'cuda')
    del checkpoint
    gc.collect()
    model.frame_microbatch = 32
    model.eval().requires_grad_(False)
    schema = make_state_schema(metadata)
    adapter = CrForgeObservationAdapter(schema, model.cfg.vocab_size, entity_bucket=16)
    columns = schema.header['decision_ticks']
    sigma = settings['tile_label_sigma']
    if args.prior_cache:
        prior_protocol = json.loads((args.prior_cache.parent / 'protocol.json').read_text())
        for key in ('checkpoint_sha256', 'manifest_sha256', 'source_sha256', 'training_config'):
            if prior_protocol[key] != protocol[key]:
                raise ValueError('Cached prior is not bound to the exact diagnostic protocol')
        cached = json.loads(args.prior_cache.read_text())
        if cached['sigma'] != sigma or cached['fit_games'] != sum(part == 'train' for part in assignments.values()):
            raise ValueError('Cached prior has different training cohort or targets')
        prior_mass = {int(key): np.asarray(value) for key, value in cached['mass'].items()}
        card_counts = Counter({int(key): value for key, value in cached['card_counts'].items()})
        original_counts = cached['valid_original_counts']
    else:
        prior_mass, card_counts, original_counts = build_prior(args.source, assignments, metadata, sigma)
    atomic_json(args.out / 'train_prior.json', {'card_counts': card_counts,
        'mass': {key: value.tolist() for key, value in prior_mass.items()},
        'fit_games': sum(part == 'train' for part in assignments.values()), 'sigma': sigma,
        'valid_original_counts': {key: dict(value) for key, value in original_counts.items()}})
    # Whole-partition row denominators come from the immutable 0.4 audit;
    # recomputed below and checked at completion. No minibatch prior estimates.
    audit = json.loads((args.source.parent.parent.parent /
        'bc_diagnostic_20261110/supervision/bc100/audit.json').read_text())
    if audit['source_npz_sha256'] != protocol['source_sha256']:
        raise ValueError('Full original source SHA differs from immutable supervision audit')
    priors = {part: audit['stats'][part + '/resampled']['valid_nonwait'] /
        audit['stats'][part + '/resampled']['valid_rows'] for part in set(assignments.values())}
    if abs(priors['train'] - settings['sampled_action_prior']) > 1e-12:
        raise ValueError('Original training action prior mismatch')
    known = {int(record['replay_offset']): record for record in manifest['shards']}
    hand = np.asarray(schema.header['hand_card'])
    cost_columns = np.asarray(schema.header['hand_cost'])
    started = time.perf_counter()
    observed_counts = defaultdict(Counter)
    checked_shards = []
    summary_path = args.out / 'seat_metrics.jsonl'
    completed = {}
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            row = json.loads(line)
            if row['checkpoint_sha256'] != checkpoint_sha:
                raise ValueError('Mixed checkpoint results')
            completed[(row['replay_id'], row['seat'])] = row
    with summary_path.open('a', encoding='utf-8') as output:
        for game_index, (rid, offset, original) in enumerate(stream_games(args.source)):
            if args.max_games and game_index >= args.max_games:
                break
            arrays, resampling_audit = resample_game(original, decision_ticks=8,
                decision_header_columns=columns, source_row_ids=np.arange(offset, offset + len(original['actions'])))
            if not resampling_audit['all_nonwait_structurally_preserved']:
                raise ValueError('Resampling dropped original action supervision')
            del original
            record = known[rid]
            if len(arrays['actions']) != record['rows']:
                raise ValueError('Resampled rows differ from checkpoint-bound dataset')
            local_shard = args.dataset.parent / record['path']
            if local_shard.exists():
                if digest(local_shard) != record['sha256']:
                    raise ValueError('Downloaded reference shard SHA differs')
                with np.load(local_shard, allow_pickle=False) as reference:
                    for field in (*FIELDS, 'source_ticks', 'source_rows'):
                        expected = arrays[field] if field != 'replay_ids' else np.zeros_like(arrays[field])
                        if not np.array_equal(reference[field], expected):
                            raise ValueError(f'Local resampling differs from trained shard: {rid}/{field}')
                checked_shards.append(rid)
            part = assignments[rid]
            y = arrays['actions']
            valid = (arrays['flags'] == 0) & arrays['masks'][np.arange(len(y)), y].astype(bool)
            observed_counts[part].update(rows=int(valid.sum()), act=int(((y != 0) & valid).sum()),
                play=int(((y > 0) & (y < 2305) & valid).sum()))
            for seat in (0, 1):
                if (rid, seat) in completed:
                    continue
                seat_started = time.perf_counter()
                rows = np.flatnonzero(arrays['seats'] == seat)
                episode = StateEpisode(rid, seat, rows, arrays, 8)
                state = model.initial_state(1, device='cuda', dtype=torch.float16)
                totals = dict.fromkeys(('loss', 'timing_loss', 'conditional_slot_loss', 'position_loss'), 0.)
                timing = defaultdict(list)
                plays = defaultdict(list)
                correct = 0
                with torch.inference_mode():
                    for start, selected_rows in episode.chunks(32):
                        check_ram_budget(args.ram_limit_gib)
                        from cr_rl.training.state_bc import prepare_state_chunk
                        inputs = prepare_state_chunk(episode, selected_rows, adapter, 'cuda')
                        with torch.autocast('cuda', dtype=torch.float16):
                            lp, _, state = model.forward_state(**inputs, state=state, compute_value=False)
                        if not torch.isfinite(lp).all():
                            raise FloatingPointError('Nonfinite frozen model output')
                        state = model.detach_state(state)
                        selected = episode.supervised[start:start + len(selected_rows)]
                        pick = torch.as_tensor(selected, device='cuda')
                        labels = episode.labels[start:start + len(selected_rows)][selected]
                        labels_t = torch.as_tensor(labels, device='cuda')
                        lp = lp[pick].float()
                        legal_t = inputs['legal_action_mask'][pick]
                        loss, metrics = hierarchical_bc_loss(lp, labels_t, legal_t,
                            position_coef=settings['position_coef'], tile_label_sigma=sigma,
                            conditional_card_loss=True, normalization=episode.normalization,
                            timing_prior=settings['timing_target_action_prior'], sampled_action_prior=priors[part])
                        totals['loss'] += float(loss)
                        for key in totals.keys() - {'loss'}:
                            totals[key] += float(metrics[key])
                        cards = torch.cat((lp[:, :1], torch.logsumexp(lp[:, 1:2305].reshape(-1, 4, 576), -1), lp[:, 2305:]), -1)
                        probabilities = (-torch.expm1(lp[:, 0])).cpu().numpy()
                        choice = legal_t[:, 1:].any(-1).cpu().numpy()
                        timing['labels'].append(labels != 0)
                        timing['probabilities'].append(probabilities)
                        timing['choice'].append(choice)
                        legal = legal_t.cpu().numpy()
                        play = (labels > 0) & (labels < 2305)
                        if not play.any():
                            continue
                        truth_slot = (labels[play] - 1) // 576
                        truth_tile = (labels[play] - 1) % 576
                        selected_original_rows = selected_rows[selected][play]
                        card_ids = arrays['observations'][selected_original_rows, hand[truth_slot]].astype(np.int64)
                        costs = arrays['observations'][selected_original_rows, cost_columns[truth_slot]].astype(np.int64)
                        conditional = lp[torch.as_tensor(play, device='cuda'), 1:2305].reshape(-1, 4, 576)
                        gather = torch.as_tensor(truth_slot, device='cuda')
                        tile_lp = conditional[torch.arange(len(gather), device='cuda'), gather]
                        tile_lp = (tile_lp - torch.logsumexp(tile_lp, -1, keepdim=True)).cpu().numpy()
                        tile_legal = legal[play, 1:2305].reshape(-1, 4, 576)[np.arange(len(gather)), truth_slot]
                        targets = soft_targets(truth_tile, tile_legal, sigma)
                        entropy = -(targets * np.log(np.maximum(targets, 1e-300))).sum(-1)
                        prediction = tile_lp.argmax(-1)
                        distance = abs(prediction % 18 - truth_tile % 18) + abs(prediction // 18 - truth_tile // 18)
                        predicted_slot = cards[:, 1:5].argmax(-1).cpu().numpy()[play]
                        correct += int((predicted_slot == truth_slot).sum())
                        values = {'cards': card_ids, 'costs': costs, 'slot_correct': predicted_slot == truth_slot,
                            'tile_ce': -(targets * tile_lp).sum(-1), 'target_entropy': entropy,
                            'tile_hard_nll': -tile_lp[np.arange(len(truth_tile)), truth_tile],
                            'tile_top1': prediction == truth_tile, 'tile_within2': distance <= 2,
                            'tile_distance': distance, 'uniform_tile_ce': np.log(tile_legal.sum(-1)),
                            'uniform_slot_accuracy': 1 / legal[play, 1:2305].reshape(-1, 4, 576).any(-1).sum(-1)}
                        for alpha in (.01, 1.):
                            marginal = np.stack([prior_mass.get(int(card), np.zeros(576)) for card in card_ids]) + alpha
                            marginal *= tile_legal
                            marginal /= marginal.sum(-1, keepdims=True)
                            values['prior_ce_' + str(alpha)] = -(targets * np.log(np.maximum(marginal, 1e-300))).sum(-1)
                        for key, value in values.items():
                            plays[key].append(value)
                timing_arrays = {key: np.concatenate(value) for key, value in timing.items()}
                play_arrays = {key: np.concatenate(value) for key, value in plays.items()}
                np.savez_compressed(args.out / 'seats' / f'replay_{rid:06d}_seat{seat}.npz',
                    **{'timing_' + key: value for key, value in timing_arrays.items()},
                    **{'play_' + key: value for key, value in play_arrays.items()})
                replay = metadata['replays'][rid]
                row = {'checkpoint_sha256': checkpoint_sha, 'replay_id': rid, 'seat': seat, 'partition': part,
                    'valid_rows': episode.normalization.rows, 'action_rows': episode.normalization.act,
                    'play_rows': episode.normalization.play, 'choice_rows': int(timing_arrays['choice'].sum()),
                    'head_metrics': totals, 'slot_accuracy': correct / max(1, episode.normalization.play),
                    'play_metrics': {key: float(np.mean(value)) for key, value in play_arrays.items()
                        if key not in ('cards', 'costs')},
                    'deck': sorted(replay['decks'][seat]), 'source': replay.get('source'),
                    'seconds': time.perf_counter() - seat_started,
                    'rss_gib': check_ram_budget(args.ram_limit_gib) / 1024**3,
                    'vram_gib': torch.cuda.max_memory_allocated() / 1024**3}
                output.write(json.dumps(row, allow_nan=False) + '\n')
                output.flush()
                completed[(rid, seat)] = row
                print(json.dumps({'phase': 'frozen_inference', 'game': rid, 'seat': seat, 'partition': part,
                    'seats_done': len(completed), 'loss': totals['loss'], 'tile_ce': totals['position_loss'],
                    'seconds': row['seconds'], 'rss_gib': row['rss_gib']}), flush=True)
                del state, episode
            del arrays
    if not args.max_games:
        for part, counts in observed_counts.items():
            if counts['act'] != original_counts[part]['act'] or counts['play'] != original_counts[part]['play']:
                raise ValueError('Original action supervision counts changed')
            if abs(counts['act'] / counts['rows'] - priors[part]) > 1e-12:
                raise ValueError('Partition prior differs from recomputed complete dataset')
        if len(completed) != 2 * len(assignments):
            raise ValueError('Missing one or more complete seats')
    atomic_json(args.out / 'completion.json', {'completed': True, 'full_dataset': not args.max_games,
        'seconds_this_invocation': time.perf_counter() - started, 'seats': len(completed),
        'partition_counts': {key: dict(value) for key, value in observed_counts.items()},
        'exact_reference_shard_matches': checked_shards, 'model_unchanged': True,
        'checkpoint_sha256_after': digest(args.checkpoint), 'optimizer_created': False})


if __name__ == '__main__':
    main()
