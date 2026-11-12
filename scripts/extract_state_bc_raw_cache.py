"""Frozen full-history raw placement cache, streamed one tiny block to G:.

Skips placement attention/readout entirely. Captures encoded entity tokens and
temporal contexts for GT ordinary-play rows; no optimizer or fitted weights.
GPU execution is explicit in main; --self-test is a tiny CPU-only helper check.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

from diagnose_state_bc_generalization import atomic_json, digest, soft_targets


RAW_FIELDS = ('context', 'entities', 'entity_mask', 'card_features', 'slots')
FLOAT_FIELDS = ('context', 'entities', 'card_features', 'targets', 'entity_xy')


def build_block(context, entities, entity_mask, card_features, *, labels, play,
                legal, entity_xy, hand_card_ids, frame_indices, partition, replay_id, seat, sigma):
    """Select aligned GT rows only; returned CPU storage never aliases GPU tensors."""
    take = torch.as_tensor(np.flatnonzero(play), device=context.device)
    if not len(take):
        return None
    chosen_labels = np.asarray(labels)[np.asarray(play)]
    slots_np, tiles_np = (chosen_labels - 1) // 576, (chosen_labels - 1) % 576
    slots = torch.as_tensor(slots_np, device=context.device, dtype=torch.long)
    chosen_legal = np.asarray(legal)[np.asarray(play), 1:2305].reshape(-1, 4, 576)[
        np.arange(len(slots_np)), slots_np].astype(bool)
    if not chosen_legal[np.arange(len(tiles_np)), tiles_np].all():
        raise ValueError('A supervised GT play lost its legal target tile')
    targets = torch.from_numpy(soft_targets(tiles_np, chosen_legal, sigma))
    targets = targets.float()
    selected_cards = hand_card_ids.index_select(0, take).gather(1, slots[:, None]).squeeze(1)
    if (selected_cards < 0).any():
        raise ValueError('Supervised ordinary play has an unknown raw GT hand-card ID')
    def cpu(value):
        return value.detach().to(device='cpu', copy=True).contiguous()
    block = {
        'context': cpu(context.index_select(0, take)), 'entities': cpu(entities.index_select(0, take)),
        'entity_mask': cpu(entity_mask.index_select(0, take)),
        'card_features': cpu(card_features.index_select(0, take)), 'slots': cpu(slots),
        'legal': torch.from_numpy(chosen_legal.copy()), 'targets': targets,
        'entropy': -(targets * targets.clamp_min(1e-30).log()).sum(-1),
        'tiles': torch.from_numpy(tiles_np.astype(np.int64, copy=True)),
        'entity_xy': cpu(entity_xy.index_select(0, take)), 'card_ids': cpu(selected_cards.long()),
        'episode_frame_indices': torch.from_numpy(np.asarray(frame_indices)[np.asarray(play)].astype(np.int64, copy=True)),
        'partition': partition, 'replay_id': int(replay_id), 'seat': int(seat),
    }
    if any(not torch.isfinite(value).all() for value in block.values() if isinstance(value, torch.Tensor)):
        raise FloatingPointError('Nonfinite cached raw representation')
    return block


class ReferenceChecks:
    """Match per-seat GT-action ordinal, then check labels before float values."""
    def __init__(self, root, *, checkpoint_sha, parent_sha, rows_per_seat=4, atol=.002, rtol=.002):
        self.records = defaultdict(list)
        self.progress = defaultdict(int)
        self.checked = defaultdict(int)
        self.atol, self.rtol, self.rows_per_seat = atol, rtol, rows_per_seat
        self.summary = {'enabled': root is not None, 'match': 'replay/seat/chronological GT-play ordinal + exact slot/tile/target',
            'rows': 0, 'atol': atol, 'rtol': rtol, 'max_abs_by_field': {}, 'outside_tolerance_rows': 0}
        if root is None:
            return
        root = Path(root)
        protocol = json.loads((root.parent / 'protocol.json').read_text())
        if protocol['checkpoint_sha256'] != checkpoint_sha or protocol['parent_manifest_sha256'] != parent_sha:
            raise ValueError('Reference cache belongs to another checkpoint or original manifest')
        self.summary['reference_protocol_sha256'] = digest(root.parent / 'protocol.json')
        totals = defaultdict(int)
        for path in sorted(root.glob('block_*.pt')):
            value = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            key = (int(value['replay_id']), int(value['seat']))
            start, stop = totals[key], totals[key] + len(value['tiles'])
            if start < rows_per_seat:
                self.records[key].append((path, start, stop))
            totals[key] = stop
            del value
        self.summary['reference_seats'] = len(totals)

    def compare(self, block):
        key = (block['replay_id'], block['seat'])
        start, stop = self.progress[key], self.progress[key] + len(block['tiles'])
        self.progress[key] = stop
        if self.checked[key] >= self.rows_per_seat:
            return
        for path, ref_start, ref_stop in self.records[key]:
            left, right = max(start, ref_start), min(stop, ref_stop, self.rows_per_seat)
            if left >= right:
                continue
            reference = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            for ordinal in range(left, right):
                a, b = ordinal - start, ordinal - ref_start
                for field in ('slots', 'tiles', 'legal', 'targets'):
                    if not torch.equal(block[field][a], reference[field][b]):
                        raise ValueError('Reference GT action/target alignment differs; refusing a float comparison')
                mask, ref_mask = block['entity_mask'][a].bool(), reference['entity_mask'][b].bool()
                if not torch.equal(mask[:min(len(mask),len(ref_mask))], ref_mask[:min(len(mask),len(ref_mask))]) or mask.sum() != ref_mask.sum():
                    raise ValueError('Reference valid entity rows differ')
                outside = False
                for field in FLOAT_FIELDS:
                    if field not in reference:
                        continue
                    current, old = block[field][a], reference[field][b]
                    if field in ('entities', 'entity_xy'):
                        current, old = current[mask], old[ref_mask]
                    difference = float((current.float() - old.float()).abs().max())
                    self.summary['max_abs_by_field'][field] = max(difference,
                        self.summary['max_abs_by_field'].get(field,0.))
                    outside |= not torch.allclose(current.float(), old.float(), atol=self.atol, rtol=self.rtol)
                self.summary['outside_tolerance_rows'] += int(outside)
                self.summary['rows'] += 1
                self.checked[key] += 1
            del reference


def run(args):
    from cr_rl.models.state_policy import state_model_from_checkpoint
    from cr_rl.data.observations import CrForgeObservationAdapter
    from cr_rl.training.state_bc import ReplaySequenceDataset, check_ram_budget, prepare_state_chunk
    if args.out.resolve().drive.upper() != 'G:':
        raise ValueError('Cache and all scratch outputs must stay on G:')
    args.out.mkdir(parents=True, exist_ok=True)
    cache_root = args.out / 'feature_cache'
    cache_root.mkdir(exist_ok=True)
    if any(cache_root.iterdir()):
        raise ValueError('Raw cache output must be empty; no overwrite or implicit resume')
    if not torch.cuda.is_available():
        raise RuntimeError('Existing CUDA required; no CPU fallback for the actual extraction')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(20261111)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,args.gpu_memory_limit_gib*1024**3/total_vram),0)
    manifest = json.loads(args.dataset.read_text())
    parent_sha, checkpoint_sha = digest(args.parent_manifest), digest(args.checkpoint)
    cohort = json.loads(args.cohort.read_text())
    if (cohort['reference_checkpoint_sha256'] != checkpoint_sha or
        cohort['checkpoint_bound_dataset_manifest_sha256'] != parent_sha or
        cohort['source_dataset_sha256'] != manifest['source_sha256']):
        raise ValueError('Frozen full cohort/checkpoint/source binding differs')
    games = {part: sorted(map(int,cohort['partitions'][part]['game_ids']))
        for part in ('train','validation','heldout')}
    assignments = {rid: part for part, ids in games.items() for rid in ids}
    if len(assignments) != sum(map(len,games.values())):
        raise ValueError('Cohort partitions overlap')
    checkpoint = torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if (checkpoint['dataset_fingerprint'] != parent_sha or checkpoint.get('model_architecture_version') != 3 or
        checkpoint.get('decision_ticks') != 8 or any(checkpoint['split'].get(str(rid)) != part for rid,part in assignments.items())):
        raise ValueError('Checkpoint architecture, interval, original split, or dataset fingerprint differs')
    settings = dict(checkpoint['training_config'])
    model = state_model_from_checkpoint(checkpoint,'cuda')
    progress = dict(checkpoint['progress'])
    del checkpoint
    gc.collect()
    model.eval().requires_grad_(False)
    model.frame_microbatch = 32
    dataset = ReplaySequenceDataset(args.dataset,args.out/'.replay-cache',ram_limit_gib=args.ram_limit_gib)
    if dataset.replay_ids != set(assignments):
        raise ValueError('Prepared dataset must contain exactly the frozen cohort games')
    adapter = CrForgeObservationAdapter(dataset.schema,model.cfg.vocab_size,entity_bucket=16)
    xy_columns = [dataset.schema.entity_fields.index(field) for field in ('x','y')]
    reference = ReferenceChecks(args.reference_cache,checkpoint_sha=checkpoint_sha,parent_sha=parent_sha)
    report = {'checkpoint_sha256':checkpoint_sha,'checkpoint_progress':progress,
        'parent_manifest_sha256':parent_sha,'source_sha256':manifest['source_sha256'],
        'dataset_sha256':digest(args.dataset),'cohort_sha256':digest(args.cohort),'games':games,
        'scope':'Frozen raw-only capture; full history from tick zero; no placement attention/readout, fitting or exported weights',
        'precision':'fp16 autocast; configured FP32 frame encoder preserved','float32_matmul_precision':'high',
        'ram_limit_gib':args.ram_limit_gib,'gpu_memory_limit_gib':args.gpu_memory_limit_gib,
        'extraction':{'blocks':0,'bytes':0,'disk_bytes':0,'history_frames':0,'play_rows':0,'seats':0,
            'raw_capture_retained':list(RAW_FIELDS)+['entity_xy','card_ids'],
            'all_frames_from_tick_zero':True,'features':'Raw context/tokens only; no per-tile feature tensors',
            'placement_attention_evaluated':False,'fitted_weights_saved':False},
        'reference_check':reference.summary,'completed':False}
    active = {}
    original_placement = model.placement_logits
    def capture(context,entities,entity_mask,card_features):
        start, stop = active['cursor'], active['cursor'] + len(context)
        active['cursor'] = stop
        block = build_block(context,entities,entity_mask,card_features,
            labels=active['labels'][start:stop],play=active['play'][start:stop],legal=active['legal'][start:stop],
            entity_xy=active['entity_xy'][start:stop],hand_card_ids=active['hands'][start:stop],
            frame_indices=active['frames'][start:stop],partition=active['partition'],replay_id=active['rid'],
            seat=active['seat'],sigma=settings['tile_label_sigma'])
        if block is not None:
            reference.compare(block)
            count = report['extraction']['blocks'] + 1
            path = cache_root / f'block_{count:06d}.pt'
            pending = path.with_suffix('.pt.tmp')
            torch.save(block,pending)
            pending.replace(path)
            report['extraction']['blocks'] = count
            report['extraction']['bytes'] += sum(v.numel()*v.element_size() for v in block.values() if isinstance(v,torch.Tensor))
            report['extraction']['disk_bytes'] += path.stat().st_size
            report['extraction']['play_rows'] += len(block['tiles'])
            del block
        return context.new_zeros(len(context),4,576,dtype=torch.float32)
    model.placement_logits = capture
    episodes = dataset.episodes({rid:'probe' for rid in assignments},'probe')
    seen = set()
    started = time.perf_counter()
    atomic_json(args.out/'protocol.json',report)
    try:
        with torch.inference_mode():
            for episode in episodes:
                state = model.initial_state(1,device='cuda',dtype=torch.float16)
                for start,rows in episode.chunks(32):
                    check_ram_budget(args.ram_limit_gib)
                    inputs = prepare_state_chunk(episode,rows,adapter,'cuda')
                    labels = episode.labels[start:start+len(rows)]
                    active.update(cursor=0,labels=labels,play=episode.supervised[start:start+len(rows)]&(labels>0)&(labels<2305),
                        legal=episode.arrays['masks'][rows],entity_xy=inputs['entities'][...,xy_columns],
                        hands=inputs['hand_card_ids'],frames=np.arange(start,start+len(rows)),
                        partition=assignments[episode.replay_id],rid=episode.replay_id,seat=episode.seat)
                    with torch.autocast('cuda',dtype=torch.float16):
                        lp,_,state = model.forward_state(**inputs,state=state,compute_value=False)
                    if active['cursor'] != len(rows) or not torch.isfinite(lp).all():
                        raise ValueError('Raw capture lost frame alignment or generated nonfinite probabilities')
                    state = model.detach_state(state)
                    report['extraction']['history_frames'] += len(rows)
                    active.clear()
                    del inputs,lp
                del state
                seen.add((episode.replay_id,episode.seat))
                report['extraction']['seats'] = len(seen)
                report['seconds'] = time.perf_counter()-started
                atomic_json(args.out/'protocol.json',report)
                print(json.dumps({'phase':'raw_extract','replay_id':episode.replay_id,'seat':episode.seat,
                    'seats':len(seen),'blocks':report['extraction']['blocks'],'play_rows':report['extraction']['play_rows'],
                    'disk_gib':report['extraction']['disk_bytes']/1024**3,'rss_gib':check_ram_budget(args.ram_limit_gib)/1024**3}),flush=True)
    finally:
        episodes.close()
        model.placement_logits = original_placement
    expected = {(rid,seat) for rid in assignments for seat in (0,1)}
    if seen != expected or report['extraction']['play_rows'] != sum(value['play'] for value in dataset.supervision_counts.values()):
        raise ValueError('Extracted seats or GT-play total differs from admitted full dataset')
    report.update(completed=True,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
    if digest(args.checkpoint) != checkpoint_sha:
        raise ValueError('Parent checkpoint changed during extraction')
    atomic_json(args.out/'protocol.json',report)


def cpu_self_test():
    context=torch.randn(4,8)
    entities=torch.randn(4,3,6)
    mask=torch.ones(4,3,dtype=torch.bool)
    cards=torch.randn(4,4,6)
    labels=np.array([0,1+2*576+31,2305,1+8])
    legal=np.ones((4,2307),dtype=bool)
    block=build_block(context,entities,mask,cards,labels=labels,play=np.array([False,True,False,True]),
        legal=legal,entity_xy=entities[...,:2],hand_card_ids=torch.tensor([[1,2,3,4]]*4),
        frame_indices=np.array([32,33,34,35]),partition='train',replay_id=9,seat=1,sigma=1.)
    assert block['slots'].tolist()==[2,0] and block['card_ids'].tolist()==[3,1]
    assert block['episode_frame_indices'].tolist()==[33,35]
    assert torch.equal(block['context'],context[[1,3]]) and block['targets'].shape==(2,576)
    assert torch.allclose(block['targets'].sum(-1),torch.ones(2))
    assert all(value.device.type=='cpu' for value in block.values() if isinstance(value,torch.Tensor))
    assert 'features' not in block
    scratch=Path(__file__).resolve().parents[1]/'artifacts/bc_capacity_20261111/tmp'
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='raw-helper-',dir=scratch) as temporary:
        root=Path(temporary)
        reference_root=root/'feature_cache'
        reference_root.mkdir()
        (root/'protocol.json').write_text(json.dumps({'checkpoint_sha256':'cpu-fixture','parent_manifest_sha256':'parent-fixture'}))
        torch.save(block,reference_root/'block_000001.pt')
        checker=ReferenceChecks(reference_root,checkpoint_sha='cpu-fixture',parent_sha='parent-fixture')
        for index in range(2):
            piece={key:value[index:index+1].clone() if isinstance(value,torch.Tensor) else value for key,value in block.items()}
            # Different padded bucket sizes are semantically identical.
            for field in ('entities','entity_xy'):
                piece[field]=torch.nn.functional.pad(piece[field],(0,0,0,2))
            piece['entity_mask']=torch.nn.functional.pad(piece['entity_mask'],(0,2))
            checker.compare(piece)
        assert checker.summary['rows']==2 and checker.summary['outside_tolerance_rows']==0
        checker=ReferenceChecks(reference_root,checkpoint_sha='cpu-fixture',parent_sha='parent-fixture')
        changed={key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in block.items()}
        changed['context']+=.1
        checker.compare(changed)
        assert checker.summary['outside_tolerance_rows']==2
        checker=ReferenceChecks(reference_root,checkpoint_sha='cpu-fixture',parent_sha='parent-fixture')
        changed['tiles'][0]+=1
        try:
            checker.compare(changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Misaligned reference labels must be rejected')
    return {'passed':True,'device':'cpu','raw_alignment':True,'gt_card_identity':True,'normalized_targets':True,
        'no_per_tile_features':True,'reference_cross_block_and_padding_match':True,
        'reference_numeric_drift_recorded':True,'reference_label_mismatch_rejected':True,
        'limitation':'Tiny CPU helper test; actual full-model GPU extraction not executed'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint','parent-manifest','dataset','cohort','out'):
        parser.add_argument('--'+flag,type=Path)
    parser.add_argument('--ram-limit-gib',type=float,default=6.)
    parser.add_argument('--gpu-memory-limit-gib',type=float,default=4.)
    parser.add_argument('--reference-cache',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(cpu_self_test()))
    else:
        if any(getattr(args,name) is None for name in ('checkpoint','parent_manifest','dataset','cohort','out')):
            parser.error('Provide checkpoint, parent-manifest, dataset, cohort, and out')
        if min(args.ram_limit_gib,args.gpu_memory_limit_gib)<=0:
            parser.error('Positive memory limits required')
        run(args)
