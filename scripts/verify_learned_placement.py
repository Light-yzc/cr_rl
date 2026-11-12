"""Strict-load the retained small decoder and check a real all-hand fixture."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

from diagnose_state_bc_generalization import atomic_json, digest
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.learned_placement import attach_learned_placement
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.state_bc import make_state_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    if not torch.cuda.is_available():
        raise RuntimeError('Use the existing CUDA environment; install nothing')
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    source_sha = digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model = state_model_from_checkpoint(checkpoint, 'cuda').eval()
    del checkpoint
    gc.collect()
    payload = torch.load(args.decoder, map_location='cpu', mmap=True, weights_only=False)
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=False)
    values = {key: value.to('cuda') for key, value in fixture.items() if isinstance(value, torch.Tensor)}
    hands = values['hand_card_ids']
    legal = values['full_legal_action_mask'].bool()
    context, entities = values['context'].float(), values['entities'].float()
    mask, features = values['entity_mask'], values['card_features'].float()
    raw = np.asarray(fixture['raw_observations'])
    if raw.ndim == 1:
        raw = raw[None]
    metadata = json.loads(args.metadata.read_text())
    observation_adapter = CrForgeObservationAdapter(make_state_schema(metadata), model.cfg.vocab_size, entity_bucket=16)
    h, e, m = observation_adapter(raw)
    online = {'header': torch.as_tensor(h, device='cuda'), 'entities': torch.as_tensor(e, device='cuda'),
              'entity_mask': torch.as_tensor(m, device='cuda'), 'legal_action_mask': legal,
              'hand_card_ids': hands}
    base_parameters = sum(value.numel() for value in model.parameters())
    with torch.inference_mode():
        base_tile = model.placement_logits(context, entities, mask, features)
        slot = model.slot_actor(torch.cat((context[:, None].expand(-1, 4, -1), features), -1)).squeeze(-1)
        cards = torch.cat((model.wait_actor(context), slot, model.ability_actor(context)), -1)
        base_lp = model._joint_log_probs(cards, base_tile, legal)
        with torch.autocast('cuda', dtype=torch.float16):
            online_before = model.step(**online)[0]
    learned = attach_learned_placement(model, payload, source_checkpoint_sha256=source_sha)
    model.requires_grad_(False)
    with torch.inference_mode():
        all_tiles = learned(context, entities, mask, features, entity_xy=values['entity_xy'], card_ids=hands)
        expected_gt = learned.forward_gt(context, entities, mask, features, values['slots'],
            entity_xy=values['entity_xy'], card_ids=hands)
        actual_gt = all_tiles[torch.arange(len(context), device='cuda'), values['slots']]
        if not torch.allclose(expected_gt, actual_gt, atol=2e-4, rtol=1e-4):
            raise ValueError('Production all-hand path differs from supervised GT-slot path')
        candidate_lp = model._joint_log_probs(cards, all_tiles, legal)
        with torch.autocast('cuda', dtype=torch.float16):
            online_after = model.step(**online)[0]
    def masses(lp):
        return torch.cat((lp[:, :1], lp[:, 1:2305].reshape(-1, 4, 576).logsumexp(-1), lp[:, 2305:]), -1)
    mass_error = float((masses(base_lp) - masses(candidate_lp)).abs().max())
    online_error = float((masses(online_before) - masses(online_after)).abs().max())
    if max(mass_error, online_error) > 3e-6:
        raise ValueError('Placement update changed card/WAIT probability mass')
    if not torch.isfinite(candidate_lp).all() or not torch.isfinite(online_after).all():
        raise FloatingPointError('Nonfinite production policy output')
    if not torch.allclose(candidate_lp.exp().sum(-1), torch.ones(len(context), device='cuda'), atol=2e-6):
        raise ValueError('Production joint policy is not normalized')
    # Tensor/state key roundtrip is tested without a second full GPU trunk or
    # another multi-GB checkpoint. CPU tiny-policy tests cover strict factory IO.
    definition = model.checkpoint_config()['learned_placement_definition']
    report = {'pass': True, 'checkpoint_sha256': source_sha, 'decoder_sha256': digest(args.decoder),
        'real_hand_ids': hands.cpu().tolist(), 'temperature': definition['temperature'],
        'all_four_slot_shape': list(all_tiles.shape), 'gt_vs_all_logits_max_error': float((actual_gt - expected_gt).abs().max()),
        'real_full_legal_card_wait_logmass_max_error': mass_error, 'end_to_end_step_card_wait_logmass_max_error': online_error,
        'joint_sum_error': float((candidate_lp.exp().sum(-1) - 1).abs().max()),
        'base_parameters': base_parameters, 'registered_parameters': sum(value.numel() for value in model.parameters()),
        'active_parameter_note': 'Registered count includes the retained unused original placement head for checkpoint compatibility',
        'source_checkpoint_unchanged': digest(args.checkpoint) == source_sha,
        'definition': definition, 'optimizer_created': False, 'new_checkpoint_saved': False}
    atomic_json(args.out, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
