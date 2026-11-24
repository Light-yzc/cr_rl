"""Train a separate actor candidate on one full-data Sparky BC phase."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from cr_rl.envs.crforge import CrForgeVector
from cr_rl.training.expert_bc import ExpertBC
from cr_rl.training.inference import StateInference
from cr_rl.training.state_selfplay import _extract_decks
from run_nulls_dryrun import load_policy_inference


def collect_probes(actor, inference, decks_path):
    _, names, decks, forms = _extract_decks(decks_path)
    focal = [i for i, deck in enumerate(decks) if 'ZapMachine' in deck]
    general = [i for i, deck in enumerate(decks) if 'ZapMachine' not in deck]
    if not focal or len(general) < 3:
        raise ValueError('PPO probe deck manifest needs Sparky and general decks')
    chosen = [(focal[0], general[0]), (general[1], general[2])]
    with CrForgeVector(2, 2, independent=True, decision_ticks=8) as env:
        env.set_autoreset(False)
        for lane, (left, right) in enumerate(chosen):
            env.vector.reset_env(lane, 2026112000 + lane, decks[left], decks[right], forms[left], forms[right])
        policy = StateInference(actor, inference.adapter, inference.device, 4,
                                precision=inference.precision, compute_value=False)
        observations, masks, actions = [], [], []
        for _ in range(96):
            raw = env.observations.copy().reshape(4, -1)
            legal = env.masks.copy().reshape(4, -1)
            selected, *_ = policy.sample(raw, legal, collect_stats=False)
            observations.append(raw)
            masks.append(legal)
            actions.append(selected.astype(np.int64))
            env.step(selected.reshape(2, 2))
            if any(env.vector.info(lane).get('fault', 0) for lane in range(2)):
                raise RuntimeError('Native PPO probe simulation fault')
            if env.dones.any():
                break
    raw, legal, selected = (np.stack(x, axis=1) for x in (observations, masks, actions))
    return [dict(raw_obs=obs, legal_masks=mask, actions=action, actor_keep=mask.sum(-1) > 1)
            for obs, mask, action in zip(raw, legal, selected) if (mask.sum(-1) > 1).any()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--decks', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expect-update', type=int, required=True)
    parser.add_argument('--sequences', type=int, default=48)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--max-kl', type=float, default=.01)
    args = parser.parse_args()
    if args.sequences < 1:
        parser.error('Choose at least one BC sequence')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists() or (args.out / 'actor_export.pt').exists():
        raise FileExistsError('Candidate output has already been produced')
    torch.manual_seed(20261120)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        if free < 4 * 1024**3:
            raise RuntimeError('Insufficient GPU room for the isolated candidate process')
        torch.cuda.set_per_process_memory_fraction(2.5 * 1024**3 / total, device)
    pace = time.perf_counter()
    snapshot = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if snapshot.get('updates') != args.expect_update or 'critic_optimizer' in snapshot:
        raise ValueError('Use the requested immutable actor-only checkpoint')
    actor, inference, _ = load_policy_inference(args.checkpoint, args.device, compute_value=False)
    bc = ExpertBC(args.dataset, actor, experimental=True)
    if bc.validation_enabled or bc.focus_card != 'ZapMachine':
        raise ValueError('Use the full-data Sparky snapshot')
    print(json.dumps({'stage': 'loaded', 'source_update': args.expect_update, 'train_seats': len(bc.train)}), flush=True)
    probes = collect_probes(actor, inference, args.decks)
    print(json.dumps({'stage': 'probes_ready', 'ppo_probe_choice_frames': sum(int(p['actor_keep'].sum()) for p in probes)}), flush=True)
    result = bc.run_phase(actor, inference.adapter, probes, update=args.expect_update,
                          device=args.device, precision=inference.precision,
                          sequences=args.sequences, lr=args.learning_rate, max_kl=args.max_kl)
    result.update(source_update=args.expect_update, device=args.device, elapsed_seconds=time.perf_counter() - pace)
    if result.get('accepted'):
        weights = {name: value.detach().cpu() if torch.is_tensor(value) else value
                   for name, value in actor.state_dict().items()}
        model_key = 'model' if 'model' in snapshot else 'actor'
        snapshot[model_key] = weights
        snapshot['sparky_bc_candidate'] = dict(source_update=args.expect_update,
            snapshot_id=bc.snapshot_id, trained_sequences=result['trained_sequences'])
        temporary = args.out / 'actor_export.pt.partial'
        torch.save(snapshot, temporary)
        temporary.replace(args.out / 'actor_export.pt')
    report = args.out / 'report.json'
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
