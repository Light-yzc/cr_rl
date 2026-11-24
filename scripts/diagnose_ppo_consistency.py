"""Replay frozen full-game observations to compare streaming and PPO arithmetic.

This tool never updates weights. Replay bundles contain ``segments`` as emitted
by PersistentStateSelfplay.collect(), and must match the checkpoint version.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.state_ppo import _adapt_segment_batch, prepare_state_rollout_targets
from cr_rl.training.ppo import factorized_entropy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--replay', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mode', default='baseline',
                   choices=['baseline', 'fp32', 'unified', 'encoder', 'temporal', 'heads', 'residual'])
    p.add_argument('--gradients', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model = state_model_from_checkpoint(payload, 'cuda:0')
    if args.mode == 'unified':
        model.ppo_actor_fp32 = True
        torch.backends.mha.set_fastpath_enabled(False)
    version = payload['updates']
    del payload
    replay = torch.load(args.replay, map_location='cpu', weights_only=False)
    assert replay['version'] == version, 'Replay and checkpoint must share weights'
    # The bundle carries the adapter schema to avoid simulator startup.
    adapter = CrForgeObservationAdapter(replay['schema'], replay['vocabulary_size'], entity_bucket=16)
    segments = replay['segments'][:4]
    length = max(len(s['raw_obs']) for s in segments)
    device = torch.device('cuda:0')
    h, e, em, tm, hands, _ = _adapt_segment_batch(segments, adapter, start=0, end=length, device=device)
    legal = torch.zeros(len(segments), length, model.cfg.action_count, dtype=torch.bool, device=device)
    legal[..., 0] = True
    actions = torch.zeros(len(segments), length, dtype=torch.long, device=device)
    old = torch.zeros_like(actions, dtype=torch.float32)
    mask = torch.zeros_like(actions, dtype=torch.bool)
    for i, s in enumerate(segments):
        n = len(s['raw_obs'])
        legal[i, :n] = torch.as_tensor(s['legal_masks'], device=device)
        actions[i, :n] = torch.as_tensor(s['actions'], device=device)
        old[i, :n] = torch.as_tensor(s['old_log_probs'], device=device)
        mask[i, :n] = torch.as_tensor(s['valid'], device=device) & legal[i, :n, 1:].any(-1)

    def fp32_call(function):
        def wrapped(*a, **kw):
            a = tuple(x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in a)
            with torch.autocast('cuda', enabled=False):
                return function(*a, **kw)
        return wrapped

    if args.mode == 'encoder':
        model.encode_frames = fp32_call(model.encode_frames)
    if args.mode == 'temporal':
        model.to_temporal.forward = fp32_call(model.to_temporal.forward)
        model.temporal.prefill = fp32_call(model.temporal.prefill)
    if args.mode == 'heads':
        for module in (model.slot_actor, model.wait_actor, model.wait_hand_gate, model.ability_actor, model.learned_placement):
            module.forward = fp32_call(module.forward)
    if args.mode == 'residual':
        for block in model.temporal.blocks:
            original = block.forward
            def wrapped(x, *a, _original=original, **kw):
                return _original(x.float(), *a, **kw)
            block.forward = wrapped

    report = dict(version=version, mode=args.mode, frames=int(tm.sum()), lengths=[len(s['raw_obs']) for s in segments], results={})
    chosen = {}
    for name in ('steps', 'eval_sequence', 'train_sequence'):
        model.train(name == 'train_sequence')
        state = model.initial_state(len(segments), device=device,
            dtype=torch.float32 if args.mode in ('fp32', 'unified', 'temporal') else torch.float16)
        began = time.perf_counter()
        gate_outputs = []
        hook = model.wait_hand_gate.register_forward_hook(lambda m, a, out: gate_outputs.append(out.detach().flatten()))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=args.mode != 'fp32'):
            if name == 'steps':
                outputs = []
                for t in range(length):
                    lp, _, state = model.forward_state(h[:, t:t+1], e[:, t:t+1], em[:, t:t+1], legal[:, t:t+1],
                        hand_card_ids=hands[:, t:t+1], time_mask=tm[:, t:t+1], state=state, compute_value=False)
                    outputs.append(lp.gather(-1, actions[:, t:t+1, None]).squeeze(-1))
                chosen[name] = torch.cat(outputs, 1)
            else:
                lp, _, _ = model.forward_state(h, e, em, legal, hand_card_ids=hands, time_mask=tm,
                    state=state, compute_value=False)
                chosen[name] = lp.gather(-1, actions[..., None]).squeeze(-1)
                card_h, weighted_tile_h = factorized_entropy(lp[mask])
                play_mass = lp[mask][:, 1:2305].exp().sum(-1)
                report['entropy'] = dict(
                    card=float(card_h.mean()), weighted_tile=float(weighted_tile_h.mean()),
                    play_mass=float(play_mass.mean()),
                    conditional_tile=float(weighted_tile_h.sum()/play_mass.sum().clamp_min(1e-9)),
                )
        hook.remove()
        torch.cuda.synchronize()
        vals = torch.cat(gate_outputs).float().abs()
        error = (chosen[name]-old).abs()[mask]
        report['results'][name] = dict(seconds=time.perf_counter()-began,
            vs_rollout_max=float(error.max()), vs_rollout_mean=float(error.mean()),
            gate_abs_mean=float(vals.mean()), gate_abs_max=float(vals.max()))
        print(json.dumps({name: report['results'][name]}), flush=True)
    for name in ('eval_sequence', 'train_sequence'):
        error = (chosen[name]-chosen['steps']).abs()[mask]
        report['results'][name].update(vs_steps_max=float(error.max()), vs_steps_mean=float(error.mean()),
            vs_steps_p99=float(error.quantile(.99)))
    if args.gradients:
        prepare_state_rollout_targets(replay['segments'], gamma=.9995, gae_lambda=.985,
            critic_gae_lambda=.992, choice_only=True)
        advantages = torch.zeros_like(old)
        for i, s in enumerate(segments):
            advantages[i, :len(s['raw_obs'])] = torch.as_tensor(s['advantages'], device=device)
        model.train()
        model.gradient_checkpointing = True
        model.temporal_checkpoint_mode = 'block'
        model.zero_grad(set_to_none=True)
        state = model.initial_state(len(segments), device=device,
            dtype=torch.float32 if args.mode in ('fp32', 'unified', 'temporal') else torch.float16)
        with torch.autocast('cuda', dtype=torch.float16, enabled=args.mode != 'fp32'):
            lp, _, _ = model.forward_state(h, e, em, legal, hand_card_ids=hands, time_mask=tm,
                state=state, compute_value=False)
        selected = lp[mask]
        ratios = (selected.gather(-1, actions[mask][:, None]).squeeze(-1) - old[mask]).exp()
        adv = advantages[mask]
        pg = -torch.minimum(ratios*adv, ratios.clamp(.8, 1.2)*adv).mean()
        ch, th = factorized_entropy(selected)
        params = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
        vectors = {}
        stats = {}
        for label, loss in (('ppo', pg), ('card_entropy', -.003*ch.mean()), ('tile_entropy', -.0005*th.mean())):
            grads = torch.autograd.grad(loss, [v for _, v in params], retain_graph=True, allow_unused=True)
            vector = torch.cat([(g.detach().float().flatten() if g is not None else torch.zeros(v.numel(), device=device))
                                for (_, v), g in zip(params, grads)])
            vectors[label] = vector.cpu()
            stats[label] = dict(loss=float(loss.detach()), grad_norm=float(vector.norm()),
                gate_grad_norm=float(torch.stack([g.float().square().sum() for (n, _), g in zip(params, grads)
                    if n.startswith('wait_hand_gate.') and g is not None]).sum().sqrt()))
            del vector, grads
        for label in ('card_entropy', 'tile_entropy'):
            stats[label]['cosine_with_ppo'] = float(torch.nn.functional.cosine_similarity(vectors[label], vectors['ppo'], dim=0))
            stats[label]['norm_ratio_to_ppo'] = stats[label]['grad_norm']/max(stats['ppo']['grad_norm'], 1e-12)
        report['gradients'] = stats
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
