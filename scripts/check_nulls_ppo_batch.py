"""Validate live PPO episodes against the exact frozen Actor in a full checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from cr_rl.client.rl_recording import actor_fingerprint, as_ppo_segment, load_episode
from run_nulls_dryrun import load_policy_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    controller_lock = None
    if sys.platform == 'win32':
        import msvcrt
        lock_path = Path(__file__).resolve().parents[1]/'artifacts/nulls_live/policy-control.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        controller_lock = lock_path.open('a+b')
        if lock_path.stat().st_size == 0:
            controller_lock.write(b'0'); controller_lock.flush()
        controller_lock.seek(0)
        try:
            msvcrt.locking(controller_lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            parser.error('Live controller is active; run local replay only after collection exits')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if 'critic' not in checkpoint or 'critic_optimizer' not in checkpoint:
        parser.error('Specify the matching full training checkpoint with Critic and optimizer')
    fingerprint = actor_fingerprint(checkpoint)
    del checkpoint
    infer = None
    seen = set()
    for run in args.runs:
        episode = load_episode(run)
        if episode['session_id'] in seen or episode['behavior_version'] != fingerprint:
            raise ValueError(f'{run}: duplicate episode or behavior checkpoint mismatch')
        seen.add(episode['session_id'])
        precision = 'fp16' if str(episode['manifest']['arguments'].get('device','')).startswith('cuda') else 'fp32'
        if precision == 'fp16' and not args.device.startswith('cuda'):
            raise ValueError('Replay CUDA FP16 recordings on CUDA to check behavior probabilities')
        if infer is None:
            _, infer, _ = load_policy_inference(args.checkpoint, args.device, compute_value=False)
        infer.precision = precision
        infer.reset()
        segment = as_ppo_segment(episode)
        worst = 0.
        for obs, legal, action, old in zip(segment['raw_obs'], segment['legal_masks'],
                                           segment['actions'], segment['old_log_probs'], strict=True):
            logits, _ = infer.evaluate(obs[None], legal[None], compute_value=False)
            logp = float(logits.float().log_softmax(-1)[0, action])
            worst = max(worst, abs(logp - float(old))) if np.isfinite(logp) else float('inf')
        if worst > .02:
            raise ValueError(f'{run}: old_logp replay error {worst:.6f} exceeds .02')
        print(json.dumps({'session':run.name, 'frames':len(segment['actions']),
                          'outcome':episode['manifest']['outcome'], 'max_old_logp_error':worst}))


if __name__ == '__main__':
    main()
