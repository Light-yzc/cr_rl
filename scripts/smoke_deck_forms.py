"""Bounded legal-action simulator acceptance for a deck/form pool; no training."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from cr_rl.envs.crforge import CrForgeVector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decks', type=Path, default=Path('configs/decks_augmented.yaml'))
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--out', type=Path, default=Path('artifacts/deck_forms/smoke.json'))
    args = p.parse_args()
    pool = yaml.safe_load(args.decks.read_text(encoding='utf-8'))
    env = CrForgeVector(count=1, threads=1, decision_ticks=8)
    rng = np.random.default_rng(20261115)
    results = []
    try:
        for i, (name, cards) in enumerate(pool['decks'].items()):
            flags = pool.get('slots', {}).get(name, [0]*8)
            env.vector.reset([10000+i], cards, cards, flags, flags)
            fault = None
            steps = 0
            for _ in range(args.steps):
                if env.vector.dones[0]:
                    break
                action = np.zeros((1, 2), dtype=np.int32)
                for seat in range(2):
                    action[0, seat] = rng.choice(np.flatnonzero(env.vector.masks[0, seat]))
                env.vector.step(action)
                steps += 1
            info = env.vector.info(0)
            # Fault info is explicit; natural episode endings are not faults.
            fault = env.vector.fault(0)
            results.append({'name': name, 'steps': steps, 'fault': str(fault) if fault else None,
                            'info': info})
            if (i+1) % 25 == 0:
                print(f'{i+1}/{len(pool["decks"])} checked', flush=True)
    finally:
        env.vector.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    failures = [x for x in results if x['fault']]
    print(json.dumps({'checked': len(results), 'failures': failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
