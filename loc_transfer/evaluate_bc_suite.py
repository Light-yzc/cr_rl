"""Fixed-budget paired evaluation against the migrated BC anchor export."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = (ROOT / 'artifacts/local_dual_v100').resolve()
OUT = ROOT / 'artifacts/local_dual_v100_checks/bc_suite_179'
PIN = RUN / 'bc_suite_candidate_179.pt'
OPPONENT = ROOT / 'artifacts/v100_handoff/opponents/opponent_000000.pt'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def aggregate():
    reports = [json.loads(p.read_text()) for p in sorted(OUT.glob('seed*/chunk*/evaluation.json'))]
    matches = [dict(m, seed_group=i // 4) for i, r in enumerate(reports) for m in r['matchups']]
    valid = [m for m in matches if m['valid']]
    pairs = {}
    for m in valid:
        key = (m['seed_group'], m['pair'])
        pairs.setdefault(key, []).append((m['reward'] + 1) / 2)
    scores = np.array([np.mean(v) for v in pairs.values() if len(v) == 2])
    interval = None
    if len(scores):
        rng = np.random.default_rng(20261108)
        means = rng.choice(scores, (20000, len(scores)), replace=True).mean(1)
        interval = np.percentile(means, [2.5, 97.5]).tolist()
    result = dict(
        candidate_update=179, candidate_sha256=PROTOCOL['candidate_sha256'],
        reference='migrated training-pool BC anchor export; original BC5096 file not independently compared',
        reference_sha256=PROTOCOL['reference_sha256'],
        planned_games=128, completed_games=len(valid), invalid_games=len(matches)-len(valid),
        chunks_complete=len(reports), complete=len(reports)==8,
        wins=sum(m['reward'] > 0 for m in valid), losses=sum(m['reward'] < 0 for m in valid),
        draws=sum(m['reward'] == 0 for m in valid),
        mean_score=float(scores.mean()) if len(scores) else None,
        paired_bootstrap_95_interval=interval,
        by_seed=[dict(group=i, games=sum(m['seed_group']==i for m in valid),
                      wins=sum(m['seed_group']==i and m['reward']>0 for m in valid),
                      losses=sum(m['seed_group']==i and m['reward']<0 for m in valid)) for i in range(2)],
        by_seat=[dict(seat=i, games=sum(m['learner_seat']==i for m in valid),
                      wins=sum(m['learner_seat']==i and m['reward']>0 for m in valid)) for i in range(2)],
        deck_names=sorted({m['deck0'] for m in valid} | {m['deck1'] for m in valid}),
        limitations=['Fixed training deck pool with fresh evaluation seeds; not unseen-deck generalization',
                      'Anchor documented as BC in migration; raw BC5096 hash differs and weights not yet compared',
                      'Paired bootstrap clusters seat swaps; seed groups reuse deck pairings'],
        matches=matches, updated_unix=time.time(),
    )
    temporary=OUT/'summary.tmp.json'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(OUT/'summary.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ['matches','deck_names','limitations']}), flush=True)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    PROTOCOL=dict(candidate_update=179,candidate_sha256=digest(PIN),
        reference_sha256=digest(OPPONENT),planned_games=128,
        pairs_per_chunk=8, chunks_per_seed=4, seed_groups=2,
        pairing_seed=20261109, pair_start=20, device='physical GPU0 RTX4090',
        no_wait_bias=True, original_bc_download='Drive API returned project quota exceeded',
        predeclared_unix=time.time())
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    try:
        for group in range(2):
            for chunk in range(4):
                target=OUT/f'seed{group}'/f'chunk{chunk}'
                subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_policy.py'),
                    '--policy',str(PIN),'--opponent',str(OPPONENT),
                    '--decks',str(ROOT/'configs/decks_expansion12.yaml'), '--out',str(target),
                    '--pairs','8','--pair-start',str(20+chunk*8),
                    '--sampling-seed',str(20261109+group*10000),
                    '--simulator-seed-base',str(9200000+group*100000),
                    '--pairing-seed','20261109','--device','cuda:0',
                    '--frame-cache','--forced-wait-fastpath'],check=True)
                aggregate()
    finally:
        PIN.unlink(missing_ok=True)
