"""Replay all finished BC-suite games natively and compare actual executed card costs."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from cr_rl.evaluation.strategy_audit import summarize_strategy

root=Path(__file__).resolve().parents[1]
base=root/'artifacts/local_dual_v100_checks/bc_suite_179'
reports=sorted(base.glob('seed*/chunk*/evaluation.json'))
assert len(reports)==8, 'Run only after the fixed suite has completed'
items=[]
for report in reports:
    audit=report.parent/'execution_audit.json'
    subprocess.run([sys.executable,str(root/'scripts/audit_policy_replays.py'),
                    '--evaluation',str(report),'--out',str(audit)],check=True)
    items.append((json.loads(report.read_text()),json.loads(audit.read_text())))
strategy=summarize_strategy(items)
(base/'strategy.json').write_text(json.dumps(strategy,indent=2)+'\n')
# Resample physical deck-pair clusters, keeping both seats AND both seed groups together.
pairs={}
for evaluation,_ in items:
    for m in evaluation['matchups']:
        assert m['valid']
        pairs.setdefault(m['pair'],[]).append((m['reward']+1)/2)
assert len(pairs)==32 and all(len(scores)==4 for scores in pairs.values())
scores=np.array([np.mean(v) for v in pairs.values()])
rng=np.random.default_rng(20261109)
means=rng.choice(scores,(20000,len(scores)),replace=True).mean(1)
summary=json.loads((base/'summary.json').read_text())
summary['deck_pair_cluster_bootstrap_95_interval']=np.percentile(means,[2.5,97.5]).tolist()
summary['native_outcome_parity_games']=sum(g['native_outcome_parity'] for _,a in items for g in a['games'])
summary['executed_strategy']={side:{k:v for k,v in strategy[side].items() if k!='by_deck'} for side in ['learner','opponent']}
(base/'final_report.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['matches','deck_names','limitations']},indent=2))
