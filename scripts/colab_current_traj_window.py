import json,math
from pathlib import Path
root=Path('/content/crrl_training/runs/longrun'); latest=json.loads((root/'summary.json').read_text())['update']
p=root/'console.log'
with p.open('rb') as f:
 f.seek(max(0,p.stat().st_size-5000000));lines=f.read().splitlines()
last={}
for line in lines:
 try:d=json.loads(line)
 except (ValueError,UnicodeDecodeError):continue
 if d.get('phase')=='state_ppo' and 941<=d['update']<=latest:last[d['update']]=d
rows=[last[k] for k in sorted(last)]
def stats(rr):
 roles={}
 for role in ('anchor','historical','current'):
  games=sum(d['training_win_rates'].get(role+'_games',0) for d in rr);wins=sum(d['training_win_rates'].get(role+'_wins',0) for d in rr)
  roles[role]=dict(games=games,wins=wins,win_rate=wins/games if games else None)
 games=sum(d.get('completed_training_games',0) for d in rr)
 wins=sum(d.get('terminal_seat_wins',0) for d in rr)
 samples=sum(d['valid_learner_samples'] for d in rr);seconds=sum(d['seconds'] for d in rr)
 requests=sum(d['decision_coverage'].get('card_requests',0) for d in rr);high=sum(d['decision_coverage'].get('high_cost_requests',0) for d in rr)
 return dict(updates=[d['update'] for d in rr],games=games,wins=wins,win_rate=wins/games if games else None,roles=roles,
   learner_samples=samples,weighted_sps=samples/seconds if seconds else None,actor_overflows=sum(d['actor_overflow_steps'] for d in rr),critic_overflows=sum(d['critic_overflow_steps'] for d in rr),faulted_samples=sum(d['faulted_samples'] for d in rr),
   card_requests=requests,high_cost_requests=high,high_cost_request_fraction=high/requests if requests else None,
   mean_card_cost=sum(d['decision_coverage'].get('requested_card_cost_mean',0)*d['decision_coverage'].get('card_requests',0) for d in rr)/requests if requests else None,
   play_fraction=sum(d['decision_coverage']['play_fraction']*d['valid_learner_samples'] for d in rr)/samples if samples else None)
print('SINCE_RESTART',json.dumps(stats(rows)))
print('RECENT5',json.dumps(stats(rows[-5:])))
print('WINDOWS',json.dumps([dict(first=rr[0]['update'],last=rr[-1]['update'],stats=stats(rr),value_loss=sum(d['metrics']['value_loss'] for d in rr)/len(rr),ev=sum(d['metrics']['explained_variance'] for d in rr)/len(rr),kl=sum(d['metrics']['approx_kl'] for d in rr)/len(rr)) for rr in (rows[:5],rows[-5:]) if rr]))
p=root/'awbc_latest_report.json'
if p.exists():
 a=json.loads(p.read_text());print('AWBC',{k:a.get(k) for k in ('accepted','best_epoch','batch_games','actor_train_indices','actor_holdout_indices','baseline','epochs','holdout_summary')})
print('SAVED_TRAJ',dict(rollout_npz=len(list((root/'rollouts').glob('*.npz')))))
