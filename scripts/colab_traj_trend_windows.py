import json,math
from pathlib import Path
root=Path('/content/crrl_training/runs/longrun');latest=json.loads((root/'summary.json').read_text())['update']
p=root/'console.log'
with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000000));lines=f.read().splitlines()
last={}
for line in lines:
 try:d=json.loads(line)
 except (ValueError,UnicodeDecodeError):continue
 if d.get('phase')=='state_ppo' and 941<=d['update']<=latest:last[d['update']]=d
rows=[last[k] for k in sorted(last)]
def wilson(w,n):
 if not n:return None
 z=1.96;p=w/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [round(100*(c-h),1),round(100*(c+h),1)]
def window(rr):
 if not rr:return None
 roles={}
 for role in ('anchor','historical','current'):
  n=sum(d['training_win_rates'].get(role+'_games',0) for d in rr);w=sum(d['training_win_rates'].get(role+'_wins',0) for d in rr)
  roles[role]=dict(w=w,n=n,pct=round(100*w/n,2) if n else None,ci95=wilson(w,n))
 n=sum(d['completed_training_games'] for d in rr);w=sum(d['terminal_seat_wins'] for d in rr)
 req=sum(d['decision_coverage']['card_requests'] for d in rr);high=sum(d['decision_coverage']['high_cost_requests'] for d in rr)
 metrics={k:sum(d['metrics'].get(k,0) for d in rr)/len(rr) for k in ('value_loss','explained_variance','approx_kl','anchor_kl','card_entropy')}
 return dict(start=rr[0]['update'],end=rr[-1]['update'],games=n,win_pct=100*w/n,roles=roles,metrics=metrics,
  mean_cost=sum(d['decision_coverage']['requested_card_cost_mean']*d['decision_coverage']['card_requests'] for d in rr)/req if req else None,high_cost_pct=100*high/req if req else None,
  balanced_legacy=rr[-1]['training_win_rates'].get('deck_balanced_vs_bc_win_rate'),anchor_pool_games=rr[-1]['training_win_rates'].get('anchor_pool_games'),
  awbc=[d['update'] for d in rr if d.get('awbc_playbuffer',{}).get('ran')],bc=[d['update'] for d in rr if d.get('expert_bc',{}).get('ran')])
print('LATEST',latest)
for start in range(941,latest+1,10):
 print('BLOCK',json.dumps(window([d for d in rows if start<=d['update']<start+10])))
print('FIRST20',json.dumps(window(rows[:20])))
print('RECENT20',json.dumps(window(rows[-20:])))
print('RECENT5',json.dumps(window(rows[-5:])))
print('LATEST_WIN_RATES',json.dumps(rows[-1]['training_win_rates']))
