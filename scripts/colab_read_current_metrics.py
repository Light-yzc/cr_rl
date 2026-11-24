"""Bounded read-only summary of current training, auxiliary phases and heldout results."""
import json
from pathlib import Path

root=Path('/content/crrl_training/runs/longrun')
for name in ('summary.json','awbc_latest_report.json','expert_bc_latest_report.json'):
    path=root/name
    if not path.is_file(): continue
    data=json.loads(path.read_text())
    if name=='summary.json':
        keys=('update','completed_training_games_total','learner_samples_total','metrics',
              'decision_coverage','training_win_rates','learning_rates','army_reward_distribution')
    else:
        keys=('accepted','actor_attempted','reason','best_epoch','baseline','validation',
              'ppo_forward_kl','value_report','phase_count')
    print(name,json.dumps({k:data.get(k) for k in keys},ensure_ascii=False)[:16000])
path=root/'console.log'
if path.exists():
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-1500000))
        tail=stream.read().decode(errors='replace').splitlines()
    summaries=[]
    for line in tail:
        try: data=json.loads(line)
        except ValueError: continue
        if data.get('phase')=='state_ppo':
            m=data.get('metrics',{})
            summaries.append(dict(update=data['update'],sps=data.get('samples_per_second'),
                value_loss=m.get('value_loss'),explained_variance=m.get('explained_variance'),
                card_entropy=m.get('card_entropy'),tile_entropy=m.get('tile_entropy'),
                kl=m.get('approx_kl'),critic_overflow=data.get('critic_overflow_steps'),
                actor_overflow=data.get('actor_overflow_steps')))
    print('RECENT_UPDATES',json.dumps(summaries[-8:]))
for path in sorted((root/'validation').glob('update_*/summary.json'))[-1:]:
    print('LATEST_VALIDATION',str(path),path.read_text()[:7000])
