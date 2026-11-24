"""Summarize local PPO windows without treating changing self-play as fixed validation."""
import argparse
import json
import math
import statistics
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/ppo_v3_v100'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/colab_upload_20261116/training_trend.json'))
    args = parser.parse_args()
    latest = {}
    raw_count = 0
    for line in (args.run / 'metrics.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('phase') == 'state_ppo':
            latest[row['update']] = row
            raw_count += 1
    windows = []
    for low, high in [(9,128), (129,256), (257,448), (449,640), (641,697),
                      (698,708), (709,800), (801,864), (865,929)]:
        rows = [r for u, r in sorted(latest.items()) if low <= u <= high]
        if not rows:
            continue
        def mean_metric(name, weight_key):
            samples = [(r.get('metrics', {}).get(name), r.get(weight_key, 0)) for r in rows]
            pairs = [(v, w) for v, w in samples if finite(v) and w > 0]
            return sum(v*w for v,w in pairs)/sum(w for _,w in pairs) if pairs else None
        def median(get):
            values = [get(r) for r in rows]
            values = [v for v in values if finite(v)]
            return statistics.median(values) if values else None
        decisions = sum(r.get('valid_learner_samples', 0) for r in rows)
        choices = sum(r.get('decision_coverage', {}).get('choice_rows', 0) for r in rows)
        plays = sum(r.get('decision_coverage', {}).get('card_requests', 0) for r in rows)
        expensive = sum(r.get('decision_coverage', {}).get('at_least_five_cost_requests', 0) for r in rows)
        cost = sum(r.get('decision_coverage', {}).get('requested_card_cost_mean', 0) *
                   r.get('decision_coverage', {}).get('card_requests', 0) for r in rows)
        roles = {}
        for role in ['anchor', 'historical', 'current']:
            games = sum(r.get('training_win_rates', {}).get(role+'_games', 0) for r in rows)
            wins = sum(r.get('training_win_rates', {}).get(role+'_wins', 0) for r in rows)
            roles[role] = {'games': games, 'wins': wins, 'win_rate': wins/games if games else None}
        windows.append({
            'updates': [low, high], 'records': len(rows), 'learner_decisions': decisions,
            'value_loss': mean_metric('value_loss', 'critic_training_samples'),
            'explained_variance_median': median(lambda r:r.get('metrics',{}).get('explained_variance')),
            'policy_loss': mean_metric('policy_loss', 'actor_training_samples'),
            'card_entropy': mean_metric('card_entropy', 'actor_training_samples'),
            'tile_entropy': mean_metric('tile_entropy', 'actor_training_samples'),
            'approx_kl_median': median(lambda r:r.get('metrics',{}).get('approx_kl')),
            'anchor_kl': mean_metric('anchor_kl', 'actor_training_samples'),
            'clip_fraction': mean_metric('clip_fraction', 'actor_training_samples'),
            'actor_lr_used_median': median(lambda r:r.get('actor_lr_schedule',{}).get('lr_used')),
            'actor_overflows': sum(r.get('actor_overflow_steps',0) for r in rows),
            'critic_overflows': sum(r.get('critic_overflow_steps',0) for r in rows),
            'kl_stopped_updates': sum(bool(r.get('actor_kl_stopped')) for r in rows),
            'rejected_groups': sum(r.get('rejected_groups',0) for r in rows),
            'initial_logp_error_max': max(r.get('initial_logp_max_error',0) for r in rows),
            'choice_fraction': choices/decisions if decisions else None,
            'card_requests': plays, 'high_cost_requests': expensive,
            'high_cost_fraction': expensive/plays if plays else None,
            'requested_cost_mean': cost/plays if plays else None,
            'seconds_per_update_median': median(lambda r:r.get('seconds')),
            'samples_per_second_median': median(lambda r:r.get('samples_per_second')),
            'training_roles': roles,
        })
    validations = {}
    for path in sorted((args.run/'validation').glob('*/summary.json')):
        content = json.loads(path.read_text())
        validations[path.parent.name] = {role:{k:v for k,v in data.items() if k in
            ('games','wins','win_rate','budget_passed','latency_violations','warning')}
            for role,data in content.items() if isinstance(data,dict)}
    result = {'raw_records': raw_count, 'unique_updates': len(latest),
              'deduplication': 'Last recorded metric per update retained; retries counted once.',
              'last_update': max(latest), 'windows': windows, 'validations': validations,
              'limitations': ['Metrics have no per-row wall timestamp: these are update windows, not a precise calendar-day slice.',
                              'Opponent mix, episode semantics, deck distribution and learning rate changed: windows are not controlled experiments.',
                              'Training role win rates are not fixed independent validation; absent role metrics are null, not zero.',
                              'Explained variance is against stored TD(lambda) targets, not an independent long-horizon value test.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
