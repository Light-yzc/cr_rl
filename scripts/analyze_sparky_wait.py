"""Read completed Nulls recordings and diagnose Sparky affordability and selection."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean

from cr_rl.client.schema_v4 import SchemaV4 as S


def analyze(run):
    events = [json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf8').splitlines()]
    decisions = {e['decision_id']:e for e in events if e['event']=='decision'}
    confirmed = {e['decision_id'] for e in events if e['event']=='receipt' and e['receipt']['status']=='confirmed'}
    groups = {'near': [], 'affordable': []}
    windows, current = [], []
    def close(reason):
        if current:
            windows.append(dict(end=reason,tick=current[0]['tick'],frames=len(current),
                                mean_wait=mean(x['wait'] for x in current),
                                wait_majority_frames=sum(x['wait']>.5 for x in current)))
            current.clear()
    count = Counter()
    for path in sorted((run/'diagnostics').glob('frames_*.jsonl.gz')):
        with gzip.open(path,'rt',encoding='utf8') as f:
            for line in f:
                r=json.loads(line); d=decisions[r['decision_id']]; o=r['obs']; p=r['probabilities']
                count['frames']+=1
                hand=d['hand']; slot=hand.index('ZapMachine') if 'ZapMachine' in hand else None
                available=(o[S.H_ELIXIR]-max(0,o[S.H_ELIXIR_RESERVED]))/S.ELIXIR_SCALE
                threat=any(o[a+S.E_SIDE]==1 and o[a+S.E_KIND]==S.KIND_CHARACTER and
                           o[a+S.E_HP]>0 and 0<=o[a+S.E_Y]<=18000
                           for a in range(S.HEADER_INTS,S.HEADER_INTS+o[S.H_ENTITY_COUNT]*S.ENTITY_INTS,S.ENTITY_INTS))
                if slot is None:
                    close('card_left_hand'); continue
                count['sparky_in_hand']+=1
                if available>=6: count['sparky_in_hand_ge6']+=1
                mass=[sum(p[1+k*S.TILES:1+(k+1)*S.TILES]) for k in range(4)]
                proposal=r.get('action_selection',{}).get('proposed_action',0)
                actual=d['action'] if r['decision_id'] in confirmed else 0
                actual_slot=(actual-1)//S.TILES if 1<=actual<S.FIRST_ABILITY else None
                if actual_slot==slot: count['sparky_confirmed']+=1
                free=d['reason'] in ('action_ready','policy_wait')
                if available>=6:
                    close('reached_six')
                elif threat:
                    close('visible_threat')
                elif available<5:
                    close('below_five_or_other')
                if not free: continue
                row=dict(tick=r['tick'],wait=p[0],sparky=mass[slot],
                         sparky_top_card=mass[slot]>=max(mass),
                         sparky_top_option=mass[slot]>=max(p[0],*mass,*p[S.FIRST_ABILITY:]),
                         sparky_sampled=1+slot*S.TILES<=proposal<1+(slot+1)*S.TILES,
                         confirmed_card=hand[actual_slot] if actual_slot is not None else None,
                         threat=threat,available=available)
                if available>=6 and d['legal_play_tiles'][slot]>0:
                    groups['affordable'].append(row)
                if 5<=available<6 and not threat:
                    groups['near'].append(row); current.append(row)
                    if actual_slot is not None and 0<o[S.H_HAND_COST+actual_slot]<6:
                        close('confirmed_cheaper_card')
    close('match_ended')
    return dict(match=run.name,counts=count,windows=windows,**groups)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('batch',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    ledger=json.loads((a.batch/'batch.json').read_text(encoding='utf8'))
    results=[analyze(a.batch/name) for name,info in ledger['completed'].items() if info['slot']==2]
    a.out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print('Matches:',len(results),'counts:',sum((Counter(r['counts']) for r in results),Counter()))
    ws=[w for r in results for w in r['windows']]
    print('Window endings:',Counter(w['end'] for w in ws))
    for group in ('near','affordable'):
        rows=[x for r in results for x in r[group]]
        if not rows: continue
        sent=[x for x in rows if x['confirmed_card']]
        print(group,len(rows),'mean_wait',mean(x['wait'] for x in rows),'wait_majority',sum(x['wait']>.5 for x in rows),
              'mean_sparky',mean(x['sparky'] for x in rows),'sparky_top_card',sum(x['sparky_top_card'] for x in rows),
              'sparky_top_option',sum(x['sparky_top_option'] for x in rows),'sparky_sampled',sum(x['sparky_sampled'] for x in rows),
              'confirmed',Counter(x['confirmed_card'] for x in sent),'wait_majority_on_confirmed',sum(x['wait']>.5 for x in sent))
        if sent: print('Confirmed rows:',[(x['tick'],x['confirmed_card'],round(x['wait'],3),round(x['sparky'],3)) for x in sent][:35])
    for r in results:
        print(r['match'],r['counts'],len(r['near']),len(r['affordable']))
