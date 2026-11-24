"""Recover private-server outcomes from ended status, with optional UI evidence.

No winner is inferred from a missing entity alone. The already-adapted terminal
score is used only with ended clock, valid seat and previously known tower HP
scales. This is explicitly recorded as status inference, not a native winner.
"""
import json
from pathlib import Path

import numpy as np

from .rl_recording import OUTCOMES, set_result, sha256_file, atomic_json
from .schema_v4 import SchemaV4 as S


def screen_score_bounds(path):
    """Independent fixed-UI crown evidence; ambiguous animation stays unknown.

    Calibrated for the existing Chinese 900x1600 result layout. This reads only
    the post-match scoreboard; it is not a visual gameplay observation path.
    """
    from PIL import Image
    boxes=((160,210,320,360),(370,180,530,330),(580,210,740,360),
           (160,680,320,840),(370,650,530,810),(580,680,740,840))
    counts=[]
    with Image.open(path) as image:
        if image.size!=(900,1600):raise ValueError('Uncalibrated result screenshot size')
        image=image.convert('RGB')
        for box in boxes:
            a=np.asarray(image.crop(box),dtype=np.int32);r,g,b=a[...,0],a[...,1],a[...,2]
            counts.append(int(((r>190)&(g>120)&(b<140)&(r>g*1.05)&(g>b*1.3)).sum()))
    def bounds(values):
        return (sum(v>=5000 for v in values),sum(v>=400 for v in values))
    enemy,own=bounds(counts[:3]),bounds(counts[3:])
    outcome='win' if own[0]>enemy[1] else 'loss' if enemy[0]>own[1] else None
    return dict(own_bounds=own,enemy_bounds=enemy,gold_pixels=counts,outcome=outcome)


def infer_terminal_outcome(terminal, screenshot=None):
    snapshot=terminal.get('snapshot',{})
    if (terminal.get('clock_ended') is not True or snapshot.get('clock_ended') is not True
            or terminal.get('own_seat') not in (0,1)
            or snapshot.get('own_seat')!=terminal['own_seat']
            or snapshot.get('tick')!=terminal.get('tick')):
        raise ValueError('No consistent ended battle status')
    obs=terminal.get('obs',[])
    if len(obs)!=S.SEAT_INTS or obs[S.H_TICK]!=terminal['tick']:
        raise ValueError('Terminal observation/status mismatch')
    for base in (S.H_TOWERS,S.H_TOWERS+6):
        hp,maximum=obs[base:base+2]
        if not (isinstance(hp,int) and isinstance(maximum,int) and maximum>0 and 0<=hp<=maximum):
            raise ValueError('Unknown/inconsistent terminal king HP scale')
    own,enemy=obs[S.H_CROWNS],obs[S.H_CROWNS_ENEMY]
    if type(own) is not int or type(enemy) is not int or not 0<=own<=3 or not 0<=enemy<=3:
        raise ValueError('Unknown terminal crown score')
    evidence=dict(method='ended_status_crown_score_v1',clock_ended=True,tick=terminal['tick'],
                  own_seat=terminal['own_seat'],own_crowns=own,enemy_crowns=enemy)
    status='win' if own>enemy else 'loss' if own<enemy else None
    screen=None
    if screenshot is not None and Path(screenshot).is_file():
        screen=screen_score_bounds(screenshot)
        evidence.update(result_screen_sha256=sha256_file(screenshot),screen=screen)
        if screen['outcome'] and status and screen['outcome']!=status:
            raise ValueError('Terminal score and result screen disagree')
    outcome=status or (screen['outcome'] if screen else None)
    if outcome is None:
        raise ValueError('Equal/unknown score: do not invent a draw or a tiebreak winner')
    evidence['decisive_source']='terminal_score' if status else 'result_screen'
    return outcome,evidence


def finalize_result(run):
    run=Path(run)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf8'))
    if (run/'result.json').exists():
        result=json.loads((run/'result.json').read_text(encoding='utf8'))
        # Existing labels must retain their evidence and pass the normal importer.
        if (result.get('outcome')!=manifest.get('outcome') or not result.get('outcome_verified')
                or not manifest.get('outcome_verified')
                or result.get('reward')!=OUTCOMES.get(result.get('outcome'))
                or result.get('terminal_sha256')!=sha256_file(run/'terminal.json')):
            raise ValueError('Existing outcome integrity mismatch')
        if result.get('source')=='terminal_status_inference_v1':
            evidence=json.loads(result['evidence'])
            screenshot=run/'result_screen.png'
            if screenshot.is_file() and not evidence.get('result_screen_sha256'):
                terminal=json.loads((run/'terminal.json').read_text(encoding='utf8'))
                outcome,proof=infer_terminal_outcome(terminal,screenshot)
                if outcome!=result['outcome']:raise ValueError('Late screenshot conflicts with existing outcome')
                return set_result(run,outcome,source=result['source'],evidence=json.dumps(proof,sort_keys=True))
        return result
    terminal=json.loads((run/'terminal.json').read_text(encoding='utf8'))
    outcome,evidence=infer_terminal_outcome(terminal,run/'result_screen.png')
    return set_result(run,outcome,source='terminal_status_inference_v1',
                      evidence=json.dumps(evidence,sort_keys=True))


def seal_recording(run):
    """Publish small immutable metadata only after the recording writer has closed."""
    run=Path(run)
    result=finalize_result(run)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('collection_complete') or manifest.get('stop_reason')!='battle_end':
        raise ValueError('Cannot seal an incomplete battle')
    atomic_json(run/'collection_sealed.json',dict(format='crrl-sealed-session-v1',
                                                manifest=manifest,result=result))
