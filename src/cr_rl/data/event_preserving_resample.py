"""Event-preserving offline BC timing augmentation; never a native rollout.

Only WAIT rows may be merged. Each original non-WAIT row is copied exactly
once, including flagged/illegal labels. Logical time is a delayed fixed grid;
the simulator observation clock remains the original pre-action state clock.
"""
from __future__ import annotations

import hashlib
import numpy as np


FIELDS = ("observations", "masks", "actions", "ticks", "seats", "flags", "replay_ids")


def _identity_hash(arrays, rows, source_rows, source_ticks, excluded_observation_columns=()):
    digest=hashlib.sha256()
    keep=np.ones(arrays['observations'].shape[1],dtype=bool)
    keep[list(excluded_observation_columns)]=False
    for row,source_row,source_tick in zip(rows,source_rows,source_ticks):
        digest.update(np.asarray([source_row,source_tick,int(arrays['seats'][row]),
            int(arrays['actions'][row]),int(arrays['flags'][row])],dtype='<i8').tobytes())
        observation=np.ascontiguousarray(arrays['observations'][row,keep])
        mask=np.ascontiguousarray(arrays['masks'][row])
        digest.update(observation.dtype.str.encode());digest.update(observation.tobytes())
        digest.update(mask.dtype.str.encode());digest.update(mask.tobytes())
    return digest.hexdigest()


def resample_game(arrays, *, decision_ticks=8, decision_header_columns=(),source_row_ids=None):
    """Return one game's delayed logical sequence and exact action audit.

    A WAIT slot takes the latest original WAIT not later than logical time,
    after the previously emitted source row, and strictly before the next
    not-yet-emitted source action. Repeated safe WAIT frames are permitted.
    If no such row exists, fail rather than relabel an action or use future
    state. This protects unusual inputs instead of silently fabricating data.
    """
    if decision_ticks<1:
        raise ValueError('decision_ticks must be positive')
    count=len(arrays['actions'])
    if not count or any(len(arrays[name])!=count for name in FIELDS):
        raise ValueError('Nonempty equally sized BC fields are required')
    if len(np.unique(arrays['replay_ids']))!=1:
        raise ValueError('Resample one complete game per shard, not a full dataset array')
    labels=np.asarray(arrays['actions']);ticks=np.asarray(arrays['ticks'])
    row_ids=(np.arange(count,dtype=np.int64) if source_row_ids is None else
        np.asarray(source_row_ids,dtype=np.int64))
    if row_ids.shape!=(count,) or len(np.unique(row_ids))!=count:
        raise ValueError('Original source row identities must be unique')
    seats=np.asarray(arrays['seats'])
    if np.any((labels<0)|(labels>=arrays['masks'].shape[1])):
        raise ValueError('Action outside the source mask codec')
    if set(map(int,np.unique(seats)))!={0,1}:
        raise ValueError('A complete game must contain both original seats')
    planned={};source_by_seat={};schedule_end=0
    for seat in (0,1):
        rows=np.flatnonzero(seats==seat)
        if ticks[rows[0]]!=0 or np.any(np.diff(ticks[rows])<=0):
            raise ValueError('Each source seat must start at tick zero and increase strictly')
        action_rows=rows[labels[rows]!=0]
        previous=-decision_ticks;scheduled=[]
        for row in action_rows:
            logical=max((int(ticks[row])+decision_ticks-1)//decision_ticks*decision_ticks,
                previous+decision_ticks)
            scheduled.append((logical,int(row)));previous=logical
        planned[seat]=scheduled;source_by_seat[seat]=rows
        schedule_end=max(schedule_end,(int(ticks[rows[-1]])+decision_ticks-1)//decision_ticks*decision_ticks,
            previous if scheduled else 0)
    selections=[];logical_ticks=[];output_seats=[];fallback_repeats=0
    for seat in (0,1):
        rows=source_by_seat[seat]
        waiting=rows[labels[rows]==0]
        pending=planned[seat];next_action=0;previous_source=-1
        for logical in range(0,schedule_end+1,decision_ticks):
            if next_action<len(pending) and pending[next_action][0]==logical:
                row=pending[next_action][1];next_action+=1
            else:
                next_source=pending[next_action][1] if next_action<len(pending) else count
                candidates=waiting[(ticks[waiting]<=logical)&(waiting>=previous_source)&(waiting<next_source)]
                if not len(candidates):
                    raise ValueError(f'No causal original WAIT for seat={seat}, logical_tick={logical}; '
                        'refusing future observation or action-to-WAIT relabelling')
                row=int(candidates[-1]);fallback_repeats+=int(row==previous_source)
            if row<previous_source or int(ticks[row])>logical:
                raise ValueError('Source observations would move backwards or into the future')
            previous_source=row
            selections.append(row);logical_ticks.append(logical);output_seats.append(seat)
        if next_action!=len(pending):raise AssertionError('An original action was not scheduled')
    order=np.lexsort((output_seats,logical_ticks))
    chosen=np.asarray(selections,dtype=np.int64)[order]
    logical=np.asarray(logical_ticks,dtype=arrays['ticks'].dtype)[order]
    output={name:np.array(arrays[name][chosen],copy=True) for name in FIELDS}
    output['source_ticks']=np.asarray(arrays['ticks'][chosen]).copy()
    output['source_rows']=row_ids[chosen].copy()
    output['ticks']=logical
    for column in decision_header_columns:
        output['observations'][:,column]=decision_ticks
    audits=[]
    for seat in (0,1):
        before=source_by_seat[seat][labels[source_by_seat[seat]]!=0]
        after=np.flatnonzero((output['seats']==seat)&(output['actions']!=0))
        if not np.array_equal(output['source_rows'][after],row_ids[before]):
            raise AssertionError('Non-WAIT source row identities or per-seat order changed')
        original_hash=_identity_hash(arrays,before,row_ids[before],ticks[before],decision_header_columns)
        result_hash=_identity_hash(output,after,output['source_rows'][after],output['source_ticks'][after],
            decision_header_columns)
        if original_hash!=result_hash:raise AssertionError('Non-WAIT pre-state/mask/action/flags identity changed')
        audits.append({'seat':seat,'nonwait':len(before),'source_identity_sha256':original_hash,
            'output_identity_sha256':result_hash,'identity_equal':True})
    before_legal=np.asarray(arrays['masks'][np.arange(count),labels],dtype=bool)
    before_valid=(np.asarray(arrays['flags'])==0)&before_legal
    after_labels=output['actions'];after_legal=np.asarray(
        output['masks'][np.arange(len(after_labels)),after_labels],dtype=bool)
    after_valid=(output['flags']==0)&after_legal
    totals=lambda y,valid,flags,legal:{'rows':len(y),'nonwait':int((y!=0).sum()),
        'card':int(((y>0)&(y<2305)).sum()),'ability':int((y>=2305).sum()),
        'valid_nonwait':int(((y!=0)&valid).sum()),'valid_card':int(((y>0)&(y<2305)&valid).sum()),
        'valid_ability':int(((y>=2305)&valid).sum()),'wait':int((y==0).sum()),
        'valid_rows':int(valid.sum()),'flagged_rows':int((flags!=0).sum()),
        'flagged_nonwait':int(((flags!=0)&(y!=0)).sum()),'illegal_label_rows':int((~legal).sum())}
    before_totals=totals(labels,before_valid,arrays['flags'],before_legal)
    after_totals=totals(after_labels,after_valid,output['flags'],after_legal)
    for key in ('nonwait','card','ability','valid_nonwait','valid_card','valid_ability'):
        if before_totals[key]!=after_totals[key]:raise AssertionError(f'Source label count changed: {key}')
    action_output=np.flatnonzero(after_labels!=0)
    delay=output['ticks'][action_output].astype(np.int64)-output['source_ticks'][action_output].astype(np.int64)
    if np.any(delay<0):raise AssertionError('An action was moved earlier')
    return output,{'before':before_totals,'after':after_totals,'seats':audits,
        'logical_decision_ticks':decision_ticks,'source_clock_preserved':True,
        'all_nonwait_structurally_preserved':True,'only_observation_columns_modified':list(decision_header_columns),
        'repeated_safe_wait_frames':fallback_repeats,'source_rows_monotonic_per_seat':True,
        'early_actions':0,'action_delay_ticks':{'min':int(delay.min()) if len(delay) else None,
            'p50':float(np.quantile(delay,.5)) if len(delay) else None,
            'p95':float(np.quantile(delay,.95)) if len(delay) else None,
            'max':int(delay.max()) if len(delay) else None},
        'semantic_kind':'offline_delayed_bc_supervision','native_trajectory':False,
        'ppo_rollouts_allowed':False,'native_parity_claim_allowed':False}
