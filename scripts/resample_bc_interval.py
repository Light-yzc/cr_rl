"""Keep every original recorded BC action while merging WAITs onto a logical grid.

Offline delayed supervision only: copied source pre-states are not freshly
simulated 0.4s native trajectories and must not be used as PPO rollouts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

from cr_rl.data.event_preserving_resample import FIELDS,resample_game


def file_hash(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def checked_path(root,name):
    result=(root/name).resolve()
    if not result.is_relative_to(root.resolve()):raise ValueError('Source artifact escapes its dataset directory')
    return result


def source_games(path,scratch,replay_offset):
    """Stream a compressed whole dataset to disk, then yield only one game in RAM."""
    scratch.mkdir(parents=True,exist_ok=False)
    maps={}
    try:
        with zipfile.ZipFile(path) as archive:
            for field in FIELDS:
                member=field+'.npy'
                if member not in archive.namelist():raise ValueError(f'Missing BC array: {field}')
                target=scratch/member
                with archive.open(member) as source,target.open('wb') as output:
                    shutil.copyfileobj(source,output,4*1024**2)
                maps[field]=np.load(target,mmap_mode='r',allow_pickle=False)
        ids=np.unique(maps['replay_ids'])
        for local_id in ids:
            rows=np.flatnonzero(maps['replay_ids']==local_id)
            arrays={field:np.array(value[rows],copy=True) for field,value in maps.items()}
            yield int(local_id)+replay_offset,rows,arrays
    finally:
        for value in maps.values():value._mmap.close()
        resolved=scratch.resolve()
        if resolved.name.startswith('shard_') and resolved.parent.name=='.source_arrays':
            shutil.rmtree(resolved)
        else:raise ValueError('Refusing scratch cleanup outside owned source arrays')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--decision-ticks',type=int,default=8)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.decision_ticks<1:parser.error('Positive decision interval required')
    if args.out.exists() and any(args.out.iterdir()):parser.error('Use a fresh output directory')
    manifest=json.loads(args.source_manifest.read_text(encoding='utf-8'))
    if manifest.get('format')!='crrl-bc-shards-v1':raise ValueError('Expected per-game BC shards manifest')
    root=args.source_manifest.resolve().parent
    metadata_path=checked_path(root,manifest['metadata'])
    if file_hash(metadata_path)!=manifest['metadata_sha256']:raise ValueError('Source metadata SHA mismatch')
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    columns=[i for i,name in enumerate(metadata['schema']['header_fields']) if name=='decision_ticks']
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'shards').mkdir();(args.out/'audit').mkdir()
    (args.out/'.source_arrays').mkdir()
    before={};after={};shards=[];seen=set();summaries=[]
    source_fingerprint=file_hash(args.source_manifest)
    for shard_index,record in enumerate(manifest['shards']):
        path=checked_path(root,record['path'])
        if file_hash(path)!=record['sha256']:raise ValueError(f'Source shard SHA mismatch: {path.name}')
        scratch=args.out/'.source_arrays'/f'shard_{shard_index:06d}'
        for replay_id,source_rows,arrays in source_games(path,scratch,int(record.get('replay_offset',0))):
            if replay_id in seen:raise ValueError('Source game appears in multiple shards')
            seen.add(replay_id)
            output,audit=resample_game(arrays,decision_ticks=args.decision_ticks,decision_header_columns=columns,
                source_row_ids=source_rows)
            # New files each have one local replay ID, preserving original global ID as offset.
            output['replay_ids']=np.zeros(len(output['actions']),dtype=np.int32)
            audit.update(source_shard=record['path'],source_shard_sha256=record['sha256'],
                source_replay_id=replay_id,source_manifest_sha256=source_fingerprint)
            target=args.out/'shards'/f'replay_{replay_id:06d}.npz'
            np.savez_compressed(target,**output)
            with np.load(target,allow_pickle=False) as saved:
                for field in (*FIELDS,'source_rows','source_ticks'):
                    if not np.array_equal(saved[field],output[field]):raise AssertionError('Published shard differs from verified output')
            shards.append({'path':str(target.relative_to(args.out)).replace('\\','/'),
                'sha256':file_hash(target),'rows':len(output['actions']),'replays':1,'replay_offset':replay_id})
            for key,value in audit['before'].items():before[key]=before.get(key,0)+value
            for key,value in audit['after'].items():after[key]=after.get(key,0)+value
            atomic_json(args.out/'audit'/f'replay_{replay_id:06d}.json',audit)
            summaries.append({'replay_id':replay_id,'source_shard_sha256':record['sha256'],
                'before':audit['before'],'after':audit['after'],'identity_equal':True,
                'action_delay_ticks':audit['action_delay_ticks']})
            print(json.dumps({'phase':'resampled','game':replay_id,'done':len(shards),
                'nonwait':audit['after']['nonwait'],'rows_before':len(arrays['actions']),
                'rows_after':len(output['actions'])}),flush=True)
            del arrays,output
    (args.out/'.source_arrays').rmdir()
    expected_games=int(manifest.get('replays',len(metadata['replays'])))
    if len(seen)!=expected_games:raise AssertionError('Not every original game was preserved')
    for key in ('nonwait','card','ability','valid_nonwait','valid_card','valid_ability'):
        if before.get(key)!=after.get(key):raise AssertionError(f'Dataset action preservation failed: {key}')
    semantics={'kind':'offline_delayed_bc_supervision','source_manifest_sha256':source_fingerprint,
        'source_metadata_sha256':manifest['metadata_sha256'],'source_decision_ticks':metadata['decision_ticks'],
        'logical_decision_ticks':args.decision_ticks,'all_original_games_preserved':True,
        'all_original_nonwait_rows_preserved':True,'observation_clock':'original native source tick/time_ms retained',
        'mutable_observation_columns':columns,'native_trajectory':False,'ppo_rollouts_allowed':False,
        'native_parity_claim_allowed':False,'mapping':'causal ceil plus per-seat FIFO; only WAIT rows merged',
        'source_rows_definition':'original row index within source_shard, not a resimulated event ID'}
    summary={'semantics':semantics,'games':len(shards),'before':before,'after':after,
        'training_ready':True,'all_identity_checks_passed':True,'shards':summaries}
    atomic_json(args.out/'resampling_summary.json',summary)
    # Replay identities and split membership retain original indices unchanged.
    metadata.update(rows=after['rows'],decision_ticks=args.decision_ticks,
        source_decision_ticks=semantics['source_decision_ticks'],label_source='offline_delayed_bc_supervision',
        interval_resampling=semantics,training_ready=True,smoke_only=False,native_trajectory=False,
        ppo_rollouts_allowed=False,native_parity_claim_allowed=False)
    metadata['import_policy']={**metadata.get('import_policy',{}),
        'offline_augmentation':'event_preserving_delayed_supervision',
        'logical_decision_ticks':args.decision_ticks,'source_state_clock_preserved':True}
    metadata['label_totals']={**metadata.get('label_totals',{}),'rows':after['rows'],
        'labelled':after['nonwait'],'flagged':after['flagged_rows'],'label_masked':after['illegal_label_rows'],
        'resampling_scope':'only WAIT merging; original nonWAIT flags/legal status preserved'}
    atomic_json(args.out/'dataset_metadata.json',metadata)
    result={**manifest,'rows':after['rows'],'replays':len(shards),'decision_ticks':args.decision_ticks,
        'metadata':'dataset_metadata.json','metadata_sha256':file_hash(args.out/'dataset_metadata.json'),
        'shards':shards,'training_ready':True,'smoke_only':False,'label_source':'offline_delayed_bc_supervision',
        'native_trajectory':False,'ppo_rollouts_allowed':False,'native_parity_claim_allowed':False,
        'source_manifest_sha256':source_fingerprint}
    atomic_json(args.out/'dataset.manifest.json',result)
    print(json.dumps({'phase':'complete','games':len(shards),'before':before,'after':after,
        'training_ready':True,'native_trajectory':False}),flush=True)


if __name__=='__main__':main()
