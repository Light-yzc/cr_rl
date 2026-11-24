"""Stream public replay rows into strictly screened native 0.4s BC shards.

No hand/placement repair or extra resource wait is allowed. This verifies
internal reconstruction evidence, not fidelity to the original battlefield.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from crforge_env import LadderConfig, VectorBattle
from cr_rl.data.bc_quality import assess_game
from cr_rl.data.interval_rebuild import canonicalise_events, EVENT_KINDS
from cr_rl.data.native_replay import NativeReplayObserver
from cr_rl.data.opening_cycle import infer_opening_cycle, native_opening_deals, reorder_for_deal
from cr_rl.data.replay_execution import ExecutionConfig, execute_repaired_events
from cr_rl.training.persistence import atomic_json, file_sha256
from import_il_replay import ALIASES, game_signature, normalise, row_name, slot_flag

REVISION = '059d43a02138a34b1b3009cc2acc7630fb99a638'


def partition(signature, decks, heldout, registry):
    old = registry.get('assignments', {}).get(signature)
    if old is not None:
        return old
    if any(tuple(sorted(deck)) in heldout for deck in decks):
        return 'heldout'
    return 'validation' if int(signature[:8], 16) % 5 == 0 else 'train'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--install', type=Path, required=True)
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--prior-split', type=Path)
    p.add_argument('--eval-decks', type=Path, default=Path('configs/decks_eval.yaml'))
    p.add_argument('--max-source-rows', type=int, default=128)
    p.add_argument('--target-games', type=int, default=32)
    p.add_argument('--start-row', type=int, default=0)
    a = p.parse_args()
    if min(a.max_source_rows, a.target_games) < 1 or a.start_row < 0:
        p.error('Invalid bounded scan size')
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Use a fresh output directory')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'shards').mkdir()
    (a.out/'audit').mkdir()
    registry = json.loads(a.prior_split.read_text()) if a.prior_split else {'assignments':{}}
    heldout = {tuple(sorted(deck)) for deck in yaml.safe_load(a.eval_decks.read_text())['decks'].values()}
    cards = set()
    for name in ('spells_characters','spells_buildings','spells_other'):
        cards.update(json.loads((a.tables/(name+'.json')).read_text())['rows'])
    vocabulary = {normalise(name):name for name in cards}
    source_hash = file_sha256(a.parquet)
    rejected = Counter()
    records, shards, seen = [], [], set()
    total_rows = scanned = 0
    config = LadderConfig(decision_ticks=8, level=11, tower_level=11, mask_version=2, arena='Arena_L')
    with VectorBattle(1, 1, install=a.install, tables=a.tables, config=config) as vector:
        vector.set_autoreset(False)
        native_metadata = dict(vector.metadata)
        for batch in pq.ParquetFile(a.parquet).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                row_index = scanned
                scanned += 1
                if row_index < a.start_row:
                    continue
                if scanned > a.start_row+a.max_source_rows or len(records) >= a.target_games:
                    break
                if row.get('game_mode') != 'Ranked':
                    rejected['not_ranked'] += 1
                    continue
                payload = json.loads(row['payload_json'])
                if payload.get('schema_version') != 'royaleapi-battle-actions.v1':
                    rejected['unsupported_schema'] += 1
                    continue
                decks, forms, raw_cards = [], [], []
                for side in ('team','opponent'):
                    players = payload.get('battle',{}).get(side,{}).get('players',[])
                    hand = players[0].get('deck',[]) if len(players)==1 else []
                    names = [row_name(card, vocabulary, cards) for card in hand]
                    if len(hand)!=8 or None in names or len(set(names))!=8:
                        break
                    decks.append(names);forms.append([slot_flag(card) for card in hand]);raw_cards.append(hand)
                if len(decks)!=2:
                    rejected['unsupported_deck'] += 1
                    continue
                signature = game_signature(decks, forms, payload)
                if signature is None or signature in seen:
                    rejected['invalid_or_duplicate_identity'] += 1
                    continue
                seen.add(signature)
                events = [e for e in payload.get('events',[]) if e.get('kind') in EVENT_KINDS]
                if len(events)<20:
                    rejected['too_few_source_events'] += 1
                    continue
                try:
                    events = canonicalise_events(events,decks,ALIASES)
                except ValueError:
                    rejected['unmapped_event'] += 1
                    continue
                tag = str(payload.get('source',{}).get('replay_tag',row.get('replay_tag',signature)))
                seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4],'big') & 0x7fffffff
                vector.reset([seed],*decks,*forms)
                runtime, runtime_forms = [list(d) for d in decks], [list(f) for f in forms]
                native = native_opening_deals(vector)
                inference = []
                for seat,side in enumerate(('team','opponent')):
                    sequence = []
                    for event in sorted(events,key=lambda e:int(e['replay_tick_20hz'])):
                        if event.get('side')==side and event['kind']=='play_card':
                            keys = event.get('deck_card_key_candidates') or [event.get('card_key')]
                            # canonicalise_events resolves candidates to native deck names.
                            sequence.append(next((i for i,c in enumerate(decks[seat]) if c in keys),-1))
                    inferred = infer_opening_cycle(sequence) if native[seat]['ordinary'] else None
                    inference.append(inferred)
                    if inferred:
                        runtime[seat],runtime_forms[seat] = reorder_for_deal(decks[seat],forms[seat],
                            native[seat]['order'],inferred['order'])
                vector.reset([seed],*runtime,*runtime_forms)
                event_map = {}
                for event in events:
                    event_map.setdefault(int(event['replay_tick_20hz']),[]).append(event)
                stats, arrays, _, _ = execute_repaired_events(vector,event_map,decks,decks,runtime,
                    ExecutionConfig(decision_ticks=8,repair_hand=False,repair_tile=False,
                        repair_time=True,time_tolerance_ticks=7,max_delay_ticks=7,
                        max_omissions=0,skip_unavailable_abilities=False,serial_native_confirmation=False),
                    native_observer=NativeReplayObserver(vector))
                stats.update(game_mode='Ranked',omitted_commands=len(stats.get('omitted_command_ticks',[])))
                reasons,_ = assess_game(stats,max_snap_ticks=7)
                if arrays is None:
                    reasons.append('missing_arrays')
                else:
                    indices = np.arange(len(arrays['actions']))
                    legal = arrays['masks'][indices,arrays['actions']].astype(bool)
                    from cr_rl.data.replay_supervision import GENERATED_WAIT_TAIL
                    bad_flags = arrays['flags'] & ~GENERATED_WAIT_TAIL
                    if (not np.all(legal) or np.any(bad_flags)
                            or np.any(arrays['flags'][arrays['actions'] != 0])):
                        reasons.append('flagged_or_illegal_rows')
                atomic_json(a.out/'audit'/f'row_{row_index:06d}.json',
                            dict(source_row=row_index,signature=signature,reasons=reasons,stats=stats))
                if reasons:
                    rejected.update(reasons)
                else:
                    replay_id = len(records)
                    arrays['replay_ids'] = np.zeros(len(arrays['actions']),dtype=np.int32)
                    target = a.out/'shards'/f'game_{replay_id:06d}.npz'
                    np.savez_compressed(target,**arrays)
                    split = partition(signature,decks,heldout,registry)
                    registry.setdefault('assignments',{})[signature] = split
                    records.append(dict(stats,canonical_game_signature=signature,decks=decks,slots=forms,
                        source='VanguardX101/IL_Replay',source_revision=REVISION,source_row_index=row_index,
                        source_file_sha256=source_hash,opening_cycle_inference=inference,
                        repair_ledger_path=f'audit/row_{row_index:06d}.json'))
                    shards.append(dict(path=str(target.relative_to(a.out)),sha256=file_sha256(target),
                        rows=len(arrays['actions']),replays=1,replay_offset=replay_id))
                    total_rows += len(arrays['actions'])
                print(json.dumps(dict(stage='quality_bc_scan',source_row=row_index,
                    accepted_games=len(records),last_reasons=reasons)),flush=True)
                del arrays
            if scanned >= a.start_row+a.max_source_rows or len(records)>=a.target_games:
                break
    metadata = dict(schema=native_metadata['schema'],vocabulary=native_metadata['vocabulary'],
        dataset_kind='repaired',label_source='live_pre_action_capture_interval_v1',
        decision_ticks=8,mask_version=2,level=11,tower_level=11,rows=total_rows,replays=records,
        source_revision=REVISION,source_file_sha256=source_hash,backend_provenance=native_metadata,
        training_ready=False,expert_skill_verified=False,original_battlefield_parity_verified=False,
        import_policy=dict(repair_hand=False,repair_tile=False,repair_time=True,
            time_tolerance_ticks=7,max_delay_ticks=7,serial_native_confirmation=False),
        native_trajectory=True,ppo_rollouts_allowed=False)
    atomic_json(a.out/'dataset_metadata.json',metadata)
    atomic_json(a.out/'split_registry.json',registry)
    atomic_json(a.out/'dataset.manifest.json',dict(format='crrl-bc-shards-v1',dataset_kind='repaired',
        rows=total_rows,replays=len(records),metadata='dataset_metadata.json',
        metadata_sha256=file_sha256(a.out/'dataset_metadata.json'),shards=shards,
        decision_ticks=8,training_ready=False,ppo_rollouts_allowed=False))
    summary = dict(scanned_rows=scanned,accepted_games=len(records),rows=total_rows,
        rejection_counts=dict(rejected),partition_counts=dict(Counter(
            registry['assignments'][r['canonical_game_signature']] for r in records)),
        training_ready=False,remaining_gate='Expert/context quality review and model contract validation')
    atomic_json(a.out/'quality_summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
