"""Bounded v3 policy control of Nulls private server via existing client APIs.

No training, memory writes, injection, or automatic ladder/menu navigation.
Use --execute only for the explicitly calibrated private-server device.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback
import faulthandler
import io
import zipfile
import platform
import os
from contextlib import ExitStack

os.environ.setdefault('CUDA_MODULE_LOADING','LAZY')

import numpy as np
import torch

from run_nulls_dryrun import build_policy_inference
from cr_rl.client.adb_actions import ActionObservation,AdbActionBridge,mumu_900x1600_calibration
from cr_rl.client.nulls_reader import (NullsMemoryReader, NotInBattleError, ReadBudgetExceededError,
                                       resolve_pid_and_libg_base)
from cr_rl.client.persistent_memory import PersistentAdbMemReader,UnmappedGamePageError
from cr_rl.client.schema_adapter import NullsToSchemaV4Adapter
from cr_rl.client.schema_v4 import SchemaV4
from cr_rl.client.live_control import (PendingPlay,PendingAbility,reconcile_play,reconcile_ability,
                                      BattleClockGate,pending_targets_for)
from cr_rl.client.action_selection import HazardTempoState, select_live_action
from cr_rl.client.diagnostics import DiagnosticRecorder,decision_reason,model_input_quality
from cr_rl.client.persistent_touch import PersistentAdbTouch
from cr_rl.client.runtime import prepare_private_device
from cr_rl.client.menu_navigation import guarded_tap, ensure_lobby
from cr_rl.client.rl_recording import (CONTRACT, actor_fingerprint, atomic_json,
                                       record_terminal, set_result, export_episode, sha256_file)
from cr_rl.training.awbc_inbox import exclusive_lock


# Map unseen private-server tower/hero/custom entities to their closest trained Schema v4 rows
# so entity & card embeddings never hit untrained out-of-distribution row indices in live play.
NULLS_OOD_ALIASES: dict[str, str] = {
    'ChefTower': 'PrincessTower',
    'ChefTowerKing': 'KingTower',
    'Cannoneer': 'PrincessTower',
    'CannoneerTower': 'PrincessTower',
    'TowerCannoneer': 'PrincessTower',
    'DaggerDuchess': 'PrincessTower',
    'DaggerDuchessTower': 'PrincessTower',
    'TowerKnifeThrower': 'PrincessTower',
    'TowerPrincess': 'PrincessTower',
    'GoblinQueenTower': 'KingTower',
    'GoblinQueenTowerTopCharacter': 'KingTower',
    'GoblinQueenSecondaryTower': 'PrincessTower',
    'TowerKnifeThrowerProjectile': 'TowerPrincessProjectile',
    'CannoneerProjectile': 'TowerPrincessProjectile',
    'CannoneerProjectile_AoE': 'TowerPrincessProjectile',
    'TowerCannonball': 'TowerPrincessProjectile',
    'ChefTower_pancake_projectile': 'KingProjectile',
    'ChefTower_spatula_projectile': 'TowerPrincessProjectile',
    'GoblinQueenSecondaryTowerProjectile': 'TowerPrincessProjectile',
    'Chef': 'PrincessTower',
    'Ronin': 'Prince',
    'GoblinMachine': 'GiantSkeleton',
    'MinionGiant': 'MegaMinion',
    'MiniPekkaHero': 'MiniPekka',
    'ValkyrieHero': 'Valkyrie',
    'GiantHero': 'Giant',
    'MusketeerHero': 'Musketeer',
    'KnightHero': 'Knight',
    'ElectroWizardHero': 'ElectroWizard',
    'BalloonHero': 'Balloon',
    'WizardHero': 'Wizard',
    'WizardHero_air': 'BabyDragon',
    'MegaMinionHero': 'MegaMinion',
    'IceGolemiteHero': 'IceGolemite',
    'BowlerHero': 'Bowler',
    'DarkPrinceHero': 'DarkPrince',
    'DarkPrinceHero_Walking': 'DarkPrince',
    'DarkPrinceHero_Mount': 'DarkPrince',
    'IceWizardHero': 'IceWizard',
    'BerserkerHero': 'Berserker',
    'BerserkerHero_BearForm': 'GiantSkeleton',
    'EliteArcherHero': 'Archer',
    'GoblinHero': 'Goblin',
    'GoblinHero_Base': 'Goblin',
    'GoblinHero_Flag_Building': 'GoblinHut',
    'BarbLogBarbarianHero': 'Barbarian',
    'BarbLogBarbarianHeroNaked': 'Barbarian',
    'TombstoneHero': 'Tombstone',
    'TombstoneHeroSkeleton': 'Skeleton',
    'TombstoneHero_Monster_Passive': 'GiantSkeleton',
    'TombstoneHero_Monster_Active': 'GiantSkeleton',
    'BalloonHero_Bomb': 'GiantSkeletonBomb',
    'IceWizardHero_IceCube': 'IceWizard',
    'IceWizardHeroProjectile': 'chr_wizardProjectile',
    'IceWizard_EV1': 'IceWizard',
    'Princess_EV1': 'Princess',
    'Princess_EV1_Projectile': 'PrincessProjectile',
    'Princess_EV1_FreezeProjectile': 'PrincessProjectile',
    'BabyDragon_EV1': 'BabyDragon',
    'Furnace_EV1': 'FirespiritHut',
    'BlowdartGoblin_EV1': 'BlowdartGoblin',
    'AngryBarbarian_EV1': 'AngryBarbarian',
    'AngryBarbarian_EV1_2': 'AngryBarbarian',
    'MinionHorde_EV1': 'Minion',
    'Giant_EV1': 'Giant',
    'ElectroGiant_EV1': 'ElectroGiant',
    'GoblinGiant_EV1': 'GoblinGiant',
    'IceSpirits_EV1': 'IceSpirits',
    'Pekka_EV1': 'Pekka',
    'Recruit_EV1': 'Recruit',
    'RoyalGiant_EV1': 'RoyalGiant',
    'Ghost_EV1_Clone': 'Ghost',
    'Ghost_EV1_Summon_Base': 'Ghost',
    'Ghost_EV1_Summon_Left': 'Ghost',
    'Ghost_EV1_Summon_Right': 'Ghost',
    'GoblinCage_EV1_GoblinBrawler': 'GoblinBrawler',
    'RoyalHog_EV1_Grounded': 'RoyalHog',
    'SkeletonArmy_EV1_General_Base': 'Skeleton',
    'SkeletonArmy_EV1_Spectral': 'Skeleton',
    'Witch_EV1_Healing_Skeleton': 'Skeleton',
    'Witch_EV1_Skeleton_Interval': 'Skeleton',
}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial',default='127.0.0.1:16416')
    p.add_argument('--adb',default='G:/sdk/platform-tools/adb.exe')
    p.add_argument('--checkpoint',type=Path,default=Path('artifacts/ppo_v3_v100/actor_export.pt'))
    p.add_argument('--tables',type=Path,default=Path.home()/'.crforge/tables/16.635.3')
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--out-root',type=Path,default=None,
                   help='Parent directory for per-match play_<timestamp> folders in persistent multi-match mode')
    p.add_argument('--matches',type=int,default=1,
                   help='Consecutive battles to run in one process reusing GPU weights (0 = unlimited until stopped)')
    p.add_argument('--device',default='cuda')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--max-plays',type=int,default=12)
    p.add_argument('--max-state-age',type=float,default=.4)
    p.add_argument('--seed',type=int,default=20261115)
    p.add_argument('--touch-transport',choices=['auto','persistent','shell'],default='auto')
    p.add_argument('--selection-mode',choices=['policy_sample','confidence','argmax','hazard'],default='policy_sample',
                   help='Sample the trained joint policy, take pure argmax, use cumulative hazard tempo, or confidence thresholds')
    p.add_argument('--action-mode',choices=['argmax','policy_sample','confidence','hazard'],default=None,
                   help='Alias overriding --selection-mode (e.g. --action-mode hazard)')
    p.add_argument('--play-threshold',type=float,default=.45)
    p.add_argument('--slot-threshold',type=float,default=.28)
    p.add_argument('--tile-temperature',type=float,default=0.0)
    p.add_argument('--enable-abilities',action='store_true')
    p.add_argument('--ood-aliases',dest='ood_aliases',action='store_true',default=True,
                   help='Alias private-server custom/hero/chef entities to trained CrForge vocabulary rows (default)')
    p.add_argument('--no-ood-aliases',dest='ood_aliases',action='store_false',
                   help='Disable private-server OOD entity aliasing')
    p.add_argument('--reserve-to-wait',dest='reserve_to_wait',action='store_true',default=None,
                   help='Force folding unaffordable hand-card logits into WAIT (default: match checkpoint config)')
    p.add_argument('--no-reserve-to-wait',dest='reserve_to_wait',action='store_false',
                   help='Force masking unaffordable hand-card logits before softmax instead of folding into WAIT')
    p.add_argument('--auto-battle',action='store_true',help='Click verified private-server battle and result buttons')
    p.add_argument('--prompt-result',action='store_true',
                   help='After a recorded terminal, ask the operator for win/loss/draw; unknown stays unlabeled')
    p.add_argument('--ability-calibration',type=Path,default=Path('configs/nulls_abilities_900x1600.json'))
    a=p.parse_args(argv)
    if a.action_mode is not None:
        a.selection_mode = a.action_mode
    if a.matches < 0:
        p.error('--matches must be >= 0')
    if a.out is None and a.out_root is None:
        p.error('Specify --out or --out-root')
    if a.auto_battle and not a.execute:
        p.error('--auto-battle requires --execute')
    if not (0 <= a.play_threshold <= 1 and 0 <= a.slot_threshold <= 1 and np.isfinite(a.tile_temperature) and a.tile_temperature >= 0):
        p.error('Invalid confidence threshold or tile temperature')
    if not 0<a.seconds<=600 or not 0<a.max_plays<=100 or not 0<a.max_state_age<=.4:
        p.error('Use 0..600 seconds, 1..100 plays, and freshness <=400ms')
    return a


class PolicyRuntime:
    """Model and static tables for a frozen Actor. Match state is reset for each run."""

    def __init__(self, args, checkpoint_sha=None):
        self.checkpoint = args.checkpoint.resolve(strict=True)
        self.checkpoint_stat = self.checkpoint.stat()
        self.checkpoint_sha = checkpoint_sha or sha256_file(self.checkpoint)
        self.source_archive = None
        self.infer = None

    def prepare(self, args, emit):
        stat = self.checkpoint.stat()
        if (args.checkpoint.resolve() != self.checkpoint or
                (stat.st_size, stat.st_mtime_ns, stat.st_ino) !=
                (self.checkpoint_stat.st_size, self.checkpoint_stat.st_mtime_ns,
                 self.checkpoint_stat.st_ino)):
            raise ValueError('Frozen Actor changed during collection')
        torch.manual_seed(args.seed)
        if self.infer is not None:
            self.infer.reset()
            self.adapter.reset_match()
            emit('actor_reused')
            return

        torch.set_num_threads(2)
        checkpoint = torch.load(self.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        self.fingerprint = actor_fingerprint(checkpoint)
        is_cuda = str(args.device).startswith('cuda')
        model, infer, ticks = build_policy_inference(
            checkpoint, str(args.device), compute_value=False,
            precision='fp16' if is_cuda else 'fp32')
        rtw_override = getattr(args, 'reserve_to_wait', None)
        if rtw_override is not None and bool(rtw_override) != bool(model.cfg.reserve_to_wait):
            from dataclasses import replace
            model.cfg = replace(model.cfg, reserve_to_wait=bool(rtw_override))
        del checkpoint
        if ticks != 8:
            raise ValueError('Private-server controller requires 8-tick v3 policy')
        emit('warming_up', message='Initializing CUDA kernels before accepting a battle')
        warm_mask = np.zeros((1, 2307), dtype=bool)
        warm_mask[:, 0] = True
        warm_obs = np.zeros((1, SchemaV4.SEAT_INTS), dtype=np.int32)
        infer.evaluate(warm_obs, warm_mask, compute_value=False)
        warm_mask[:, 1:SchemaV4.FIRST_ABILITY] = True
        infer.evaluate(warm_obs, warm_mask, compute_value=False)
        rows = warm_obs[0, SchemaV4.HEADER_INTS:].reshape(SchemaV4.MAX_ENTITIES, SchemaV4.ENTITY_INTS)
        for count in (6, 32, 96, 200):
            warm_obs[:, SchemaV4.H_ENTITY_COUNT] = count
            rows[:count, SchemaV4.E_ROW] = 1
            rows[:count, SchemaV4.E_KIND] = SchemaV4.KIND_CHARACTER
            rows[:count, SchemaV4.E_HP] = 1000
            rows[:count, SchemaV4.E_MAX_HP] = 1000
            infer.evaluate(warm_obs, warm_mask, compute_value=False)
        if is_cuda:
            torch.cuda.synchronize()
        adapter = NullsToSchemaV4Adapter(
            args.tables,
            aliases=NULLS_OOD_ALIASES if getattr(args, 'ood_aliases', True) else None,
            allow_abilities=args.enable_abilities,
        )
        if adapter.vocab.names and len(adapter.vocab.names) != model.cfg.vocab_size:
            raise ValueError('Checkpoint and table vocabularies differ')
        self.adapter = adapter
        self.metadata = model.deployment_metadata
        infer.reset()
        self.infer = infer

    def write_source(self, out):
        if self.source_archive is None:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                sources = sorted(Path('src/cr_rl').rglob('*.py'))
                sources += [Path(name) for name in (
                    'scripts/run_nulls_policy.py', 'scripts/run_nulls_dryrun.py',
                    'scripts/native/CrrlTouch.java', 'scripts/native/nulls_readonly_pages.c')]
                for source in sources:
                    archive.write(source, source.as_posix())
            self.source_archive = buffer.getvalue()
        (out/'source.zip').write_bytes(self.source_archive)


def run_battle(a, runtime, resources):
    selection_rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'manifest.json').exists() or (a.out/'events.jsonl').exists():
        raise ValueError('Use a new session output directory; never append a second recurrent rollout')
    log = resources.enter_context((a.out/'events.jsonl').open('a', encoding='utf-8', buffering=1))
    diagnostics = DiagnosticRecorder(a.out/'diagnostics', keep=None, durable_members=True)
    resources.callback(diagnostics.close)
    from cr_rl.client.async_recording import OrderedRecorder
    recorder = OrderedRecorder()
    resources.callback(recorder.close)

    def write_event(row):
        encoded = json.dumps(row, default=str)
        log.write(encoded+'\n')
        print(encoded, flush=True)

    def emit(event, **data):
        recorder.submit(write_event, {'event':event, 'unix':time.time(), **data})

    emit('device_ready', **prepare_private_device(a.serial, adb=a.adb))
    size = subprocess.check_output([a.adb, '-s', a.serial, 'shell', 'wm', 'size'], timeout=5).decode()
    if '900x1600' not in size or ('Override size:' in size and not size.strip().endswith('900x1600')):
        raise ValueError('Only the validated 900x1600 calibration is supported')
    started = time.monotonic()
    if runtime is None:
        runtime = PolicyRuntime(a)
    runtime.prepare(a, emit)
    runtime.write_source(a.out)
    infer, adapter = runtime.infer, runtime.adapter
    manifest={'format':'crrl-null-client-session-v1','checkpoint':str(a.checkpoint.resolve()),
              'checkpoint_sha256':runtime.checkpoint_sha,'python':platform.python_version(),
              'torch':torch.__version__,'numpy':np.__version__,'arguments':vars(a),
              'outcome':None,'outcome_verified':False,'collection_complete':False,
              'note':'Observations are partial and non-atomic; not an exact server-state replay. '
                     'Gated deployment actions are off-policy and are not execution labels. Join confirmed receipts by decision_id.'}
    manifest['input_contract']={'version':3,'schema_ints':7024,'action_mask_size':2307,
        'entity_layout_policy':'verified_vtables_only','buff_count':'total_active_buffs',
        'ability_states_as_inputs':True,'ability_actions_enabled':a.enable_abilities,
        'pending_target_source':'local_requested_point_matched_to_server_pending_card',
        'refusal_fields':'unknown_without_verified_client_equivalent',
        'live_validation_pending':['sustained_area_effects','evolution_cycles']}
    manifest['outcome_policy']='ended_status_crown_score_v1_with_result_screen_crosscheck'
    manifest['recording_io']='bounded_ordered_background_writer_v1'
    manifest['rl_contract']={'version':CONTRACT,'action_semantics':'policy_proposal',
        'actor_fingerprint':runtime.fingerprint,'initial_memory_reset':True,
        'decision_ticks':8,'ms_per_tick':50,'reward_scheme':'verified_terminal_only',
        'gzip_layout':'independent_members_per_frame','memory_advances_on_all_frames':True}
    torch.save({'cpu':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state_all() if str(a.device).startswith('cuda') else []},
               a.out/'sampling_rng.pt')
    manifest['checkpoint_metadata']=runtime.metadata
    manifest['sampling_rng']='sampling_rng.pt'
    (a.out/'manifest.json').write_text(json.dumps(manifest,default=str,indent=2),encoding='utf-8')
    ability_points=None
    if a.enable_abilities:
        profile=json.loads(a.ability_calibration.read_text(encoding='utf-8'))
        if not profile.get('validated') or profile.get('size')!=[900,1600] or profile.get('serial')!=a.serial:
            raise ValueError('Validated ability calibration for this device is required')
        ability_points=tuple(tuple(p) if p is not None else None for p in profile['points'])
        manifest['ability_calibration']=profile
    pid,base=resolve_pid_and_libg_base(a.serial,adb=a.adb)
    # Busy battles with burst spawns and buffs can exceed the 1 MiB default.
    transports = resources.enter_context(ExitStack())
    memory=PersistentAdbMemReader(a.serial,pid,adb=a.adb,max_poll_bytes=2*1024*1024)
    transports.callback(memory.close)
    # Keep ended-clock snapshots for terminal evidence. The control loop never
    # evaluates or sends an action from them.
    reader=NullsMemoryReader(memory,base,require_running_clock=False)
    touch=None
    if a.execute and a.touch_transport!='shell':
        try:
            touch=PersistentAdbTouch(a.serial,adb=a.adb)
            transports.callback(touch.close)
            emit('touch_transport',mode='persistent',gap_ms=20)
        except Exception as error:
            if a.touch_transport=='persistent':raise
            emit('touch_transport',mode='shell',startup_fallback=str(error))
    current=None
    bridge=AdbActionBridge(mumu_900x1600_calibration(a.serial,validated=True,ability_points=ability_points),
        adb=a.adb,state_source=lambda:current,max_age=a.max_state_age,tap_transport=touch)
    pending=None;last_tick=-1;epoch=0;in_battle=False;plays=0;frames=0;confirmed=0
    stop_reason='time_limit';last_heartbeat=0.;wait_since=None;last_warning=0.
    clock_gate=BattleClockGate()
    tempo_state=HazardTempoState()
    read_failures=0;terminal_record=None;previous_played_count=None
    emit('configuration',arguments=vars(a),checkpoint_sha256=runtime.checkpoint_sha,
         diagnostic_retention='complete bounded session, 32MiB segments')
    deadline=time.monotonic()+a.seconds
    emit('ready',pid=pid,device=a.device,execute=a.execute,load_seconds=time.monotonic()-started)
    def battle_active():
        try:
            snapshot=reader.read_snapshot()
            return not snapshot.clock_ended
        except NotInBattleError:
            return False
        except (OSError,ValueError,EOFError,TimeoutError):
            return None
    try:
        if a.auto_battle:
            ensure_lobby(a.adb,a.serial,timeout=60,failure_dir=a.out,
                         battle_probe=battle_active,emit=emit)
            guarded_tap(a.adb,a.serial,'battle',timeout=60,failure_dir=a.out)
            emit('menu_battle_tapped')
        while time.monotonic()<deadline:
            t=time.monotonic()
            try:snap=reader.read_snapshot()
            except (UnmappedGamePageError, ReadBudgetExceededError) as error:
                read_failures+=1
                current=None  # Never send from a partial or previous snapshot.
                emit('snapshot_discarded',reason=str(error),consecutive_failures=read_failures,
                     retry_limit=5)
                if read_failures>=5:
                    raise RuntimeError('Five consecutive unreadable snapshots; stopping for reader diagnosis') from error
                time.sleep(.1);continue
            except NotInBattleError as error:
                if in_battle:
                    stop_reason='battle_unavailable'
                    emit('battle_unavailable',reason=str(error),terminal_verified=False);break
                if time.monotonic()-last_heartbeat>5:
                    emit('waiting_for_battle',reason=str(error));last_heartbeat=time.monotonic()
                time.sleep(.4);continue
            except (EOFError, TimeoutError, BrokenPipeError, OSError) as error:
                current=None
                stop_reason='read_transport_lost'
                emit('read_transport_lost',reason=str(error),last_tick=last_tick)
                break
            if read_failures:
                emit('snapshot_recovered',discarded=read_failures,tick=snap.tick)
                read_failures=0
            read_s=time.monotonic()-t
            if snap.clock_ended:
                if not in_battle:
                    if time.monotonic()-last_heartbeat>5:
                        emit('waiting_for_battle',reason='Previous battle result overlay');last_heartbeat=time.monotonic()
                    time.sleep(.4);continue
                terminal_match=f'{pid}:{epoch}:{snap.local_acc}:{tuple(s.acc_id for s in snap.summoners)}'
                if terminal_match!=match_id or snap.tick<last_tick:
                    stop_reason='terminal_identity_mismatch';break
                if pending is not None and snap.own_seat in (0,1):
                    final_own=snap.summoners[snap.own_seat]
                    if isinstance(pending,PendingAbility):
                        final_status=reconcile_ability(pending,match_id=match_id,
                            abilities=final_own.abilities,now=time.monotonic())
                    else:
                        final_status=reconcile_play(pending,match_id=match_id,
                            played_count=final_own.played_count,hand=final_own.hand,now=time.monotonic())
                    if final_status=='confirmed':
                        emit('receipt',receipt=bridge.resolve('confirmed'),decision_id=pending.decision_id,
                             played_count=final_own.played_count,confirmation_tick=snap.tick)
                        confirmed+=1;pending=None
                    elif (isinstance(pending,PendingPlay)
                          and final_own.played_count==pending.played_count
                          and final_own.hand==pending.hand):
                        emit('receipt',receipt=bridge.resolve('rejected'),decision_id=pending.decision_id,
                             played_count=final_own.played_count,confirmation_tick=snap.tick,
                             proof='battle_ended_unchanged_native_count_and_hand')
                        pending=None
                terminal_adapted=adapter.adapt(snap)
                terminal_adapted.legal_mask[1:]=False
                terminal_record=record_terminal(a.out,match_id=match_id,snapshot=snap,
                    obs=terminal_adapted.obs,legal_mask=terminal_adapted.legal_mask)
                stop_reason='battle_end'
                emit('battle_end',tick=snap.tick,terminal_file='terminal.json',outcome_known=False)
                break
            if snap.own_seat not in (0,1):
                emit('unknown_seat');time.sleep(.4);continue
            if not in_battle or snap.tick<last_tick:
                epoch+=1;infer.reset();adapter.reset_match();pending=None
                previous_played_count=None
                clock_gate=BattleClockGate()
                tempo_state.reset()
                if bridge.pending:bridge.resolve('timed_out')
                emit('battle_start',epoch=epoch,seat=snap.own_seat,tick=snap.tick)
            in_battle=True
            match_id=f'{pid}:{epoch}:{snap.local_acc}:{tuple(s.acc_id for s in snap.summoners)}'
            own=snap.summoners[snap.own_seat]
            if previous_played_count is not None and own.played_count>previous_played_count:
                delta=own.played_count-previous_played_count
                if delta!=1 or not isinstance(pending,PendingPlay):
                    emit('unattributed_play',tick=snap.tick,delta=delta,
                         reason='No matching exclusive-controller pending play')
            previous_played_count=own.played_count
            if isinstance(pending,PendingAbility):
                result=reconcile_ability(pending,match_id=match_id,abilities=own.abilities,now=time.monotonic())
            else:
                result=reconcile_play(pending,match_id=match_id,played_count=own.played_count,
                                      hand=own.hand,now=time.monotonic())
            if result:
                receipt=bridge.resolve(result);emit('receipt',receipt=receipt,played_count=own.played_count,
                                                    abilities=own.abilities,
                                                    decision_id=pending.decision_id)
                confirmed+=int(result=='confirmed');pending=None
            if snap.tick==last_tick:
                if time.monotonic()-last_heartbeat>5:
                    emit('clock_stalled',tick=snap.tick);last_heartbeat=time.monotonic()
                time.sleep(.05);continue
            gap=None if last_tick<0 else snap.tick-last_tick
            last_tick=snap.tick
            slots=tuple(isinstance(pending,PendingPlay) and pending.slot==i for i in range(4))
            targets=pending_targets_for(pending,own.hand)
            adapted=adapter.adapt(snap,pending_slots=slots,pending_targets=targets)
            if frames == 0:
                emit('own_deck',tick=snap.tick,match_id=match_id,
                     cards=[{'name':card.name,'scid':card.scid,'level_raw':card.level_raw,
                             'slot_flags':card.slot_flags} for card in own.deck])
            executable_abilities=[False,False]
            # The current profile was verified for one active champion button.
            alive={e.eid for e in snap.entities if e.hp is not None and e.hp>0}
            active=[ab for ab in own.abilities or () if ab.unit_eid in alive]
            for i in range(2):
                executable_abilities[i]=bool(ability_points and ability_points[i] is not None
                    and (profile.get('layout')!='single_active_champion' or len(active)==1))
                if not executable_abilities[i]:adapted.legal_mask[2305+i]=False
            quality=model_input_quality(adapted.obs)
            # Gate the opening animation until the battle clock is advancing.
            opening_gate=not clock_gate.allow(tick=snap.tick,time_ms=snap.time_ms,
                                             running_flag=snap.clock_running,ended=snap.clock_ended)
            legal_before_gate=int(adapted.legal_mask[1:].sum())
            if opening_gate:
                adapted.legal_mask[1:]=False
            if pending is not None:
                # Pending confirmation is a controller constraint. Expose it in
                # the behavior mask instead of sampling actions then losing them.
                adapted.legal_mask[1:]=False
            if adapted.unmatched_names:
                emit('unmatched_names',names=adapted.unmatched_names)
            if adapted.rejected_entities:
                emit('entities_rejected',tick=snap.tick,entities=adapted.rejected_entities)
            ability_ids=tuple(next((ab.unit_eid for ab in own.abilities or () if ab.slot==i),None) for i in range(2))
            current=ActionObservation(match_id,t,adapted.hand_card_ids,tuple(map(bool,adapted.legal_mask)),900,1600,True,ability_ids)
            inference_t=time.monotonic()
            logits,_=infer.evaluate(adapted.obs[None],adapted.legal_mask[None],compute_value=False)
            distribution=torch.distributions.Categorical(logits=logits)
            probabilities=distribution.probs[0].cpu().numpy()
            action,selection_info=select_live_action(probabilities,adapted.legal_mask,adapted.obs,
                adapted.hand_cards,play_threshold=a.play_threshold,slot_threshold=a.slot_threshold,
                tile_temperature=a.tile_temperature,rng=selection_rng,selection_mode=a.selection_mode,
                tempo_state=tempo_state)
            proposal=int(selection_info.get('proposed_action',action))
            proposal_logp=float(distribution.log_prob(torch.as_tensor([proposal],device=logits.device))[0])
            play_probabilities=probabilities[1:2305].reshape(4,576)
            slot_probabilities=play_probabilities.sum(axis=1)
            stats={**selection_info,'entropy':float(distribution.entropy()[0]),
                   'max_action_probability':float(probabilities.max()),
                   'wait_probability':float(probabilities[0]),
                   'slot_probabilities':slot_probabilities.tolist(),
                   'ability_probabilities':probabilities[2305:].tolist(),
                   'selected_action_probability':float(probabilities[action])}
            inference_s=time.monotonic()-inference_t;frames+=1
            decoded=SchemaV4.decode_action(action)
            if decoded['kind']=='play':
                slot=int(decoded['slot'])
                stats['card']=adapted.hand_cards[slot]
                stats['tile_probability_given_card']=float(probabilities[action]/max(float(slot_probabilities[slot]),1e-30))
            elif decoded['kind']=='ability':
                stats['ability_slot']=int(decoded['slot'])
            top=np.argsort(probabilities)[-5:][::-1]
            stats['top_actions']=[{'action':int(i),'probability':float(probabilities[i]),
                                  'decoded':SchemaV4.decode_action(int(i))} for i in top]
            age=time.monotonic()-t
            reason=decision_reason(action,adapted.legal_mask,opening_gate=opening_gate,
                                   pending=pending is not None,stale=age>a.max_state_age)
            stats.update(reason=reason,legal_before_gate=legal_before_gate,
                         legal_play_tiles=adapted.legal_mask[1:2305].reshape(4,576).sum(axis=1).tolist(),
                         clock_running=snap.clock_running,own_seat=snap.own_seat,
                         tower_hp=adapted.obs[SchemaV4.H_TOWERS:SchemaV4.H_TOWERS+12].tolist(),
                         pending_slot=pending.slot if pending else None)
            stats['input_quality']=quality
            recorder.submit(diagnostics.write,snapshot=snap,obs=adapted.obs.copy(),legal_mask=adapted.legal_mask.copy(),
                              probabilities=probabilities.copy(),tick=snap.tick,reason=reason,
                              action=action,hand=adapted.hand_cards,decision_id=frames,
                              adapter_context={'pending_slots':slots,'pending_targets_battle':targets,
                                               'pending_target_source':'local_requested_point',
                                               'allow_abilities':a.enable_abilities,
                                               'executable_abilities':executable_abilities},
                              input_quality=quality,
                              action_selection=selection_info,
                              proposal_log_probability=proposal_logp,
                              rejected_entities=adapted.rejected_entities,
                              match_id=match_id,observed_monotonic=t,read_finished_monotonic=t+read_s,
                              inference_finished_monotonic=inference_t+inference_s,
                              selected_log_probability=float(distribution.log_prob(torch.as_tensor([action],device=logits.device))[0]))
            emit('decision',decision_id=frames,tick=snap.tick,tick_gap=gap,read_ms=read_s*1000,inference_ms=inference_s*1000,
                 state_age_ms=(time.monotonic()-t)*1000,bytes=snap.bytes_read,
                 read_exchanges=memory.poll_exchanges,action=action,decoded=decoded,
                 hand=adapted.hand_cards,elixir=own.elixir,played_count=own.played_count,**stats)
            if action==0:
                if wait_since is None:wait_since=time.monotonic()
                if time.monotonic()-wait_since>=10 and time.monotonic()-last_warning>=10:
                    emit('no_play_warning',seconds=time.monotonic()-wait_since,reason=reason,
                         elixir=own.elixir,wait_probability=stats['wait_probability'],
                         legal_play_tiles=stats['legal_play_tiles']);last_warning=time.monotonic()
            else:wait_since=None
            if action and pending is None:
                if time.monotonic()-t>a.max_state_age:
                    emit('stale_action_skipped',action=action,decision_id=frames)
                elif plays<a.max_plays:
                    slot=int(decoded['slot'])
                    skill=decoded['kind']=='ability'
                    ab=next((ab for ab in own.abilities or () if ab.slot==slot),None) if skill else None
                    receipt=bridge.submit(action,match_id=match_id,
                        expected_card_id=None if skill else adapted.hand_card_ids[slot],
                        expected_ability_unit_id=ab.unit_eid if skill else None,execute=a.execute)
                    emit('submitted',receipt=receipt,card=ab.champion_name if skill else adapted.hand_cards[slot],
                         kind=decoded['kind'],decision_id=frames,
                         probability=stats['selected_action_probability'],
                         slot_probability=float(probabilities[action] if skill else slot_probabilities[slot]))
                    if a.execute:
                        plays+=1
                        if skill:
                            pending=PendingAbility(match_id,slot,ab.deck_idx,ab.unit_eid,ab.charges,
                                                   ab.cooldown_ms,time.monotonic(),frames)
                        else:
                            target=(SchemaV4.game_x(snap.own_seat,int(decoded['col'])),
                                    SchemaV4.game_y(snap.own_seat,int(decoded['row'])))
                            pending=PendingPlay(match_id,own.played_count,own.hand,slot,
                                                time.monotonic(),frames,target)
            if plays>=a.max_plays and pending is None:
                stop_reason='max_plays';break
            delay=.4-(time.monotonic()-t)
            if delay>0:time.sleep(delay)
    except KeyboardInterrupt:
        stop_reason='keyboard_interrupt'
    except Exception:
        stop_reason='error';emit('error',traceback=traceback.format_exc());raise
    finally:
        if bridge.pending:emit('receipt',receipt=bridge.resolve('timed_out'),
                               decision_id=pending.decision_id if pending else None)
        emit('stopped',reason=stop_reason,frames=frames,sent=plays,confirmed=confirmed)
        # Publish completion only after all forensic frames/receipts are durable.
        recorder.flush()
        manifest.update(stop_reason=stop_reason,collection_complete=stop_reason=='battle_end',
                        frames=frames,sent=plays,confirmed=confirmed)
        (a.out/'manifest.json').write_text(json.dumps(manifest,default=str,indent=2),encoding='utf-8')
        diagnostics.close()
        transports.close()
        if terminal_record is not None and stop_reason=='battle_end':
            from cr_rl.client.outcome import finalize_result
            if a.auto_battle:
                try:
                    guarded_tap(a.adb,a.serial,'result',a.out/'result_screen.png',tap=False,
                                timeout=30,failure_dir=a.out)
                except (RuntimeError,OSError,ValueError) as error:
                    emit('result_screen_unavailable',reason=str(error))
            try:
                verified=finalize_result(a.out)
                emit('result_recorded',outcome=verified['outcome'],reward=verified['reward'],
                     source=verified['source'])
            except (ValueError,OSError,KeyError) as error:
                emit('result_unresolved',reason=str(error))
        if terminal_record is not None and a.prompt_result and not (a.out/'result.json').exists():
            try:
                outcome=input('Result for THIS battle (win/loss/draw, Enter=unknown): ').strip().lower()
                if outcome in ('win','loss','draw'):
                    set_result(a.out,outcome,evidence='Operator confirmed this battle post-match result',
                               source='operator_confirmation')
                    emit('result_recorded',outcome=outcome,source='operator_confirmation')
            except (EOFError,KeyboardInterrupt):
                emit('result_unknown',reason='No operator confirmation')
        try:
            report=export_episode(a.out)
            emit('rl_export',**report)
        except (ValueError,OSError,EOFError) as error:
            emit('rl_not_ready',reason=str(error))
        try:
            if a.auto_battle and terminal_record is not None and stop_reason=='battle_end':
                if not (a.out/'result.json').is_file():
                    raise RuntimeError('Battle result unresolved; preserving scoreboard and stopping before next match')
                guarded_tap(a.adb,a.serial,'result',a.out/'result_screen.png',timeout=30,failure_dir=a.out)
                emit('menu_return_tapped')
                ensure_lobby(a.adb,a.serial,timeout=60,failure_dir=a.out,emit=emit)
                guarded_tap(a.adb,a.serial,'battle',tap=False,timeout=60,failure_dir=a.out)
                emit('menu_lobby_verified')
        finally:
            recorder.close()
            if terminal_record is not None and stop_reason=='battle_end' and (a.out/'result.json').is_file():
                from cr_rl.client.outcome import seal_recording
                seal_recording(a.out)


def run_logged_battle(args, runtime=None):
    lock_path = Path(__file__).resolve().parents[1]/'artifacts/nulls_live/policy-control.lock'
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'crash.log').open('a', encoding='utf-8', buffering=1) as crash:
        faulthandler.enable(file=crash)
        try:
            with exclusive_lock(lock_path), ExitStack() as resources:
                run_battle(args, runtime, resources)
        except BaseException:
            traceback.print_exc(file=crash)
            raise
        finally:
            faulthandler.disable()


def main():
    args = parse_args()
    if args.out_root is None and args.matches == 1:
        run_logged_battle(args)
        return
    from copy import copy
    from datetime import datetime
    out_root = (args.out_root or args.out.parent).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    runtime = PolicyRuntime(args)
    total = int(args.matches)
    match_idx = 0
    while total == 0 or match_idx < total:
        match_idx += 1
        while True:
            now = datetime.now()
            stamp = now.strftime('%Y%m%d_%H%M%S_') + f'{now.microsecond // 1000:03d}'
            run_dir = out_root / f'play_{stamp}'
            if not run_dir.exists():
                break
            time.sleep(0.002)
        match_args = copy(args)
        match_args.out = run_dir
        match_args.seed = int(args.seed) + match_idx - 1
        tag = f'{match_idx}/{total}' if total > 0 else f'{match_idx}/unlimited'
        print(f'Match {tag} | Log directory: {run_dir}', flush=True)
        run_logged_battle(match_args, runtime=runtime)
        manifest_path = run_dir / 'manifest.json'
        if not manifest_path.is_file():
            break
        mf = json.loads(manifest_path.read_text(encoding='utf-8'))
        stop_reason = mf.get('stop_reason')
        transient_idle = total == 0 and (
            stop_reason == 'time_limit'
            or (stop_reason == 'read_transport_lost' and int(mf.get('frames', 0) or 0) == 0)
        )
        if stop_reason != 'battle_end' and not transient_idle:
            print(f'Session ended with {stop_reason}. Stopping batch; see {run_dir}', flush=True)
            break
        if transient_idle:
            time.sleep(2.0)


if __name__ == '__main__':
    main()
