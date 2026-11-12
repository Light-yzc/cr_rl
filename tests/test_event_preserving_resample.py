"""CPU-only preservation/causality checks for delayed BC supervision."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cr_rl.data.event_preserving_resample import resample_game


def source_game(game=0):
    ticks=np.repeat(np.arange(0,35,5),2).astype(np.int32)
    n=len(ticks);seats=np.tile([0,1],7).astype(np.int32)
    observations=np.zeros((n,5),dtype=np.int32)
    observations[:,0]=ticks;observations[:,1]=ticks*50
    observations[:,2]=5;observations[:,3]=np.arange(n)
    actions=np.zeros(n,dtype=np.int32)
    actions[2]=1;actions[4]=577;actions[6]=2305;actions[9]=2306
    flags=np.zeros(n,dtype=np.int32);flags[6]=65536
    masks=np.ones((n,2307),dtype=np.uint8);masks[9,2306]=False
    return {'observations':observations,'masks':masks,'actions':actions,'flags':flags,
        'ticks':ticks,'seats':seats,'replay_ids':np.full(n,game,dtype=np.int32)}


class EventPreservationTests(unittest.TestCase):
    def test_all_actions_skills_flagged_and_illegal_rows_preserved_with_exact_prestate(self):
        source=source_game();output,audit=resample_game(source,decision_header_columns=[2])
        self.assertEqual(audit['before']['nonwait'],4)
        self.assertEqual(audit['after']['nonwait'],4)
        self.assertEqual(audit['before']['ability'],2)
        self.assertEqual(audit['after']['valid_nonwait'],2)
        for seat in (0,1):
            old=np.flatnonzero((source['seats']==seat)&(source['actions']!=0))
            new=np.flatnonzero((output['seats']==seat)&(output['actions']!=0))
            np.testing.assert_array_equal(output['source_rows'][new],old)
            for original,current in zip(old,new):
                for field in ('actions','flags','masks'):
                    np.testing.assert_array_equal(output[field][current],source[field][original])
                np.testing.assert_array_equal(output['observations'][current,[0,1,3,4]],
                    source['observations'][original,[0,1,3,4]])
                self.assertEqual(output['observations'][current,2],8)
        self.assertTrue(all(row['identity_equal'] for row in audit['seats']))

    def test_causal_fifo_and_wait_never_cross_next_unemitted_action(self):
        source=source_game();output,audit=resample_game(source)
        side=np.flatnonzero(output['seats']==0)
        act=side[output['actions'][side]!=0]
        np.testing.assert_array_equal(output['ticks'][act],[8,16,24])
        np.testing.assert_array_equal(output['source_ticks'][act],[5,10,15])
        for seat in (0,1):
            rows=np.flatnonzero(output['seats']==seat)
            self.assertEqual(output['ticks'][rows[0]],0)
            self.assertTrue(np.all(np.diff(output['ticks'][rows])==8))
            self.assertTrue(np.all(np.diff(output['source_rows'][rows])>=0))
            self.assertTrue(np.all(output['source_ticks'][rows]<=output['ticks'][rows]))
        self.assertEqual(audit['early_actions'],0)
        self.assertFalse(audit['native_trajectory'])
        self.assertFalse(audit['ppo_rollouts_allowed'])

    def test_global_source_rows_trace_to_original_monolithic_shard(self):
        source=source_game();global_rows=np.arange(len(source['actions']))+12345
        output,audit=resample_game(source,source_row_ids=global_rows)
        self.assertGreaterEqual(output['source_rows'].min(),12345)
        self.assertTrue(all(row['identity_equal'] for row in audit['seats']))

    def test_missing_safe_wait_is_rejected_instead_of_future_or_action_relabelling(self):
        source=source_game()
        source['actions'][:]=1
        # Seat1 has one fewer action, requiring an unavailable safe WAIT at the tail.
        keep=np.arange(len(source['actions']))!=13
        source={key:value[keep] for key,value in source.items()}
        with self.assertRaisesRegex(ValueError,'No causal original WAIT'):
            resample_game(source)

    def test_whole_dataset_shard_is_streamed_one_game_and_scratch_removed(self):
        script=Path(__file__).resolve().parents[1]/'scripts/resample_bc_interval.py'
        spec=importlib.util.spec_from_file_location('resample_cli',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        parent=Path(__file__).resolve().parents[1]/'artifacts/event_resampling_tests'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root=Path(temp);games=[source_game(0),source_game(1)]
            np.savez_compressed(root/'all.npz',**{key:np.concatenate([g[key] for g in games]) for key in games[0]})
            scratch=root/'.source_arrays'/'shard_000000'
            generated=module.source_games(root/'all.npz',scratch,0)
            count=0
            for game_id,rows,arrays in generated:
                self.assertEqual(len(arrays['actions']),14)
                self.assertEqual(game_id,count)
                np.testing.assert_array_equal(rows,np.arange(14)+count*14)
                result,_=resample_game(arrays,source_row_ids=rows)
                self.assertTrue((result['source_rows']>=count*14).all())
                count+=1
            self.assertEqual(count,2)
            self.assertFalse(scratch.exists())

    def test_cli_monolithic_source_publishes_all_games_and_identity_gates(self):
        script=Path(__file__).resolve().parents[1]/'scripts/resample_bc_interval.py'
        spec=importlib.util.spec_from_file_location('resample_cli_complete',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        parent=Path(__file__).resolve().parents[1]/'artifacts/event_resampling_tests'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root=Path(temp);games=[source_game(0),source_game(1)]
            np.savez_compressed(root/'all.npz',**{key:np.concatenate([g[key] for g in games]) for key in games[0]})
            meta={'schema':{'header_fields':['tick','time_ms','decision_ticks','test','reserved']},
                'replays':[{'canonical_game_signature':'train-game'},{'canonical_game_signature':'validation-game'}],
                'decision_ticks':5,'rows':28,'dataset_kind':'repaired'}
            (root/'metadata.json').write_text(json.dumps(meta))
            manifest={'format':'crrl-bc-shards-v1','metadata':'metadata.json',
                'metadata_sha256':module.file_hash(root/'metadata.json'),'rows':28,'replays':2,
                'shards':[{'path':'all.npz','sha256':module.file_hash(root/'all.npz'),
                    'rows':28,'replays':2,'replay_offset':0}]}
            (root/'dataset.manifest.json').write_text(json.dumps(manifest))
            with patch('sys.argv',['resample_bc_interval','--source-manifest',str(root/'dataset.manifest.json'),
                '--decision-ticks','8','--out',str(root/'new')]):
                module.main()
            summary=json.loads((root/'new/resampling_summary.json').read_text())
            self.assertEqual(summary['games'],2)
            self.assertEqual(summary['before']['nonwait'],8)
            self.assertEqual(summary['after']['nonwait'],8)
            self.assertEqual(summary['after']['valid_nonwait'],4)
            self.assertTrue(summary['all_identity_checks_passed'])
            self.assertFalse(summary['semantics']['native_trajectory'])
            self.assertFalse((root/'new/.source_arrays').exists())
            newmeta=json.loads((root/'new/dataset_metadata.json').read_text())
            self.assertEqual(newmeta['replays'],meta['replays'])
            self.assertEqual(newmeta['decision_ticks'],8)


if __name__=='__main__':unittest.main()
