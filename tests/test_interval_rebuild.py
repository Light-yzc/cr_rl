"""Causal 0.4s command reconstruction arithmetic, without JVM/GPU."""
import unittest

from cr_rl.data.interval_rebuild import (
    canonicalise_events,
    causal_grid_plan,
    execution_interval_audit,
    recorded_source_events,
    strict_command_admission,
)


def event(tick, side="team", card="fireball"):
    return {"kind":"play_card","side":side,"card_key":card,"replay_tick_20hz":tick}


class IntervalRebuildTests(unittest.TestCase):
    def test_causal_fifo_collisions_preserve_every_command_and_cross_seat_parallelism(self):
        events=[event(1),event(2),event(2,"opponent"),event(8),event(17)]
        plan,summary=causal_grid_plan(events,8)
        self.assertEqual([r['planned_submission_tick'] for r in plan],[8,16,8,24,32])
        self.assertEqual(len(plan),len(events))
        self.assertEqual(summary['same_seat_collision_events'],2)
        self.assertEqual(summary['early_commands'],0)
        self.assertTrue(all(r['source_tick']<=r['planned_submission_tick'] for r in plan))
        self.assertEqual([e['replay_tick_20hz'] for e in events],[1,2,2,8,17])

    def test_ledger_missing_original_commands_fails_closed(self):
        replay={'total_play_events':2,'repair_records':[{'category':'execution','event_id':0,
            'source_tick':1,'source_event':event(1)}]}
        with self.assertRaisesRegex(ValueError,'Missing'):
            recorded_source_events(replay)
        events,provenance=recorded_source_events(replay,allow_incomplete=True)
        self.assertEqual(len(events),1)
        self.assertFalse(provenance['source_complete'])
        self.assertEqual(provenance['source_command_coverage'],.5)

    def test_complete_ledger_deduplicates_retry_without_changing_source_time(self):
        row={'category':'execution','event_id':0,'source_tick':7,'source_event':event(7)}
        events,provenance=recorded_source_events({'total_play_events':1,'repair_records':[row,row]})
        self.assertTrue(provenance['source_complete'])
        self.assertEqual(events[0]['replay_tick_20hz'],7)
        self.assertEqual(events[0]['original_event_id'],0)
        bad={**row,'source_tick':8}
        with self.assertRaisesRegex(ValueError,'time'):
            recorded_source_events({'total_play_events':1,'repair_records':[bad]})

    def test_form_key_resolution_retains_original_identity_and_spell_time(self):
        original=event(7,card='elite-barbarians')
        original['deck_card_key_candidates']=['elite-barbarians-ev1']
        transformed=canonicalise_events([original],[['AngryBarbarians'],['Fireball']],
            {'elitebarbarians':'AngryBarbarians'})[0]
        self.assertEqual(transformed['deck_card_key_candidates'],['AngryBarbarians'])
        self.assertEqual(transformed['original_deck_card_key_candidates'],['elite-barbarians-ev1'])
        self.assertEqual(transformed['replay_tick_20hz'],7)
        self.assertEqual(original['deck_card_key_candidates'],['elite-barbarians-ev1'])

    def test_submission_quantisation_and_native_delay_are_reported_separately(self):
        stats={'repair_records':[{'category':'execution','native_success':True,'event_id':0,
            'source_tick':7,'simulator_tick':8,'actual_run_tick':28}],
            'trajectory_completion_kind':'all_source_events','simulator_fault':0}
        report=execution_interval_audit(stats,1)
        self.assertEqual(report['command_retention'],1.)
        self.assertEqual(report['submission_minus_source_ticks']['max'],1)
        self.assertEqual(report['native_run_minus_submission_ticks']['max'],20)
        stats['repair_records'][0]['simulator_tick']=6
        with self.assertRaisesRegex(ValueError,'before'):
            execution_interval_audit(stats,1)

    def test_reject_invalid_interval_and_source(self):
        with self.assertRaises(ValueError):causal_grid_plan([event(1)],0)
        with self.assertRaises(ValueError):causal_grid_plan([event(-1)],8)
        with self.assertRaises(ValueError):causal_grid_plan([event(1,'third-seat')],8)

    def test_strict_admission_requires_one_native_confirmed_legal_row_per_source(self):
        import numpy as np
        source=[event(1),event(2,'opponent')]
        arrays={'ticks':np.array([0,0,8,8]),'seats':np.array([0,1,0,1]),
            'actions':np.array([0,0,1,577]),'flags':np.zeros(4,dtype=int),
            'masks':np.ones((4,2307),dtype=bool)}
        stats={'all_source_events_executed':True,'pending_commands_settled':True,
            'repair_records':[{'category':'execution','event_id':i,'seat':i,
                'source_tick':i+1,'simulator_tick':8,'executed_action':action,
                'actual_run_tick':28,
                'accepted':True,'native_success':True,'execution_status':'confirmed'}
                for i,action in enumerate([1,577])]}
        audit=strict_command_admission(stats,arrays,source,8)
        self.assertTrue(audit['accepted'])
        self.assertEqual(audit['retention'],1.)
        arrays['flags'][2]=1
        self.assertFalse(strict_command_admission(stats,arrays,source,8)['accepted'])
        arrays['flags'][2]=0;arrays['masks'][2,1]=False
        self.assertFalse(strict_command_admission(stats,arrays,source,8)['accepted'])
        arrays['masks'][2,1]=True;stats['repair_records'].pop()
        self.assertFalse(strict_command_admission(stats,arrays,source,8)['accepted'])

    def test_strict_admission_rejects_terminal_missing_tail_and_ability_skip(self):
        import numpy as np
        arrays={'ticks':np.array([0,0]),'seats':np.array([0,1]),'actions':np.array([0,0]),
            'flags':np.zeros(2,dtype=int),'masks':np.ones((2,2307),dtype=bool)}
        stats={'source_or_native_terminal_completed':True,'normal_simulator_terminal':True,
            'all_source_events_executed':False,'pending_commands_settled':True,
            'repair_records':[{'category':'ability_skipped','event_id':0}]}
        audit=strict_command_admission(stats,arrays,[event(100)],8)
        self.assertFalse(audit['accepted'])
        self.assertIn('source_stream_incomplete',audit['reasons'])
        self.assertIn('ability_skipped',audit['reasons'])


if __name__=='__main__':unittest.main()
