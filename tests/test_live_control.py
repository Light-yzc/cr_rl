import unittest

from cr_rl.client.live_control import PendingPlay, reconcile_play, BattleClockGate, pending_targets_for


class BattleClockGateTest(unittest.TestCase):
    def test_false_running_flag_does_not_block_advancing_battle(self):
        gate = BattleClockGate()
        self.assertTrue(gate.allow(tick=2643, time_ms=132150, running_flag=True, ended=False))
        self.assertTrue(gate.allow(tick=2651, time_ms=132550, running_flag=False, ended=False))
        self.assertTrue(gate.allow(tick=3596, time_ms=179800, running_flag=False, ended=False))
        self.assertFalse(gate.allow(tick=3600, time_ms=180000, running_flag=True, ended=True))

    def test_opening_frozen_clock_and_tick_rewind(self):
        gate = BattleClockGate()
        self.assertFalse(gate.allow(tick=0, time_ms=0, running_flag=True, ended=False))
        self.assertTrue(gate.allow(tick=8, time_ms=400, running_flag=False, ended=False))
        self.assertFalse(gate.allow(tick=8, time_ms=400, running_flag=False, ended=False))
        self.assertFalse(gate.allow(tick=0, time_ms=0, running_flag=False, ended=False))
        self.assertFalse(BattleClockGate().allow(tick=100, time_ms=5000, running_flag=False, ended=False))


class ReconcilePlayTest(unittest.TestCase):
    def setUp(self):
        self.pending = PendingPlay('match', 3, (0, 1, 2, 3), 1, 10.)

    def check(self, **changes):
        state = dict(match_id='match', played_count=3, hand=(0, 1, 2, 3), now=10.5)
        state.update(changes)
        return reconcile_play(self.pending, **state)

    def test_confirmation_requires_count_and_expected_slot_rotation(self):
        self.assertIsNone(self.check(played_count=4))
        self.assertIsNone(self.check(hand=(4, 1, 2, 3), played_count=4))
        self.assertIsNone(self.check(hand=(0, 4, 2, 3)))
        self.assertEqual(self.check(hand=(0, 4, 2, 3), played_count=4), 'confirmed')

    def test_timeout_and_match_change_release_pending(self):
        self.assertEqual(self.check(now=12.), 'timed_out')
        self.assertEqual(self.check(match_id='next'), 'timed_out')

    def test_multiple_unattributed_plays_cannot_confirm_one_pending_card(self):
        self.assertIsNone(self.check(hand=(0,4,2,3),played_count=5))

    def test_no_pending(self):
        self.pending = None
        self.assertIsNone(self.check())

    def test_pending_target_tracks_card_identity_not_reused_hand_slot(self):
        self.pending.battle_target=(3500,22000)
        self.assertEqual(pending_targets_for(self.pending,(0,1,2,3)),(None,(3500,22000),None,None))
        self.assertEqual(pending_targets_for(self.pending,(0,4,1,3)),(None,None,(3500,22000),None))
        self.assertEqual(pending_targets_for(self.pending,(0,4,2,3)),(None,)*4)
        self.assertEqual(pending_targets_for(None,(0,1,2,3)),(None,)*4)
