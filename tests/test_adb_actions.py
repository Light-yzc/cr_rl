import unittest
from dataclasses import replace

from cr_rl.client.adb_actions import (
    ActionObservation,
    AdbActionBridge,
    MUMU_900X1600,
    TouchCalibration,
    mumu_900x1600_calibration,
)


class AdbActionsTest(unittest.TestCase):
    def setUp(self):
        self.calibration = TouchCalibration('test', 900, 1600,
            ((250,1400),(420,1400),(590,1400),(760,1400)),
            ((90,100),(810,100),(810,1250),(90,1250)), validated=True)
        self.state = ActionObservation('game1', 10., (11,22,33,44),
            (True,)*2307, 900, 1600, True)
        self.calls = []
        self.bridge = AdbActionBridge(self.calibration, adb='adb',
            state_source=lambda: self.state, clock=lambda:10.1, transport=self.calls.append)

    def test_dry_run_and_wait_never_tap(self):
        self.bridge.submit(1, match_id='game1')
        self.bridge.submit(0, match_id='game1', execute=True)
        self.assertEqual(self.calls, [])

    def test_slot_mapping_receipt_and_duplicate_guard(self):
        result = self.bridge.submit(577, match_id='game1', expected_card_id=22, execute=True)
        self.assertEqual(result['taps'][0], (420,1400))
        self.assertEqual(result['status'], 'sent_unconfirmed')
        self.assertEqual(len(self.calls),1)
        with self.assertRaises(RuntimeError):
            self.bridge.submit(577, match_id='game1', expected_card_id=22, execute=True)
        self.assertEqual(self.bridge.resolve('timed_out')['status'], 'timed_out')

    def test_stale_or_changed_state_never_sends(self):
        for change in ({'observed_monotonic':9.}, {'match_id':'other'},
                       {'hand_card_ids':(99,22,33,44)}, {'verified':False}, {'width':1000}):
            self.state=replace(ActionObservation('game1',10.,(11,22,33,44),(True,)*2307,900,1600,True),**change)
            with self.assertRaises(ValueError):
                self.bridge.submit(1, match_id='game1', expected_card_id=11, execute=True)
        self.assertEqual(self.calls,[])

    def test_illegal_and_uncalibrated_rejected(self):
        self.state=replace(self.state,legal_actions=(True,)+(False,)*2306)
        with self.assertRaises(ValueError):
            self.bridge.submit(1,match_id='game1',expected_card_id=11,execute=True)
        self.bridge.calibration=replace(self.calibration,validated=False)
        with self.assertRaises(ValueError):
            self.bridge.submit(1,match_id='game1',expected_card_id=11,execute=True)
        self.assertEqual(self.calls,[])

    def test_row_zero_maps_bottom_and_row_31_maps_top(self):
        # row=0 (own back-left) -> bottom-left of screen board_quad
        _, bottom_left = self.calibration.taps(1)
        self.assertEqual(bottom_left, (110, 1232))
        # row=31, col=17 (enemy back-right) -> top-right of screen board_quad
        _, top_right = self.calibration.taps(576)
        self.assertEqual(top_right, (790, 118))

        rotated = replace(self.calibration, rotate_board=True)
        _, rot_xy = rotated.taps(1)
        self.assertEqual(rot_xy, (790, 118))

    def test_abilities_calibrated(self):
        cal = replace(
            self.calibration,
            ability_points=((110.0, 1280.0), (200.0, 1280.0)),
        )
        self.assertEqual(cal.taps(2305), ((110, 1280),))
        self.assertEqual(cal.taps(2306), ((200, 1280),))
        calls = []
        bridge = AdbActionBridge(
            cal,
            adb='adb',
            state_source=lambda: self.state,
            clock=lambda: 10.1,
            transport=calls.append,
        )
        dry = bridge.submit(2305, match_id='game1', execute=False)
        self.assertEqual(dry['status'], 'dry_run')
        self.assertEqual(dry['taps'], ((110, 1280),))
        self.assertEqual(calls, [])

        sent = bridge.submit(2305, match_id='game1', execute=True)
        self.assertEqual(sent['status'], 'sent_unconfirmed')
        self.assertEqual(sent['taps'], ((110, 1280),))
        self.assertEqual(len(calls), 1)
        self.assertIn('input tap 110 1280', calls[0][-1])

    def test_mumu_900x1600_preset(self):
        self.assertEqual((MUMU_900X1600.width, MUMU_900X1600.height), (900, 1600))
        self.assertEqual(MUMU_900X1600.hand_points, ((280, 1448), (450, 1448), (619, 1448), (788, 1448)))
        self.assertEqual(MUMU_900X1600.board_quad, ((68, 129), (832, 129), (832, 1229), (68, 1229)))
        cal = mumu_900x1600_calibration(validated=True)
        hand_xy, tile_xy = cal.taps(1)
        self.assertEqual(hand_xy, (280, 1448))
        self.assertEqual(tile_xy, (89, 1212))

    def test_abilities_rejected(self):
        with self.assertRaises(ValueError):
            self.bridge.submit(2305,match_id='game1')

    def test_partial_transport_failure_stays_pending(self):
        def fail(_): raise OSError('offline')
        self.bridge.transport=fail
        with self.assertRaises(OSError):
            self.bridge.submit(1,match_id='game1',expected_card_id=11,execute=True)
        self.assertEqual(self.bridge.pending['status'],'transport_failed_unconfirmed')


if __name__ == '__main__':
    unittest.main()
