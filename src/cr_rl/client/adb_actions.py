"""Calibrated ADB taps with explicit fresh-state checks and execution receipts.

ADB success only means the touch command ran, never that the game accepted it.
State snapshots must come from a validated observation source.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import time
from typing import Callable


@dataclass(frozen=True)
class TouchCalibration:
    serial: str
    width: int
    height: int
    hand_points: tuple[tuple[float, float], ...]
    # Clockwise: top-left, top-right, bottom-right, bottom-left in screen pixels.
    board_quad: tuple[tuple[float, float], ...]
    validated: bool = False
    rotate_board: bool = False
    ability_points: tuple[tuple[float, float] | None, ...] | None = None

    def __post_init__(self):
        if self.width < 1 or self.height < 1 or len(self.hand_points) != 4 or len(self.board_quad) != 4:
            raise ValueError('Four hand points, four board corners, and positive dimensions required')
        points = [*self.hand_points, *self.board_quad]
        if self.ability_points is not None:
            if len(self.ability_points) != 2:
                raise ValueError('Two ability points required when ability_points is provided')
            points.extend(p for p in self.ability_points if p is not None)
        for x, y in points:
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError('Calibration point outside screen')

    def taps(self, action: int):
        if action == 0:
            return ()
        if 2305 <= action <= 2306:
            if self.ability_points is None:
                raise ValueError('Ability buttons require separate calibration; unsupported action')
            point = self.ability_points[action - 2305]
            if point is None:
                raise ValueError('This ability slot has no validated button position')
            x, y = point
            return ((round(x), round(y)),)
        if not 1 <= action <= 2304:
            raise ValueError('Ability buttons require separate calibration; unsupported action')
        slot, tile = divmod(action - 1, 576)
        u, v = (tile % 18 + .5) / 18, 1.0 - (tile // 18 + .5) / 32
        if self.rotate_board:
            u, v = 1.0 - u, (tile // 18 + .5) / 32
        tl, tr, br, bl = self.board_quad
        xy = tuple((1-u)*(1-v)*tl[k] + u*(1-v)*tr[k] + u*v*br[k] + (1-u)*v*bl[k] for k in (0, 1))
        return tuple((round(x), round(y)) for x, y in (self.hand_points[slot], xy))


MUMU_900X1600 = TouchCalibration(
    serial='127.0.0.1:16416',
    width=900,
    height=1600,
    hand_points=((280, 1448), (450, 1448), (619, 1448), (788, 1448)),
    board_quad=((68, 129), (832, 129), (832, 1229), (68, 1229)),
    validated=False,
    rotate_board=False,
    ability_points=None,
)


def mumu_900x1600_calibration(
    serial: str = '127.0.0.1:16416',
    *,
    validated: bool = False,
    ability_points: tuple[tuple[float, float] | None, ...] | None = None,
) -> TouchCalibration:
    return TouchCalibration(
        serial=serial,
        width=MUMU_900X1600.width,
        height=MUMU_900X1600.height,
        hand_points=MUMU_900X1600.hand_points,
        board_quad=MUMU_900X1600.board_quad,
        validated=validated,
        rotate_board=False,
        ability_points=ability_points,
    )


@dataclass(frozen=True)
class ActionObservation:
    match_id: str
    observed_monotonic: float
    hand_card_ids: tuple[int, int, int, int]
    legal_actions: tuple[bool, ...]
    width: int
    height: int
    verified: bool
    ability_unit_ids: tuple[int | None, int | None] = (None,None)


class AdbActionBridge:
    def __init__(self, calibration: TouchCalibration, *, adb: str | Path,
                 state_source: Callable[[], ActionObservation], max_age=.4,
                 clock=time.monotonic, transport=None, tap_transport=None):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError('Positive finite freshness limit required')
        self.calibration, self.adb, self.state_source = calibration, str(adb), state_source
        self.max_age, self.clock = max_age, clock
        self.transport = transport or self._transport
        self.tap_transport = tap_transport
        self.pending = None

    @staticmethod
    def _transport(argv):
        return subprocess.run(argv, capture_output=True, check=True, timeout=5)

    def submit(self, action: int, *, match_id: str, expected_card_id=None,
               expected_ability_unit_id=None, execute=False):
        if not isinstance(action, int) or isinstance(action, bool):
            raise ValueError('Action must be an integer')
        taps = self.calibration.taps(action)
        receipt = {'action': action, 'match_id': match_id, 'taps': taps, 'status': 'dry_run'}
        if not execute:
            return receipt
        if action == 0:
            return {**receipt, 'status': 'wait', 'sent': False}
        if self.pending is not None:
            raise RuntimeError('Previous action is still awaiting game confirmation')
        if not self.calibration.validated:
            raise ValueError('Touch calibration has not been validated')
        state = self.state_source()
        age = self.clock() - state.observed_monotonic
        if not state.verified or not math.isfinite(age) or not 0 <= age <= self.max_age:
            raise ValueError('Observation source is unverified or stale')
        if state.match_id != match_id:
            raise ValueError('Match changed since action was selected')
        if (state.width, state.height) != (self.calibration.width, self.calibration.height):
            raise ValueError('Screen dimensions changed since calibration')
        if 1 <= action <= 2304:
            slot = (action - 1) // 576
            if expected_card_id is None or state.hand_card_ids[slot] != expected_card_id:
                raise ValueError('Hand slot identity changed or was not supplied')
        if action>=2305 and expected_ability_unit_id is not None:
            if state.ability_unit_ids[action-2305]!=expected_ability_unit_id:
                raise ValueError('Champion identity changed since skill was selected')
        if len(state.legal_actions) != 2307 or not state.legal_actions[action]:
            raise ValueError('Action is not legal in the current observation')
        shell = ' && '.join(f'input tap {x} {y}' for x, y in taps)
        # Even a transport failure can occur after the first tap. Keep pending.
        self.pending = {**receipt, 'status': 'sending', 'submitted_monotonic': self.clock()}
        try:
            if self.tap_transport is not None and len(taps)==2:
                self.tap_transport(taps)
            else:
                self.transport([self.adb, '-s', self.calibration.serial, 'shell', shell])
        except Exception:
            self.pending['status'] = 'transport_failed_unconfirmed'
            raise
        self.pending['status'] = 'sent_unconfirmed'
        self.pending['transport_ms'] = (self.clock()-self.pending['submitted_monotonic'])*1000
        return dict(self.pending)

    def resolve(self, status: str):
        """Caller must supply actual game evidence or explicitly mark timeout."""
        if status not in ('confirmed', 'rejected', 'timed_out') or self.pending is None:
            raise ValueError('A pending action and valid resolution required')
        result = {**self.pending, 'status': status}
        self.pending = None
        return result
