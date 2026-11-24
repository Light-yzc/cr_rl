import gzip
import json
from dataclasses import dataclass

import numpy as np

from cr_rl.client.diagnostics import DiagnosticRecorder, decision_reason, model_input_quality
from cr_rl.client.schema_v4 import SchemaV4 as S


def test_input_quality_ignores_projectile_non_applicable_unit_fields():
    obs=np.zeros(S.SEAT_INTS,dtype=np.int32)
    obs[S.H_ENTITY_COUNT]=2
    rows=obs[S.HEADER_INTS:].reshape(S.MAX_ENTITIES,S.ENTITY_INTS)
    rows[0,S.E_KIND]=S.KIND_CHARACTER
    rows[0,S.E_LOAD_MS]=-1
    rows[0,S.E_SPEED_PERCENT]=100
    rows[1,S.E_KIND]=S.KIND_PROJECTILE
    rows[1,S.E_LOAD_MS]=-1
    result=model_input_quality(obs)
    assert result['unit_rows']==1
    assert result['unknown_unit_fields']['load_ms']==1
    assert result['unknown_unit_fields']['speed_percent']==0


@dataclass
class Snapshot:
    tick: int


def test_rotating_frames_are_readable_and_bounded(tmp_path):
    recorder = DiagnosticRecorder(tmp_path, segment_bytes=1, keep=2)
    for tick in range(5):
        recorder.write(snapshot=Snapshot(tick), obs=np.array([1, -1]),
                       legal_mask=np.array([True, False, True]),
                       probabilities=np.array([.4, 0., .6]))
    recorder.close()
    paths = sorted(tmp_path.glob('*.gz'))
    assert len(paths) == 2
    frames = [json.loads(gzip.decompress(path.read_bytes())) for path in paths]
    assert [x['snapshot']['tick'] for x in frames] == [3, 4]
    assert frames[-1]['legal_actions'] == [0, 2]
    assert frames[-1]['obs'] == [1, -1]
    assert frames[-1]['probabilities'] == [.4, 0., .6]


def test_forced_mask_is_distinct_from_policy_wait_or_transport_wait():
    def reason(action, mask, **kwargs):
        return decision_reason(action, np.array(mask), opening_gate=kwargs.get('opening', False),
                               pending=kwargs.get('pending', False), stale=kwargs.get('stale', False))
    assert reason(0, [True, False]) == 'forced_wait_mask'
    assert reason(0, [True, True]) == 'policy_wait'
    assert reason(1, [True, True], pending=True) == 'pending_confirmation'
    assert reason(0, [True, False], opening=True) == 'opening_gate'
    assert reason(1, [True, True], stale=True) == 'stale_state'
    assert reason(1, [True, True]) == 'action_ready'
