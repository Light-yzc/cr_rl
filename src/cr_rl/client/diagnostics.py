"""Bounded compressed forensic frames for real-client inference failures."""
import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from .schema_v4 import SchemaV4


def model_input_quality(obs):
    """Report real missing unit values without treating non-applicable rows as gaps."""
    count=int(obs[SchemaV4.H_ENTITY_COUNT])
    rows=np.asarray(obs[SchemaV4.HEADER_INTS:]).reshape(SchemaV4.MAX_ENTITIES,SchemaV4.ENTITY_INTS)[:count]
    units=rows[np.isin(rows[:,SchemaV4.E_KIND],[SchemaV4.KIND_CHARACTER,SchemaV4.KIND_TOWER])]
    fields={'state':SchemaV4.E_STATE,'attack_ms':SchemaV4.E_ATTACK_MS,
            'load_ms':SchemaV4.E_LOAD_MS,'speed_percent':SchemaV4.E_SPEED_PERCENT,
            'hit_speed_percent':SchemaV4.E_HIT_SPEED_PERCENT}
    return {'unit_rows':len(units),'unknown_unit_fields':{
        name:int((units[:,index]==SchemaV4.UNKNOWN).sum()) for name,index in fields.items()},
        'unknown_elixir_reserved':bool(obs[SchemaV4.H_ELIXIR_RESERVED]==SchemaV4.UNKNOWN),
        'unknown_hand_refill':bool(obs[SchemaV4.H_HAND_REFILL_MS]==SchemaV4.UNKNOWN),
        'unknown_pending_count':bool(obs[SchemaV4.H_PENDING_COUNT]==SchemaV4.UNKNOWN)}


class DiagnosticRecorder:
    def __init__(self, directory, *, segment_bytes=32*1024*1024, keep=4, durable_members=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment_bytes, self.keep = segment_bytes, keep
        self.index = 0
        self.durable_members = durable_members
        self.size = 0
        self.stream = None

    def write(self, *, snapshot, obs, legal_mask, probabilities, **metadata):
        row = dict(metadata, snapshot=asdict(snapshot), obs=obs.tolist(),
                   legal_actions=np.flatnonzero(legal_mask).tolist(),
                   probabilities=probabilities.tolist())
        data = (json.dumps(row, separators=(',', ':'), default=str)+'\n').encode('utf-8')
        if self.stream is None or self.size + len(data) > self.segment_bytes:
            self.close()
            self.index += 1
            self.size = 0
            path = self.directory/f'frames_{self.index:04d}.jsonl.gz'
            self.stream = path.open('wb') if self.durable_members else gzip.open(path, 'wb')
            if self.keep is not None:
                for old in sorted(self.directory.glob('frames_*.jsonl.gz'))[:-self.keep]:
                    old.unlink()
        # Each flushed frame has its own complete gzip footer. A process killed
        # between frames no longer leaves all previous frames in an open member.
        self.stream.write(gzip.compress(data, compresslevel=1, mtime=0) if self.durable_members else data)
        self.stream.flush()  # Keep the latest complete frame available during a failure.
        self.size += len(data)

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def decision_reason(action, legal_mask, *, opening_gate, pending, stale):
    if opening_gate:
        return 'opening_gate'
    if pending:
        return 'pending_confirmation'
    if not np.any(legal_mask[1:]):
        return 'forced_wait_mask'
    if action == 0:
        return 'policy_wait'
    if stale:
        return 'stale_state'
    return 'action_ready'
