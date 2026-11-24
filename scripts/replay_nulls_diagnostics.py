"""Rebuild action masks from saved memory frames without ADB, model or game taps."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from cr_rl.client.live_control import BattleClockGate
from cr_rl.client.nulls_reader import NullsBattleSnapshot
from cr_rl.client.schema_adapter import NullsToSchemaV4Adapter
from cr_rl.client.schema_v4 import SchemaV4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--tables', type=Path, default=Path.home()/'.crforge/tables/16.635.3')
    args = p.parse_args()
    adapter = NullsToSchemaV4Adapter(args.tables)
    gate = BattleClockGate()
    report = {'frames': 0, 'incorrectly_blocked_frames': 0, 'restored_playable_frames': 0,
              'old_model_obs_mismatches': 0, 'first_restored_tick': None}
    for file in sorted((args.run/'diagnostics').glob('frames_*.jsonl.gz')):
        with gzip.open(file, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                snap = NullsBattleSnapshot.from_dict(row['snapshot'])
                context=row.get('adapter_context',{})
                pending = tuple(context.get('pending_slots',
                    [bool(x) for x in row['obs'][SchemaV4.H_HAND_PENDING:SchemaV4.H_HAND_PENDING+4]]))
                targets=context.get('pending_targets_battle')
                adapter.allow_abilities=bool(context.get('allow_abilities',False))
                adapted = adapter.adapt(snap, pending_slots=pending,pending_targets=targets)
                allowed = gate.allow(tick=snap.tick,time_ms=snap.time_ms,
                                     running_flag=snap.clock_running,ended=snap.clock_ended)
                old_allowed = snap.clock_running and snap.tick >= 8
                report['frames'] += 1
                report['old_model_obs_mismatches'] += int(not np.array_equal(adapted.obs, row['obs']))
                if allowed and not old_allowed:
                    report['incorrectly_blocked_frames'] += 1
                    if adapted.legal_mask[1:].any():
                        report['restored_playable_frames'] += 1
                        if report['first_restored_tick'] is None:
                            report['first_restored_tick'] = snap.tick
    print(json.dumps(report, indent=2))
    (args.run/'gate_replay_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
