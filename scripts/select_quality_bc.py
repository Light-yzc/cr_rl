"""Audit metadata and publish a small immutable auxiliary-BC selection index."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from cr_rl.data.bc_quality import select_games


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--split-registry', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-snap-ticks', type=int, default=4)
    parser.add_argument('--min-source-events', type=int, default=20)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    registry = json.loads(args.split_registry.read_text(encoding='utf-8'))
    if args.min_source_events < 1:
        parser.error('--min-source-events must be positive')
    decisions = select_games(metadata, registry, max_snap_ticks=args.max_snap_ticks,
                             min_source_events=args.min_source_events)
    selected = [x['replay_id'] for x in decisions if x['selected']]
    rejections = Counter(reason for x in decisions for reason in x['reasons'])
    report = dict(format='crrl-aux-bc-selection-v1', metadata_path=str(args.metadata.resolve()),
                  metadata_sha256=sha(args.metadata), split_registry_sha256=sha(args.split_registry),
                  source_decision_ticks=metadata.get('decision_ticks'),
                  max_snap_ticks=args.max_snap_ticks, min_source_events=args.min_source_events,
                  selected_replay_ids=selected,
                  total_games=len(decisions), selected_games=len(selected),
                  rejection_counts=dict(rejections), decisions=decisions,
                  ppo_rollouts_allowed=False, expert_skill_verified=False,
                  original_battlefield_parity_verified=False,
                  training_ready=False,
                  next_gate='Verify array hashes, labels, backend and 0.4s conversion before use')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError('Use a fresh audit path; selections are immutable')
    args.out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('total_games', 'selected_games',
                      'selected_replay_ids', 'rejection_counts', 'training_ready')}))


if __name__ == '__main__':
    main()
