"""Conservative whole-game admission for auxiliary BC, never PPO rollouts.

These checks measure reconstruction fidelity evidence, not player skill or
agreement between the reconstructed battlefield and the original client.
"""
from collections import Counter


def assess_game(record, *, max_snap_ticks=4, min_source_events=20):
    if max_snap_ticks < 0:
        raise ValueError('max_snap_ticks must be nonnegative')
    reasons = []
    for key in ('source_event_stream_completed', 'all_source_events_executed',
                'pending_commands_settled'):
        if record.get(key) is not True:
            reasons.append(key)
    if record.get('coverage') != 1 or record.get('missing_source_events') != 0:
        reasons.append('incomplete_source_coverage')
    if record.get('simulator_fault') is not False:
        reasons.append('fault_or_missing_fault_evidence')
    if record.get('omitted_commands') != 0:
        reasons.append('omitted_or_unknown_commands')
    if record.get('game_mode') != 'Ranked':
        reasons.append('not_ranked')
    ledger = record.get('repair_records')
    if not isinstance(ledger, list) or not ledger:
        return sorted(set(reasons + ['missing_execution_ledger'])), {}
    allowed = {'execution', 'native_run', 'time_snap', 'ability_source_resolution'}
    categories = Counter()
    executions = set()
    for item in ledger:
        category = item.get('category')
        categories[str(category)] += 1
        if category not in allowed:
            reasons.append('ledger:' + str(category))
        if category == 'time_snap':
            delta = item.get('delta_ticks')
            source, actual = item.get('source_tick'), item.get('simulator_tick')
            if (not isinstance(delta, int) or not isinstance(source, int)
                    or not isinstance(actual, int) or delta != actual-source
                    or not 0 <= delta <= max_snap_ticks):
                reasons.append('time_snap_out_of_bounds')
        if category == 'execution':
            if item.get('accepted') is not True or item.get('native_success') is not True:
                reasons.append('unconfirmed_execution')
            executions.add(item.get('event_id'))
        if category == 'native_run' and item.get('success') is not True:
            reasons.append('native_run_failed')
    total = record.get('total_play_events')
    if not isinstance(total, int) or total < min_source_events:
        reasons.append('too_few_source_events')
    if not isinstance(total, int) or total <= 0 or len(executions) != total or None in executions:
        reasons.append('execution_count_mismatch')
    # Summary evidence must agree even if a producer dropped detailed records.
    summary = record.get('repair_summary', {})
    for key in ('hand_replacements', 'tile_snaps'):
        if summary.get(key, 0):
            reasons.append('summary:' + key)
    return sorted(set(reasons)), dict(categories)


def select_games(metadata, registry, *, max_snap_ticks=4, min_source_events=20):
    assignments = registry.get('assignments', {})
    seen = set()
    decisions = []
    for index, record in enumerate(metadata['replays']):
        identity = record.get('canonical_game_signature')
        reasons, categories = assess_game(record, max_snap_ticks=max_snap_ticks,
                                          min_source_events=min_source_events)
        partition = assignments.get(identity)
        if not identity or identity in seen:
            reasons.append('missing_or_duplicate_identity')
        seen.add(identity)
        if partition != 'train':
            reasons.append('partition:' + str(partition))
        decisions.append(dict(replay_id=index, identity=identity, partition=partition,
                              selected=not reasons, reasons=sorted(set(reasons)),
                              ledger_categories=categories))
    return decisions
