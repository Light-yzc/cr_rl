from copy import deepcopy

from cr_rl.data.bc_quality import assess_game, select_games


def good():
    return dict(canonical_game_signature='a', game_mode='Ranked',
                source_event_stream_completed=True, all_source_events_executed=True,
                pending_commands_settled=True, coverage=1., missing_source_events=0,
                simulator_fault=False, omitted_commands=0, total_play_events=20,
                repair_records=[dict(category='execution', event_id=0, accepted=True,
                                     native_success=True),
                                dict(category='time_snap', source_tick=1,
                                     simulator_tick=5, delta_ticks=4)] +
                               [dict(category='execution', event_id=i, accepted=True,
                                     native_success=True) for i in range(1, 20)])


def test_small_snap_admitted_but_delay_or_mutation_rejects_whole_game():
    assert assess_game(good())[0] == []
    for category in ('time_delay', 'hand_replacement', 'tile_snap', 'new_unknown_repair'):
        record = good()
        record['repair_records'].append(dict(category=category))
        assert 'ledger:'+category in assess_game(record)[0]


def test_no_future_or_inconsistent_time():
    for delta, actual in ((-1, 0), (5, 6), (4, 6)):
        record = good()
        record['repair_records'][1].update(delta_ticks=delta, simulator_tick=actual)
        assert 'time_snap_out_of_bounds' in assess_game(record)[0]


def test_validation_missing_identity_duplicates_and_missing_evidence_fail_closed():
    records = [good(), deepcopy(good()), good()]
    records[2]['canonical_game_signature'] = 'v'
    result = select_games(dict(replays=records), dict(assignments={'a':'train','v':'validation'}))
    assert [x['selected'] for x in result] == [True, False, False]
    record = good()
    del record['repair_records']
    assert 'missing_execution_ledger' in assess_game(record)[0]


def test_failed_native_or_missing_execution_rejected():
    record = good()
    record['repair_records'][0]['native_success'] = False
    assert 'unconfirmed_execution' in assess_game(record)[0]
    record = good()
    record['total_play_events'] = 2
    assert 'execution_count_mismatch' in assess_game(record)[0]


def test_nearly_empty_replay_is_not_a_quality_demonstration():
    record = good()
    record['total_play_events'] = 2
    assert 'too_few_source_events' in assess_game(record)[0]
