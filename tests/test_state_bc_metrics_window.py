"""Weighted ratios and logging intervals do not average incompatible denominators."""
import pytest
from cr_rl.training.state_bc_metrics import BCMetricWindow


def seat(play,correct,distance=0,loss=2):
    return {'rows':play+10,'act_rows':play,'play_rows':play,'wait_rows':10,
        'history_frames':play+10,'slot_accuracy':correct/play if play else 0,
        'tile_mean_distance':distance,'exact_accuracy':.5,'wait_accuracy':.8,
        'act_recall':.5,'ability_accuracy':0,**dict.fromkeys(BCMetricWindow.losses,loss)}


def report(seats,overflow=False):
    return {'reports':seats,'seconds':2,'grad_norm':3,'optimizer_stepped':not overflow,
            'amp_overflow_skipped':overflow}


def test_slot_and_distance_are_weighted_by_play_rows_not_seat_count():
    window=BCMetricWindow(1)
    window.add(report([seat(1,1,0,1),seat(9,0,10,3),seat(0,0,0,5)]),
               epoch=1,episode_index=2,optimizer_steps=1)
    result=window.flush()
    assert result['slot_accuracy']==pytest.approx(.1)
    assert result['tile_mean_distance']==pytest.approx(9)
    assert result['loss']==3  # Preserve equal-seat training-objective averaging.
    assert result['play_rows']==10 and result['window_seats']==3
    assert result['episode_index']==2 and result['episode_index_start']==0


def test_no_play_window_omits_undefined_ratios_instead_of_reporting_zero():
    window=BCMetricWindow(1)
    window.add(report([seat(0,0)]),epoch=1,episode_index=0,optimizer_steps=1)
    result=window.flush()
    for name in ('slot_accuracy','tile_mean_distance','act_recall','ability_accuracy'):
        assert name not in result
    assert result['play_rows']==0 and result['wait_accuracy']==.8


def test_32_successful_updates_produce_one_256_seat_record():
    window=BCMetricWindow(32,initial_step=10)
    for index in range(32):
        window.add(report([seat(2,1) for _ in range(8)]),
            epoch=3,episode_index=index*8+7,optimizer_steps=11+index)
        assert window.ready()==(index==31)
    result=window.flush()
    assert result['window_optimizer_updates']==32 and result['window_seats']==256
    assert result['play_rows']==512 and result['slot_accuracy']==.5
    assert result['optimizer_steps']==42 and result['episode_index']==255
    assert window.flush() is None


def test_overflow_is_reported_immediately_and_partial_checkpoint_window_can_flush():
    window=BCMetricWindow(32,initial_step=100)
    window.add(report([seat(3,2)],True),epoch=2,episode_index=11,optimizer_steps=100)
    assert window.ready()
    first=window.flush()
    assert first['amp_overflow_skipped'] and first['window_optimizer_updates']==0
    window.add(report([seat(3,2)]),epoch=2,episode_index=12,optimizer_steps=101)
    assert not window.ready()
    assert window.flush()['slot_accuracy']==pytest.approx(2/3)
