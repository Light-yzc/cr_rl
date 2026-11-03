import json

from scripts.run_all_bc import stage_training_plan


def test_streaming_stage_keeps_target_when_producer_finishes_before_resume(tmp_path):
    # Reproduces a Colab restart after the producer finished, while BC was only
    # scheduled for a one-pass streaming stage.
    (tmp_path / 'training_config.json').write_text(
        json.dumps({'command': ['python', 'train_bc.py', '--epochs', '1']})
    )
    plan = stage_training_plan(tmp_path, source_complete=True, final_epochs=8)
    assert plan == {'epochs': 1, 'source_complete_at_start': False}
    assert stage_training_plan(tmp_path, False, 8) == plan


def test_final_snapshot_gets_full_target_and_persists_it_across_resume(tmp_path):
    plan = stage_training_plan(tmp_path, source_complete=True, final_epochs=8)
    assert plan == {'epochs': 8, 'source_complete_at_start': True}
    assert stage_training_plan(tmp_path, False, 1) == plan
