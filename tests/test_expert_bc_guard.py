import json

from cr_rl.training.expert_bc_guard import plan_b_health


def test_plan_b_health_checks_previous_ppo_and_latest_argmax(tmp_path):
    summary = tmp_path / 'summary.json'
    summary.write_text(json.dumps(dict(update=1384, metrics=dict(entropy=.85, anchor_kl=.80))))
    assert plan_b_health(tmp_path, 1385)['passed']
    panel = tmp_path / 'validation' / 'update_00001375'
    panel.mkdir(parents=True)
    (panel / 'summary.json').write_text(json.dumps({
        'bc': {'action_mode': 'argmax', 'mean_card_request_interval_seconds': 11.01}
    }))
    assert plan_b_health(tmp_path, 1385)['reasons'] == ['recent_argmax_card_request_interval_above_limit']
    summary.write_text(json.dumps(dict(update=1384, metrics=dict(entropy=.84, anchor_kl=.81))))
    assert len(plan_b_health(tmp_path, 1385)['reasons']) == 3
    assert 'missing_previous_entropy' in plan_b_health(tmp_path, 1390)['reasons']
