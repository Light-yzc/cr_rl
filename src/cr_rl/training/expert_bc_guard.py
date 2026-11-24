"""Check the last PPO update before starting the specialist BC phase."""

import json
import math
from pathlib import Path


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def plan_b_health(
    out: Path,
    update: int,
    *,
    min_entropy: float = 0.85,
    max_anchor_kl: float = 0.80,
) -> dict:
    reasons = []
    try:
        previous = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        previous = {}
    metrics = previous.get('metrics', {}) if previous.get('update') == update - 1 else {}
    entropy = metrics.get('entropy')
    anchor_kl = metrics.get('anchor_kl')
    if not _finite_number(entropy):
        reasons.append('missing_previous_entropy')
        entropy = None
    elif entropy < min_entropy:
        reasons.append('previous_entropy_below_limit')
    if not _finite_number(anchor_kl):
        reasons.append('missing_previous_anchor_kl')
        anchor_kl = None
    elif anchor_kl > max_anchor_kl:
        reasons.append('previous_anchor_kl_above_limit')
    report = dict(previous_ppo_update=update - 1, entropy=entropy, anchor_kl=anchor_kl,
                  argmax_card_request_interval_seconds=None)
    for path in sorted((out / 'validation').glob('update_*/summary.json'), reverse=True):
        try:
            validation_update = int(path.parent.name.removeprefix('update_'))
            if validation_update >= update:
                continue
            anchor = json.loads(path.read_text(encoding='utf-8')).get('bc', {})
        except (FileNotFoundError, ValueError):
            continue
        interval = anchor.get('mean_card_request_interval_seconds')
        if anchor.get('action_mode') != 'argmax':
            continue
        if not _finite_number(interval):
            if anchor.get('games', 0) > 0:
                report['argmax_update'] = validation_update
                reasons.append('recent_argmax_missing_card_request_interval')
                break
            continue
        report.update(argmax_update=validation_update, argmax_card_request_interval_seconds=interval)
        if interval > 11.0:
            reasons.append('recent_argmax_card_request_interval_above_limit')
        break
    return dict(report, passed=not reasons, reasons=reasons)
