import importlib
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from cr_rl.models import ActorPolicy, ModelConfig


def test_evaluation_wait_bias_preserves_conditional_actions_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    evaluator = importlib.import_module('evaluate_policy')
    cfg = ModelConfig.preset('tiny')
    model = ActorPolicy(cfg)
    path = tmp_path / 'actor.pt'
    torch.save({'model': model.state_dict(), 'config': asdict(cfg),
                'model_architecture_version': 2, 'policy_only': True}, path)
    original_bytes = path.read_bytes()
    plain, _ = evaluator.load_policy(path, 'cpu')
    biased, _ = evaluator.load_policy(path, 'cpu', 2.5)
    torch.testing.assert_close(biased.wait_actor.bias, plain.wait_actor.bias + 2.5)
    for key, tensor in plain.state_dict().items():
        if key != 'wait_actor.bias':
            torch.testing.assert_close(biased.state_dict()[key], tensor)
    card = torch.randn(3, 7)
    tile = torch.randn(3, 4, 576)
    legal = torch.rand(3, 2307) > 0.2
    legal[:, 0] = True
    shifted = card.clone()
    shifted[:, 0] += 2.5
    p = model._joint_log_probs(card, tile, legal)
    q = model._joint_log_probs(shifted, tile, legal)
    torch.testing.assert_close(
        q[:, 0] - torch.logsumexp(q[:, 1:], -1),
        p[:, 0] - torch.logsumexp(p[:, 1:], -1) + 2.5,
    )
    torch.testing.assert_close(
        q[:, 1:] - torch.logsumexp(q[:, 1:], -1, keepdim=True),
        p[:, 1:] - torch.logsumexp(p[:, 1:], -1, keepdim=True),
    )
    assert path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match='finite'):
        evaluator.load_policy(path, 'cpu', float('nan'))
