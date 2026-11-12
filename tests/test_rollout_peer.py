import numpy as np
import pytest
import torch
from test_separate_critic import setup_models

from cr_rl.training.rollout_peer import actor_replica, actor_rows, merge_stats
from cr_rl.training.separate import forward_inputs


def test_stable_stream_partition_and_weighted_stats():
    indices = np.array([0, 1, 3, 4, 6, 7])
    own, peer = actor_rows(indices, True)
    np.testing.assert_array_equal(own, [0, 1, 4])
    np.testing.assert_array_equal(peer, [3, 6, 7])
    own, peer = actor_rows(indices, False)
    np.testing.assert_array_equal(own, indices)
    assert not len(peer)
    assert merge_stats({"entropy": 2.0}, 1, {"entropy": 4.0}, 3) == {"entropy": 3.5}
    assert merge_stats({}, 0, {"entropy": 4.0}, 3) == {"entropy": 4.0}


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
def test_replica_preserves_weights_and_old_action_logprobs(monkeypatch):
    monkeypatch.setenv("CRRL_AMP_DTYPE", "fp16")
    _, actor, _, inputs = setup_models()
    actor = actor.to("cuda:0").eval()
    replica = actor_replica(actor, None, "cuda:1", False)
    inputs0 = tuple(t.to("cuda:0") for t in inputs)
    inputs1 = tuple(t.to("cuda:1") for t in inputs)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        primary, _ = forward_inputs(actor, inputs0)
        peer, _ = forward_inputs(replica.model, inputs1)
    torch.testing.assert_close(primary.cpu(), peer.cpu(), rtol=0, atol=1e-3)
    for a, b in zip(actor.parameters(), replica.model.parameters(), strict=True):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        assert not b.requires_grad


def test_training_cache_trim_preserves_most_recent_models():
    from collections import OrderedDict

    from cr_rl.training.rollout_peer import trim_opponent_cache

    cache = OrderedDict((i, object()) for i in range(8))
    assert trim_opponent_cache(cache, None) == 0
    assert trim_opponent_cache(cache, 2) == 6
    assert list(cache) == [6, 7]
    assert trim_opponent_cache(cache, 2) == 0
    with pytest.raises(ValueError):
        trim_opponent_cache(cache, 0)
