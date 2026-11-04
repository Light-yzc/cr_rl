import numpy as np
import pytest
import torch

from cr_rl.training.reference_cache import ReferenceDistributionCache


def test_reference_is_lazy_reorders_hits_and_keeps_training_features_uncached():
    cache = ReferenceDistributionCache(4)
    original = torch.randn(4, 7, requires_grad=True)
    calls = []

    def evaluate(rows):
        def forward():
            calls.append(list(rows))
            return original[rows]

        return cache.evaluate(rows, forward)

    torch.testing.assert_close(evaluate([2, 0]), original[[2, 0]])
    torch.testing.assert_close(evaluate([0, 2]), original[[0, 2]])
    assert calls == [[2, 0]]
    assert not evaluate([2]).requires_grad
    # A partial miss computes the unchanged entire batch, never a different crop.
    torch.testing.assert_close(evaluate([1, 2]), original[[1, 2]])
    assert calls[-1] == [1, 2]
    assert cache.forward_calls == 2 and cache.hit_rows == 3
    with pytest.raises(IndexError):
        evaluate([4])


def test_reference_budget_falls_back_without_extra_model_calls():
    cache = ReferenceDistributionCache(6, max_bytes=4)
    for _ in range(2):
        result = cache.evaluate(np.array([0, 1]), lambda: torch.ones(2, 4))
        assert result.shape == (2, 4)
    assert cache.disabled and cache.values is None and cache.forward_calls == 2
