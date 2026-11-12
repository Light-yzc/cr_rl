from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.training.episode_trace import EpisodeTrace
from cr_rl.training.host_memory import detach_optimizer_counters


def test_shared_trace_preserves_values_and_detaches_full_lane_buffers():
    raw = np.arange(4 * 2 * 12).reshape(4, 2, 12)
    masks = np.ones((4, 2, 7), dtype=bool)
    trace = EpisodeTrace(1, 42)
    trace.append(raw[2], masks[2], np.zeros(2, dtype=np.float32), copy=False)
    assert np.shares_memory(trace.observations[0], raw)
    assert np.shares_memory(trace.masks[0], masks)
    expected = raw[2].copy()
    trace.detach_views()
    assert not np.shares_memory(trace.observations[0], raw)
    assert trace.observations[0].base is None
    raw[:] = -1
    np.testing.assert_array_equal(trace.observations[0], expected)


def test_default_trace_still_copies_mutable_input():
    raw = np.ones((2, 12))
    trace = EpisodeTrace(1, 42)
    trace.append(raw, np.ones((2, 7)), [0, 0])
    raw[:] = 0
    assert trace.observations[0].all()


def test_optimizer_counters_do_not_keep_checkpoint_storage_alive(tmp_path):
    path = tmp_path / "state.pt"
    torch.save({"step": torch.tensor(15.0), "moment": torch.arange(4.0)}, path)
    source = torch.load(path, mmap=True, weights_only=True)
    optimizer = SimpleNamespace(state=defaultdict(dict, {0: dict(source)}))
    detach_optimizer_counters(optimizer)
    assert optimizer.state[0]["step"].data_ptr() != source["step"].data_ptr()
    optimizer.state[0]["step"].add_(1)
    assert source["step"].item() == 15
    assert optimizer.state[0]["step"].item() == 16
    assert optimizer.state[0]["moment"].data_ptr() == source["moment"].data_ptr()
