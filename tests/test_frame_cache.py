from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.models.frame_cache import FrameFeatureCache


def test_cache_reuses_frames_matches_full_history_and_invalidates_after_weight_change():
    fields = ("id", "kind", "side", "row", "x", "y", "state", "flags", "buff_0", "buff_1", "buff_2")
    headers = ("entity_count", "hand_card", "deck", "enemy_seen")
    schema = SimpleNamespace(
        entity_fields=fields,
        header_fields=headers,
        header_ints=4,
        entity_ints=11,
        max_entities=4,
        seat_ints=48,
        header={name: (i,) for i, name in enumerate(headers)},
    )
    adapter = CrForgeObservationAdapter(schema)
    cfg = ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=16,
        temporal_heads=2,
        temporal_layers=2,
        context_len=3,
        header_dim=4,
        entity_feature_dim=11,
    ).for_v2(schema)
    model = ActorCritic(cfg).eval()
    cache = FrameFeatureCache(model, adapter, "cpu", capacity=3, encode_batch=2)
    raw = np.zeros((5, 48), dtype=np.int32)
    raw[:, 0] = 2
    raw[:, 1] = np.arange(5) + 1
    raw[:, 4 + 3] = 17
    time_mask = np.ones((1, 3), dtype=bool)

    def compare(start):
        windows = raw[None, start : start + 3]
        h, e, mask = (torch.as_tensor(v) for v in adapter(windows))
        legal = torch.ones(1, 2307, dtype=torch.bool)
        hand = torch.as_tensor(windows[:, -1, 1:2]).expand(-1, 4)
        with torch.inference_mode():
            full = model(h, e, mask, legal, torch.as_tensor(time_mask), hand_card_ids=hand)[0]
            encoded = cache.encode(windows, time_mask)
            cached = model(
                h,
                e,
                mask,
                legal,
                torch.as_tensor(time_mask),
                hand_card_ids=hand,
                encoded_history=encoded,
            )[0]
            torch.testing.assert_close(full, cached, atol=2e-5, rtol=2e-5)
        assert len(cache.entries) <= cache.capacity

    compare(0)
    assert cache.encoded_frames == 3
    compare(1)
    assert cache.encoded_frames == 4
    with torch.no_grad():
        model.frame.entity_in.weight.add_(0.01)
    compare(2)
    assert cache.encoded_frames == 7 and cache.invalidations == 2
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        cache.encode(raw[None, :3], time_mask)


def test_automatic_cache_keeps_all_parallel_rolling_windows():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.cfg = SimpleNamespace(context_len=128, entity_dim=4)

        def encode_frames(self, header, entities, mask):
            return header[:, :1].expand(-1, 4), entities

        def encode_temporal(self, frames, time_mask, **kwargs):
            return frames.mean(1)

    class Adapter:
        schema = SimpleNamespace(header={'deck': (1,), 'enemy_seen': (2,)})

        def __call__(self, raw):
            n = len(raw)
            return raw.astype(np.float32), np.zeros((n, 1, 4), np.float32), np.ones((n, 1), bool)

    cache = FrameFeatureCache(Model().eval(), Adapter(), 'cpu')
    raw = np.zeros((6, 131, 3), np.int32)
    raw[:, :, 0] = np.arange(6 * 131).reshape(6, 131)
    for start in range(3):
        windows = raw[:, start:start + 128]
        context, _, _ = cache.encode(windows, np.ones((6, 128), bool))
        torch.testing.assert_close(context[:, 0], torch.as_tensor(windows[:, :, 0].mean(1)).float())
        assert cache.encoded_frames == 6 * (128 + start)
        assert len(cache.entries) <= cache.capacity == 6 * 129
