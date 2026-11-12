"""Frozen Actor replica and weighted diagnostics for two-device rollout inference."""

import copy

import torch

from cr_rl.training.inference import FrozenInference


def actor_rows(indices, enabled):
    # Keep stream ownership stable across steps, so incremental caches remain useful.
    peer = ((indices // 2) % 2 == 1) if enabled else indices < 0
    return indices[~peer], indices[peer]


def merge_stats(primary, primary_count, peer, peer_count):
    if not primary:
        return peer
    if not peer:
        return primary
    return {
        key: (primary[key] * primary_count + peer[key] * peer_count) / (primary_count + peer_count)
        for key in primary
    }


def actor_replica(actor, adapter, device, fastpath):
    if torch.device(device) == next(actor.parameters()).device:
        raise ValueError("Rollout replica requires a second device")
    model = copy.deepcopy(actor).to(device).eval().requires_grad_(False)
    return FrozenInference(model, adapter, device, forced_wait_fastpath=fastpath)


def trim_opponent_cache(cache, limit):
    """Release frozen models once collection is finished, retaining recent entries."""
    if limit is None:
        return 0
    if limit < 1:
        raise ValueError("Training opponent cache limit must be positive")
    evicted = 0
    while len(cache) > limit:
        cache.popitem(last=False)
        evicted += 1
    return evicted
