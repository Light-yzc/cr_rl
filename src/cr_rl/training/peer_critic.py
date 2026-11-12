"""A persistent Critic replica that uses the Actor GPU after Actor optimization.

Only the primary owns Adam/scaler state. Replica gradients are summed before the
single global normalization/clipping/update; parameters are then broadcast P2P.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import torch

from cr_rl.training.fp16 import backward_loss
from cr_rl.training.separate import (
    autocast_for,
    bounded_microbatches,
    forward_inputs,
    move_inputs,
)


def buckets(pairs, maximum_bytes=32 * 2**20):
    chunk, size = [], 0
    for pair in pairs:
        nbytes = pair[0].numel() * pair[0].element_size()
        if chunk and size + nbytes > maximum_bytes:
            yield chunk
            chunk, size = [], 0
        chunk.append(pair)
        size += nbytes
    if chunk:
        yield chunk


def transfer_bucket(values, destination):
    flat = torch.cat([value.detach().reshape(-1) for value in values])
    transferred = flat.to(destination, non_blocking=True)
    parts = transferred.split([value.numel() for value in values])
    return [part.view_as(value) for part, value in zip(parts, values, strict=True)]


class PeerCritic:
    def __init__(self, primary, peer_device, peer_microbatch=16):
        if peer_microbatch < 1:
            raise ValueError("Peer microbatch must be positive")
        self.peer_microbatch = peer_microbatch
        self.primary = primary
        self.primary_device = next(primary.parameters()).device
        self.peer_device = torch.device(peer_device)
        if self.peer_device == self.primary_device or not torch.cuda.can_device_access_peer(
            self.primary_device.index, self.peer_device.index
        ):
            raise ValueError("Critic balancing requires CUDA peer access between two GPUs")
        self.replica = copy.deepcopy(primary).to(self.peer_device)
        self.primary_parameters = list(primary.parameters())
        self.peer_parameters = list(self.replica.parameters())
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peer-critic")
        self.actor_done = Event()
        self.dirty = True
        self.rows = self.primary_rows = self.gradient_bytes = self.weight_bytes = 0

    @property
    def ready(self):
        return self.actor_done.is_set()

    def begin_update(self, actor_enabled, optimizer):
        self.rows = self.primary_rows = self.gradient_bytes = self.weight_bytes = 0
        self.replica.zero_grad(set_to_none=True)
        self.replica.train()
        self.dirty = True
        self.actor_done.clear()
        if not actor_enabled:
            self.actor_done.set()
        scaler = getattr(optimizer, "amp_scaler", None)
        if scaler is not None and scaler.is_enabled():
            # Initialize once on the owner thread before concurrent .scale() calls.
            scaler.scale(torch.zeros((), device=self.primary_device))

    @torch.no_grad()
    def ensure_synced(self):
        if not self.dirty:
            return
        for group in buckets(list(zip(self.primary_parameters, self.peer_parameters, strict=True))):
            source, target = zip(*group, strict=True)
            transferred = transfer_bucket(source, self.peer_device)
            torch._foreach_copy_(list(target), transferred)
            self.weight_bytes += sum(value.numel() * value.element_size() for value in source)
        self.dirty = False

    @staticmethod
    def shard(model, device, make_batch, indices, targets, optimizer):
        with torch.cuda.device(device):
            inputs = move_inputs(make_batch(indices), device)
            selected_targets = targets[indices].to(device, non_blocking=True)
            with autocast_for(device):
                prediction = forward_inputs(model, inputs).float()
                loss = 0.5 * (prediction - selected_targets).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite data-parallel value loss")
            backward_loss(loss * len(indices), optimizer)
            return loss.detach()

    def replica_backward(self, make_batch, indices, targets, optimizer):
        # Actor/Adam, frozen reference and opponents remain resident on this GPU.
        # Bound activation memory independently of the primary's larger microbatch.
        losses = []
        for rows, _ in bounded_microbatches(
            indices, self.peer_microbatch, len(indices), make_batch, 16 * 48
        ):
            loss = self.shard(self.replica, self.peer_device, make_batch, rows, targets, optimizer)
            losses.append(loss * len(rows))
        return sum(losses) / len(indices)

    def loss_and_backward(self, make_batch, indices, targets, optimizer):
        if not self.ready or len(indices) == 1:
            self.primary_rows += len(indices)
            return self.shard(
                self.primary, self.primary_device, make_batch, indices, targets, optimizer
            )
        self.ensure_synced()
        middle = (len(indices) + 1) // 2
        own, peer = indices[:middle], indices[middle:]
        future = self.executor.submit(self.replica_backward, make_batch, peer, targets, optimizer)
        try:
            own_loss = self.shard(
                self.primary, self.primary_device, make_batch, own, targets, optimizer
            )
        finally:
            # A failed primary still waits for the peer; no background model mutation escapes.
            peer_loss = future.result()
        self.primary_rows += len(own)
        self.rows += len(peer)
        return (own_loss * len(own) + peer_loss.to(self.primary_device) * len(peer)) / len(indices)

    @torch.no_grad()
    def reduce_gradients(self):
        pairs = []
        for primary, replica in zip(self.primary_parameters, self.peer_parameters, strict=True):
            if replica.grad is not None:
                if primary.grad is None:
                    primary.grad = torch.zeros_like(primary)
                pairs.append((replica.grad, primary.grad))
        for group in buckets(pairs):
            source, target = zip(*group, strict=True)
            transferred = transfer_bucket(source, self.primary_device)
            torch._foreach_add_(list(target), transferred)
            self.gradient_bytes += sum(value.numel() * value.element_size() for value in source)
        self.replica.zero_grad(set_to_none=True)

    def primary_updated(self):
        self.dirty = True

    def metadata(self):
        return {
            "enabled": True,
            "peer_microbatch": self.peer_microbatch,
            "peer_entity_budget": 16 * 48,
            "peer_rows": self.rows,
            "primary_rows": self.primary_rows,
            "gradient_p2p_bytes": self.gradient_bytes,
            "weight_p2p_bytes": self.weight_bytes,
        }

    def close(self):
        self.executor.shutdown(wait=True)
