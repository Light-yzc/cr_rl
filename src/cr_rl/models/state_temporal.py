"""Causal matrix-delta memory interleaved with full-history grouped attention.

This is an independent temporal core, not a checkpoint-compatible replacement
for the windowed Transformer. KDA matrix arithmetic and normalization accumulate
in FP32 even when projections, activations and the GQA KV cache use FP16. The
implementation uses ordinary PyTorch operations and has no Triton/FLA dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class StateTemporalConfig:
    width: int = 768
    heads: int = 12
    kv_heads: int = 3
    head_dim: int = 64
    pattern: tuple[str, ...] = ("kda", "kda", "kda", "gqa") * 2
    ffn_multiplier: float = 8 / 3
    chunk_size: int = 32
    cache_initial_capacity: int = 128
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        if min(self.width, self.heads, self.kv_heads, self.head_dim) <= 0:
            raise ValueError("Temporal dimensions must be positive")
        if self.width != self.heads * self.head_dim:
            raise ValueError("width must equal heads * head_dim")
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("GQA requires heads divisible by kv_heads and even head_dim")
        if not self.pattern or any(p not in ("kda", "gqa") for p in self.pattern):
            raise ValueError("pattern must contain kda and/or gqa blocks")
        if self.chunk_size <= 0 or self.cache_initial_capacity <= 0:
            raise ValueError("chunk size and cache initial capacity must be positive")
        if self.ffn_multiplier <= 0 or self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("FFN multiplier, RoPE theta and norm epsilon must be positive")


@dataclass
class KDAState:
    matrix: torch.Tensor  # [batch, heads, key_dim, value_dim], always FP32

    def detach(self) -> KDAState:
        return KDAState(self.matrix.detach())


@dataclass
class GQAState:
    key: torch.Tensor  # [batch, kv_heads, allocated_capacity, head_dim]
    value: torch.Tensor

    def detach(self) -> GQAState:
        return GQAState(self.key.detach(), self.value.detach())


@dataclass
class TemporalState:
    layers: list[KDAState | GQAState]
    positions: torch.Tensor  # Valid token count per independent stream.
    length_bound: int = 0  # CPU upper bound, avoids a GPU max().item() each step.

    def detach(self) -> TemporalState:
        """Keep all history, but truncate gradients at this chunk boundary."""
        return TemporalState([s.detach() for s in self.layers], self.positions.detach(), self.length_bound)

    def reset(self, mask: torch.Tensor | None = None) -> TemporalState:
        """Clear selected streams; their stale KV storage is never read again.

        A reset does not mutate the supplied state. New state is required for each
        environment, seat and model; this helper only supports batched resets.
        """
        if mask is None:
            mask = torch.ones_like(self.positions, dtype=torch.bool)
            bound = 0
        else:
            mask = mask.to(device=self.positions.device, dtype=torch.bool)
            if mask.shape != self.positions.shape:
                raise ValueError("reset mask must have shape [batch]")
            bound = self.length_bound
        layers = []
        for state in self.layers:
            if isinstance(state, KDAState):
                layers.append(KDAState(state.matrix * (~mask)[:, None, None, None]))
            else:
                # Positions define the only accessible KV prefix. Keeping the
                # allocation also avoids an allocation on each episode reset.
                layers.append(state)
        return TemporalState(layers, self.positions.masked_fill(mask, 0), bound)


def _check_kda_inputs(q, k, v, log_decay, beta, state, time_mask):
    if q.ndim != 4 or k.shape != q.shape or log_decay.shape != q.shape:
        raise ValueError("KDA q, k and log_decay must be [batch, heads, time, key_dim]")
    if v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("KDA value/beta dimensions do not match q")
    expected = (q.shape[0], q.shape[1], q.shape[-1], v.shape[-1])
    if state is not None and state.shape != expected:
        raise ValueError("KDA initial matrix shape does not match q/value dimensions")
    if time_mask is not None and time_mask.shape != (q.shape[0], q.shape[2]):
        raise ValueError("KDA time mask must be [batch, time]")


def _prepare_kda(q, k, v, log_decay, beta, state, time_mask):
    _check_kda_inputs(q, k, v, log_decay, beta, state, time_mask)
    q, k, v, log_decay, beta = (t.float() for t in (q, k, v, log_decay, beta))
    if state is None:
        state = q.new_zeros(q.shape[0], q.shape[1], q.shape[-1], v.shape[-1])
    else:
        state = state.float()
    if time_mask is not None:
        valid = time_mask[:, None, :, None].to(device=q.device, dtype=torch.bool)
        # Padding neither forgets nor writes memory and produces zero output.
        q = q.masked_fill(~valid, 0)
        k = k.masked_fill(~valid, 0)
        v = v.masked_fill(~valid, 0)
        log_decay = log_decay.masked_fill(~valid, 0)
        beta = beta.masked_fill(~valid.squeeze(-1), 0)
    return q, k, v, log_decay, beta, state


def kda_recurrent(q, k, v, log_decay, beta, state=None, time_mask=None):
    """Reference FP32 matrix-delta recurrence; q/k are already normalized.

    S_bar = diag(exp(log_decay)) S
    S = S_bar + beta k (v - S_bar.T k).T; output = S.T q.
    The function is differentiable through every time step and initial matrix.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _kda_recurrent_fp32(q, k, v, log_decay, beta, state, time_mask)


def _kda_recurrent_fp32(q, k, v, log_decay, beta, state, time_mask):
    q, k, v, log_decay, beta, state = _prepare_kda(
        q, k, v, log_decay, beta, state, time_mask
    )
    outputs = []
    for t in range(q.shape[2]):
        decayed = state * log_decay[:, :, t].exp().unsqueeze(-1)
        predicted = torch.einsum("bhkv,bhk->bhv", decayed, k[:, :, t])
        residual = beta[:, :, t, None] * (v[:, :, t] - predicted)
        state = decayed + k[:, :, t, :, None] * residual[:, :, None, :]
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, :, t]))
    output = torch.stack(outputs, dim=2) if outputs else v[:, :, :0]
    return output, state


def kda_chunk(q, k, v, log_decay, beta, state=None, time_mask=None, chunk_size=32, checkpoint_chunks=False):
    """Exact block-parallel delta rule using a lower-triangular solve.

    Within a chunk, residual writes solve (I + C) e = beta (v - k A S0),
    where C[t,j] = beta[t] <k[t], A[t]/A[j] k[j]> for j<t. Decay
    ratios are computed in log space only in the forward direction, avoiding
    division by underflowed products. Chunk boundaries retain the full matrix
    graph; callers can explicitly detach the returned state for truncated BPTT.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _kda_chunk_fp32(q, k, v, log_decay, beta, state, time_mask, chunk_size, checkpoint_chunks)


def _kda_chunk_fp32(q, k, v, log_decay, beta, state, time_mask, chunk_size, checkpoint_chunks=False):
    if chunk_size <= 0:
        raise ValueError("KDA chunk_size must be positive")
    q, k, v, log_decay, beta, state = _prepare_kda(
        q, k, v, log_decay, beta, state, time_mask
    )
    outputs = []
    for start in range(0, q.shape[2], chunk_size):
        stop = min(start + chunk_size, q.shape[2])
        qc, kc, vc = q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop]
        bc = beta[:, :, start:stop]
        if checkpoint_chunks and torch.is_grad_enabled():
            output, state = checkpoint(_kda_block, qc, kc, vc,
                log_decay[:, :, start:stop], bc, state, use_reentrant=False)
        else:
            output, state = _kda_block(qc, kc, vc, log_decay[:, :, start:stop], bc, state)
        outputs.append(output)
    return (torch.cat(outputs, dim=2) if outputs else v[:, :, :0]), state


def _kda_block(qc, kc, vc, decay, bc, state):
    cumulative = decay.cumsum(dim=2)
    length = qc.shape[2]
    causal = torch.ones(length, length, dtype=torch.bool, device=qc.device).tril()
    # [batch, heads, query_time, write_time, key_dim]. Upper triangle
    # is explicitly zero; clamp prevents overflow before applying its mask.
    ratio = (cumulative.unsqueeze(-2) - cumulative.unsqueeze(-3)).clamp_max(0).exp()
    ratio = ratio * causal[None, None, :, :, None]
    key_interaction = (kc.unsqueeze(-2) * kc.unsqueeze(-3) * ratio).sum(-1)
    lower = (bc.unsqueeze(-1) * key_interaction).tril(diagonal=-1)
    identity = torch.eye(length, dtype=torch.float32, device=qc.device)
    initial_prediction = torch.matmul(kc * cumulative.exp(), state)
    rhs = bc.unsqueeze(-1) * (vc - initial_prediction)
    writes = torch.linalg.solve_triangular(identity + lower, rhs, upper=False, unitriangular=True)
    read_weights = (qc.unsqueeze(-2) * kc.unsqueeze(-3) * ratio).sum(-1)
    output = torch.matmul(qc * cumulative.exp(), state) + torch.matmul(read_weights, writes)
    final_decay = cumulative[:, :, -1]
    decayed_keys = kc * (final_decay.unsqueeze(-2) - cumulative).clamp_max(0).exp()
    state = state * final_decay.exp().unsqueeze(-1) + torch.matmul(decayed_keys.transpose(-1, -2), writes)
    return output, state


class _RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        result = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (result * self.weight.float()).to(dtype)


class _KDALayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.decay = nn.Linear(cfg.width, cfg.width, bias=True)
        self.beta = nn.Linear(cfg.width, cfg.heads, bias=True)
        self.gate = nn.Linear(cfg.width, cfg.width, bias=False)
        self.output_norm = _RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.width, cfg.width, bias=False)
        nn.init.constant_(self.decay.bias, -4.0)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, state, positions, time_mask, capacity, reference=False, fresh_prefix=False):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.cfg.heads, self.cfg.head_dim).unbind(2)
        q = F.normalize(q.float(), dim=-1, eps=self.cfg.norm_eps).transpose(1, 2)
        k = F.normalize(k.float(), dim=-1, eps=self.cfg.norm_eps).transpose(1, 2)
        v = v.transpose(1, 2)
        log_decay = -F.softplus(self.decay(x).float()).reshape(
            batch, length, self.cfg.heads, self.cfg.head_dim
        ).transpose(1, 2)
        beta = self.beta(x).float().sigmoid().transpose(1, 2)
        # Single-token inference is already a constant-size matrix update; a
        # one-by-one triangular solve would add unnecessary kernel launches.
        recurrent = reference or length == 1
        function = kda_recurrent if recurrent else kda_chunk
        kwargs = {} if recurrent else {"chunk_size": self.cfg.chunk_size,
            "checkpoint_chunks": bool(getattr(self, "gradient_checkpointing", False))}
        output, matrix = function(q, k, v, log_decay, beta, state.matrix, time_mask, **kwargs)
        output = self.output_norm(output.transpose(1, 2)).reshape(batch, length, self.cfg.width)
        output = output * self.gate(x).float().sigmoid()
        return self.output(output.to(x.dtype)), KDAState(matrix)


class _GQALayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.query = nn.Linear(cfg.width, cfg.width, bias=False)
        self.key_value = nn.Linear(cfg.width, 2 * cfg.kv_heads * cfg.head_dim, bias=False)
        self.output = nn.Linear(cfg.width, cfg.width, bias=False)
        self.register_buffer("inv_frequency", cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim), persistent=False)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        # model.half() must not quantize RoPE frequencies before long-position
        # multiplication. Rebuild rather than upcasting an already rounded copy.
        self.inv_frequency = self.cfg.rope_theta ** (
            -torch.arange(0, self.cfg.head_dim, 2, dtype=torch.float32,
                          device=self.inv_frequency.device) / self.cfg.head_dim
        )
        return result

    def _rope(self, x, positions):
        angles = positions[:, None, :, None].float() * self.inv_frequency.float()[None, None, None]
        cosine, sine = angles.cos(), angles.sin()
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2).to(x.dtype)

    @staticmethod
    def _append(old, new, positions, mask, capacity):
        if old.dtype != new.dtype:
            old = old.to(dtype=new.dtype)
        if old.shape[2] < capacity:
            old = F.pad(old, (0, 0, 0, capacity - old.shape[2]))
        slots = torch.arange(capacity, device=old.device)
        # Remove storage past the old prefix, including a reset stream's stale
        # entries. scatter_add safely ignores repeated padded indices via zeros.
        old = old * (slots[None, None, :, None] < positions[:, None, None, None])
        indices = positions[:, None] + mask.long().cumsum(-1) - 1
        indices = indices.clamp_min(0)[:, None, :, None].expand_as(new)
        values = new * mask[:, None, :, None]
        return old.scatter_add(2, indices, values)

    def forward(self, x, state, positions, time_mask, capacity, reference=False, fresh_prefix=False):
        batch, length, _ = x.shape
        q = self.query(x).reshape(batch, length, self.cfg.heads, self.cfg.head_dim).transpose(1, 2)
        k, v = self.key_value(x).reshape(batch, length, 2, self.cfg.kv_heads, self.cfg.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        token_positions = positions[:, None] + time_mask.long().cumsum(-1) - 1
        q = self._rope(q, token_positions.clamp_min(0))
        k = self._rope(k, token_positions.clamp_min(0))
        key = self._append(state.key, k, positions, time_mask, capacity)
        value = self._append(state.value, v, positions, time_mask, capacity)
        slots = torch.arange(capacity, device=x.device)
        allowed = (slots[None, None, :] <= token_positions[:, :, None]) & time_mask[:, :, None]
        attention_key, attention_value = key, value
        if fresh_prefix and length > 1:
            # Whole-game prefill attends to actual tokens, not the power-of-two
            # streaming cache allocation. Keep the cache output for step parity.
            indices = torch.arange(length, device=x.device)
            allowed = ((indices[None, :] <= indices[:, None])[None]
                       & time_mask[:, :, None] & time_mask[:, None, :])
            attention_key, attention_value = k, v
        # This explicit KV-head expansion works on CPU, V100 and 4090 without
        # assuming availability of a newer fused GQA kernel.
        repeats = self.cfg.heads // self.cfg.kv_heads
        output = F.scaled_dot_product_attention(
            q, attention_key.repeat_interleave(repeats, dim=1), attention_value.repeat_interleave(repeats, dim=1),
            attn_mask=allowed[:, None], dropout_p=0.0,
        )
        output = output.transpose(1, 2).reshape(batch, length, self.cfg.width)
        return self.output(output), GQAState(key, value)


class _TemporalBlock(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.norm = _RMSNorm(cfg.width, cfg.norm_eps)
        self.mixer = _KDALayer(cfg) if kind == "kda" else _GQALayer(cfg)
        self.ffn_norm = _RMSNorm(cfg.width, cfg.norm_eps)
        hidden = math.ceil(cfg.width * cfg.ffn_multiplier / 8) * 8
        self.ffn_in = nn.Linear(cfg.width, 2 * hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, cfg.width, bias=False)

    def forward(self, x, state, positions, time_mask, capacity, reference=False, fresh_prefix=False):
        mixed, state = self.mixer(self.norm(x), state, positions, time_mask, capacity, reference, fresh_prefix)
        x = x + mixed
        gate, value = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=-1)
        x = x + self.ffn_out(F.silu(gate) * value)
        return x * time_mask[:, :, None], state


class StateTemporalCore(nn.Module):
    """KDA/GQA stack with independent, explicit stream memory.

    ``forward`` starts from an empty game. ``prefill`` continues an existing
    state and retains gradients unless detach_state=True. ``step`` is the same
    computation for one new token. No sliding-window truncation occurs. Cached
    state belongs to the weights that created it: after an optimizer update it
    must be rebuilt, or its approximation must be managed by the caller.
    """

    def __init__(self, cfg: StateTemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([_TemporalBlock(cfg, kind) for kind in cfg.pattern])
        self.final_norm = _RMSNorm(cfg.width, cfg.norm_eps)

    def init_state(self, batch, device=None, dtype=None):
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        layers = []
        for kind in self.cfg.pattern:
            if kind == "kda":
                layers.append(KDAState(torch.zeros(batch, self.cfg.heads, self.cfg.head_dim,
                                                  self.cfg.head_dim, device=device, dtype=torch.float32)))
            else:
                shape = (batch, self.cfg.kv_heads, self.cfg.cache_initial_capacity, self.cfg.head_dim)
                layers.append(GQAState(torch.zeros(shape, device=device, dtype=dtype),
                                       torch.zeros(shape, device=device, dtype=dtype)))
        return TemporalState(layers, torch.zeros(batch, device=device, dtype=torch.long))

    def forward(self, x, time_mask=None):
        return self.prefill(x, time_mask=time_mask)[0]

    def step(self, x, state, time_mask=None):
        if x.ndim != 2:
            raise ValueError("step expects [batch, width]")
        mask = None if time_mask is None else time_mask[:, None]
        output, state = self.prefill(x[:, None], state, mask)
        return output[:, 0], state

    def prefill(self, x, state=None, time_mask=None, detach_state=False, reference=False):
        if x.ndim != 3 or x.shape[-1] != self.cfg.width:
            raise ValueError("prefill expects [batch, time, width]")
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        if len(state.layers) != len(self.blocks) or state.positions.shape != (x.shape[0],):
            raise ValueError("Temporal state does not match this model/batch")
        if time_mask is None:
            time_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            if time_mask.shape != x.shape[:2]:
                raise ValueError("time_mask must have shape [batch, time]")
            time_mask = time_mask.to(device=x.device, dtype=torch.bool)
        if not x.shape[1]:
            return x, state.detach() if detach_state else state
        new_positions = state.positions + time_mask.long().sum(-1)
        bound = state.length_bound + x.shape[1]
        capacity = max((s.key.shape[2] for s in state.layers if isinstance(s, GQAState)), default=0)
        if capacity and bound > capacity:
            # Only synchronize on an allocation boundary. Resets/padding may
            # make the CPU upper bound pessimistic, so first recover its value.
            bound = int(new_positions.max().item())
            while capacity < bound:
                capacity *= 2
        x = x * time_mask[:, :, None]
        layers = []
        for block, memory in zip(self.blocks, state.layers):
            block.mixer.gradient_checkpointing = (bool(getattr(self, "gradient_checkpointing", False))
                and bool(getattr(self, "checkpoint_kda_chunks", True)))
            if self.training and torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
                # Non-reentrant checkpoint supports our explicit dataclass memory outputs.
                x, memory = checkpoint(block, x, memory, state.positions, time_mask,
                                       capacity, reference, state.length_bound == 0, use_reentrant=False)
            else:
                x, memory = block(x, memory, state.positions, time_mask, capacity, reference, state.length_bound == 0)
            layers.append(memory)
        output = self.final_norm(x) * time_mask[:, :, None]
        result = TemporalState(layers, new_positions, bound)
        return output, result.detach() if detach_state else result
