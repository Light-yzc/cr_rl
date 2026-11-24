"""Production PPO optimization engine for explicit-state architecture-v3 models."""
from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist

from cr_rl.models.state_policy import (
    PolicyMemory,
    clone_memory,
    rebind_memory,
    slice_memory,
    stack_memories,
)
from cr_rl.training.advantages import compute_actor_critic_targets
from cr_rl.training.fp16 import backward_loss
from cr_rl.training.ppo import (
    factorized_entropy,
    factorized_reverse_kl,
    stop_for_factorized_kl,
    stop_for_kl,
)
from cr_rl.training.sample_selection import normalize_actor_advantages
from cr_rl.training.separate import (
    _optimizer_step,
    autocast_for,
    complete_episode_returns,
)


@dataclass
class StatePPOConfig:
    """Hyperparameters for architecture-v3 PPO with TBPTT and factorized guards."""

    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    precision: str = "fp16"
    max_initial_logp_error: float | None = 0.05
    epochs: int = 2
    critic_epochs: int = 4
    gamma: float = 0.9995
    gae_lambda: float = 0.98
    critic_gae_lambda: float | None = None
    card_entropy_coef: float | None = None
    tile_entropy_coef: float | None = None
    reference_kl_coef: float = 0.01
    anchor_kl_coef: float = 0.0
    target_kl: float = 0.02
    target_card_kl: float | None = None
    target_tile_kl: float | None = None
    continue_after_kl_reject: bool = False
    actor_choice_only: bool = False
    full_episode_bptt: bool = False
    tbptt_chunk: int = 32
    actor_batch_size: int = 4
    critic_batch_size: int = 4
    actor_effective_batch_size: int = 256
    critic_effective_batch_size: int = 256
    critic_max_grad_norm: float = 0.5
    cache_reference: bool = True
    defer_scalar_metrics: bool = True


def _amp_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return autocast_for(device)


_CUDA_LINALG_LOCK = threading.Lock()
_CUDA_LINALG_WARMED: set[int] = set()


def ensure_cuda_linalg_initialized(device: torch.device | str | None) -> None:
    """Warm PyTorch's one-shot lazy `torch.linalg` CUDA backend before worker threads start."""
    if device is None:
        return
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    idx = 0 if dev.index is None else int(dev.index)
    if idx in _CUDA_LINALG_WARMED:
        return
    with _CUDA_LINALG_LOCK:
        if idx in _CUDA_LINALG_WARMED:
            return
        with torch.cuda.device(dev):
            eye = torch.eye(2, device=dev, dtype=torch.float32)
            rhs = torch.ones(2, 1, device=dev, dtype=torch.float32)
            torch.linalg.solve_triangular(eye, rhs, upper=False, unitriangular=True)
        _CUDA_LINALG_WARMED.add(idx)


def _rank_allreduce_scalars(values: list[float], device: torch.device) -> list[float]:
    if not (dist.is_available() and dist.is_initialized()):
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


@contextmanager
def state_frame_cache(segments, adapter, devices, *, max_bytes_per_device=768 * 1024**2):
    """Update-scoped FP32 observation cache, shared by reference and PPO passes.

    Cache tensors contain no trainable activations. Encoder/temporal computation
    and all gradients are recomputed normally. Release on success or exception.
    """
    began = time.perf_counter()
    devices = list(dict.fromkeys(torch.device(d) for d in devices if d is not None))
    used = {str(d): 0 for d in devices}
    hand_indices = tuple(adapter.schema.header["hand_card"])
    try:
        nonempty = [seg for seg in segments if len(seg["raw_obs"])]
        from concurrent.futures import ThreadPoolExecutor as _SFCExec
        with _SFCExec(max_workers=min(16, max(1, (os.cpu_count() or 4) // 2))) as _ex:
            adapted_list = list(_ex.map(lambda s: adapter(s["raw_obs"][None]), nonempty))
        for seg, (header, entities, mask) in zip(nonempty, adapted_list):
            source = (torch.as_tensor(header[0]), torch.as_tensor(entities[0]),
                      torch.as_tensor(mask[0]),
                      torch.as_tensor(seg["raw_obs"][:, list(hand_indices)], dtype=torch.long))
            size = sum(t.numel() * t.element_size() for t in source)
            entries = seg.setdefault("_state_frame_cache", {})
            for device in devices:
                key = str(device)
                if used[key] + size > max_bytes_per_device:
                    continue
                if device.type == "cuda" and torch.cuda.mem_get_info(device)[0] < size + 2 * 1024**3:
                    continue
                entries[key] = tuple(t.to(device=device) for t in source)
                used[key] += size
        yield {"bytes_per_device": used, "prepare_seconds": time.perf_counter() - began}
    finally:
        for seg in segments:
            seg.pop("_state_frame_cache", None)


def _cached_segment_batch(batch_segs, adapter, start, end, device, hand_indices):
    if hand_indices is not None and tuple(hand_indices) != tuple(adapter.schema.header["hand_card"]):
        return None
    rows = [s.get("_state_frame_cache", {}).get(str(device)) for s in batch_segs]
    if any(row is None for row in rows):
        return None
    lengths = [max(0, min(end, len(s["raw_obs"])) - start) for s in batch_segs]
    steps = max(lengths)
    entities = max(row[1].shape[1] for row in rows)
    batch = len(rows)
    h = rows[0][0].new_zeros(batch, steps, rows[0][0].shape[-1])
    e = rows[0][1].new_zeros(batch, steps, entities, rows[0][1].shape[-1])
    m = torch.zeros(batch, steps, entities, dtype=torch.bool, device=device)
    hands = torch.full((batch, steps, rows[0][3].shape[-1]), -1, dtype=torch.long, device=device)
    time_mask = np.zeros((batch, steps), dtype=bool)
    for i, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        if not length:
            continue
        h[i, :length] = row[0][start:start+length]
        count = row[1].shape[1]
        e[i, :length, :count] = row[1][start:start+length]
        m[i, :length, :count] = row[2][start:start+length]
        hands[i, :length] = row[3][start:start+length]
        time_mask[i, :length] = True
    return h, e, m, torch.as_tensor(time_mask, device=device), hands, time_mask


def _adapt_segment_batch(
    batch_segs: list[dict],
    adapter,
    *,
    start: int,
    end: int,
    device: torch.device,
    hand_indices: np.ndarray | None = None,
):
    """Adapt `[B, T_chunk]` raw observations on CPU and copy to `device`."""
    cached = _cached_segment_batch(batch_segs, adapter, start, end, device, hand_indices)
    if cached is not None:
        return cached
    raw_slices = [seg["raw_obs"][start:end] for seg in batch_segs]
    max_t = max(len(x) for x in raw_slices)
    batch_size = len(batch_segs)
    seat_ints = raw_slices[0].shape[-1]
    raw_batch = np.zeros((batch_size, max_t, seat_ints), dtype=np.int32)
    time_mask = np.zeros((batch_size, max_t), dtype=bool)
    if hand_indices is None:
        hand_indices = np.asarray(adapter.schema.header["hand_card"], dtype=np.int64)
    hand_ids = np.full((batch_size, max_t, len(hand_indices)), -1, dtype=np.int64)
    for i, sl in enumerate(raw_slices):
        t_len = len(sl)
        raw_batch[i, :t_len] = sl
        time_mask[i, :t_len] = True
        hand_ids[i, :t_len] = sl[:, hand_indices]
    header, entities, entity_mask = adapter(raw_batch)
    non_blocking = device.type == "cuda"
    return (
        torch.as_tensor(header).to(device=device, non_blocking=non_blocking),
        torch.as_tensor(entities).to(device=device, non_blocking=non_blocking),
        torch.as_tensor(entity_mask).to(device=device, non_blocking=non_blocking),
        torch.as_tensor(time_mask).to(device=device, non_blocking=non_blocking),
        torch.as_tensor(hand_ids).to(device=device, non_blocking=non_blocking),
        time_mask,
    )


def burn_in_memory(
    model,
    adapter,
    raw_prefixes: list[np.ndarray],
    *,
    device: torch.device | str,
    precision: str = "fp16",
    batch_size: int = 8,
    chunk_size: int = 64,
    legal_masks: list[np.ndarray] | None = None,
) -> list[PolicyMemory]:
    """Rebuild exact current-weight recurrent states for continuing episode prefixes."""
    device = torch.device(device)
    if not raw_prefixes:
        return []
    was_training = model.training
    model.eval()
    hand_indices = np.asarray(adapter.schema.header["hand_card"], dtype=np.int64)
    results: list[PolicyMemory | None] = [None] * len(raw_prefixes)
    order = sorted(range(len(raw_prefixes)), key=lambda idx: len(raw_prefixes[idx]), reverse=True)
    non_empty = [idx for idx in order if len(raw_prefixes[idx]) > 0]
    for idx in order:
        if len(raw_prefixes[idx]) == 0:
            results[idx] = model.initial_state(1, device=device)
    with torch.inference_mode():
        for left in range(0, len(non_empty), max(1, batch_size)):
            group = non_empty[left : left + max(1, batch_size)]
            max_len = max(len(raw_prefixes[idx]) for idx in group)
            state = model.initial_state(len(group), device=device)
            for start in range(0, max_len, max(1, chunk_size)):
                end = min(max_len, start + max(1, chunk_size))
                active_mask = np.zeros((len(group), end - start), dtype=bool)
                raw_chunk = np.zeros((len(group), end - start, raw_prefixes[group[0]].shape[-1]), dtype=np.int32)
                hand_chunk = np.full((len(group), end - start, len(hand_indices)), -1, dtype=np.int64)
                legal_chunk = None
                if legal_masks is not None:
                    legal_chunk = np.zeros((len(group), end - start, legal_masks[group[0]].shape[-1]), dtype=bool)
                    legal_chunk[..., 0] = True
                for pos, seg_idx in enumerate(group):
                    prefix = raw_prefixes[seg_idx]
                    if start < len(prefix):
                        seg_end = min(len(prefix), end)
                        span = seg_end - start
                        raw_chunk[pos, :span] = prefix[start:seg_end]
                        hand_chunk[pos, :span] = prefix[start:seg_end, hand_indices]
                        active_mask[pos, :span] = True
                        if legal_masks is not None:
                            legal_chunk[pos, :span] = legal_masks[seg_idx][start:seg_end]
                if not active_mask.any():
                    break
                header, entities, entity_mask = adapter(raw_chunk)
                ht = torch.as_tensor(header, device=device)
                et = torch.as_tensor(entities, device=device)
                emt = torch.as_tensor(entity_mask, device=device)
                tmt = torch.as_tensor(active_mask, device=device)
                with _amp_context(device, precision):
                    if hasattr(model, "learned_placement") or hasattr(model, "slot_actor"):
                        lt = (
                            None
                            if legal_chunk is None
                            else torch.as_tensor(legal_chunk, device=device)
                        )
                        hdt = torch.as_tensor(hand_chunk, device=device)
                        _, _, state = model.forward_state(
                            ht, et, emt, lt, state=state, hand_card_ids=hdt, time_mask=tmt, compute_value=False
                        )
                    else:
                        _, state = model.forward_state(ht, et, emt, state=state, time_mask=tmt)
                state = model.detach_state(state)
            for pos, seg_idx in enumerate(group):
                results[seg_idx] = slice_memory(state, [pos], device=device, detach=True, trim_gqa=True)
    if was_training:
        model.train()
    return [res for res in results if res is not None]


def _evaluate_critic_slice(
    critic,
    adapter,
    segments: list[dict],
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    chunk_size: int,
) -> list[PolicyMemory]:
    """Evaluate a shard of segments on one device and write `old_values`/`h0_critic` in place."""
    if not segments:
        return []
    was_training = critic.training
    critic.eval()
    prefixes = [
        seg.get("prefix_raw", np.zeros((0, adapter.schema.seat_ints), dtype=np.int32))
        for seg in segments
    ]
    h0_list = burn_in_memory(
        critic,
        adapter,
        prefixes,
        device=device,
        precision=precision,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )
    order = sorted(range(len(segments)), key=lambda i: len(segments[i]["raw_obs"]), reverse=True)
    with torch.inference_mode():
        for left in range(0, len(order), max(1, batch_size)):
            group = order[left : left + max(1, batch_size)]
            group_seqs = []
            has_tail = []
            for idx in group:
                seg = segments[idx]
                obs = seg["raw_obs"]
                tail = seg.get("bootstrap_raw")
                if tail is not None:
                    group_seqs.append(np.concatenate([obs, tail[None]], axis=0))
                    has_tail.append(True)
                else:
                    group_seqs.append(obs)
                    has_tail.append(False)
            max_len = max(len(x) for x in group_seqs)
            state = rebind_memory(stack_memories([h0_list[idx] for idx in group], device=device), critic)
            pred_chunks = [[] for _ in group]
            for start in range(0, max_len, max(1, chunk_size)):
                end = min(max_len, start + max(1, chunk_size))
                span_max = end - start
                raw_chunk = np.zeros((len(group), span_max, group_seqs[0].shape[-1]), dtype=np.int32)
                active_mask = np.zeros((len(group), span_max), dtype=bool)
                for pos, seq in enumerate(group_seqs):
                    if start < len(seq):
                        seg_end = min(len(seq), end)
                        span = seg_end - start
                        raw_chunk[pos, :span] = seq[start:seg_end]
                        active_mask[pos, :span] = True
                if not active_mask.any():
                    break
                header, entities, entity_mask = adapter(raw_chunk)
                with _amp_context(device, precision):
                    values, state = critic.forward_state(
                        torch.as_tensor(header, device=device),
                        torch.as_tensor(entities, device=device),
                        torch.as_tensor(entity_mask, device=device),
                        state=state,
                        time_mask=torch.as_tensor(active_mask, device=device),
                    )
                state = critic.detach_state(state)
                vals_np = values.float().cpu().numpy()
                for pos, seq in enumerate(group_seqs):
                    if start < len(seq):
                        seg_end = min(len(seq), end)
                        pred_chunks[pos].append(vals_np[pos, : seg_end - start])
            for pos, idx in enumerate(group):
                full_vals = (
                    np.concatenate(pred_chunks[pos], axis=0)
                    if pred_chunks[pos]
                    else np.zeros(0, dtype=np.float32)
                )
                t_len = len(segments[idx]["raw_obs"])
                segments[idx]["old_values"] = full_vals[:t_len].astype(np.float32)
                segments[idx]["h0_critic"] = (
                    clone_memory(h0_list[idx], device=device, detach=True, trim_gqa=True)
                    if len(segments[idx].get("prefix_raw", ())) > 0
                    else None
                )
                if has_tail[pos]:
                    segments[idx]["bootstrap_value"] = float(full_vals[t_len])
                else:
                    segments[idx]["bootstrap_value"] = float(segments[idx].get("bootstrap_value", 0.0))
    if was_training:
        critic.train()
    return h0_list


def evaluate_critic_segments(
    critic,
    adapter,
    segments: list[dict],
    *,
    device: torch.device | str,
    precision: str = "fp16",
    batch_size: int = 8,
    chunk_size: int = 64,
    peer_critic=None,
    peer_device: torch.device | str | None = None,
) -> tuple[list[PolicyMemory], float]:
    """Evaluate `old_values`, optional truncation/tail `bootstrap_value`, and cache `h0_critic`."""
    device = torch.device(device)
    if not segments:
        return [], 0.0
    began = time.perf_counter()
    if peer_critic is not None and peer_device is not None and len(segments) >= 2:
        peer_dev = torch.device(peer_device)
        ensure_cuda_linalg_initialized(device)
        ensure_cuda_linalg_initialized(peer_dev)
        with torch.no_grad():
            for dst, src in zip(peer_critic.parameters(), critic.parameters(), strict=True):
                dst.data.copy_(src.data, non_blocking=True)
        if peer_dev.type == "cuda":
            torch.cuda.synchronize(peer_dev)
        left_segs = segments[0::2]
        right_segs = segments[1::2]
        right_h0: list[PolicyMemory] = []
        peer_err: list[BaseException] = []

        def run_peer_eval():
            try:
                ctx = torch.cuda.device(peer_dev) if peer_dev.type == "cuda" else nullcontext()
                with ctx:
                    right_h0.extend(
                        _evaluate_critic_slice(
                            peer_critic,
                            adapter,
                            right_segs,
                            device=peer_dev,
                            precision=precision,
                            batch_size=batch_size,
                            chunk_size=chunk_size,
                        )
                    )
            except BaseException as exc:  # pragma: no cover - propagated cleanly
                peer_err.append(exc)

        thread = threading.Thread(target=run_peer_eval)
        thread.start()
        left_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with left_ctx:
            left_h0 = _evaluate_critic_slice(
                critic,
                adapter,
                left_segs,
                device=device,
                precision=precision,
                batch_size=batch_size,
                chunk_size=chunk_size,
            )
        thread.join()
        if peer_err:
            raise peer_err[0]
        h0_merged: list[PolicyMemory] = [None] * len(segments)  # type: ignore[list-item]
        for idx, h0 in enumerate(left_h0):
            h0_merged[2 * idx] = h0
        for idx, h0 in enumerate(right_h0):
            h0_merged[2 * idx + 1] = h0
        return h0_merged, time.perf_counter() - began

    h0_list = _evaluate_critic_slice(
        critic,
        adapter,
        segments,
        device=device,
        precision=precision,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )
    return h0_list, time.perf_counter() - began


def prepare_state_rollout_targets(
    segments: list[dict],
    *,
    gamma: float,
    gae_lambda: float,
    critic_gae_lambda: float | None = None,
    choice_only: bool = False,
    critic_warmup: bool = False,
) -> dict:
    """Compute per-segment GAE / MC value targets and normalize actor advantages globally."""
    if not segments:
        return {
            "actor_eligible_count": 0,
            "critic_eligible_count": 0,
            "explained_variance": float("nan"),
            "advantage_std": 0.0,
            "warmup_mc_games": 0,
        }
    all_raw_adv: list[torch.Tensor] = []
    actor_masks: list[np.ndarray] = []
    critic_masks: list[np.ndarray] = []
    all_old_values: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    warmup_games = 0

    for seg in segments:
        t_len = len(seg["raw_obs"])
        rewards = torch.as_tensor(seg["rewards"], dtype=torch.float32).unsqueeze(-1)
        values = torch.as_tensor(seg["old_values"], dtype=torch.float32).unsqueeze(-1)
        terminated = torch.as_tensor(seg["terminated"], dtype=torch.float32).unsqueeze(-1)
        dones_arr = (
            seg["dones"]
            if "dones" in seg
            else np.logical_or(seg["terminated"], seg.get("truncated", False))
        )
        dones = torch.as_tensor(dones_arr, dtype=torch.float32).unsqueeze(-1)
        last_val = torch.tensor([float(seg.get("bootstrap_value", 0.0))], dtype=torch.float32)
        next_vals = torch.cat([values[1:, 0], last_val], dim=0).unsqueeze(-1)
        adv, returns = compute_actor_critic_targets(
            rewards,
            values,
            terminated,
            last_val,
            gamma=gamma,
            gae_lambda=gae_lambda,
            critic_gae_lambda=critic_gae_lambda,
            next_values=next_vals,
            episode_ends=dones,
            step_ticks=(torch.as_tensor(seg['step_ticks'],dtype=torch.float32).unsqueeze(-1)
                        if 'step_ticks' in seg else None),
            reference_ticks=int(seg.get('decision_ticks',8)),
        )
        valid = np.asarray(seg["valid"], dtype=bool)
        learner = np.asarray(seg.get("learner_mask", np.ones(t_len, dtype=bool)), dtype=bool)
        legal = np.asarray(seg["legal_masks"], dtype=bool)
        eligible = valid & learner
        if choice_only:
            actor_mask = eligible & (legal.sum(axis=-1) > 1)
        else:
            actor_mask = eligible.copy()
        if 'policy_loss_mask' in seg:
            actor_mask &= np.asarray(seg['policy_loss_mask'], dtype=bool)

        seg_targets = returns[:, 0].numpy().astype(np.float32)
        critic_mask = eligible.copy()
        if critic_warmup:
            truncated = np.asarray(seg.get("truncated", np.zeros(t_len, dtype=bool)), dtype=bool)
            # A complete fault-free episode starts with episode_start[0]=True (no missing prefix)
            # and ends with terminated[-1]=True.
            is_complete = (
                bool(seg.get("has_complete_prefix", len(seg.get("prefix_raw", ())) == 0))
                and t_len > 0
                and bool(seg["terminated"][-1])
                and not bool(truncated.any())
                and bool(valid.all())
            )
            mc_ret, mc_sel, games = complete_episode_returns(
                seg["rewards"][:, None],
                seg["terminated"][:, None],
                truncated[:, None],
                eligible[:, None],
                gamma,
            )
            if is_complete and games > 0:
                warmup_games += games
                seg_targets = mc_ret[:, 0].astype(np.float32)
                critic_mask = mc_sel[:, 0] & eligible
            else:
                critic_mask = np.zeros(t_len, dtype=bool)

        seg["advantages"] = adv[:, 0].numpy().astype(np.float32)
        seg["returns"] = seg_targets
        seg["actor_mask"] = actor_mask
        seg["critic_mask"] = critic_mask

        all_raw_adv.append(adv[:, 0])
        actor_masks.append(actor_mask)
        critic_masks.append(critic_mask)
        if critic_mask.any():
            all_old_values.append(np.asarray(seg["old_values"], dtype=np.float32)[critic_mask])
            all_targets.append(seg_targets[critic_mask])

    flat_adv = torch.cat(all_raw_adv, dim=0)
    flat_actor_mask = np.concatenate(actor_masks, axis=0)
    actor_indices = np.flatnonzero(flat_actor_mask)
    if len(actor_indices):
        normalized = normalize_actor_advantages(flat_adv, actor_indices)
        adv_std = float(flat_adv[actor_indices].std(unbiased=False)) if len(actor_indices) > 1 else 0.0
    else:
        normalized = torch.zeros_like(flat_adv)
        adv_std = 0.0

    offset = 0
    for seg in segments:
        t_len = len(seg["raw_obs"])
        seg["advantages"] = normalized[offset : offset + t_len].numpy().astype(np.float32)
        offset += t_len

    explained_var = float("nan")
    if all_targets:
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_old_values)
        var_y = float(np.var(y_true))
        if var_y > 1e-8:
            explained_var = float(1.0 - np.var(y_true - y_pred) / var_y)

    return {
        "actor_eligible_count": int(flat_actor_mask.sum()),
        "critic_eligible_count": int(sum(int(m.sum()) for m in critic_masks)),
        "explained_variance": explained_var,
        "advantage_std": adv_std,
        "warmup_mc_games": int(warmup_games),
    }


def _build_microbatch_groups(
    segments: list[dict],
    *,
    mask_key: str,
    batch_segments: int,
    tbptt_chunk: int,
    effective_batch_samples: int,
    rng: np.random.Generator,
) -> list[list[tuple[list[dict], int, int, int, int]]]:
    """Group `(segment_batch, start, end, effective_max, eligible_count)` chunks into step groups."""
    eligible_segs = [seg for seg in segments if int(seg[mask_key].sum()) > 0]
    if not eligible_segs:
        return []
    # Shuffle segment order across epochs, then bucket similar lengths inside sliding windows
    # to preserve stochasticity while cutting right-padding waste.
    perm = rng.permutation(len(eligible_segs))
    shuffled = [eligible_segs[i] for i in perm]
    window = max(batch_segments * 4, batch_segments)
    ordered: list[dict] = []
    for w_start in range(0, len(shuffled), window):
        chunk = shuffled[w_start : w_start + window]
        chunk.sort(key=lambda s: len(s["raw_obs"]), reverse=True)
        ordered.extend(chunk)

    chunk_items: list[tuple[list[dict], int, int, int, int]] = []
    for left in range(0, len(ordered), max(1, batch_segments)):
        batch_segs = ordered[left : left + max(1, batch_segments)]
        max_len = max(len(s["raw_obs"]) for s in batch_segs)
        last_needed = 0
        for s in batch_segs:
            nz = np.flatnonzero(s[mask_key])
            if len(nz):
                last_needed = max(last_needed, int(nz[-1]) + 1)
        effective_max = min(max_len, last_needed)
        chunk_length = effective_max if tbptt_chunk == 0 else max(1, tbptt_chunk)
        for start in range(0, effective_max, max(1, chunk_length)):
            end = min(effective_max, start + chunk_length)
            cnt = sum(int(s[mask_key][start:end].sum()) for s in batch_segs)
            chunk_items.append((batch_segs, start, end, effective_max, cnt))

    groups: list[list[tuple[list[dict], int, int, int, int]]] = []
    cur_group: list[tuple[list[dict], int, int, int, int]] = []
    cur_count = 0
    for item in chunk_items:
        cur_group.append(item)
        cur_count += item[4]
        if cur_count >= effective_batch_samples:
            groups.append(cur_group)
            cur_group = []
            cur_count = 0
    if cur_group and cur_count > 0:
        groups.append(cur_group)
    return groups


def _sync_peer_weights(
    dst_model, src_model, dst_device: torch.device, src_device: torch.device
) -> None:
    if src_device.type == "cuda":
        torch.cuda.synchronize(src_device)
    with torch.no_grad():
        for p_dst, p_src in zip(dst_model.parameters(), src_model.parameters(), strict=True):
            p_dst.data.copy_(p_src.data, non_blocking=True)
    if hasattr(dst_model, "mark_weights_updated"):
        dst_model.mark_weights_updated()
    if dst_device.type == "cuda":
        torch.cuda.synchronize(dst_device)


def _reduce_peer_grads(
    dst_model, src_model, dst_device: torch.device, src_device: torch.device
) -> None:
    if src_device.type == "cuda":
        torch.cuda.synchronize(src_device)
    with torch.no_grad():
        for p_dst, p_src in zip(dst_model.parameters(), src_model.parameters(), strict=True):
            if p_src.grad is not None:
                transferred = p_src.grad.to(device=dst_device)
                if p_dst.grad is None:
                    p_dst.grad = transferred
                else:
                    p_dst.grad.add_(transferred)
    src_model.zero_grad(set_to_none=True)


def _shard_group_items(
    group: list[tuple[list[dict], int, int, int, int]],
    seg_last: dict[int, int],
    *,
    can_shard: bool,
) -> tuple[list[tuple[list[dict], int, int, int]], list[tuple[list[dict], int, int, int]]]:
    primary_items: list[tuple[list[dict], int, int, int]] = []
    peer_items: list[tuple[list[dict], int, int, int]] = []
    for item_index, (batch_segs, start, end, eff_max, _) in enumerate(group):
        if can_shard and len(batch_segs) >= 2:
            left_segs = batch_segs[0::2]
            right_segs = batch_segs[1::2]
            left_eff = max((seg_last[id(s)] for s in left_segs), default=0)
            right_eff = max((seg_last[id(s)] for s in right_segs), default=0)
            if start < left_eff:
                primary_items.append((left_segs, start, min(end, left_eff), left_eff))
            if start < right_eff:
                peer_items.append((right_segs, start, min(end, right_eff), right_eff))
        elif can_shard and start == 0 and end == eff_max and item_index % 2:
            # An entire one-episode item has no recurrent state dependency on
            # another item; distribute these too when using memory-light B=1.
            peer_items.append((batch_segs, start, end, eff_max))
        else:
            primary_items.append((batch_segs, start, end, eff_max))
    return primary_items, peer_items


def _actor_run_items(
    actor,
    behavior,
    optimizer,
    adapter,
    items: list[tuple[list[dict], int, int, int]],
    seg_states: dict[int, tuple[object, object | None]],
    seg_last: dict[int, int],
    seg_actor_prefix: dict[int, np.ndarray],
    cfg: StatePPOConfig,
    *,
    device: torch.device,
    hand_indices: np.ndarray,
    ref_cache_per_seg: dict[int, torch.Tensor] | None,
    anchor_cache_per_seg: dict[int, torch.Tensor] | None = None,
    shared_critic: bool,
    actor_enabled: bool,
    val_weight: float,
    loss_scale: float = 1.0,
    check_init_logp: bool = False,
) -> dict:
    """Run actor (+ optional shared value) forward/backward for `(segs, start, end, eff_max)` items."""
    raw_actor = actor.module if hasattr(actor, "module") else actor
    need_inline_behavior = (
        actor_enabled and ref_cache_per_seg is None and cfg.reference_kl_coef > 0 and behavior is not None
    )
    grp_samples = 0
    grp_val_samples = 0
    grp_card_kl = 0.0
    grp_tile_kl = 0.0
    grp_approx_kl = 0.0
    grp_policy_loss = 0.0
    grp_value_loss = 0.0
    grp_entropy = 0.0
    grp_card_entropy = 0.0
    grp_tile_entropy = 0.0
    grp_ref_kl = 0.0
    grp_anchor_kl = 0.0
    grp_clip_frac = 0.0
    grp_ratio = 0.0
    init_errs: list[float] = []

    for batch_segs, start, end, _eff_max in items:
        h0_a_list = []
        h0_b_list = []
        for s in batch_segs:
            cached = seg_states.get(id(s))
            if cached is not None:
                a_s, b_s = cached
            else:
                a_s = (
                    s["h0_actor"]
                    if s.get("h0_actor") is not None
                    else raw_actor.initial_state(1, device=device)
                )
                b_s = (
                    (
                        s["h0_behavior"]
                        if s.get("h0_behavior") is not None
                        else behavior.initial_state(1, device=device)
                    )
                    if need_inline_behavior
                    else None
                )
            h0_a_list.append(a_s)
            if need_inline_behavior:
                h0_b_list.append(b_s)

        a_state = rebind_memory(stack_memories(h0_a_list, device=device), raw_actor)
        b_state = (
            rebind_memory(stack_memories(h0_b_list, device=device), behavior)
            if need_inline_behavior
            else None
        )

        ht, et, emt, tmt, hdt, _ = _adapt_segment_batch(
            batch_segs, adapter, start=start, end=end, device=device, hand_indices=hand_indices
        )
        span_max = end - start
        legal_np = np.zeros((len(batch_segs), span_max, raw_actor.cfg.action_count), dtype=bool)
        legal_np[..., 0] = True
        actor_mask_np = np.zeros((len(batch_segs), span_max), dtype=bool)
        critic_mask_np = np.zeros((len(batch_segs), span_max), dtype=bool)
        actions_np = np.zeros((len(batch_segs), span_max), dtype=np.int64)
        old_lp_np = np.zeros((len(batch_segs), span_max), dtype=np.float32)
        adv_np = np.zeros((len(batch_segs), span_max), dtype=np.float32)
        ret_np = np.zeros((len(batch_segs), span_max), dtype=np.float32)

        ref_slices = []
        anchor_slices = []
        for pos, seg in enumerate(batch_segs):
            if start < len(seg["raw_obs"]):
                seg_end = min(len(seg["raw_obs"]), end)
                span = seg_end - start
                legal_np[pos, :span] = seg["legal_masks"][start:seg_end]
                if actor_enabled:
                    actor_mask_np[pos, :span] = seg["actor_mask"][start:seg_end]
                critic_mask_np[pos, :span] = seg["critic_mask"][start:seg_end]
                actions_np[pos, :span] = seg["actions"][start:seg_end]
                old_lp_np[pos, :span] = seg["old_log_probs"][start:seg_end]
                adv_np[pos, :span] = seg["advantages"][start:seg_end]
                ret_np[pos, :span] = seg["returns"][start:seg_end]
                if actor_enabled and (
                    (ref_cache_per_seg is not None and id(seg) in ref_cache_per_seg)
                    or (anchor_cache_per_seg is not None and id(seg) in anchor_cache_per_seg)
                ):
                    pref = seg_actor_prefix[id(seg)]
                    lo = int(pref[start - 1]) if start > 0 else 0
                    hi = int(pref[seg_end - 1])
                    if hi > lo:
                        if ref_cache_per_seg is not None and id(seg) in ref_cache_per_seg:
                            ref_slices.append(ref_cache_per_seg[id(seg)][lo:hi])
                        if anchor_cache_per_seg is not None and id(seg) in anchor_cache_per_seg:
                            anchor_slices.append(anchor_cache_per_seg[id(seg)][lo:hi])

        has_actor = actor_enabled and bool(actor_mask_np.any())
        has_value_loss = shared_critic and bool(critic_mask_np.any())

        if not has_actor and not has_value_loss:
            # Only advance temporal state; zero tile legals so placement head is skipped.
            legal_np[..., 1:] = False
            legal_np[..., 0] = True
            lt = torch.as_tensor(legal_np, device=device)
            with torch.no_grad(), _amp_context(device, cfg.precision):
                _, _, next_a = actor.forward_state(
                    ht, et, emt, lt, state=a_state, hand_card_ids=hdt, time_mask=tmt, compute_value=False
                )
                if b_state is not None:
                    _, _, b_state = behavior.forward_state(
                        ht, et, emt, lt, state=b_state, hand_card_ids=hdt, time_mask=tmt, compute_value=False
                    )
                    b_state = behavior.detach_state(b_state)
            next_a_det = raw_actor.detach_state(next_a)
            for pos, seg in enumerate(batch_segs):
                if end < seg_last[id(seg)]:
                    seg_states[id(seg)] = (
                        slice_memory(next_a_det, [pos], device=device, detach=True, trim_gqa=True),
                        slice_memory(b_state, [pos], device=device, detach=True, trim_gqa=True)
                        if b_state is not None
                        else None,
                    )
                else:
                    seg_states.pop(id(seg), None)
            continue

        if (~actor_mask_np).any():
            # Non-actor frames only advance temporal state; never read placement logits there.
            legal_np[~actor_mask_np, 1:] = False
            legal_np[~actor_mask_np, 0] = True
        lt = torch.as_tensor(legal_np, device=device)
        with _amp_context(device, cfg.precision):
            logp, values, next_a = actor.forward_state(
                ht,
                et,
                emt,
                lt,
                state=a_state,
                hand_card_ids=hdt,
                time_mask=tmt,
                compute_value=shared_critic,
            )
        next_a_det = raw_actor.detach_state(next_a)
        del next_a

        ref_logp_selected = None
        if b_state is not None:
            with torch.no_grad(), _amp_context(device, cfg.precision):
                ref_full, _, next_b = behavior.forward_state(
                    ht, et, emt, lt, state=b_state, hand_card_ids=hdt, time_mask=tmt, compute_value=False
                )
            b_state = behavior.detach_state(next_b)
            del next_b
            if has_actor:
                ref_logp_selected = ref_full[torch.as_tensor(actor_mask_np, device=device)].float()
            del ref_full
        elif ref_slices:
            ref_logp_selected = torch.cat(ref_slices, dim=0).to(device=device, dtype=torch.float32)

        chunk_loss = logp.new_zeros(())
        c_actor = int(actor_mask_np.sum())
        if c_actor > 0:
            sel_t = torch.as_tensor(actor_mask_np, device=device)
            sel_logp = logp[sel_t].float()
            sel_actions = torch.as_tensor(actions_np[actor_mask_np], dtype=torch.long, device=device)
            sel_old_lp = torch.as_tensor(old_lp_np[actor_mask_np], dtype=torch.float32, device=device)
            sel_adv = torch.as_tensor(adv_np[actor_mask_np], dtype=torch.float32, device=device)

            new_chosen_lp = sel_logp.gather(-1, sel_actions[:, None]).squeeze(-1)
            if check_init_logp:
                init_err = float((new_chosen_lp - sel_old_lp).abs().max().detach().cpu())
                init_errs.append(init_err)
                if cfg.max_initial_logp_error is not None and (
                    not math.isfinite(init_err) or init_err > cfg.max_initial_logp_error
                ):
                    import os
                    debug_path = os.environ.get("CRRL_LOGP_DEBUG")
                    if debug_path:
                        torch.save({"header": ht.detach().cpu(), "entities": et.detach().cpu(),
                            "entity_mask": emt.detach().cpu(), "legal": lt.detach().cpu(),
                            "hands": hdt.detach().cpu(), "time_mask": tmt.detach().cpu(),
                            "actions": actions_np, "old_logp": old_lp_np, "actor_mask": actor_mask_np,
                            "start": start, "end": end,
                            "h0": clone_memory(a_state, device="cpu", detach=True, trim_gqa=True),
                            "episode_ids": [s["episode_id"] for s in batch_segs]}, debug_path+"."+str(device).replace(":","_")+".pt")
                    raise RuntimeError(
                        f"Actor initial log-prob mismatch {init_err:.6f} exceeds limit {cfg.max_initial_logp_error}; "
                        f"device={device} chunk={start}:{end} episodes={[s['episode_id'] for s in batch_segs]}"
                    )

            log_ratio = new_chosen_lp - sel_old_lp
            ratio = log_ratio.exp()
            unclipped = ratio * sel_adv
            clipped = ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * sel_adv
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            card_h, tile_h = factorized_entropy(sel_logp)
            card_h_mean = card_h.mean()
            tile_h_mean = tile_h.mean()
            entropy_mean = card_h_mean + tile_h_mean
            if cfg.card_entropy_coef is not None and cfg.tile_entropy_coef is not None:
                entropy_term = cfg.card_entropy_coef * card_h_mean + cfg.tile_entropy_coef * tile_h_mean
            else:
                entropy_term = cfg.entropy_coef * entropy_mean

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
                ratio_mean = ratio.mean()

            if ref_logp_selected is not None:
                card_kl_t, tile_kl_t = factorized_reverse_kl(sel_logp, ref_logp_selected)
                card_kl_mean = card_kl_t.mean()
                tile_kl_mean = tile_kl_t.mean()
                ref_kl = card_kl_mean + tile_kl_mean
            else:
                card_kl_mean = torch.zeros_like(approx_kl)
                tile_kl_mean = torch.zeros_like(approx_kl)
                ref_kl = torch.zeros_like(approx_kl)

            if anchor_slices:
                anchor_logp_selected = torch.cat(anchor_slices, dim=0).to(device=device, dtype=torch.float32)
                anc_card_kl_t, anc_tile_kl_t = factorized_reverse_kl(sel_logp, anchor_logp_selected)
                anchor_kl = anc_card_kl_t.mean() + anc_tile_kl_t.mean()
            else:
                anchor_kl = torch.zeros_like(approx_kl)

            actor_chunk_loss = policy_loss - entropy_term
            if cfg.reference_kl_coef > 0 and ref_logp_selected is not None:
                actor_chunk_loss = actor_chunk_loss + cfg.reference_kl_coef * ref_kl
            if cfg.anchor_kl_coef > 0 and anchor_slices:
                actor_chunk_loss = actor_chunk_loss + cfg.anchor_kl_coef * anchor_kl
            chunk_loss = chunk_loss + actor_chunk_loss * float(c_actor)

            grp_samples += c_actor
            grp_policy_loss += float(policy_loss.detach()) * c_actor
            grp_entropy += float(entropy_mean.detach()) * c_actor
            grp_card_entropy += float(card_h_mean.detach()) * c_actor
            grp_tile_entropy += float(tile_h_mean.detach()) * c_actor
            grp_approx_kl += float(approx_kl.detach()) * c_actor
            grp_card_kl += float(card_kl_mean.detach()) * c_actor
            grp_tile_kl += float(tile_kl_mean.detach()) * c_actor
            grp_ref_kl += float(ref_kl.detach()) * c_actor
            grp_anchor_kl += float(anchor_kl.detach()) * c_actor
            grp_clip_frac += float(clip_frac.detach()) * c_actor
            grp_ratio += float(ratio_mean.detach()) * c_actor

        c_val = int(critic_mask_np.sum()) if shared_critic else 0
        if shared_critic and c_val > 0:
            val_mask = torch.as_tensor(critic_mask_np, device=device)
            v_pred = values[val_mask].float()
            v_ret = torch.as_tensor(ret_np[critic_mask_np], dtype=torch.float32, device=device)
            v_loss = 0.5 * (v_pred - v_ret).square().mean()
            chunk_loss = chunk_loss + cfg.value_coef * v_loss * float(c_val) * val_weight
            grp_value_loss += float(v_loss.detach()) * c_val
            grp_val_samples += c_val

        if optimizer is not None:
            backward_loss(chunk_loss, optimizer)
        else:
            (chunk_loss * float(loss_scale)).backward()
        del chunk_loss, logp, values

        # Slice next recurrent states only after backward frees forward activations.
        for pos, seg in enumerate(batch_segs):
            if end < seg_last[id(seg)]:
                seg_states[id(seg)] = (
                    slice_memory(next_a_det, [pos], device=device, detach=True, trim_gqa=True),
                    slice_memory(b_state, [pos], device=device, detach=True, trim_gqa=True)
                    if b_state is not None
                    else None,
                )
            else:
                seg_states.pop(id(seg), None)
        del next_a_det, b_state

    return {
        "samples": grp_samples,
        "val_samples": grp_val_samples,
        "policy_loss": grp_policy_loss,
        "value_loss": grp_value_loss,
        "entropy": grp_entropy,
        "card_entropy": grp_card_entropy,
        "tile_entropy": grp_tile_entropy,
        "approx_kl": grp_approx_kl,
        "card_kl": grp_card_kl,
        "tile_kl": grp_tile_kl,
        "ref_kl": grp_ref_kl,
        "anchor_kl": grp_anchor_kl,
        "clip_frac": grp_clip_frac,
        "ratio": grp_ratio,
        "init_errs": init_errs,
    }


def _build_actor_logp_cache(
    model,
    adapter,
    segments: list[dict],
    cfg: StatePPOConfig,
    *,
    model_device: torch.device,
    store_device: torch.device,
    h0_key: str,
    hand_indices: np.ndarray,
    rollout_check: list[float] | None = None,
) -> dict[int, torch.Tensor]:
    """Precompute selected-step FP16 log-probs for a frozen reference or anchor policy."""
    cache_per_seg: dict[int, torch.Tensor] = {}
    eligible_segs = [seg for seg in segments if int(seg["actor_mask"].sum()) > 0]
    if not eligible_segs:
        return cache_per_seg
    raw_model = model.module if hasattr(model, "module") else model
    action_count = raw_model.cfg.action_count
    was_training = raw_model.training
    raw_model.eval()
    order = sorted(eligible_segs, key=lambda s: len(s["raw_obs"]), reverse=True)
    ctx = torch.cuda.device(model_device) if model_device.type == "cuda" else nullcontext()
    errors = []
    try:
        with ctx, torch.inference_mode():
            cache_batch = (
                max(16, int(cfg.critic_batch_size or 16))
                if (rollout_check is None and model_device.type == "cuda")
                else max(1, cfg.actor_batch_size)
            )
            for left in range(0, len(order), cache_batch):
                batch_segs = order[left : left + cache_batch]
                h0s = [
                    seg[h0_key]
                    if seg.get(h0_key) is not None
                    else raw_model.initial_state(1, device=model_device)
                    for seg in batch_segs
                ]
                state = rebind_memory(stack_memories(h0s, device=model_device), raw_model)
                last_needed = max(int(np.flatnonzero(s["actor_mask"])[-1]) + 1 for s in batch_segs)
                seg_chunks = [[] for _ in batch_segs]
                for start in range(0, last_needed, max(1, cfg.tbptt_chunk)):
                    end = min(last_needed, start + max(1, cfg.tbptt_chunk))
                    ht, et, emt, tmt, hdt, _ = _adapt_segment_batch(
                        batch_segs, adapter, start=start, end=end, device=model_device, hand_indices=hand_indices
                    )
                    legal_np = np.zeros((len(batch_segs), end - start, action_count), dtype=bool)
                    legal_np[..., 0] = True
                    for pos, seg in enumerate(batch_segs):
                        if start < len(seg["legal_masks"]):
                            seg_end = min(len(seg["legal_masks"]), end)
                            span = seg_end - start
                            seg_legal = seg["legal_masks"][start:seg_end].copy()
                            skip_tile = ~seg["actor_mask"][start:seg_end]
                            if skip_tile.any():
                                seg_legal[skip_tile, 1:] = False
                                seg_legal[skip_tile, 0] = True
                            legal_np[pos, :span] = seg_legal
                    lt = torch.as_tensor(legal_np, device=model_device)
                    with _amp_context(model_device, cfg.precision):
                        logp, _, state = model.forward_state(
                            ht, et, emt, lt, state=state, hand_card_ids=hdt, time_mask=tmt, compute_value=False
                        )
                    state = raw_model.detach_state(state)
                    for pos, seg in enumerate(batch_segs):
                        if start < len(seg["raw_obs"]):
                            seg_end = min(len(seg["raw_obs"]), end)
                            m_slice = seg["actor_mask"][start:seg_end]
                            if m_slice.any():
                                ref_t = logp[pos, : seg_end - start][
                                    torch.as_tensor(m_slice, device=model_device)
                                ]
                                if rollout_check is not None:
                                    selected = torch.as_tensor(seg["actions"][start:seg_end][m_slice],
                                                               device=model_device, dtype=torch.long)
                                    old = torch.as_tensor(seg["old_log_probs"][start:seg_end][m_slice],
                                                          device=model_device, dtype=torch.float32)
                                    errors.append((ref_t.gather(1, selected[:, None]).squeeze(1) - old).abs().amax())
                                ref_t = ref_t.to(torch.float16)
                                if store_device.type == "cuda":
                                    seg_chunks[pos].append(ref_t.to(device=store_device, non_blocking=True))
                                else:
                                    seg_chunks[pos].append(ref_t.cpu())
                for pos, seg in enumerate(batch_segs):
                    if seg_chunks[pos]:
                        cache_per_seg[id(seg)] = torch.cat(seg_chunks[pos], dim=0)
            if errors:
                max_error = float(torch.stack(errors).amax())
                rollout_check.append(max_error)
                if not math.isfinite(max_error) or max_error > cfg.max_initial_logp_error:
                    raise RuntimeError(
                        f"Frozen behavior rollout log-prob mismatch {max_error:.6f} "
                        f"exceeds limit {cfg.max_initial_logp_error}; checked all actor rows before updating"
                    )
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
            if store_device != model_device and store_device.type == "cuda":
                torch.cuda.synchronize(store_device)
    finally:
        if was_training:
            raw_model.train()
    return cache_per_seg


def _run_actor_optimization(
    actor,
    behavior,
    actor_optimizer,
    adapter,
    segments: list[dict],
    cfg: StatePPOConfig,
    *,
    device: torch.device,
    rng: np.random.Generator,
    shared_critic: bool = False,
    actor_enabled: bool = True,
    lr_controller=None,
    peer_actor=None,
    peer_device: torch.device | None = None,
    peer_behavior=None,
    peer_ready_event: threading.Event | None = None,
    anchor_cache_per_seg: dict[int, torch.Tensor] | None = None,
) -> dict:
    raw_actor = actor.module if hasattr(actor, "module") else actor
    raw_peer = (peer_actor.module if hasattr(peer_actor, "module") else peer_actor) if peer_actor is not None else None
    hand_indices = np.asarray(adapter.schema.header["hand_card"], dtype=np.int64)
    began = time.perf_counter()
    actor.train()
    if peer_actor is not None and peer_device is not None:
        ensure_cuda_linalg_initialized(device)
        ensure_cuda_linalg_initialized(peer_device)
        peer_actor.train()
    behavior.eval()
    if peer_behavior is not None:
        peer_behavior.eval()

    ref_cache_per_seg: dict[int, torch.Tensor] | None = None
    ref_time = 0.0
    initial_logp_errors: list[float] = []
    if actor_enabled and cfg.cache_reference and cfg.reference_kl_coef > 0:
        t0 = time.perf_counter()
        ref_cache_per_seg = _build_actor_logp_cache(
            behavior,
            adapter,
            segments,
            cfg,
            model_device=device,
            store_device=device,
            h0_key="h0_behavior",
            hand_indices=hand_indices,
            rollout_check=initial_logp_errors if (
                cfg.full_episode_bptt and cfg.max_initial_logp_error is not None
                and getattr(behavior, "ppo_actor_fp32", False)
            ) else None,
        )
        ref_time = time.perf_counter() - t0

    actor_steps = 0
    critic_steps = 0
    overflow_steps = 0
    kl_stops = 0
    rejected_groups = 0
    stopped_early = False
    actor_samples = 0
    critic_samples = 0

    sum_policy_loss = 0.0
    sum_value_loss = 0.0
    sum_entropy = 0.0
    sum_card_entropy = 0.0
    sum_tile_entropy = 0.0
    sum_approx_kl = 0.0
    sum_card_kl = 0.0
    sum_tile_kl = 0.0
    sum_ref_kl = 0.0
    sum_anchor_kl = 0.0
    sum_clip_frac = 0.0
    sum_ratio = 0.0
    stat_count = 0
    stat_val_count = 0
    grad_norms: list[float] = []

    seg_actor_prefix: dict[int, np.ndarray] = {}
    for seg in segments:
        seg_actor_prefix[id(seg)] = np.cumsum(seg["actor_mask"].astype(np.int64))

    mask_key = "actor_mask" if actor_enabled else "critic_mask"
    seg_last = {
        id(s): (
            max(
                int(np.flatnonzero(s[mask_key])[-1]) + 1 if s[mask_key].any() else 0,
                int(np.flatnonzero(s["critic_mask"])[-1]) + 1 if (shared_critic and s["critic_mask"].any()) else 0,
            )
        )
        for s in segments
    }
    need_inline_behavior = (
        actor_enabled and ref_cache_per_seg is None and cfg.reference_kl_coef > 0 and behavior is not None
    )
    peer_capable = (
        peer_actor is not None
        and peer_device is not None
        and (not need_inline_behavior or peer_behavior is not None)
    )
    scaler = getattr(actor_optimizer, "amp_scaler", None)
    peer_synced = False
    epochs = cfg.epochs if actor_enabled else cfg.critic_epochs
    for _epoch in range(epochs):
        groups = _build_microbatch_groups(
            segments,
            mask_key=mask_key,
            batch_segments=cfg.actor_batch_size,
            tbptt_chunk=0 if cfg.full_episode_bptt else cfg.tbptt_chunk,
            effective_batch_samples=cfg.actor_effective_batch_size,
            rng=rng,
        )
        if not groups:
            break

        # Epoch-scoped TBPTT state maps so recurrent states flow cleanly across effective-batch splits.
        primary_states: dict[int, tuple[object, object | None]] = {}
        peer_states: dict[int, tuple[object, object | None]] = {}

        for group in groups:
            actor_optimizer.zero_grad(set_to_none=True)
            grp_actor_total = 0
            grp_val_total = 0
            for batch_segs, start, end, _eff_max, _ in group:
                for seg in batch_segs:
                    if start < len(seg["raw_obs"]):
                        seg_end = min(len(seg["raw_obs"]), end)
                        if actor_enabled:
                            grp_actor_total += int(seg["actor_mask"][start:seg_end].sum())
                        if shared_critic:
                            grp_val_total += int(seg["critic_mask"][start:seg_end].sum())

            scale_base = float(grp_actor_total if (actor_enabled and grp_actor_total > 0) else max(1, grp_val_total))
            val_weight = (scale_base / float(grp_val_total)) if (shared_critic and grp_val_total > 0) else 0.0
            check_init_logp = actor_steps == 0

            can_shard = peer_capable and (peer_ready_event is None or peer_ready_event.is_set())
            primary_items, peer_items = _shard_group_items(group, seg_last, can_shard=can_shard)

            for segs, _, _, _ in primary_items:
                for s in segs:
                    sid = id(s)
                    if sid in peer_states and sid not in primary_states:
                        primary_states[sid] = peer_states.pop(sid)

            if peer_items and raw_peer is not None and peer_device is not None:
                moved_states = False
                for segs, _, _, _ in peer_items:
                    for s in segs:
                        sid = id(s)
                        if sid in primary_states and sid not in peer_states:
                            peer_states[sid] = primary_states.pop(sid)
                            moved_states = True
                if not peer_synced:
                    _sync_peer_weights(raw_peer, raw_actor, peer_device, device)
                    peer_synced = True
                elif moved_states and device.type == "cuda":
                    torch.cuda.synchronize(device)

                raw_peer.zero_grad(set_to_none=True)
                peer_out: dict = {}
                peer_err: list[BaseException] = []
                peer_scale = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.0

                def run_peer(
                    items=peer_items,
                    p_scale=peer_scale,
                    chk_init=check_init_logp,
                    out_buf=peer_out,
                    err_buf=peer_err,
                ):
                    try:
                        p_ctx = torch.cuda.device(peer_device) if peer_device.type == "cuda" else nullcontext()
                        with p_ctx:
                            out_buf.update(
                                _actor_run_items(
                                    peer_actor,
                                    peer_behavior,
                                    None,
                                    adapter,
                                    items,
                                    peer_states,
                                    seg_last,
                                    seg_actor_prefix,
                                    cfg,
                                    device=peer_device,
                                    hand_indices=hand_indices,
                                    ref_cache_per_seg=ref_cache_per_seg,
                                    anchor_cache_per_seg=anchor_cache_per_seg,
                                    shared_critic=shared_critic,
                                    actor_enabled=actor_enabled,
                                    val_weight=val_weight,
                                    loss_scale=p_scale,
                                    check_init_logp=chk_init,
                                )
                            )
                    except BaseException as exc:  # pragma: no cover - re-raised on main thread
                        err_buf.append(exc)

                thread = threading.Thread(target=run_peer)
                thread.start()
                m_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
                with m_ctx:
                    main_out = _actor_run_items(
                        actor,
                        behavior,
                        actor_optimizer,
                        adapter,
                        primary_items,
                        primary_states,
                        seg_last,
                        seg_actor_prefix,
                        cfg,
                        device=device,
                        hand_indices=hand_indices,
                        ref_cache_per_seg=ref_cache_per_seg,
                        anchor_cache_per_seg=anchor_cache_per_seg,
                        shared_critic=shared_critic,
                        actor_enabled=actor_enabled,
                        val_weight=val_weight,
                        check_init_logp=check_init_logp,
                    )
                thread.join()
                if peer_err:
                    raise peer_err[0]
                _reduce_peer_grads(raw_actor, raw_peer, device, peer_device)

                grp_samples = main_out["samples"] + peer_out["samples"]
                grp_val_samples = main_out["val_samples"] + peer_out["val_samples"]
                grp_policy_loss = main_out["policy_loss"] + peer_out["policy_loss"]
                grp_value_loss = main_out["value_loss"] + peer_out["value_loss"]
                grp_entropy = main_out["entropy"] + peer_out["entropy"]
                grp_card_entropy = main_out["card_entropy"] + peer_out["card_entropy"]
                grp_tile_entropy = main_out["tile_entropy"] + peer_out["tile_entropy"]
                grp_approx_kl = main_out["approx_kl"] + peer_out["approx_kl"]
                grp_card_kl = main_out["card_kl"] + peer_out["card_kl"]
                grp_tile_kl = main_out["tile_kl"] + peer_out["tile_kl"]
                grp_ref_kl = main_out["ref_kl"] + peer_out["ref_kl"]
                grp_anchor_kl = main_out["anchor_kl"] + peer_out["anchor_kl"]
                grp_clip_frac = main_out["clip_frac"] + peer_out["clip_frac"]
                grp_ratio = main_out["ratio"] + peer_out["ratio"]
                initial_logp_errors.extend(main_out["init_errs"])
                initial_logp_errors.extend(peer_out["init_errs"])
            else:
                main_out = _actor_run_items(
                    actor,
                    behavior,
                    actor_optimizer,
                    adapter,
                    primary_items,
                    primary_states,
                    seg_last,
                    seg_actor_prefix,
                    cfg,
                    device=device,
                    hand_indices=hand_indices,
                    ref_cache_per_seg=ref_cache_per_seg,
                    anchor_cache_per_seg=anchor_cache_per_seg,
                    shared_critic=shared_critic,
                    actor_enabled=actor_enabled,
                    val_weight=val_weight,
                    check_init_logp=check_init_logp,
                )
                grp_samples = main_out["samples"]
                grp_val_samples = main_out["val_samples"]
                grp_policy_loss = main_out["policy_loss"]
                grp_value_loss = main_out["value_loss"]
                grp_entropy = main_out["entropy"]
                grp_card_entropy = main_out["card_entropy"]
                grp_tile_entropy = main_out["tile_entropy"]
                grp_approx_kl = main_out["approx_kl"]
                grp_card_kl = main_out["card_kl"]
                grp_tile_kl = main_out["tile_kl"]
                grp_ref_kl = main_out["ref_kl"]
                grp_anchor_kl = main_out["anchor_kl"]
                grp_clip_frac = main_out["clip_frac"]
                grp_ratio = main_out["ratio"]
                initial_logp_errors.extend(main_out["init_errs"])

            if grp_samples == 0 and grp_val_samples == 0:
                actor_optimizer.zero_grad(set_to_none=True)
                continue

            rejected = False
            if actor_enabled and grp_samples > 0:
                reduced = _rank_allreduce_scalars(
                    [float(grp_samples), grp_card_kl, grp_tile_kl, grp_approx_kl], device
                )
                total_grp_samples = max(1.0, reduced[0])
                mean_card_kl = reduced[1] / total_grp_samples
                mean_tile_kl = reduced[2] / total_grp_samples
                mean_approx_kl = reduced[3] / total_grp_samples

                use_factorized_kl = cfg.target_card_kl is not None and cfg.target_tile_kl is not None
                if use_factorized_kl:
                    rejected = stop_for_factorized_kl(
                        actor_optimizer, mean_card_kl, mean_tile_kl, cfg.target_card_kl, cfg.target_tile_kl
                    )
                else:
                    rejected = stop_for_kl(actor_optimizer, mean_approx_kl, cfg.target_kl)

                if lr_controller is not None:
                    t_card = cfg.target_card_kl if cfg.target_card_kl is not None else cfg.target_kl
                    t_tile = cfg.target_tile_kl if cfg.target_tile_kl is not None else cfg.target_kl
                    lr_controller.observe(
                        card_kl=mean_card_kl if use_factorized_kl else mean_approx_kl,
                        tile_kl=mean_tile_kl if use_factorized_kl else 0.0,
                        target_card_kl=t_card,
                        target_tile_kl=t_tile,
                        samples=int(round(total_grp_samples)),
                        rejected=rejected,
                    )

            if rejected:
                kl_stops += 1
                rejected_groups += 1
                if cfg.continue_after_kl_reject:
                    continue
                stopped_early = True
                break

            grad_norm = _optimizer_step(raw_actor, actor_optimizer, scale_base, cfg.max_grad_norm)
            if grad_norm is None:
                overflow_steps += 1
                continue

            raw_actor.mark_weights_updated()
            peer_synced = False
            if actor_enabled:
                actor_steps += 1
                actor_samples += grp_samples
            if shared_critic:
                critic_steps += 1
                critic_samples += grp_val_samples
            grad_norms.append(float(grad_norm))
            sum_policy_loss += grp_policy_loss
            sum_value_loss += grp_value_loss
            sum_entropy += grp_entropy
            sum_card_entropy += grp_card_entropy
            sum_tile_entropy += grp_tile_entropy
            sum_approx_kl += grp_approx_kl
            sum_card_kl += grp_card_kl
            sum_tile_kl += grp_tile_kl
            sum_ref_kl += grp_ref_kl
            sum_anchor_kl += grp_anchor_kl
            sum_clip_frac += grp_clip_frac
            sum_ratio += grp_ratio
            stat_count += grp_samples
            stat_val_count += grp_val_samples

        if stopped_early:
            primary_states.clear()
            peer_states.clear()
            break
        primary_states.clear()
        peer_states.clear()

    if ref_cache_per_seg is not None:
        ref_cache_per_seg.clear()
        ref_cache_per_seg = None
    if anchor_cache_per_seg is not None:
        anchor_cache_per_seg.clear()
        anchor_cache_per_seg = None

    denom = float(max(1, stat_count))
    val_denom = float(max(1, stat_val_count))
    actor_lr_event = None
    if lr_controller is not None:
        actor_lr_event = lr_controller.finish_update(
            actor_optimizer,
            actor_enabled=actor_enabled,
            committed_steps=actor_steps,
            overflow_steps=overflow_steps,
        )

    return {
        "actor_steps": actor_steps,
        "critic_steps": critic_steps,
        "actor_overflow_steps": overflow_steps,
        "actor_samples": actor_samples,
        "critic_samples": critic_samples,
        "kl_stops": kl_stops,
        "rejected_groups": rejected_groups,
        "stopped_early": stopped_early,
        "initial_logp_errors": initial_logp_errors,
        "actor_lr_event": actor_lr_event,
        "actor_update_seconds": time.perf_counter() - began,
        "reference_seconds": ref_time,
        "metrics": {
            "policy_loss": sum_policy_loss / denom if stat_count else 0.0,
            "shared_value_loss": sum_value_loss / val_denom if (shared_critic and stat_val_count) else 0.0,
            "entropy": sum_entropy / denom if stat_count else 0.0,
            "card_entropy": sum_card_entropy / denom if stat_count else 0.0,
            "tile_entropy": sum_tile_entropy / denom if stat_count else 0.0,
            "approx_kl": sum_approx_kl / denom if stat_count else 0.0,
            "card_kl": sum_card_kl / denom if stat_count else 0.0,
            "tile_kl": sum_tile_kl / denom if stat_count else 0.0,
            "reference_kl": sum_ref_kl / denom if stat_count else 0.0,
            "anchor_kl": sum_anchor_kl / denom if stat_count else 0.0,
            "clip_fraction": sum_clip_frac / denom if stat_count else 0.0,
            "mean_ratio": sum_ratio / denom if stat_count else 1.0,
            "actor_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        },
    }


def _critic_run_items(
    model,
    optimizer,
    adapter,
    items: list[tuple[list[dict], int, int, int]],
    batch_states: dict[tuple[int, ...], object],
    *,
    device: torch.device,
    precision: str,
    loss_scale: float = 1.0,
    hand_indices: np.ndarray | None = None,
) -> tuple[int, float]:
    """Run forward/backward on a list of `(segs, start, end, eff_max)` items on `device`."""
    samples = 0
    sum_loss = 0.0
    raw_model = model.module if hasattr(model, "module") else model
    for batch_segs, start, end, eff_max in items:
        b_key = tuple(id(s) for s in batch_segs)
        if b_key not in batch_states:
            batch_states[b_key] = rebind_memory(
                stack_memories(
                    [
                        s["h0_critic"]
                        if s.get("h0_critic") is not None
                        else raw_model.initial_state(1, device=device)
                        for s in batch_segs
                    ],
                    device=device,
                ),
                raw_model,
            )

        c_state = rebind_memory(batch_states[b_key], raw_model)
        ht, et, emt, tmt, _, _ = _adapt_segment_batch(
            batch_segs, adapter, start=start, end=end, device=device, hand_indices=hand_indices
        )
        span_max = end - start
        mask_np = np.zeros((len(batch_segs), span_max), dtype=bool)
        ret_np = np.zeros((len(batch_segs), span_max), dtype=np.float32)
        for pos, seg in enumerate(batch_segs):
            if start < len(seg["raw_obs"]):
                seg_end = min(len(seg["raw_obs"]), end)
                span = seg_end - start
                mask_np[pos, :span] = seg["critic_mask"][start:seg_end]
                ret_np[pos, :span] = seg["returns"][start:seg_end]

        c_cnt = int(mask_np.sum())
        if c_cnt == 0:
            with torch.no_grad(), _amp_context(device, precision):
                _, next_c = model.forward_state(ht, et, emt, state=c_state, time_mask=tmt)
            if end >= eff_max:
                batch_states.pop(b_key, None)
            else:
                batch_states[b_key] = raw_model.detach_state(next_c)
            continue

        with _amp_context(device, precision):
            values, next_c = model.forward_state(ht, et, emt, state=c_state, time_mask=tmt)
        if end >= eff_max:
            batch_states.pop(b_key, None)
        else:
            batch_states[b_key] = raw_model.detach_state(next_c)

        val_mask = torch.as_tensor(mask_np, device=device)
        v_pred = values[val_mask].float()
        v_ret = torch.as_tensor(ret_np[mask_np], dtype=torch.float32, device=device)
        v_loss = 0.5 * (v_pred - v_ret).square().mean()
        loss_scaled = v_loss * float(c_cnt)
        if optimizer is not None:
            backward_loss(loss_scaled, optimizer)
        else:
            (loss_scaled * float(loss_scale)).backward()
        samples += c_cnt
        sum_loss += float(v_loss.detach()) * c_cnt
    return samples, sum_loss


def _run_critic_optimization(
    critic,
    critic_optimizer,
    adapter,
    segments: list[dict],
    cfg: StatePPOConfig,
    *,
    device: torch.device,
    rng: np.random.Generator,
    peer_critic=None,
    peer_device: torch.device | None = None,
) -> dict:
    raw_critic = critic.module if hasattr(critic, "module") else critic
    raw_peer = (peer_critic.module if hasattr(peer_critic, "module") else peer_critic) if peer_critic is not None else None
    hand_indices = np.asarray(adapter.schema.header["hand_card"], dtype=np.int64)
    began = time.perf_counter()
    critic.train()
    if raw_peer is not None and peer_device is not None:
        ensure_cuda_linalg_initialized(device)
        ensure_cuda_linalg_initialized(peer_device)
        peer_critic.train()
        _sync_peer_weights(raw_peer, raw_critic, peer_device, device)

    seg_last_critic = {
        id(s): (int(np.flatnonzero(s["critic_mask"])[-1]) + 1 if s["critic_mask"].any() else 0)
        for s in segments
    }
    critic_steps = 0
    overflow_steps = 0
    critic_samples = 0
    sum_value_loss = 0.0
    grad_norms: list[float] = []
    scaler = getattr(critic_optimizer, "amp_scaler", None)

    for _epoch in range(cfg.critic_epochs):
        groups = _build_microbatch_groups(
            segments,
            mask_key="critic_mask",
            batch_segments=cfg.critic_batch_size,
            tbptt_chunk=0 if cfg.full_episode_bptt else cfg.tbptt_chunk,
            effective_batch_samples=cfg.critic_effective_batch_size,
            rng=rng,
        )
        if not groups:
            break

        primary_states: dict[tuple[int, ...], object] = {}
        peer_states: dict[tuple[int, ...], object] = {}

        for group in groups:
            critic_optimizer.zero_grad(set_to_none=True)
            primary_items, peer_items = _shard_group_items(
                group,
                seg_last_critic,
                can_shard=peer_critic is not None and peer_device is not None,
            )

            if peer_critic is not None and peer_device is not None and peer_items:
                peer_critic.zero_grad(set_to_none=True)
                peer_out: dict[str, float | int] = {"samples": 0, "loss": 0.0}
                peer_err: list[BaseException] = []
                peer_scale = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.0

                def run_peer(
                    items=peer_items,
                    p_scale=peer_scale,
                    out_buf=peer_out,
                    err_buf=peer_err,
                ):
                    try:
                        p_ctx = torch.cuda.device(peer_device) if peer_device.type == "cuda" else nullcontext()
                        with p_ctx:
                            s_p, l_p = _critic_run_items(
                                peer_critic,
                                None,
                                adapter,
                                items,
                                peer_states,
                                device=peer_device,
                                precision=cfg.precision,
                                loss_scale=p_scale,
                                hand_indices=hand_indices,
                            )
                        out_buf["samples"] = s_p
                        out_buf["loss"] = l_p
                    except BaseException as exc:  # pragma: no cover - re-raised on main thread
                        err_buf.append(exc)

                thread = threading.Thread(target=run_peer)
                thread.start()
                m_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
                with m_ctx:
                    grp_s_main, grp_loss_main = _critic_run_items(
                        critic,
                        critic_optimizer,
                        adapter,
                        primary_items,
                        primary_states,
                        device=device,
                        precision=cfg.precision,
                        hand_indices=hand_indices,
                    )
                thread.join()
                if peer_err:
                    raise peer_err[0]
                _reduce_peer_grads(raw_critic, raw_peer, device, peer_device)
                grp_samples = grp_s_main + int(peer_out["samples"])
                grp_loss = grp_loss_main + float(peer_out["loss"])
            else:
                grp_samples, grp_loss = _critic_run_items(
                    critic,
                    critic_optimizer,
                    adapter,
                    primary_items,
                    primary_states,
                    device=device,
                    precision=cfg.precision,
                    hand_indices=hand_indices,
                )

            if grp_samples == 0:
                critic_optimizer.zero_grad(set_to_none=True)
                continue

            grad_norm = _optimizer_step(
                raw_critic, critic_optimizer, float(grp_samples), cfg.critic_max_grad_norm
            )
            if grad_norm is None:
                overflow_steps += 1
                continue

            raw_critic.mark_weights_updated()
            if raw_peer is not None and peer_device is not None:
                _sync_peer_weights(raw_peer, raw_critic, peer_device, device)
            critic_steps += 1
            critic_samples += grp_samples
            sum_value_loss += grp_loss
            grad_norms.append(float(grad_norm))

    return {
        "critic_steps": critic_steps,
        "critic_overflow_steps": overflow_steps,
        "critic_samples": critic_samples,
        "critic_update_seconds": time.perf_counter() - began,
        "metrics": {
            "value_loss": sum_value_loss / float(max(1, critic_samples)) if critic_samples else 0.0,
            "critic_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        },
    }


def optimize_state_separate(
    actor,
    critic,
    behavior,
    actor_optimizer,
    critic_optimizer,
    adapter,
    segments: list[dict],
    cfg: StatePPOConfig,
    *,
    rng: np.random.Generator,
    actor_device: torch.device | str,
    critic_device: torch.device | str | None = None,
    peer_critic=None,
    peer_device: torch.device | str | None = None,
    peer_actor=None,
    peer_actor_device: torch.device | str | None = None,
    peer_behavior=None,
    anchor=None,
    anchor_device: torch.device | str | None = None,
    actor_enabled: bool = True,
    lr_controller=None,
) -> dict:
    """Run full v3 PPO optimization with separate/concurrent Actor and Critic updates."""
    actor_device = torch.device(actor_device)
    critic_device = actor_device if critic_device is None else torch.device(critic_device)
    peer_device = None if peer_device is None else torch.device(peer_device)
    peer_actor_device = None if peer_actor_device is None else torch.device(peer_actor_device)
    shared_critic = critic is None or critic is actor

    if cfg.full_episode_bptt:
        versions = {seg.get("behavior_version") for seg in segments}
        if None in versions or len(versions) > 1:
            raise ValueError("Full-episode BPTT cannot mix behavior policy versions")
        for seg in segments:
            if not seg.get("full_episode", False) or not seg.get("has_complete_prefix", False):
                raise ValueError("Full-episode BPTT requires complete, single-version episodes from reset")
            if len(seg["raw_obs"]) and not (seg["terminated"][-1] or seg["truncated"][-1]):
                raise ValueError("Full-episode BPTT requires an actual episode boundary")
            if seg.get("h0_actor") is not None:
                raise ValueError("Full-episode BPTT must start from empty Actor memory")

    targets_info = prepare_state_rollout_targets(
        segments,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        critic_gae_lambda=cfg.critic_gae_lambda,
        choice_only=cfg.actor_choice_only,
        critic_warmup=not actor_enabled,
    )

    anchor_cache_per_seg: dict[int, torch.Tensor] | None = None
    if actor_enabled and cfg.anchor_kl_coef > 0 and anchor is not None:
        anchor_dev = actor_device if anchor_device is None else torch.device(anchor_device)
        hand_indices = np.asarray(adapter.schema.header["hand_card"], dtype=np.int64)
        anchor_cache_per_seg = _build_actor_logp_cache(
            anchor,
            adapter,
            segments,
            cfg,
            model_device=anchor_dev,
            store_device=actor_device,
            h0_key="h0_anchor",
            hand_indices=hand_indices,
        )
        for seg in segments:
            seg.pop("h0_anchor", None)

    if shared_critic:
        actor_result = _run_actor_optimization(
            actor,
            behavior,
            actor_optimizer,
            adapter,
            segments,
            cfg,
            device=actor_device,
            rng=rng,
            shared_critic=True,
            actor_enabled=actor_enabled,
            lr_controller=lr_controller,
            peer_actor=peer_actor,
            peer_device=peer_actor_device,
            peer_behavior=peer_behavior,
            anchor_cache_per_seg=anchor_cache_per_seg,
        )
        metrics = {
            **actor_result["metrics"],
            "value_loss": actor_result["metrics"].get("shared_value_loss", 0.0),
            "critic_grad_norm": actor_result["metrics"]["actor_grad_norm"],
            "explained_variance": targets_info["explained_variance"],
            "advantage_std": targets_info["advantage_std"],
        }
        metrics["loss"] = (
            metrics["policy_loss"]
            + cfg.value_coef * metrics["value_loss"]
            - cfg.entropy_coef * metrics["entropy"]
        )
        return {
            **actor_result,
            "critic_steps": actor_result["critic_steps"],
            "critic_overflow_steps": actor_result["actor_overflow_steps"],
            "critic_samples": actor_result["critic_samples"],
            "critic_update_seconds": actor_result["actor_update_seconds"],
            "actor_choice_samples": targets_info["actor_eligible_count"],
            "warmup_mc_games": targets_info["warmup_mc_games"],
            "metrics": metrics,
        }

    critic_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    critic_out: dict = {}
    critic_error: list[BaseException] = []
    concurrent = critic_device != actor_device
    critic_done_event = threading.Event() if concurrent else None

    def run_critic():
        try:
            c_ctx = torch.cuda.device(critic_device) if critic_device.type == "cuda" else nullcontext()
            with c_ctx:
                critic_out.update(
                    _run_critic_optimization(
                        critic,
                        critic_optimizer,
                        adapter,
                        segments,
                        cfg,
                        device=critic_device,
                        rng=critic_rng,
                        peer_critic=peer_critic,
                        peer_device=peer_device,
                    )
                )
                critic_optimizer.zero_grad(set_to_none=True)
                if critic_device.type == "cuda":
                    torch.cuda.synchronize(critic_device)
            if critic_done_event is not None:
                critic_done_event.set()
        except BaseException as exc:  # pragma: no cover - propagated immediately
            critic_error.append(exc)

    critic_thread = None
    if concurrent:
        ensure_cuda_linalg_initialized(actor_device)
        ensure_cuda_linalg_initialized(critic_device)
        ensure_cuda_linalg_initialized(peer_device)
        ensure_cuda_linalg_initialized(peer_actor_device)
        critic_thread = threading.Thread(target=run_critic)
        critic_thread.start()

    if actor_enabled:
        peer_ready_event = (
            critic_done_event
            if (concurrent and peer_actor_device is not None and peer_actor_device in (critic_device, peer_device))
            else None
        )
        actor_result = _run_actor_optimization(
            actor,
            behavior,
            actor_optimizer,
            adapter,
            segments,
            cfg,
            device=actor_device,
            rng=rng,
            shared_critic=False,
            lr_controller=lr_controller,
            peer_actor=peer_actor,
            peer_device=peer_actor_device,
            peer_behavior=peer_behavior,
            peer_ready_event=peer_ready_event,
            anchor_cache_per_seg=anchor_cache_per_seg,
        )
    else:
        actor_lr_event = None
        if lr_controller is not None:
            actor_lr_event = lr_controller.finish_update(
                actor_optimizer, actor_enabled=False, committed_steps=0
            )
        actor_result = {
            "actor_steps": 0,
            "actor_overflow_steps": 0,
            "actor_samples": 0,
            "kl_stops": 0,
            "rejected_groups": 0,
            "stopped_early": False,
            "initial_logp_errors": [],
            "actor_lr_event": actor_lr_event,
            "actor_update_seconds": 0.0,
            "reference_seconds": 0.0,
            "metrics": {
                "policy_loss": 0.0,
                "entropy": 0.0,
                "card_entropy": 0.0,
                "tile_entropy": 0.0,
                "approx_kl": 0.0,
                "card_kl": 0.0,
                "tile_kl": 0.0,
                "reference_kl": 0.0,
                "anchor_kl": 0.0,
                "clip_fraction": 0.0,
                "mean_ratio": 1.0,
                "actor_grad_norm": 0.0,
            },
        }

    if concurrent and critic_thread is not None:
        critic_thread.join()
    else:
        run_critic()

    if critic_error:
        raise critic_error[0]

    combined_metrics = {
        **actor_result["metrics"],
        **critic_out["metrics"],
        "explained_variance": targets_info["explained_variance"],
        "advantage_std": targets_info["advantage_std"],
    }
    combined_metrics["loss"] = (
        combined_metrics["policy_loss"]
        + cfg.value_coef * combined_metrics["value_loss"]
        - cfg.entropy_coef * combined_metrics["entropy"]
    )
    return {
        "actor_steps": actor_result["actor_steps"],
        "critic_steps": critic_out["critic_steps"],
        "actor_overflow_steps": actor_result["actor_overflow_steps"],
        "critic_overflow_steps": critic_out["critic_overflow_steps"],
        "actor_samples": actor_result["actor_samples"],
        "critic_samples": critic_out["critic_samples"],
        "actor_choice_samples": targets_info["actor_eligible_count"],
        "kl_stops": actor_result["kl_stops"],
        "rejected_groups": actor_result["rejected_groups"],
        "stopped_early": actor_result["stopped_early"],
        "initial_logp_errors": actor_result["initial_logp_errors"],
        "actor_lr_event": actor_result["actor_lr_event"],
        "actor_update_seconds": actor_result["actor_update_seconds"],
        "critic_update_seconds": critic_out["critic_update_seconds"],
        "reference_seconds": actor_result["reference_seconds"],
        "warmup_mc_games": targets_info["warmup_mc_games"],
        "metrics": combined_metrics,
    }


def state_ppo_update(model, optimizer, sequences: Iterable[dict], *, config=None, scaler=None):
    """Sequence PPO update API preserved for smoke tests and shared-critic callers."""
    cfg = config or StatePPOConfig()
    seq_list = list(sequences)
    if not seq_list:
        raise ValueError("Expected at least one trajectory sequence")
    device = next(model.parameters()).device
    raw_model = model.module if hasattr(model, "module") else model
    model.train()
    if scaler is not None and getattr(optimizer, "amp_scaler", None) is None:
        optimizer.amp_scaler = scaler
    tbptt = max(1, int(getattr(cfg, "tbptt_chunk", 64)))
    epochs = max(1, int(getattr(cfg, "epochs", 1)))
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "mean_ratio": 0.0,
        "grad_norm": 0.0,
        "samples": 0,
        "sequences": len(seq_list),
    }
    updates = 0
    overflow_steps = 0

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_samples = 0
        sum_loss = 0.0
        sum_pol = 0.0
        sum_val = 0.0
        sum_ent = 0.0
        sum_ratio = 0.0
        sum_kl = 0.0

        for seq in seq_list:
            t_len = int(seq["header"].shape[0])
            if t_len == 0:
                continue
            valid = seq.get("valid")
            if valid is None:
                valid = torch.ones(t_len, dtype=torch.bool, device=device)
            else:
                valid = torch.as_tensor(valid, dtype=torch.bool, device=device)
            if not valid.any():
                continue

            h0 = seq.get("initial_state")
            state = (
                raw_model.initial_state(1, device=device)
                if h0 is None
                else PolicyMemory(h0.temporal.detach(), id(raw_model), raw_model.memory_generation)
            )
            header = seq["header"].to(device)
            entities = seq["entities"].to(device)
            entity_mask = seq["entity_mask"].to(device)
            legal = seq["legal_action_mask"].to(device)
            hands = seq["hand_card_ids"].to(device)
            actions = seq["actions"].to(device)
            old_lp = (seq["old_log_probs"] if "old_log_probs" in seq else seq["old_logp"]).to(device)
            returns = seq["returns"].to(device)
            adv = seq["advantages"].to(device)

            for start in range(0, t_len, tbptt):
                end = min(t_len, start + tbptt)
                chunk_valid = valid[start:end]
                c_cnt = int(chunk_valid.sum().item())
                state = rebind_memory(state, raw_model)
                if c_cnt == 0:
                    with torch.no_grad(), _amp_context(device, cfg.precision):
                        _, _, state = model.forward_state(
                            header[start:end],
                            entities[start:end],
                            entity_mask[start:end],
                            legal[start:end],
                            state=state,
                            hand_card_ids=hands[start:end],
                            compute_value=True,
                        )
                    state = raw_model.detach_state(state)
                    continue

                with _amp_context(device, cfg.precision):
                    logp, values, state = model.forward_state(
                        header[start:end],
                        entities[start:end],
                        entity_mask[start:end],
                        legal[start:end],
                        state=state,
                        hand_card_ids=hands[start:end],
                        compute_value=True,
                    )
                state = raw_model.detach_state(state)
                sel_logp = logp[chunk_valid].float()
                sel_val = values[chunk_valid].float()
                sel_act = actions[start:end][chunk_valid]
                sel_old = old_lp[start:end][chunk_valid].float()
                sel_ret = returns[start:end][chunk_valid].float()
                sel_adv = adv[start:end][chunk_valid].float()

                new_lp = sel_logp.gather(-1, sel_act[:, None]).squeeze(-1)
                if epoch == 0 and updates == 0 and cfg.max_initial_logp_error is not None:
                    err = float((new_lp - sel_old).abs().max().detach().cpu())
                    if err > cfg.max_initial_logp_error:
                        raise RuntimeError(
                            f"Initial log-prob mismatch {err:.6f} > {cfg.max_initial_logp_error}"
                        )

                log_ratio = new_lp - sel_old
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(
                    ratio * sel_adv,
                    ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * sel_adv,
                ).mean()
                value_loss = 0.5 * (sel_val - sel_ret).square().mean()
                card_h, tile_h = factorized_entropy(sel_logp)
                entropy = (card_h + tile_h).mean()
                if cfg.card_entropy_coef is not None and cfg.tile_entropy_coef is not None:
                    entropy_term = cfg.card_entropy_coef * card_h.mean() + cfg.tile_entropy_coef * tile_h.mean()
                else:
                    entropy_term = cfg.entropy_coef * entropy
                loss = policy_loss + cfg.value_coef * value_loss - entropy_term
                backward_loss(loss * float(c_cnt), optimizer)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    sum_kl += float(approx_kl) * c_cnt
                    sum_loss += float(loss.detach()) * c_cnt
                    sum_pol += float(policy_loss.detach()) * c_cnt
                    sum_val += float(value_loss.detach()) * c_cnt
                    sum_ent += float(entropy.detach()) * c_cnt
                    sum_ratio += float(ratio.mean().detach()) * c_cnt
                    epoch_samples += c_cnt

        if epoch_samples == 0:
            optimizer.zero_grad(set_to_none=True)
            continue

        mean_kl = sum_kl / float(epoch_samples)
        if cfg.target_kl is not None and mean_kl > cfg.target_kl:
            optimizer.zero_grad(set_to_none=True)
            break

        grad_norm = _optimizer_step(raw_model, optimizer, float(epoch_samples), cfg.max_grad_norm)
        if grad_norm is None:
            overflow_steps += 1
            continue

        raw_model.mark_weights_updated()
        updates += 1
        totals["samples"] = max(totals["samples"], epoch_samples)
        totals["loss"] += sum_loss / float(epoch_samples)
        totals["policy_loss"] += sum_pol / float(epoch_samples)
        totals["value_loss"] += sum_val / float(epoch_samples)
        totals["entropy"] += sum_ent / float(epoch_samples)
        totals["mean_ratio"] += sum_ratio / float(epoch_samples)
        totals["grad_norm"] += float(grad_norm)

    denom = max(1, updates)
    return {
        **{k: v / denom for k, v in totals.items() if k not in ("samples", "sequences")},
        "samples": totals["samples"],
        "sequences": totals["sequences"],
        "updates": updates,
        "overflow_steps": overflow_steps,
    }
