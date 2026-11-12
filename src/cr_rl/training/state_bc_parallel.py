"""Synchronous independent episode streams; one mean-gradient update per group.

Each stream retains its own whole-game causal memory, detaches at the same TBPTT
chunk boundary, and never crosses a parameter update. Padding neither contributes
loss nor advances memory. All streams finish before any optimizer step.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch

from cr_rl.training.bc import hierarchical_bc_loss
from cr_rl.training.state_bc import _autocast, _state_bc_optimizer_step, check_ram_budget, prepare_state_chunk

LOSS_KEYS = ("loss", "timing_loss", "conditional_slot_loss", "position_loss", "card_loss")
METRIC_KEYS = (
    "correct",
    "wait_correct",
    "act_correct",
    "slot_correct",
    "ability_correct",
    "abilities",
    "tile_distance_sum",
)


def prepare_episode_group(episodes, start, chunk_size, adapter, device, *, training):
    """Pad on CPU, then transfer one tensor per field instead of per stream."""
    available = [max(0, min(chunk_size, len(e.rows) - start)) for e in episodes]
    length = max(available)
    chunks = [
        prepare_state_chunk(e, e.rows[start : start + n], adapter, "cpu") if n else None
        for e, n in zip(episodes, available, strict=True)
    ]
    exemplar = next(c for c in chunks if c is not None)
    entity_count = max(c["entities"].shape[1] for c in chunks if c is not None)
    result = {}
    batch = len(episodes)
    for key, sample in exemplar.items():
        trailing = list(sample.shape[1:])
        if key in ("entities", "entity_mask"):
            trailing[0] = entity_count
        values = torch.zeros((batch, length, *trailing), dtype=sample.dtype)
        if key.endswith("card_ids"):
            values.fill_(-1)
        if key == "legal_action_mask":
            values[..., 0] = True
        for i, (chunk, count) in enumerate(zip(chunks, available, strict=True)):
            if chunk is None:
                continue
            value = chunk[key]
            if key in ("entities", "entity_mask"):
                values[i, :count, : value.shape[1]] = value
            else:
                values[i, :count] = value
        result[key] = values.to(device)
    time_mask = torch.arange(length)[None] < torch.tensor(available)[:, None]
    result["time_mask"] = time_mask.to(device)
    supervised = torch.zeros(batch, length, dtype=torch.bool)
    labels = torch.zeros(batch, length, dtype=torch.long)
    for i, (episode, count) in enumerate(zip(episodes, available, strict=True)):
        if count:
            supervised[i, :count] = torch.from_numpy(
                episode.supervised[start : start + count].copy()
            )
            labels[i, :count] = torch.from_numpy(episode.labels[start : start + count].copy())
    if training:
        result["placement_mask"] = (supervised & (labels > 0) & (labels < 2305)).to(device)
    return result, supervised, labels, available


def prediction_packet(lp, target, *, play_count):
    """Device-only diagnostics; copied to host once after the complete group."""
    chosen = lp.argmax(-1)
    wait, play = target == 0, (target > 0) & (target < 2305)
    ability = target >= 2305
    cards = torch.cat(
        (lp[:, :1], torch.logsumexp(lp[:, 1:2305].reshape(-1, 4, 576), -1), lp[:, 2305:]), -1
    )
    predicts_act = torch.logsumexp(cards[:, 1:], -1) > cards[:, 0]
    values = {
        "correct": (chosen == target).sum(),
        "wait_correct": (~predicts_act & wait).sum(),
        "act_correct": (predicts_act & ~wait).sum(),
        "ability_correct": ((chosen == target) & ability).sum(),
        "abilities": ability.sum(),
        "slot_correct": lp.new_zeros(()),
        "tile_distance_sum": lp.new_zeros(()),
    }
    if play_count:
        slot = (target[play] - 1) // 576
        tile = (
            lp[play, 1:2305]
            .reshape(-1, 4, 576)[torch.arange(play_count, device=lp.device), slot]
            .argmax(-1)
        )
        target_tile = (target[play] - 1) % 576
        values["slot_correct"] = (cards[play, 1:5].argmax(-1) == slot).sum()
        values["tile_distance_sum"] = (
            (
                (tile % 18 - target_tile % 18).float().square()
                + (tile // 18 - target_tile // 18).float().square()
            )
            .sqrt()
            .sum()
        )
    return values


def run_state_episode_group(
    model,
    episodes,
    adapter,
    *,
    chunk_size=32,
    device="cpu",
    precision="fp32",
    optimizer=None,
    scaler=None,
    position_coef=0.5,
    tile_label_sigma=1.0,
    timing_objective="natural",
    timing_target_action_prior=0.20,
    sampled_action_prior=None,
    max_grad_norm=1.0,
    ram_limit_gib=3,
    defer_metrics=True,
    collect_gradients=False,
):
    if not episodes:
        raise ValueError("An episode group cannot be empty")
    if timing_objective not in ("natural", "balanced"):
        raise ValueError("Unknown timing objective")
    training = optimizer is not None or collect_gradients
    if training and any(e.normalization is None for e in episodes):
        # Zero-label histories still advance memory; exclude them from the mean.
        active_count = sum(e.normalization is not None for e in episodes)
    else:
        active_count = len(episodes)
    model.train(training)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    memory = model.initial_state(len(episodes), device=device, dtype=dtype)
    packets, packet_streams = [], []
    started = time.perf_counter()
    positions = None
    did_backward = False
    with nullcontext() if training else torch.no_grad():
        for start in range(0, max(len(e.rows) for e in episodes), chunk_size):
            check_ram_budget(ram_limit_gib)
            inputs, supervised_cpu, labels_cpu, available = prepare_episode_group(
                episodes, start, chunk_size, adapter, device, training=training
            )
            supervised, labels = supervised_cpu.to(device), labels_cpu.to(device)
            if training and hasattr(model, 'learned_placement'):
                inputs['placement_slots'] = ((labels - 1) // 576).clamp(0, 3)
            with _autocast(device, precision):
                output, _, memory = model.forward_state(**inputs, state=memory, compute_value=False)
                losses = []
                for stream, episode in enumerate(episodes):
                    selected_np = supervised_cpu[stream].numpy()
                    if not selected_np.any():
                        continue
                    selected = supervised[stream]
                    lp, target = output[stream][selected], labels[stream][selected]
                    loss, components = hierarchical_bc_loss(
                        lp,
                        target,
                        inputs["legal_action_mask"][stream][selected],
                        position_coef=position_coef,
                        tile_label_sigma=tile_label_sigma,
                        conditional_card_loss=True,
                        normalization=episode.normalization,
                        timing_prior=timing_target_action_prior
                        if timing_objective == "natural"
                        else None,
                        sampled_action_prior=sampled_action_prior,
                    )
                    losses.append(loss / max(active_count, 1))
                    with torch.no_grad():
                        cpu_targets = labels_cpu[stream].numpy()[selected_np]
                        play_count = int(((cpu_targets > 0) & (cpu_targets < 2305)).sum())
                        metrics = prediction_packet(lp.float(), target, play_count=play_count)
                        components = {"loss": loss, **components}
                        packets.append(
                            torch.stack(
                                [
                                    *(components[key].detach().float() for key in LOSS_KEYS),
                                    *(metrics[key].float() for key in METRIC_KEYS),
                                ]
                            )
                        )
                        packet_streams.append(stream)
                total_loss = torch.stack(losses).sum() if losses else None
            if total_loss is not None:
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("Nonfinite grouped BC loss")
                if training and total_loss.requires_grad:
                    (scaler.scale(total_loss) if scaler is not None else total_loss).backward()
                    did_backward = True
            memory = model.detach_state(memory)
        positions = memory.temporal.positions.detach().cpu().tolist()
    del memory
    stepped, overflow, norm = False, False, 0.0
    if optimizer is not None and active_count and did_backward:
        stepped, overflow, norm = _state_bc_optimizer_step(model, optimizer, scaler, max_grad_norm)
        if stepped:
            model.mark_weights_updated()
    totals = [dict.fromkeys((*LOSS_KEYS, *METRIC_KEYS), 0.0) for _ in episodes]
    if packets:
        host = torch.stack(packets).detach().cpu().tolist()
        for stream, row in zip(packet_streams, host, strict=True):
            for key, value in zip((*LOSS_KEYS, *METRIC_KEYS), row, strict=True):
                totals[stream][key] += value
    seconds = time.perf_counter() - started
    reports = []
    for stream, episode in enumerate(episodes):
        sums, counts = totals[stream], episode.normalization
        n, a, p, w = (counts.rows, counts.act, counts.play, counts.wait) if counts else (0, 0, 0, 0)
        reports.append(
            {
                **{key: sums[key] for key in LOSS_KEYS},
                "rows": n,
                "act_rows": a,
                "play_rows": p,
                "wait_rows": w,
                "history_frames": len(episode.rows),
                "replay_id": episode.replay_id,
                "seat": episode.seat,
                "exact_accuracy": sums["correct"] / max(n, 1),
                "wait_accuracy": sums["wait_correct"] / max(w, 1),
                "act_recall": sums["act_correct"] / max(a, 1),
                "slot_accuracy": sums["slot_correct"] / max(p, 1),
                "ability_accuracy": sums["ability_correct"] / max(sums["abilities"], 1),
                "tile_mean_distance": sums["tile_distance_sum"] / max(p, 1),
            }
        )
    return {
        "reports": reports,
        "optimizer_stepped": stepped,
        "amp_overflow_skipped": overflow,
        "grad_norm": norm,
        "seconds": seconds,
        "episode_batch_size": len(episodes),
        "history_frames": sum(len(e.rows) for e in episodes),
        "padded_frames": len(episodes) * max(len(e.rows) for e in episodes),
        "memory_positions": positions,
        "rss_mib": check_ram_budget(ram_limit_gib) / 1024**2,
    }
