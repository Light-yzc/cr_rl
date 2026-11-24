"""Independent actor/value updates with fixed rollout targets and critic warmup."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Lock

import numpy as np
import torch

from cr_rl.training.fp16 import FP32MasterAdamW, autocast_context, backward_loss
from cr_rl.training.ppo import (
    factorized_entropy,
    factorized_reverse_kl,
    ppo_loss_with_reference,
    stop_for_factorized_kl,
    stop_for_kl,
)


def complete_episode_returns(rewards, terminated, truncated, valid, gamma):
    """MC targets only for complete, fault-free episodes contained in this rollout.

    Partial tails and truncated/faulted games are excluded, never assigned a fake
    terminal zero. The simulator is reset at each rollout collection boundary.
    """
    targets = np.zeros_like(rewards, dtype=np.float32)
    selected = np.zeros_like(valid, dtype=bool)
    games = 0
    for env in range(rewards.shape[1]):
        left = 0
        for step in range(len(rewards)):
            if terminated[step, env] or truncated[step, env]:
                if terminated[step, env] and valid[left : step + 1, env].all():
                    total = 0.0
                    for row in range(step, left - 1, -1):
                        total = float(rewards[row, env]) + gamma * total
                        targets[row, env] = total
                    selected[left : step + 1, env] = True
                    games += 1
                left = step + 1
    return targets, selected, games


def forward_inputs(model, tensors):
    options = {"frame_indices": tensors[8]} if len(tensors) > 8 else {}
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), None)
    inputs = list(tensors[:5])
    if dtype is not None:
        inputs = [value.to(dtype=dtype) if value.is_floating_point() else value for value in inputs]
    return model(
        *inputs,
        hand_card_ids=tensors[5],
        deck_card_ids=tensors[6],
        enemy_seen_card_ids=tensors[7],
        **options,
    )


def move_inputs(values, device):
    """Transfer raw histories and optional deduplicated frame layouts over CUDA P2P."""
    if isinstance(values, torch.Tensor):
        return values.to(device, non_blocking=True)
    return tuple(move_inputs(value, device) for value in values)


def autocast_for(device):
    return autocast_context(device)


def _divide_gradients(model, samples):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(samples)


def _optimizer_step(model, optimizer, samples, max_grad_norm):
    if isinstance(optimizer, FP32MasterAdamW):
        optimizer.copy_grads_to_master(samples)
        norm = torch.nn.utils.clip_grad_norm_(
            optimizer.master_parameters,
            max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(
                model.parameters(), optimizer.master_parameters, strict=True
            ):
                parameter.copy_(master)
        optimizer.zero_grad(set_to_none=True)
        return float(norm.detach().cpu())
    scaler = getattr(optimizer, "amp_scaler", None)
    if scaler is not None:
        scaler.unscale_(optimizer)
    _divide_gradients(model, samples)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
        error_if_nonfinite=not (scaler is not None and scaler.is_enabled()),
    )
    if not torch.isfinite(norm):
        if scaler is None or not scaler.is_enabled():
            raise FloatingPointError("Non-finite optimizer gradient norm")
        # unscale_ already recorded overflow before clipping touched the gradients.
        # Never apply an invalid update, including a finite-gradient norm overflow.
        next_scale = max(1.0, float(scaler.get_scale()) * float(scaler.get_backoff_factor()))
        scaler.update(new_scale=next_scale)
        optimizer.zero_grad(set_to_none=True)
        return None
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach().cpu())


def optimizer_steps(optimizer):
    return max((int(s["step"].item()) for s in optimizer.state.values()), default=0)


def bounded_microbatches(
    permutation, requested, effective, make_batch, frame_budget, parallel_factor=None
):
    """Keep exact optimizer groups while shrinking dense raw-history microbatches.

    The budget is microbatch rows times maximum entities per frame, derived from
    the measured V100 memory gate. The order and effective batch never change.
    """
    cursor = in_group = 0
    limits = getattr(make_batch, "limits", None)
    while cursor < len(permutation):
        factor = parallel_factor() if parallel_factor else 1
        size = min(requested * factor, effective - in_group, len(permutation) - cursor)
        if limits is not None:
            while (
                size > 1
                and size * int(limits[permutation[cursor : cursor + size]].max())
                > frame_budget * factor
            ):
                size = max(1, size // 2)
        indices = permutation[cursor : cursor + size]
        cursor += size
        in_group = (in_group + size) % effective
        yield indices, cursor == len(permutation)


def optimize_separate(
    actor,
    critic,
    behavior,
    actor_optimizer,
    critic_optimizer,
    make_batch,
    actor_indices,
    critic_indices,
    old_logp,
    actions,
    advantages,
    value_targets,
    *,
    rng,
    device,
    batch_size,
    accumulation,
    actor_epochs,
    critic_epochs,
    clip_coef,
    entropy_coef,
    reference_kl_coef,
    target_kl,
    max_grad_norm,
    actor_enabled=True,
    cache_reference=False,
    defer_scalar_metrics=False,
    actor_batch_size=None,
    critic_batch_size=None,
    actor_effective_batch_size=None,
    target_card_kl=None,
    target_tile_kl=None,
    max_initial_logp_error=None,
    card_entropy_coef=None,
    tile_entropy_coef=None,
    continue_after_kl_reject=False,
    emit=lambda **event: None,
    critic_device=None,
    critic_peer=None,
    actor_lr_controller=None,
):
    """Critic fits independently, even when actor is frozen or fails its KL guard.

    make_batch maps flat rollout indices to raw-observation history tensors.
    Targets/log probabilities are snapshots, never recomputed after value fitting.
    """
    emit_unlocked = emit
    event_lock = Lock()

    def emit(**event):
        with event_lock:
            emit_unlocked(**event)

    targets = value_targets.detach().clone()
    split_entropy = card_entropy_coef is not None or tile_entropy_coef is not None
    if split_entropy and (
        card_entropy_coef is None
        or tile_entropy_coef is None
        or not all(np.isfinite(c) and c >= 0 for c in (card_entropy_coef, tile_entropy_coef))
    ):
        raise ValueError("Provide two finite nonnegative entropy coefficients")
    factorized_guard = target_card_kl is not None or target_tile_kl is not None
    if factorized_guard and (target_card_kl is None or target_tile_kl is None):
        raise ValueError("Both card and tile KL limits are required")
    if factorized_guard:
        stop_for_factorized_kl(actor_optimizer, 0.0, 0.0, target_card_kl, target_tile_kl)
    if actor_lr_controller is not None and not factorized_guard:
        raise ValueError("Adaptive Actor LR requires the factorized Card/Tile KL guard")
    effective_batch = batch_size * accumulation
    actor_effective = (
        effective_batch if actor_effective_batch_size is None else actor_effective_batch_size
    )
    actor_batch = batch_size if actor_batch_size is None else actor_batch_size
    critic_batch = batch_size if critic_batch_size is None else critic_batch_size
    if min(effective_batch, actor_effective, actor_batch, critic_batch) < 1 or (
        actor_effective % actor_batch or effective_batch % critic_batch
    ):
        raise ValueError("Actor/Critic microbatches must divide their fixed effective batch")
    if actor_lr_controller is not None:
        actor_lr_controller.begin_update(actor_optimizer)
    advantages = advantages.detach().clone()
    old_logp = old_logp.detach().clone()
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    actor_steps = value_steps = 0
    value_losses, value_weights = [], []
    policy_metrics, rejected_metrics = [], []
    policy_weights, rejected_weights = [], []
    actor_presentations = critic_presentations = 0
    actor_rejected_samples = actor_proposed_samples = 0
    actor_clipped_steps = critic_clipped_steps = 0
    actor_overflow_steps = critic_overflow_steps = 0
    actor_unique = set()
    kl_rejected_any = False
    initial_logp_errors = []
    reference_cache = None
    if actor_enabled and cache_reference:
        from cr_rl.training.reference_cache import ReferenceDistributionCache

        reference_cache = ReferenceDistributionCache(len(old_logp))
    value_device = critic_device or device
    parallel = torch.device(value_device) != torch.device(device)
    if critic_peer is not None:
        if not parallel:
            raise ValueError("Balanced Critic requires separate Actor/Critic devices")
        critic_peer.begin_update(actor_enabled, critic_optimizer)
    critic_targets = targets.to(value_device)
    # Draw permutations before the worker starts: the shared NumPy RNG is never raced.
    critic_permutations = [rng.permutation(critic_indices) for _ in range(critic_epochs)]

    def fit_critic():
        nonlocal value_steps, critic_clipped_steps, critic_presentations, critic_overflow_steps
        context = (
            torch.cuda.device(value_device)
            if str(value_device).startswith("cuda")
            else nullcontext()
        )
        with context:
            critic.train()
            for epoch in range(critic_epochs):
                permutation = critic_permutations[epoch]
                accumulated = samples = 0
                for indices, last_batch in bounded_microbatches(
                    permutation,
                    critic_batch,
                    effective_batch,
                    make_batch,
                    16 * 64,
                    parallel_factor=(lambda: 2 if critic_peer.ready else 1)
                    if critic_peer
                    else None,
                ):
                    if critic_peer is not None:
                        loss = critic_peer.loss_and_backward(
                            make_batch, indices, critic_targets, critic_optimizer
                        )
                    else:
                        with autocast_for(value_device):
                            predictions = forward_inputs(
                                critic, move_inputs(make_batch(indices), value_device)
                            ).float()
                            loss = 0.5 * (predictions - critic_targets[indices]).square().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite independent value loss")
                        backward_loss(loss * len(indices), critic_optimizer)
                    samples += len(indices)
                    accumulated += 1
                    value_losses.append(
                        loss.detach() if defer_scalar_metrics else float(loss.detach().cpu())
                    )
                    value_weights.append(len(indices))
                    critic_presentations += len(indices)
                    if samples == effective_batch or last_batch:
                        if critic_peer is not None:
                            critic_peer.reduce_gradients()
                        norm = _optimizer_step(critic, critic_optimizer, samples, max_grad_norm)
                        if norm is None:
                            critic_overflow_steps += 1
                            critic_presentations -= samples
                            emit(
                                phase="amp_overflow", component="critic",
                                skipped_samples=samples,
                                scale=critic_optimizer.amp_scaler.get_scale(),
                                overflow_steps=critic_overflow_steps,
                            )
                            accumulated = samples = 0
                            continue
                        if critic_peer is not None:
                            critic_peer.primary_updated()
                        value_steps += 1
                        critic_clipped_steps += int(norm > max_grad_norm)
                        batch_value_loss = (
                            sum(
                                value * count
                                for value, count in zip(
                                    value_losses[-accumulated:],
                                    value_weights[-accumulated:],
                                    strict=True,
                                )
                            )
                            / samples
                        )
                        emit(
                            phase="critic_optimization",
                            critic_peer_rows=critic_peer.rows if critic_peer else 0,
                            epoch=epoch + 1,
                            critic_optimizer_steps=optimizer_steps(critic_optimizer),
                            value_loss=float(batch_value_loss),
                            grad_norm=norm,
                            grad_was_clipped=norm > max_grad_norm,
                            batch_samples=samples,
                            microbatches=accumulated,
                            critic_samples_committed=critic_presentations,
                            learning_rate=float(critic_optimizer.param_groups[0]["lr"]),
                        )
                        accumulated = samples = 0
            critic.eval()

    executor = ThreadPoolExecutor(max_workers=1) if parallel else None
    future = executor.submit(fit_critic) if executor else None
    if not parallel:
        fit_critic()
    try:
        actor.train()
        stopped = False
        if actor_enabled:
            for epoch in range(actor_epochs):
                permutation = rng.permutation(actor_indices)
                accumulated = samples = 0
                group_kl = group_approx_kl = group_card_kl = group_tile_kl = 0.0
                pending_metrics, pending_weights, pending_indices, pending_losses = [], [], [], []
                for indices, last_batch in bounded_microbatches(
                    permutation, actor_batch, actor_effective, make_batch, 8 * 64
                ):
                    tensors = make_batch(indices)
                    with torch.no_grad(), autocast_for(device):
                        if reference_cache is None:
                            reference, _ = forward_inputs(behavior, tensors)
                        else:
                            reference = reference_cache.evaluate(
                                indices,
                                lambda tensors=tensors: forward_inputs(behavior, tensors)[0],
                            )
                    with autocast_for(device):
                        logp_all, _ = forward_inputs(actor, tensors)
                    distribution = torch.distributions.Categorical(logits=logp_all.float())
                    logp = distribution.log_prob(actions[indices])
                    if actor_steps == 0:
                        initial_logp_errors.append(
                            (logp.detach() - old_logp[indices]).abs().max()
                            if defer_scalar_metrics
                            else float((logp.detach() - old_logp[indices]).abs().max())
                        )
                    loss, metrics = ppo_loss_with_reference(
                        logp_all,
                        reference,
                        logp,
                        old_logp[indices],
                        advantages[indices],
                        torch.zeros_like(logp),
                        torch.zeros_like(logp),
                        distribution.entropy(),
                        clip_coef=clip_coef,
                        value_coef=0.0,
                        entropy_coef=0.0 if split_entropy else entropy_coef,
                        reference_kl_coef=reference_kl_coef,
                    )
                    if split_entropy:
                        card_entropy, tile_entropy = factorized_entropy(logp_all)
                        bonus = (
                            card_entropy_coef * card_entropy.mean()
                            + tile_entropy_coef * tile_entropy.mean()
                        )
                        loss = loss - bonus
                        metrics.update(
                            card_entropy=card_entropy.detach().mean(),
                            tile_entropy=tile_entropy.detach().mean(),
                            entropy_bonus=bonus.detach(),
                        )
                    if factorized_guard:
                        with torch.no_grad():
                            card_kl, tile_kl = factorized_reverse_kl(logp_all, reference)
                            metrics.update(card_kl=card_kl.mean(), tile_kl=tile_kl.mean())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite independent policy loss")
                    row = {
                        k: v.detach() if defer_scalar_metrics else float(v.cpu())
                        for k, v in metrics.items()
                    }
                    # Value fitting has its own real loss/axis; policy updates have no value term.
                    row.pop("value_loss")
                    pending_metrics.append(row)
                    pending_weights.append(len(indices))
                    pending_indices.extend(indices.tolist())
                    pending_losses.append(loss.detach() * len(indices))
                    actor_proposed_samples += len(indices)
                    group_kl += (
                        row["reference_kl"].double()
                        if defer_scalar_metrics
                        else row["reference_kl"]
                    ) * len(indices)
                    group_approx_kl += (
                        row["approx_kl"].double() if defer_scalar_metrics else row["approx_kl"]
                    ) * len(indices)
                    if factorized_guard:
                        group_card_kl += row["card_kl"] * len(indices)
                        group_tile_kl += row["tile_kl"] * len(indices)
                    backward_loss(loss * len(indices), actor_optimizer)
                    samples += len(indices)
                    accumulated += 1
                    if samples == actor_effective or last_batch:
                        # Judge the complete proposed optimizer batch, never a single microbatch.
                        average_kl = group_kl / samples
                        average_approx_kl = group_approx_kl / samples
                        # Diagnostics describe the complete effective batch. These
                        # reductions do not change gradients or the KL decision.
                        row = {
                            name: sum(
                                item[name] * count
                                for item, count in zip(
                                    pending_metrics, pending_weights, strict=True
                                )
                            )
                            / samples
                            for name in row
                        }
                        batch_loss = torch.stack(pending_losses).sum() / samples
                        if defer_scalar_metrics:
                            names = list(row)
                            scalars = (
                                torch.stack(
                                    [row[name].double() for name in names]
                                    + [average_kl, average_approx_kl, batch_loss.double()]
                                )
                                .cpu()
                                .tolist()
                            )
                            row = dict(zip(names, scalars[: len(names)], strict=True))
                            average_kl, average_approx_kl, current_loss = scalars[-3:]
                        if actor_steps == 0 and max_initial_logp_error is not None:
                            maximum_error = max(float(x) for x in initial_logp_errors)
                            if maximum_error > max_initial_logp_error:
                                actor_optimizer.zero_grad(set_to_none=True)
                                raise ValueError(
                                    "Collected old log-prob differs from the initial Actor"
                                )
                        components = {}
                        if factorized_guard:
                            average_card = float(group_card_kl / samples)
                            average_tile = float(group_tile_kl / samples)
                            components = dict(
                                card_kl=average_card,
                                tile_kl=average_tile,
                                target_card_kl=target_card_kl,
                                target_tile_kl=target_tile_kl,
                            )
                            reject = stop_for_factorized_kl(
                                actor_optimizer,
                                average_card,
                                average_tile,
                                target_card_kl,
                                target_tile_kl,
                            )
                        else:
                            reject = stop_for_kl(actor_optimizer, average_kl, target_kl)
                        if actor_lr_controller is not None:
                            actor_lr_controller.observe(
                                card_kl=average_card,
                                tile_kl=average_tile,
                                target_card_kl=target_card_kl,
                                target_tile_kl=target_tile_kl,
                                samples=samples,
                                rejected=reject,
                            )
                        if reject:
                            kl_rejected_any = True
                            rejected_metrics.extend(pending_metrics)
                            rejected_weights.extend(pending_weights)
                            actor_rejected_samples += samples
                            emit(
                                phase="kl_stop",
                                epoch=epoch + 1,
                                approx_kl=average_approx_kl,
                                reference_kl=average_kl,
                                target_kl=target_kl,
                                discarded_microbatches=accumulated,
                                kl_samples=samples,
                                optimizer_steps=actor_steps,
                                actor_samples_committed=actor_presentations,
                                actor_samples_rejected=actor_rejected_samples,
                                actor_samples_proposed=actor_proposed_samples,
                                **components,
                            )
                            if continue_after_kl_reject:
                                # Discard only this over-limit effective group and
                                # keep scanning the rollout.  A single noisy group
                                # must not freeze the entire Actor update; the
                                # adaptive LR controller still records the reject.
                                accumulated = samples = 0
                                pending_metrics, pending_weights = [], []
                                pending_indices, pending_losses = [], []
                                group_kl = group_approx_kl = 0.0
                                group_card_kl = group_tile_kl = 0.0
                                continue
                            stopped = True
                            break
                        norm = _optimizer_step(actor, actor_optimizer, samples, max_grad_norm)
                        if norm is None:
                            actor_overflow_steps += 1
                            actor_rejected_samples += samples
                            emit(
                                phase="amp_overflow", component="actor",
                                skipped_samples=samples,
                                scale=actor_optimizer.amp_scaler.get_scale(),
                                overflow_steps=actor_overflow_steps,
                            )
                            accumulated = samples = 0
                            pending_metrics, pending_weights = [], []
                            pending_indices, pending_losses = [], []
                            group_kl = group_approx_kl = group_card_kl = group_tile_kl = 0.0
                            continue
                        actor_steps += 1
                        actor_clipped_steps += int(norm > max_grad_norm)
                        actor_presentations += samples
                        actor_unique.update(pending_indices)
                        policy_metrics.extend(pending_metrics)
                        policy_weights.extend(pending_weights)
                        emit(
                            phase="optimization",
                            optimizer_steps=actor_steps,
                            optimizer_steps_total=optimizer_steps(actor_optimizer),
                            loss=current_loss if defer_scalar_metrics else float(batch_loss),
                            grad_norm=norm,
                            grad_was_clipped=norm > max_grad_norm,
                            batch_samples=samples,
                            actor_samples_committed=actor_presentations,
                            actor_unique_samples_committed=len(actor_unique),
                            learning_rate=float(actor_optimizer.param_groups[0]["lr"]),
                            **{
                                **row,
                                "reference_kl": average_kl,
                                "approx_kl": average_approx_kl,
                                **components,
                            },
                        )
                        accumulated = samples = 0
                        pending_metrics, pending_weights = [], []
                        pending_indices, pending_losses = [], []
                        group_kl = group_approx_kl = group_card_kl = group_tile_kl = 0.0
                if stopped:
                    break
        actor.eval()

    finally:
        if critic_peer is not None:
            critic_peer.actor_done.set()
        if executor:
            executor.shutdown(wait=True)
    if future:
        future.result()

    actor_lr_schedule = None
    if actor_lr_controller is not None:
        actor_lr_schedule = actor_lr_controller.finish_update(
            actor_optimizer,
            actor_enabled=actor_enabled,
            committed_steps=actor_steps,
            overflow_steps=actor_overflow_steps,
        )
        emit(phase="actor_lr_schedule", **actor_lr_schedule)

    def mean_metric(values, weights):
        if defer_scalar_metrics:
            values = torch.stack(values).double()
            counts = torch.as_tensor(weights, device=values.device, dtype=torch.float64)
            return float(((values * counts).sum() / counts.sum()).cpu())
        return float(np.average(values, weights=weights))

    def policy_averages(rows, weights):
        if not rows:
            return {}
        return {name: mean_metric([row[name] for row in rows], weights) for name in rows[0]}

    return {
        **({"actor_lr_schedule": actor_lr_schedule} if actor_lr_schedule is not None else {}),
        "critic_parallel": critic_peer.metadata() if critic_peer else {"enabled": False},
        "actor_steps": actor_steps,
        "critic_steps": value_steps,
        "actor_kl_stopped": stopped or kl_rejected_any,
        "sample_utilization": {
            "actor_eligible_unique": len(actor_indices) if actor_enabled else 0,
            "actor_planned_presentations": (
                len(actor_indices) * actor_epochs if actor_enabled else 0
            ),
            "actor_samples_proposed": actor_proposed_samples,
            "actor_samples_committed": actor_presentations,
            "actor_unique_samples_committed": len(actor_unique),
            "actor_samples_rejected": actor_rejected_samples,
            "critic_samples_committed": critic_presentations,
            "actor_grad_clipped_steps": actor_clipped_steps,
            "critic_grad_clipped_steps": critic_clipped_steps,
            "actor_amp_overflow_steps": actor_overflow_steps,
            "critic_amp_overflow_steps": critic_overflow_steps,
        },
        "rejected_policy_metrics": policy_averages(rejected_metrics, rejected_weights),
        "initial_logp_errors": (
            torch.stack(initial_logp_errors).cpu().tolist()
            if defer_scalar_metrics and initial_logp_errors
            else initial_logp_errors
        ),
        "reference_cache": {
            "enabled": reference_cache is not None,
            "forward_calls": reference_cache.forward_calls if reference_cache else 0,
            "hit_rows": reference_cache.hit_rows if reference_cache else 0,
        },
        "metrics": {
            **policy_averages(policy_metrics, policy_weights),
            "value_loss": mean_metric(value_losses, value_weights) if value_losses else None,
        },
    }
