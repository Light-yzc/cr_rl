from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cr_rl.config import load_yaml
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.data.trajectory import TrajectoryBuffer
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.opponents import OpponentEntry, OpponentPool
from cr_rl.training.advantages import compute_gae
from cr_rl.training.ppo import ppo_loss_with_reference


def _windows(histories: list[deque], context: int, seat_ints: int) -> tuple[np.ndarray, np.ndarray]:
    batch = np.zeros((len(histories), context, seat_ints), dtype=np.int32)
    time_mask = np.zeros((len(histories), context), dtype=np.bool_)
    for index, history in enumerate(histories):
        sequence = list(history)[-context:]
        if sequence:
            batch[index, -len(sequence) :] = np.stack(sequence)
            time_mask[index, -len(sequence) :] = True
    return batch, time_mask


def _policy_inputs(raw: np.ndarray, masks: np.ndarray, histories: list[deque], adapter):
    windows, time_mask = _windows(histories, histories[0].maxlen, raw.shape[-1])
    header, entities, entity_mask = adapter(windows)
    hand_fields = adapter.schema.header["hand_card"]
    hand_ids = raw[:, list(hand_fields)]
    deck_ids = raw[:, list(adapter.schema.header["deck"])]
    enemy_seen_ids = raw[:, list(adapter.schema.header["enemy_seen"])]
    tensors = (
        torch.as_tensor(header),
        torch.as_tensor(entities),
        torch.as_tensor(entity_mask),
        torch.as_tensor(masks.astype(bool)),
        torch.as_tensor(time_mask),
        torch.as_tensor(hand_ids),
        torch.as_tensor(deck_ids),
        torch.as_tensor(enemy_seen_ids),
    )
    return tensors


def _forward(model, tensors, device):
    amp_enabled = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ),
    ):
        logits, values = model(
            *(tensor.to(device) for tensor in tensors[:5]),
            hand_card_ids=tensors[5].to(device),
            deck_card_ids=tensors[6].to(device),
            enemy_seen_card_ids=tensors[7].to(device),
        )
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        return (
            actions.cpu().numpy(),
            distribution.log_prob(actions).cpu().numpy(),
            values.cpu().numpy(),
        )


def train(args) -> None:
    from cr_rl.envs.crforge import CrForgeVector

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    cfg = ModelConfig.preset(args.model)
    model = ActorCritic(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, foreach=False)
    update_number = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if int(state.get("model_architecture_version", 0)) != 2:
            raise ValueError(
                "Checkpoint predates deck/public-seen-card embeddings; export weights and "
                "reinitialize the optimizer before migrating it to architecture v2"
            )
        if state["config"] != cfg.__dict__:
            raise ValueError("Resume checkpoint model config does not match --model")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        update_number = int(state.get("updates", 0))
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        if "numpy_rng_state" in state:
            rng.bit_generator.state = state["numpy_rng_state"]
        del state

    deck_cfg = load_yaml("decks.yaml")["decks"]
    deck_names = list(deck_cfg)
    decks = [deck_cfg[name] for name in deck_names]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    temporary_snapshot = output / f"opponent_{update_number:06d}.tmp.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "model_architecture_version": 2,
            "policy_id": update_number,
        },
        temporary_snapshot,
    )
    snapshot_hash = hashlib.sha256(temporary_snapshot.read_bytes()).hexdigest()
    snapshot_path = output / f"opponent_{update_number:06d}_{snapshot_hash[:12]}.pt"
    temporary_snapshot.replace(snapshot_path)
    pool_path = output / "opponent_pool.json"
    pool = OpponentPool.load(pool_path) if pool_path.exists() else OpponentPool()
    if not any(entry.policy_id == update_number for entry in pool.entries):
        pool.add(
            OpponentEntry(
                policy_id=update_number,
                checkpoint=str(snapshot_path.resolve()),
                generation=update_number,
                deck_id="multi_deck",
            )
        )
    selected_opponents = [
        pool.sample_league(
            rng,
            update_number,
            current_fraction=args.current_opponent_fraction,
            historical_pfsp_fraction=args.historical_pfsp_fraction,
            forgotten_fraction=args.forgotten_opponent_fraction,
        )
        for _ in range(args.envs)
    ]
    selected_opponent_hashes = {
        entry.policy_id: hashlib.sha256(Path(entry.checkpoint).read_bytes()).hexdigest()
        for entry in selected_opponents
    }
    opponent_models = {}
    for policy_id in sorted({entry.policy_id for entry in selected_opponents}):
        entry = next(item for item in selected_opponents if item.policy_id == policy_id)
        opponent_model = copy.deepcopy(model).eval()
        opponent_state = torch.load(entry.checkpoint, map_location=device, weights_only=False)
        opponent_model.load_state_dict(opponent_state["model"])
        del opponent_state
        for parameter in opponent_model.parameters():
            parameter.requires_grad_(False)
        opponent_models[policy_id] = opponent_model
    behavior_model = copy.deepcopy(model).eval()
    for parameter in behavior_model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    started = time.perf_counter()
    with CrForgeVector(args.envs, min(args.envs, 8)) as env:
        # Do not autoreset behind the trainer's back. The final pre-reset observation is
        # required to distinguish a terminal result from a time-limit/fault truncation.
        env.set_autoreset(False)
        for deck in decks:
            if len(deck) != 8 or len(set(deck)) != 8:
                raise ValueError("Every league deck must contain eight distinct cards")
            for card in deck:
                env.vector.card_id(card)  # fail fast on stale/misspelled game-table rows

        deck0_ids = np.arange(args.envs, dtype=np.int32) % len(decks)
        rng.shuffle(deck0_ids)
        deck1_ids = (deck0_ids + 1 + rng.integers(0, len(decks) - 1, size=args.envs)) % len(decks)
        deck0 = [decks[int(index)] for index in deck0_ids]
        deck1 = [decks[int(index)] for index in deck1_ids]
        seeds = rng.integers(1, 2**31 - 1, size=args.envs, dtype=np.int64).astype(np.int32)
        raw_obs, raw_masks = env.reset(seeds.tolist(), deck0, deck1)
        learner_seats = np.arange(args.envs, dtype=np.int32) % 2
        rng.shuffle(learner_seats)
        episode_ids = np.zeros(args.envs, dtype=np.int64)
        histories = [deque(maxlen=cfg.context_len) for _ in range(args.envs * 2)]
        episode_start = np.zeros((args.steps, args.envs), dtype=np.int32)
        observations = np.zeros((args.steps, args.envs, 2, env.vector.schema.seat_ints), np.int32)
        masks = np.zeros((args.steps, args.envs, 2, env.vector.schema.actions), np.uint8)
        next_observations = np.zeros_like(observations)
        next_masks = np.zeros_like(masks)
        actions = np.zeros((args.steps, args.envs, 2), np.int32)
        rewards = np.zeros((args.steps, args.envs), np.float32)
        old_log_probs = np.zeros((args.steps, args.envs, 2), np.float32)
        old_values = np.zeros_like(old_log_probs)
        raw_rewards = np.zeros((args.steps, args.envs, 2), np.float32)
        action_consumed = np.ones((args.steps, args.envs, 2), np.uint8)
        action_refusals = np.full((args.steps, args.envs, 2), -1, np.int32)
        run_refusals = np.full((args.steps, args.envs, 2), -1, np.int32)
        terminated = np.zeros((args.steps, args.envs), np.uint8)
        truncated = np.zeros_like(terminated)
        valid = np.ones((args.steps, args.envs, 2), np.uint8)
        learner_mask = np.zeros_like(valid)
        policy_ids = np.zeros_like(actions)
        opponent_ids = np.zeros_like(actions)
        policy_versions = np.zeros_like(actions)
        episode_id_rows = np.zeros((args.steps, args.envs), np.int64)
        episode_start_flags = np.zeros((args.steps, args.envs), np.uint8)
        episode_steps = np.zeros((args.steps, args.envs), np.int32)
        ticks = np.zeros((args.steps, args.envs), np.int32)
        winners = np.full((args.steps, args.envs), -1, np.int8)
        end_reasons = np.zeros((args.steps, args.envs), np.uint8)
        fault_codes = np.zeros((args.steps, args.envs), np.int16)
        seed_rows = np.zeros_like(episode_id_rows)
        truncation_bootstrap = np.zeros((args.steps, args.envs), np.float32)
        current_episode_start = np.zeros(args.envs, dtype=np.int32)
        adapter = CrForgeObservationAdapter(env.vector.schema, env.vector.vocabulary_size)
        buffer = TrajectoryBuffer(
            {
                **env.metadata(),
                "algorithm": "frozen-opponent-self-play-ppo-v1",
                "model": args.model,
                "model_architecture_version": 2,
                "model_config": cfg.__dict__,
                "model_parameters": model.parameter_count(),
                "policy_id": update_number,
                "opponent_policy_ids": [entry.policy_id for entry in selected_opponents],
                "policy_hashes": {
                    "learner": snapshot_hash,
                    "opponents": {
                        str(policy_id): policy_hash
                        for policy_id, policy_hash in selected_opponent_hashes.items()
                    },
                },
                "reward_perspective": "acting_seat",
                "reward_definition": "terminal_win_plus1_loss_minus1_draw0",
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "context_len": cfg.context_len,
                "action_codec_version": "crforge_v4_2307",
                "behaviour_temperature": 1.0,
                "league_sampling": {
                    "current": args.current_opponent_fraction,
                    "historical_pfsp": args.historical_pfsp_fraction,
                    "forgotten_uniform": args.forgotten_opponent_fraction,
                },
                "learner_seats": learner_seats.tolist(),
                "deck_names": deck_names,
                "deck0_ids": deck0_ids.tolist(),
                "deck1_ids": deck1_ids.tolist(),
                "seed_start": seeds.tolist(),
            }
        )

        for step in range(args.steps):
            observations[step] = raw_obs
            masks[step] = raw_masks
            episode_start[step] = current_episode_start
            episode_id_rows[step] = episode_ids
            episode_start_flags[step] = (current_episode_start == step).astype(np.uint8)
            seed_rows[step] = [env.vector.episode_seed(i) for i in range(args.envs)]
            current = raw_obs.reshape(args.envs * 2, -1)
            for agent in range(args.envs * 2):
                histories[agent].append(current[agent].copy())

            learner_raw = np.stack([raw_obs[i, int(learner_seats[i])] for i in range(args.envs)])
            learner_legal = np.stack(
                [raw_masks[i, int(learner_seats[i])] for i in range(args.envs)]
            )
            learner_histories = [histories[i * 2 + int(learner_seats[i])] for i in range(args.envs)]
            learner_tensors = _policy_inputs(learner_raw, learner_legal, learner_histories, adapter)
            learner_actions, learner_logp, learner_values = _forward(
                behavior_model, learner_tensors, device
            )
            opponent_actions = np.zeros(args.envs, dtype=np.int64)
            opponent_logp = np.zeros(args.envs, dtype=np.float32)
            opponent_values = np.zeros(args.envs, dtype=np.float32)
            for policy_id, opponent_model in opponent_models.items():
                indices = [
                    i for i, entry in enumerate(selected_opponents) if entry.policy_id == policy_id
                ]
                opponent_raw = np.stack([raw_obs[i, 1 - int(learner_seats[i])] for i in indices])
                opponent_legal = np.stack(
                    [raw_masks[i, 1 - int(learner_seats[i])] for i in indices]
                )
                opponent_histories = [histories[i * 2 + 1 - int(learner_seats[i])] for i in indices]
                opponent_tensors = _policy_inputs(
                    opponent_raw, opponent_legal, opponent_histories, adapter
                )
                group_actions, group_logp, group_values = _forward(
                    opponent_model, opponent_tensors, device
                )
                opponent_actions[indices] = group_actions
                opponent_logp[indices] = group_logp
                opponent_values[indices] = group_values
            step_actions = np.zeros((args.envs, 2), dtype=np.int32)
            for env_index, learner_seat in enumerate(learner_seats):
                learner_seat = int(learner_seat)
                opponent_seat = 1 - learner_seat
                step_actions[env_index, learner_seat] = learner_actions[env_index]
                step_actions[env_index, opponent_seat] = opponent_actions[env_index]
                actions[step, env_index] = step_actions[env_index]
                old_log_probs[step, env_index, learner_seat] = learner_logp[env_index]
                old_log_probs[step, env_index, opponent_seat] = opponent_logp[env_index]
                old_values[step, env_index, learner_seat] = learner_values[env_index]
                old_values[step, env_index, opponent_seat] = opponent_values[env_index]
                rewards[step, env_index] = 0.0
                learner_mask[step, env_index, learner_seat] = 1
                policy_ids[step, env_index, learner_seat] = update_number
                opponent_policy_id = selected_opponents[env_index].policy_id
                policy_ids[step, env_index, opponent_seat] = opponent_policy_id
                opponent_ids[step, env_index, learner_seat] = opponent_policy_id
                opponent_ids[step, env_index, opponent_seat] = update_number
                policy_versions[step, env_index, learner_seat] = update_number
                policy_versions[step, env_index, opponent_seat] = opponent_policy_id

            env.step(step_actions)
            terminal_obs = env.observations.copy()
            terminal_masks = env.masks.copy()
            raw_rewards[step] = env.rewards.copy()
            for env_index in range(args.envs):
                for seat in range(2):
                    action_refusals[step, env_index, seat] = terminal_obs[
                        env_index, seat, env.vector.schema.header["last_refusal"][0]
                    ]
                    run_refusals[step, env_index, seat] = terminal_obs[
                        env_index, seat, env.vector.schema.header["last_run_refusal"][0]
                    ]
            next_observations[step] = terminal_obs
            next_masks[step] = terminal_masks
            done_flags = env.dones.copy().astype(bool)
            info_rows = [env.vector.info(i) for i in range(args.envs)]
            for env_index, done in enumerate(done_flags):
                episode_steps[step, env_index] = int(info_rows[env_index]["episode_step"])
                ticks[step, env_index] = int(info_rows[env_index]["tick"])
                fault_codes[step, env_index] = int(info_rows[env_index]["fault"])
                learner_seat = int(learner_seats[env_index])
                if done:
                    info = info_rows[env_index]
                    is_fault = int(info["fault"]) != 0
                    is_truncated = bool(info["truncated"]) or is_fault
                    truncated[step, env_index] = int(is_truncated)
                    terminated[step, env_index] = int(not is_truncated)
                    end_reasons[step, env_index] = 3 if is_fault else (2 if is_truncated else 1)
                    winners[step, env_index] = -1 if is_truncated else int(info["winner"])
                    if is_truncated:
                        if is_fault:
                            # Invalidate only this episode, including rows already copied.
                            episode_first = int(current_episode_start[env_index])
                            valid[episode_first : step + 1, env_index, :] = 0
                            if episode_first < step:
                                buffer.invalidate_episode(env_index, episode_first, step - 1)
                        else:
                            final_obs = terminal_obs[env_index, learner_seat][None, :]
                            final_history = deque(
                                list(histories[env_index * 2 + learner_seat]) + [final_obs[0]],
                                maxlen=cfg.context_len,
                            )
                            final_tensor = _policy_inputs(
                                final_obs,
                                terminal_masks[env_index, learner_seat][None, :],
                                [final_history],
                                adapter,
                            )
                            _, _, final_value = _forward(model, final_tensor, device)
                            truncation_bootstrap[step, env_index] = final_value[0]
                    else:
                        rewards[step, env_index] = env.rewards[env_index, learner_seat]
                        pool.record(
                            selected_opponents[env_index].policy_id,
                            float(rewards[step, env_index]),
                        )
                    new_seed = int(rng.integers(1, 2**31 - 1))
                    env.reset_env(env_index, new_seed)
                    episode_ids[env_index] += 1
                    current_episode_start[env_index] = step + 1
                    histories[env_index * 2].clear()
                    histories[env_index * 2 + 1].clear()
                else:
                    rewards[step, env_index] = env.rewards[env_index, learner_seat]

            training_step_rewards = raw_rewards[step].copy()
            buffer.append(
                observations[step],
                masks[step],
                actions[step],
                raw_rewards[step],
                next_observations[step],
                next_masks[step],
                done_flags.astype(np.uint8),
                terminated=terminated[step],
                truncated=truncated[step],
                valid=valid[step],
                training_rewards=training_step_rewards,
                action_consumed=action_consumed[step],
                action_refusals=action_refusals[step],
                run_refusals=run_refusals[step],
                old_log_probs=old_log_probs[step],
                old_values=old_values[step],
                bootstrap_values=np.full((args.envs, 2), np.nan, dtype=np.float32),
                bootstrap_valid=np.zeros((args.envs, 2), dtype=np.uint8),
                learner_mask=learner_mask[step],
                policy_ids=policy_ids[step],
                opponent_ids=opponent_ids[step],
                policy_versions=policy_versions[step],
                episode_ids=episode_id_rows[step],
                episode_start=episode_start_flags[step],
                episode_steps=episode_steps[step],
                ticks=ticks[step],
                winner=winners[step],
                end_reason=end_reasons[step],
                fault_codes=fault_codes[step],
                seeds=seed_rows[step],
            )
            raw_obs, raw_masks = env.observations.copy(), env.masks.copy()

        # Build next-state values without ever bootstrapping across an autoreset boundary.
        next_values = np.zeros_like(rewards)
        for step in range(args.steps):
            for env_index in range(args.envs):
                if terminated[step, env_index]:
                    next_values[step, env_index] = 0.0
                elif truncated[step, env_index]:
                    next_values[step, env_index] = truncation_bootstrap[step, env_index]
                elif step + 1 < args.steps:
                    next_values[step, env_index] = old_values[
                        step + 1, env_index, int(learner_seats[env_index])
                    ]
                else:
                    final_seat = int(learner_seats[env_index])
                    final_raw = raw_obs[env_index, final_seat][None, :]
                    final_history = deque(
                        list(histories[env_index * 2 + final_seat])
                        + [raw_obs[env_index, final_seat].copy()],
                        maxlen=cfg.context_len,
                    )
                    final_tensor = _policy_inputs(
                        final_raw,
                        raw_masks[env_index, final_seat][None, :],
                        [final_history],
                        adapter,
                    )
                    _, _, final_value = _forward(model, final_tensor, device)
                    next_values[step, env_index] = final_value[0]

        for step in range(args.steps):
            for env_index, seat in enumerate(learner_seats):
                learner_seat = int(seat)
                buffer.rows["bootstrap_values"][step][env_index, learner_seat] = next_values[
                    step, env_index
                ]
                buffer.rows["bootstrap_valid"][step][env_index, learner_seat] = int(
                    valid[step, env_index, learner_seat] > 0
                )

        reward_t = torch.as_tensor(rewards, device=device)
        value_t = torch.as_tensor(
            old_values[
                np.arange(args.steps)[:, None], np.arange(args.envs)[None, :], learner_seats
            ],
            device=device,
        )
        next_value_t = torch.as_tensor(next_values, device=device)
        terminal_t = torch.as_tensor(terminated, device=device)
        boundary_t = torch.as_tensor(terminated | truncated, device=device)
        advantages, returns = compute_gae(
            reward_t,
            value_t,
            terminal_t,
            next_value_t[-1],
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            next_values=next_value_t,
            episode_ends=boundary_t,
        )
        sample_mask = np.zeros((args.steps, args.envs), dtype=bool)
        for env_index, seat in enumerate(learner_seats):
            sample_mask[:, env_index] = valid[:, env_index, int(seat)] > 0
        valid_indices = np.flatnonzero(sample_mask.reshape(-1))
        if not len(valid_indices):
            raise RuntimeError(
                "No valid learner transitions remain after simulator-fault filtering"
            )
        valid_advantages = advantages.reshape(-1)[torch.as_tensor(valid_indices, device=device)]
        advantages = torch.zeros_like(advantages)
        advantages.reshape(-1)[torch.as_tensor(valid_indices, device=device)] = (
            valid_advantages - valid_advantages.mean()
        ) / (valid_advantages.std(unbiased=False) + 1e-8)
        flat_returns = returns.reshape(-1)
        flat_old_logp = np.zeros((args.steps, args.envs), dtype=np.float32)
        flat_actions = np.zeros((args.steps, args.envs), dtype=np.int64)
        flat_masks = np.zeros((args.steps, args.envs, env.vector.schema.actions), dtype=bool)
        flat_obs = np.zeros((args.steps, args.envs, env.vector.schema.seat_ints), dtype=np.int32)
        flat_episode_start = episode_start.copy()
        for env_index, seat in enumerate(learner_seats):
            flat_old_logp[:, env_index] = old_log_probs[:, env_index, int(seat)]
            flat_actions[:, env_index] = actions[:, env_index, int(seat)]
            flat_masks[:, env_index] = masks[:, env_index, int(seat)].astype(bool)
            flat_obs[:, env_index] = observations[:, env_index, int(seat)]

        model.train()
        metric_rows = []
        schedule_scale = (1.0 + update_number) ** -0.25
        entropy_coef = args.entropy_coef * schedule_scale
        reference_kl_coef = args.reference_kl_coef * schedule_scale
        optimizer.zero_grad(set_to_none=True)
        accumulated_microbatches = 0
        optimizer_steps = 0
        total_microbatches = (len(valid_indices) + args.batch_size - 1) // args.batch_size
        for _epoch in range(args.epochs):
            permutation = rng.permutation(valid_indices)
            for batch_index, start in enumerate(range(0, len(permutation), args.batch_size)):
                indices = permutation[start : start + args.batch_size]
                batch_windows, batch_time, batch_legal = [], [], []
                batch_hand, batch_deck, batch_enemy_seen = [], [], []
                for index in indices:
                    step, env_index = divmod(int(index), args.envs)
                    left = max(int(flat_episode_start[step, env_index]), step - cfg.context_len + 1)
                    sequence = flat_obs[left : step + 1, env_index]
                    pad = cfg.context_len - len(sequence)
                    window = np.zeros((cfg.context_len, env.vector.schema.seat_ints), np.int32)
                    window[pad:] = sequence
                    time_mask = np.zeros(cfg.context_len, dtype=bool)
                    time_mask[pad:] = True
                    batch_windows.append(window)
                    batch_time.append(time_mask)
                    batch_legal.append(flat_masks[step, env_index])
                    batch_hand.append(sequence[-1, list(env.vector.schema.header["hand_card"])])
                    batch_deck.append(sequence[-1, list(env.vector.schema.header["deck"])])
                    batch_enemy_seen.append(
                        sequence[-1, list(env.vector.schema.header["enemy_seen"])]
                    )
                bh, be, bm = adapter(np.stack(batch_windows))
                batch_tensors = (
                    torch.as_tensor(bh, device=device),
                    torch.as_tensor(be, device=device),
                    torch.as_tensor(bm, device=device),
                    torch.as_tensor(np.stack(batch_legal), device=device),
                    torch.as_tensor(np.stack(batch_time), device=device),
                    torch.as_tensor(np.stack(batch_hand), device=device),
                    torch.as_tensor(np.stack(batch_deck), device=device),
                    torch.as_tensor(np.stack(batch_enemy_seen), device=device),
                )
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type="cuda" if device.startswith("cuda") else "cpu",
                        dtype=torch.bfloat16,
                        enabled=device.startswith("cuda") and torch.cuda.is_bf16_supported(),
                    ),
                ):
                    reference_logp, _ = behavior_model(
                        *batch_tensors[:5],
                        hand_card_ids=batch_tensors[5],
                        deck_card_ids=batch_tensors[6],
                        enemy_seen_card_ids=batch_tensors[7],
                    )
                with torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else "cpu",
                    dtype=torch.bfloat16,
                    enabled=device.startswith("cuda") and torch.cuda.is_bf16_supported(),
                ):
                    logp_all, new_values = model(
                        *batch_tensors[:5],
                        hand_card_ids=batch_tensors[5],
                        deck_card_ids=batch_tensors[6],
                        enemy_seen_card_ids=batch_tensors[7],
                    )
                distribution = torch.distributions.Categorical(logits=logp_all)
                new_logp = distribution.log_prob(
                    torch.as_tensor(flat_actions.reshape(-1)[indices], device=device)
                )
                loss, metrics = ppo_loss_with_reference(
                    logp_all,
                    reference_logp,
                    new_logp,
                    torch.as_tensor(flat_old_logp.reshape(-1)[indices], device=device),
                    advantages.reshape(-1)[torch.as_tensor(indices, device=device)],
                    new_values,
                    flat_returns[torch.as_tensor(indices, device=device)],
                    distribution.entropy(),
                    args.clip_coef,
                    args.value_coef,
                    entropy_coef,
                    reference_kl_coef,
                )
                loss.backward()
                accumulated_microbatches += 1
                should_step = (
                    accumulated_microbatches >= args.grad_accumulation
                    or batch_index == total_microbatches - 1
                    or float(metrics["approx_kl"].cpu()) > args.target_kl
                )
                if should_step:
                    scale = 1.0 / accumulated_microbatches
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_microbatches = 0
                    optimizer_steps += 1
                metric_rows.append({key: float(value.cpu()) for key, value in metrics.items()})
                if float(metrics["approx_kl"].cpu()) > args.target_kl:
                    break
            if metric_rows and float(metric_rows[-1]["approx_kl"]) > args.target_kl:
                break

        model.eval()
        checkpoint_path = output / "ppo_latest.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.__dict__,
                "model_architecture_version": 2,
                "updates": update_number + 1,
                "seed": args.seed,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if device.startswith("cuda")
                else None,
                "numpy_rng_state": rng.bit_generator.state,
                "backend_commit": env.metadata().get("commit"),
                "data_version": env.vector.data_version,
                "content_sha": env.vector.content_sha,
            },
            checkpoint_path,
        )
        rollout_path = buffer.save(output / "rollouts" / f"rollout_{update_number + 1:06d}.npz")
        summary = {
            "update": update_number + 1,
            "device": device,
            "model": args.model,
            "parameters": model.parameter_count(),
            "samples": int(len(valid_indices)),
            "microbatch_size": args.batch_size,
            "gradient_accumulation": args.grad_accumulation,
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "optimizer_steps": optimizer_steps,
            "learner_seats": learner_seats.tolist(),
            "opponent_policy_ids": [entry.policy_id for entry in selected_opponents],
            "deck_pairs": [
                [deck_names[int(deck0_ids[i])], deck_names[int(deck1_ids[i])]]
                for i in range(args.envs)
            ],
            "mean_reward": float(rewards.mean()),
            "terminal_games": int(terminated.sum()),
            "truncated_games": int(truncated.sum()),
            "faulted_samples": int((valid == 0).sum()),
            "metrics": {
                key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]
            },
            "entropy_coef": entropy_coef,
            "reference_kl_coef": reference_kl_coef,
            "seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint_path),
            "rollout": str(rollout_path),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary) + "\n")
        pool.save(pool_path)
        print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Symmetric frozen-opponent self-play PPO trainer")
    parser.add_argument(
        "--model",
        choices=("tiny", "small", "compact", "general", "large", "production"),
        default="general",
    )
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--reference-kl-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--current-opponent-fraction", type=float, default=0.35)
    parser.add_argument("--historical-pfsp-fraction", type=float, default=0.50)
    parser.add_argument("--forgotten-opponent-fraction", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/selfplay"))
    parser.add_argument("--updates", type=int, default=1)
    args = parser.parse_args()
    if args.updates < 1 or args.grad_accumulation < 1:
        parser.error("--updates and --grad-accumulation must be positive")
    initial_resume = args.resume
    base_seed = args.seed
    for update in range(args.updates):
        args.resume = initial_resume if update == 0 else args.out / "ppo_latest.pt"
        args.seed = base_seed + update
        train(args)


if __name__ == "__main__":
    main()
