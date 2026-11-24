from __future__ import annotations

import argparse
import copy
import gc
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
from cr_rl.models import ActorCritic, ActorPolicy, HistoryValueNetwork, ModelConfig
from cr_rl.opponents import OpponentEntry, OpponentPool
from cr_rl.training.advantages import compute_gae
from cr_rl.training.initialization import initialize_from_bc
from cr_rl.training.ppo import ppo_loss_with_reference, stop_for_kl
from cr_rl.training.rewards import RewardConfig, shaped_rewards
from cr_rl.training.separate import (
    autocast_for,
    complete_episode_returns,
    forward_inputs,
    optimize_separate,
)


def _progress(args, **event) -> None:
    if getattr(args, "progress_jsonl", None):
        event["time_unix"] = time.time()
        with Path(args.progress_jsonl).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")


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


def _forward(model, tensors, device, collect_stats=False, critic=None):
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
        if critic is not None:
            values = forward_inputs(critic, tuple(t.to(device) for t in tensors))
        distribution = torch.distributions.Categorical(logits=logits.float())
        actions = distribution.sample()
        result = (
            actions.cpu().numpy(),
            distribution.log_prob(actions).cpu().numpy(),
            values.float().cpu().numpy(),
        )
        if collect_stats:
            return (
                *result,
                {
                    "entropy": distribution.entropy().float().cpu().numpy(),
                    "max_probability": distribution.probs.max(-1).values.float().cpu().numpy(),
                },
            )
        return result


def _bootstrap_value(model, critic, tensors, device):
    with torch.inference_mode(), autocast_for(device):
        result = forward_inputs(critic or model, tuple(t.to(device) for t in tensors))
        values = result if critic is not None else result[1]
        return values.float().cpu().numpy()


def _training_batch(indices, flat_obs, flat_masks, episode_start, cfg, adapter, device):
    windows, times, legal = [], [], []
    hand, deck, seen = [], [], []
    for index in indices:
        step, env_index = divmod(int(index), flat_obs.shape[1])
        left = max(int(episode_start[step, env_index]), step - cfg.context_len + 1)
        sequence = flat_obs[left : step + 1, env_index]
        pad = cfg.context_len - len(sequence)
        window = np.zeros((cfg.context_len, flat_obs.shape[-1]), dtype=np.int32)
        window[pad:] = sequence
        time_mask = np.zeros(cfg.context_len, dtype=bool)
        time_mask[pad:] = True
        windows.append(window)
        times.append(time_mask)
        legal.append(flat_masks[step, env_index])
        for values, key in ((hand, "hand_card"), (deck, "deck"), (seen, "enemy_seen")):
            values.append(sequence[-1, list(adapter.schema.header[key])])
    h, e, m = adapter(np.stack(windows))
    return tuple(
        torch.as_tensor(v, device=device)
        for v in (
            h,
            e,
            m,
            np.stack(legal),
            np.stack(times),
            np.stack(hand),
            np.stack(deck),
            np.stack(seen),
        )
    )


def train(args) -> None:
    from cr_rl.envs.crforge import CrForgeVector

    reward_config = RewardConfig(
        enabled=getattr(args, "tower_shaping", False),
        princess=getattr(args, "princess_reward", 0.15),
        king=getattr(args, "king_reward", 0.20),
        crown=getattr(args, "crown_reward", 0.15),
        overflow=getattr(args, "overflow_penalty", 0.0),
    )
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
    reference = args.resume or getattr(args, "init_from_bc", None)
    if reference:
        header = torch.load(reference, map_location="cpu", weights_only=False, mmap=True)
        saved_cfg = ModelConfig(**header["config"])
        cfg = cfg.inherit_representation(saved_cfg)
        del header
    model = ActorCritic(cfg).to(device)
    state = None
    update_number = 0
    resume_metadata = None
    resume_league = None
    initialization = None
    if getattr(args, "init_from_bc", None):
        if args.resume:
            raise ValueError("Use either --init-from-bc or --resume")
        initialization = initialize_from_bc(model, args.init_from_bc, cfg.__dict__, device)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        previous_rewards = state.get("reward_config", RewardConfig().metadata())
        if previous_rewards != reward_config.metadata():
            raise ValueError("Resume reward configuration mismatch; start a separate reward trial")
        previous_gamma = state.get("training_args", {}).get("gamma", args.gamma)
        if previous_gamma != args.gamma:
            raise ValueError("Resume gamma mismatch changes critic targets")
        if int(state.get("model_architecture_version", 0)) != 2:
            raise ValueError(
                "Checkpoint predates deck/public-seen-card embeddings; export weights and "
                "reinitialize the optimizer before migrating it to architecture v2"
            )
        if ModelConfig(**state["config"]) != cfg:
            raise ValueError("Resume checkpoint model config does not match --model")
        if state.get("policy_only"):
            model = ActorPolicy(cfg).to(device)
        model.load_state_dict(state["model"])
        update_number = int(state.get("updates", 0))
        initialization = state.get("initialization")
        resume_league = state.get("league_config")
        resume_metadata = {
            key: state.get(key) for key in ("backend_commit", "data_version", "content_sha")
        }
        if state.get("level_profile") is not None:
            resume_metadata["level_profile"] = state["level_profile"]
        if (
            state.get("level_profile") is None
            and (
                getattr(args, "level_mixture", None)
                or getattr(args, "tower_level_mixture", None)
                or getattr(args, "level_jitter_prob", None) is not None
            )
            and not getattr(args, "allow_level_profile_migration", False)
        ):
            raise ValueError(
                "This checkpoint predates level profiles; pass --allow-level-profile-migration for an explicit curriculum migration"
            )
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_state"]])
        if "numpy_rng_state" in state:
            rng.bit_generator.state = state["numpy_rng_state"]

    requested_mode = getattr(args, "critic_mode", "auto")
    saved_mode = "independent" if state and "critic" in state else "shared"
    mode = (saved_mode if state else "independent") if requested_mode == "auto" else requested_mode
    if state and mode != saved_mode:
        raise ValueError("Resume cannot change critic mode; start a new run from BC for migration")
    critic = critic_optimizer = None
    critic_initialization = None
    warmup = {"rounds": 0, "games": 0}
    warmup_config = {"rounds": 2, "games": 4}
    critic_cfg = None
    if mode == "independent":
        if not isinstance(model, ActorPolicy):
            model = ActorPolicy.from_shared(model)
        critic_preset = getattr(args, "critic_model", "same")
        critic_cfg = (
            cfg
            if critic_preset == "same"
            else (ModelConfig.preset(critic_preset).inherit_representation(cfg))
        )
        if state:
            if critic_preset != "same" and ModelConfig(**state["critic_config"]) != critic_cfg:
                raise ValueError("Resume critic configuration mismatch")
            critic_cfg = ModelConfig(**state["critic_config"])
        if critic_cfg.context_len != cfg.context_len:
            raise ValueError("Actor and critic must currently use the same history window length")
        critic = HistoryValueNetwork(critic_cfg).to(device)
        critic_initialization = critic.initialize_encoder(model)
        critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=getattr(args, "critic_learning_rate", 1e-4),
            foreach=False,
        )
        if state:
            critic.load_state_dict(state["critic"], strict=True)
            critic_optimizer.load_state_dict(state["critic_optimizer"])
            critic_initialization = state["critic_initialization"]
            warmup = state["critic_warmup"]
            warmup_config = state["critic_warmup_config"]
        for field, arg in (("rounds", "critic_warmup_updates"), ("games", "critic_warmup_games")):
            value = getattr(args, arg, None)
            if value is not None:
                if state and value != warmup_config[field]:
                    raise ValueError("Resume warmup settings differ from checkpoint")
                warmup_config[field] = value
        critic.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, foreach=False)
    sample_offset = (
        int(state.get("learner_samples_total", update_number * args.envs * args.steps))
        if state
        else 0
    )
    if state:
        optimizer.load_state_dict(state["optimizer"])
        # Network construction consumes RNG: restore only after constructing both networks.
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_state"]])
    del state

    if getattr(args, "decks", None):
        import yaml

        league_cfg = yaml.safe_load(args.decks.read_text())
    elif resume_league:
        league_cfg = resume_league
    elif args.resume:
        saved_config = args.resume.parent / "run_config.json"
        if not saved_config.exists():
            raise ValueError(
                "Older PPO checkpoint has no deck configuration; pass --decks explicitly"
            )
        league_cfg = json.loads(saved_config.read_text())["decks"]
    else:
        league_cfg = load_yaml("decks.yaml")
    if resume_league:
        for key in ("decks", "slots"):
            if resume_league.get(key, {}) != league_cfg.get(key, {}):
                raise ValueError("PPO resume deck pool differs from the saved run")
    deck_cfg = league_cfg["decks"]
    deck_names = list(deck_cfg)
    decks = [deck_cfg[name] for name in deck_names]
    forms = [league_cfg.get("slots", {}).get(name, [0] * 8) for name in deck_names]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "ppo_latest.pt").exists():
        raise ValueError("Output already contains a checkpoint; use --resume or a new --out")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    temporary_snapshot = output / f"opponent_{update_number:06d}.tmp.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "model_architecture_version": 2,
            "policy_id": update_number,
            "policy_only": isinstance(model, ActorPolicy),
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
        opponent_cfg = ModelConfig(**opponent_state.get("config", cfg.__dict__))
        fields = ("tile_query_residual", "tile_head_fp32", "spatial_fourier")
        if {k: v for k, v in opponent_cfg.__dict__.items() if k not in fields} != {
            k: v for k, v in cfg.__dict__.items() if k not in fields
        }:
            raise ValueError("Historical opponent input/model dimensions differ")
        opponent_model.cfg = opponent_cfg
        opponent_model.frame.cfg = opponent_cfg
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
    with CrForgeVector(
        args.envs,
        min(args.envs, 8),
        independent=True,
        level_mixture=getattr(args, "level_mixture", None),
        tower_level_mixture=getattr(args, "tower_level_mixture", None),
        level_jitter_prob=getattr(args, "level_jitter_prob", None),
        level_jitter=getattr(args, "level_jitter", 2),
        level_jitter_seed=args.seed,
    ) as env:
        if initialization:
            actual = {
                "backend_commit": env.metadata().get("commit"),
                "data_version": env.vector.data_version,
                "content_sha": env.vector.content_sha,
                "level": env.level,
                "decision_ticks": 5,
                "mask_version": 2,
            }
            if "tower_level" in initialization:
                actual["tower_level"] = env.tower_level
            if "level_profile" in initialization:
                actual["level_profile"] = env.level_profile
            if any(initialization[key] != value for key, value in actual.items()):
                raise ValueError(f"BC/PPO simulator mismatch: {initialization} vs {actual}")
        if resume_metadata is not None:
            actual = {
                "backend_commit": env.metadata().get("commit"),
                "data_version": env.vector.data_version,
                "content_sha": env.vector.content_sha,
            }
            if "level_profile" in resume_metadata:
                actual["level_profile"] = env.level_profile
            if resume_metadata != actual:
                raise ValueError(f"Resume simulator/data mismatch: {resume_metadata} != {actual}")
        # Do not autoreset behind the trainer's back. The final pre-reset observation is
        # required to distinguish a terminal result from a time-limit/fault truncation.
        env.set_autoreset(False)
        for deck in decks:
            if len(deck) != 8 or len(set(deck)) != 8:
                raise ValueError("Every league deck must contain eight distinct cards")
            for card in deck:
                env.vector.card_id(card)  # fail fast on stale/misspelled game-table rows

        deck0_ids = rng.integers(0, len(decks), size=args.envs, dtype=np.int32)
        deck1_ids = rng.integers(0, len(decks), size=args.envs, dtype=np.int32)
        deck0 = [decks[int(index)] for index in deck0_ids]
        deck1 = [decks[int(index)] for index in deck1_ids]
        seeds = rng.integers(1, 2**31 - 1, size=args.envs, dtype=np.int64).astype(np.int32)
        raw_obs, raw_masks = env.reset(
            seeds.tolist(),
            deck0,
            deck1,
            [forms[i] for i in deck0_ids],
            [forms[i] for i in deck1_ids],
        )
        episode_matchups = []

        def record_matchup(index, seed):
            episode_matchups.append(
                {
                    "env": index,
                    "seed": int(seed),
                    "deck0": deck_names[int(deck0_ids[index])],
                    "deck1": deck_names[int(deck1_ids[index])],
                }
            )

        for i, seed in enumerate(seeds):
            record_matchup(i, seed)
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
        reward_components = {
            name: np.zeros_like(raw_rewards) for name in ("hp", "crown", "army", "overflow")
        }
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
        policy_stats = []
        buffer = TrajectoryBuffer(
            {
                **env.metadata(),
                "algorithm": "frozen-opponent-self-play-ppo-v1",
                "telemetry_version": 2,
                "model": args.model,
                "model_architecture_version": 2,
                "model_config": cfg.__dict__,
                "model_parameters": model.parameter_count(),
                "critic_mode": mode,
                "critic_config": critic_cfg.__dict__ if critic_cfg else None,
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
                "training_reward_config": reward_config.metadata(),
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
            if step % 256 == 0:
                print(f"update {update_number + 1}: rollout {step}/{args.steps}", flush=True)
                _progress(
                    args,
                    phase="rollout",
                    update=update_number + 1,
                    rollout_step=step,
                    rollout_total=args.steps,
                    learner_samples_total=sample_offset + step * args.envs,
                    terminal_games=int(terminated.sum()),
                    seconds=time.perf_counter() - started,
                )
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
            learner_actions, learner_logp, learner_values, learner_stats = _forward(
                behavior_model, learner_tensors, device, collect_stats=True, critic=critic
            )
            can_play = learner_legal[:, 1:2305].any(axis=1)
            for i in range(args.envs):
                policy_stats.append(
                    {
                        "entropy": float(learner_stats["entropy"][i]),
                        "max_probability": float(learner_stats["max_probability"][i]),
                        "wait": float(learner_actions[i] == 0),
                        "can_play": bool(can_play[i]),
                        "legal_actions": int(learner_legal[i].sum()),
                        "value": float(learner_values[i]),
                    }
                )
            if (step + 1) % 64 == 0 or step == args.steps - 1:
                available = [r for r in policy_stats if r["can_play"]]
                _progress(
                    args,
                    phase="policy",
                    update=update_number + 1,
                    learner_samples_total=sample_offset + (step + 1) * args.envs,
                    entropy=float(np.mean([r["entropy"] for r in policy_stats])),
                    entropy_when_play_legal=float(np.mean([r["entropy"] for r in available]))
                    if available
                    else None,
                    max_action_probability=float(
                        np.mean([r["max_probability"] for r in policy_stats])
                    ),
                    play_fraction=1 - float(np.mean([r["wait"] for r in policy_stats])),
                    wait_when_play_legal=float(np.mean([r["wait"] for r in available]))
                    if available
                    else None,
                    legal_action_count=float(np.mean([r["legal_actions"] for r in policy_stats])),
                    value_mean=float(np.mean([r["value"] for r in policy_stats])),
                    value_std=float(np.std([r["value"] for r in policy_stats])),
                )
                policy_stats.clear()
            if not np.all(learner_legal[np.arange(args.envs), learner_actions]):
                raise RuntimeError("Learner sampled an illegal action")
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
                if not np.all(opponent_legal[np.arange(len(indices)), group_actions]):
                    raise RuntimeError("Opponent sampled an illegal action")
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
                    _progress(
                        args,
                        phase="episode",
                        update=update_number + 1,
                        learner_samples_total=sample_offset + (step + 1) * args.envs,
                        reward=float(raw_rewards[step, env_index, learner_seat]),
                        episode_length=int(info["episode_step"]),
                        fault=int(is_fault),
                        truncated=int(is_truncated),
                        opponent_policy_id=selected_opponents[env_index].policy_id,
                        learner_deck=deck_names[
                            int(deck0_ids[env_index] if learner_seat == 0 else deck1_ids[env_index])
                        ],
                        opponent_deck=deck_names[
                            int(deck1_ids[env_index] if learner_seat == 0 else deck0_ids[env_index])
                        ],
                    )
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
                            final_value = _bootstrap_value(model, critic, final_tensor, device)
                            truncation_bootstrap[step, env_index] = final_value[0]
                    else:
                        rewards[step, env_index] = env.rewards[env_index, learner_seat]
                        pool.record(
                            selected_opponents[env_index].policy_id,
                            float(rewards[step, env_index]),
                        )
                    new_seed = int(rng.integers(1, 2**31 - 1))
                    deck0_ids[env_index] = rng.integers(len(decks))
                    deck1_ids[env_index] = rng.integers(len(decks))
                    d0, d1 = int(deck0_ids[env_index]), int(deck1_ids[env_index])
                    env.vector.reset_env(
                        env_index, new_seed, decks[d0], decks[d1], forms[d0], forms[d1]
                    )
                    record_matchup(env_index, new_seed)
                    episode_ids[env_index] += 1
                    current_episode_start[env_index] = step + 1
                    histories[env_index * 2].clear()
                    histories[env_index * 2 + 1].clear()
                else:
                    rewards[step, env_index] = env.rewards[env_index, learner_seat]

            training_step_rewards, components = shaped_rewards(
                observations[step],
                terminal_obs,
                step_actions,
                raw_rewards[step],
                terminated[step],
                env.vector.schema,
                args.gamma,
                reward_config,
                faulted=fault_codes[step] != 0,
            )
            for name, component in components.items():
                reward_components[name][step] = component
            rewards[step] = training_step_rewards[np.arange(args.envs), learner_seats]
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
                    final_value = _bootstrap_value(model, critic, final_tensor, device)
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
        valid_returns = returns.reshape(-1)[torch.as_tensor(valid_indices, device=device)]
        valid_values = value_t.reshape(-1)[torch.as_tensor(valid_indices, device=device)]
        target_var = valid_returns.var(unbiased=False)
        warmup_active = critic is not None and not all(
            warmup[k] >= warmup_config[k] for k in warmup_config
        )
        _progress(
            args,
            phase="targets_gae" if warmup_active else "targets",
            update=update_number + 1,
            learner_samples_total=sample_offset + args.envs * args.steps,
            advantage_mean=float(valid_advantages.mean().cpu()),
            advantage_std=float(valid_advantages.std(unbiased=False).cpu()),
            return_mean=float(valid_returns.mean().cpu()),
            return_std=float(valid_returns.std(unbiased=False).cpu()),
            explained_variance=float(
                (1 - (valid_returns - valid_values).var(unbiased=False) / target_var).cpu()
            )
            if float(target_var.cpu()) > 1e-8
            else None,
        )
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
        print(
            f"update {update_number + 1}: optimizing {len(valid_indices)} learner samples",
            flush=True,
        )
        metric_rows = []
        initial_logp_errors = []
        max_training_history_len = 0
        schedule_scale = (1.0 + update_number) ** -0.25
        entropy_coef = args.entropy_coef * schedule_scale
        reference_kl_coef = args.reference_kl_coef * schedule_scale
        separate_result = None
        actor_enabled = True
        if critic is not None:
            actor_enabled = all(warmup[k] >= warmup_config[k] for k in warmup_config)
            critic_targets = flat_returns.detach()
            critic_indices = valid_indices
            complete_games = 0
            if not actor_enabled:
                mc_returns, mc_mask, complete_games = complete_episode_returns(
                    rewards,
                    terminated,
                    truncated,
                    sample_mask,
                    args.gamma,
                )
                critic_targets = torch.as_tensor(mc_returns.reshape(-1), device=device)
                critic_indices = np.flatnonzero(mc_mask.reshape(-1))
                if len(critic_indices):
                    rows = torch.as_tensor(critic_indices, device=device)
                    target = critic_targets[rows]
                    old = value_t.reshape(-1)[rows]
                    variance = target.var(unbiased=False)
                    _progress(
                        args,
                        phase="targets",
                        update=update_number + 1,
                        learner_samples_total=sample_offset + args.envs * args.steps,
                        target_type="monte_carlo_complete_games",
                        return_mean=float(target.mean().cpu()),
                        return_std=float(target.std(unbiased=False).cpu()),
                        explained_variance=float(
                            (1 - (target - old).var(unbiased=False) / variance).cpu()
                        )
                        if float(variance.cpu()) > 1e-8
                        else None,
                    )
            _progress(
                args,
                phase="critic_warmup",
                update=update_number + 1,
                actor_enabled=actor_enabled,
                completed_games=complete_games,
                critic_samples=int(len(critic_indices)),
                warmup_rounds=warmup["rounds"],
                warmup_games=warmup["games"],
            )

            def make_batch(indices):
                nonlocal max_training_history_len
                for index in indices:
                    row, column = divmod(int(index), args.envs)
                    max_training_history_len = max(
                        max_training_history_len,
                        min(
                            cfg.context_len,
                            row - int(flat_episode_start[row, column]) + 1,
                        ),
                    )
                return _training_batch(
                    indices,
                    flat_obs,
                    flat_masks,
                    flat_episode_start,
                    cfg,
                    adapter,
                    device,
                )

            def emit(**event):
                _progress(
                    args,
                    update=update_number + 1,
                    learner_samples_total=sample_offset + args.envs * args.steps,
                    **event,
                )

            separate_result = optimize_separate(
                model,
                critic,
                behavior_model,
                optimizer,
                critic_optimizer,
                make_batch,
                valid_indices,
                critic_indices,
                torch.as_tensor(flat_old_logp.reshape(-1), device=device),
                torch.as_tensor(flat_actions.reshape(-1), device=device),
                advantages.reshape(-1),
                critic_targets,
                rng=rng,
                device=device,
                batch_size=args.batch_size,
                accumulation=args.grad_accumulation,
                actor_epochs=args.epochs,
                critic_epochs=getattr(args, "critic_epochs", 4),
                clip_coef=args.clip_coef,
                entropy_coef=entropy_coef,
                reference_kl_coef=reference_kl_coef,
                target_kl=args.target_kl,
                max_grad_norm=args.max_grad_norm,
                actor_enabled=actor_enabled,
                emit=emit,
            )
            optimizer_steps = separate_result["actor_steps"]
            initial_logp_errors = separate_result["initial_logp_errors"]
            if not actor_enabled and separate_result["critic_steps"]:
                warmup["rounds"] += 1
                warmup["games"] += complete_games
        else:
            optimizer.zero_grad(set_to_none=True)
            accumulated_microbatches = 0
            optimizer_steps = 0
            optimizer_step_offset = max(
                (int(state["step"].item()) for state in optimizer.state.values()), default=0
            )
            total_microbatches = (len(valid_indices) + args.batch_size - 1) // args.batch_size
            for _epoch in range(args.epochs):
                permutation = rng.permutation(valid_indices)
                for batch_index, start in enumerate(range(0, len(permutation), args.batch_size)):
                    indices = permutation[start : start + args.batch_size]
                    batch_windows, batch_time, batch_legal = [], [], []
                    batch_hand, batch_deck, batch_enemy_seen = [], [], []
                    for index in indices:
                        step, env_index = divmod(int(index), args.envs)
                        left = max(
                            int(flat_episode_start[step, env_index]), step - cfg.context_len + 1
                        )
                        sequence = flat_obs[left : step + 1, env_index]
                        max_training_history_len = max(max_training_history_len, len(sequence))
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
                    distribution = torch.distributions.Categorical(logits=logp_all.float())
                    new_logp = distribution.log_prob(
                        torch.as_tensor(flat_actions.reshape(-1)[indices], device=device)
                    )
                    if optimizer_steps == 0:
                        initial_logp_errors.append(
                            float(
                                (
                                    new_logp.detach()
                                    - torch.as_tensor(
                                        flat_old_logp.reshape(-1)[indices], device=device
                                    )
                                )
                                .abs()
                                .max()
                                .cpu()
                            )
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
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite PPO loss")
                    metric_rows.append({key: float(value.cpu()) for key, value in metrics.items()})
                    if stop_for_kl(optimizer, metric_rows[-1]["approx_kl"], args.target_kl):
                        _progress(
                            args,
                            phase="kl_stop",
                            update=update_number + 1,
                            microbatch=batch_index,
                            optimizer_steps=optimizer_steps,
                            discarded_microbatches=accumulated_microbatches,
                            approx_kl=metric_rows[-1]["approx_kl"],
                            target_kl=args.target_kl,
                        )
                        accumulated_microbatches = 0
                        break
                    loss.backward()
                    accumulated_microbatches += 1
                    should_step = (
                        accumulated_microbatches >= args.grad_accumulation
                        or batch_index == total_microbatches - 1
                    )
                    if should_step:
                        scale = 1.0 / accumulated_microbatches
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(scale)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_microbatches = 0
                        optimizer_steps += 1
                    if should_step:
                        _progress(
                            args,
                            phase="optimization",
                            update=update_number + 1,
                            microbatch=batch_index,
                            optimizer_steps=optimizer_steps,
                            optimizer_steps_total=optimizer_step_offset + optimizer_steps,
                            learner_samples_total=sample_offset + args.envs * args.steps,
                            grad_norm=float(grad_norm.cpu()),
                            learning_rate=float(optimizer.param_groups[0]["lr"]),
                            loss=float(loss.detach().cpu()),
                            **metric_rows[-1],
                        )
                if metric_rows and float(metric_rows[-1]["approx_kl"]) > args.target_kl:
                    break

        model.eval()
        checkpoint_path = output / "ppo_latest.pt"
        temporary_checkpoint = checkpoint_path.with_suffix(".tmp.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.__dict__,
                "model_architecture_version": 2,
                "updates": update_number + 1,
                "reward_config": reward_config.metadata(),
                "learner_samples_total": sample_offset + args.envs * args.steps,
                "training_architecture_version": 3 if critic is not None else 2,
                "policy_only": isinstance(model, ActorPolicy),
                **(
                    {
                        "critic": critic.state_dict(),
                        "critic_config": critic_cfg.__dict__,
                        "critic_optimizer": critic_optimizer.state_dict(),
                        "critic_initialization": critic_initialization,
                        "critic_warmup": warmup,
                        "critic_warmup_config": warmup_config,
                    }
                    if critic is not None
                    else {}
                ),
                "initialization": initialization,
                "league_config": league_cfg,
                "seed": args.seed,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if device.startswith("cuda")
                else None,
                "numpy_rng_state": rng.bit_generator.state,
                "backend_commit": env.metadata().get("commit"),
                "data_version": env.vector.data_version,
                "content_sha": env.vector.content_sha,
                "level_profile": env.level_profile,
                "resume_semantics": "model_optimizer_rng_restored_simulator_episodes_reset",
                "training_args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            temporary_checkpoint,
        )
        temporary_checkpoint.replace(checkpoint_path)
        if critic is not None:
            actor_export = output / "actor_export.pt"
            export_tmp = actor_export.with_suffix(".tmp.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "model_architecture_version": 2,
                    "policy_only": True,
                    "stage": "actor_export",
                    "updates": update_number + 1,
                    "backend_commit": env.metadata().get("commit"),
                    "data_version": env.vector.data_version,
                    "content_sha": env.vector.content_sha,
                    "level_profile": env.level_profile,
                },
                export_tmp,
            )
            export_tmp.replace(actor_export)
        rollout_path = buffer.save(output / "rollouts" / f"rollout_{update_number + 1:06d}.npz")
        summary = {
            "update": update_number + 1,
            "level_profile": env.level_profile,
            "device": device,
            "model": args.model,
            "parameters": model.parameter_count(),
            "samples": int(len(valid_indices)),
            "learner_samples_total": sample_offset + args.envs * args.steps,
            "microbatch_size": args.batch_size,
            "gradient_accumulation": args.grad_accumulation,
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "optimizer_steps": optimizer_steps,
            "critic_mode": mode,
            "critic_parameters": sum(p.numel() for p in critic.parameters()) if critic else 0,
            "critic_optimizer_steps": separate_result["critic_steps"] if separate_result else 0,
            "actor_enabled": actor_enabled,
            "actor_kl_stopped": separate_result["actor_kl_stopped"] if separate_result else None,
            "critic_warmup": warmup if critic else None,
            "learner_seats": learner_seats.tolist(),
            "opponent_policy_ids": [entry.policy_id for entry in selected_opponents],
            "initialization": initialization,
            "episode_matchups": episode_matchups,
            "deck_pairs": [
                [deck_names[int(deck0_ids[i])], deck_names[int(deck1_ids[i])]]
                for i in range(args.envs)
            ],
            "mean_reward": float(rewards.mean()),
            "training_reward_config": reward_config.metadata(),
            "reward_component_sums": {
                name: float(values[learner_mask.astype(bool)].sum())
                for name, values in reward_components.items()
            },
            "terminal_games": int(terminated.sum()),
            "terminal_wins": int(
                (
                    raw_rewards[learner_mask.astype(bool)].reshape(args.steps, args.envs)[
                        terminated.astype(bool)
                    ]
                    > 0
                ).sum()
            ),
            "terminal_losses": int(
                (
                    raw_rewards[learner_mask.astype(bool)].reshape(args.steps, args.envs)[
                        terminated.astype(bool)
                    ]
                    < 0
                ).sum()
            ),
            "terminal_draws": int(
                (
                    raw_rewards[learner_mask.astype(bool)].reshape(args.steps, args.envs)[
                        terminated.astype(bool)
                    ]
                    == 0
                ).sum()
            ),
            "learner_play_fraction": float((flat_actions[sample_mask] != 0).mean()),
            "nonzero_learner_rewards": int(np.count_nonzero(rewards[sample_mask])),
            "learner_reward_sum": float(rewards[sample_mask].sum()),
            "learner_raw_reward_sum": float(
                raw_rewards[learner_mask.astype(bool) & valid.astype(bool)].sum()
            ),
            "initial_logp_max_error": max(initial_logp_errors, default=0.0),
            "max_training_history_len": max_training_history_len,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
            if device.startswith("cuda")
            else None,
            "resume_semantics": "model_optimizer_rng_restored_simulator_episodes_reset",
            "truncated_games": int(truncated.sum()),
            "faulted_samples": int((valid == 0).sum()),
            "metrics": separate_result["metrics"]
            if separate_result
            else {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]},
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


def restore_league_options(args, saved, argv):
    """Restore audited league settings unless a CLI option explicitly overrides them."""
    if not getattr(args, 'resume', None) or not getattr(args, 'v3', False):
        return
    previous = saved.get('training_args', {})
    mixture = saved.get('role_mixture', {})
    options = saved.get('league_options', {})
    values = {name: options.get(name, previous.get(name)) for name in (
        'role_league', 'pool_capacity', 'history_snapshot_interval', 'snapshot_interval')}
    for role in ('current', 'historical', 'anchor'):
        name = role + '_opponent_fraction'
        values[name] = mixture.get(role, previous.get(name))
    for name, value in values.items():
        option = '--' + name.replace('_', '-')
        explicit = any(item == option or item.startswith(option + '=') for item in argv)
        if not explicit and value is not None:
            setattr(args, name, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Symmetric frozen-opponent self-play PPO trainer")
    parser.add_argument(
        "--model",
        choices=("tiny", "small", "compact", "general", "large", "production", "general_300m"),
        default="general",
    )
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--env-threads", type=int, default=1,
                        help="v2 simulator lane workers; independent of Torch CPU threads")
    parser.add_argument(
        "--level-mixture",
        type=int,
        nargs="+",
        help=(
            "Optional per-lane global card levels for a fresh curriculum pilot, e.g. 10 11 12. "
            "The pinned backend applies one level to both seats in each lane; this is not per-card level support."
        ),
    )
    parser.add_argument(
        "--tower-level-mixture",
        type=int,
        nargs="+",
        help="Optional per-lane global tower levels; one value broadcasts across the card-level schedule.",
    )
    parser.add_argument(
        "--level-jitter-prob",
        type=float,
        help="Fraction of lane base levels to perturb; perturbations are reproducible ±level-jitter.",
    )
    parser.add_argument(
        "--level-jitter",
        type=int,
        default=2,
        help="Absolute level perturbation used with --level-jitter-prob (default: 2).",
    )
    parser.add_argument(
        "--allow-level-profile-migration",
        action="store_true",
        help="Resume an older single-level checkpoint while explicitly changing the simulator level profile.",
    )
    parser.add_argument(
        "--allow-reward-migration",
        action="store_true",
        help="Resume a v3 checkpoint while explicitly changing reward shaping coefficients.",
    )
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--torch-interop-threads", type=int)
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--v3", action="store_true",
                        help="Use the explicit-memory architecture-v3 recurrent PPO driver")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--incremental-backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-checkpoint", action="store_true",
                        help="Freeze independent CPU state, then write local checkpoints in the background")
    parser.add_argument("--async-publication", action="store_true",
                        help="Publish immutable snapshots while collecting the next update")
    parser.add_argument("--publication-interval", type=int, default=1,
                        help="Publish every N updates; local checkpoints remain current")
    parser.add_argument("--cache-reference", action="store_true")
    parser.add_argument("--deduplicate-training-frames", action="store_true",
                        help="Encode each repeated raw history frame once per training microbatch")
    parser.add_argument(
        "--stream-training-batches",
        action="store_true",
        help="Keep prepared history on CPU and transfer only each microbatch to CUDA",
    )
    parser.add_argument(
        "--incremental-inference", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--forced-wait-fastpath",
        action="store_true",
        help="v2: skip frozen Actor forward when WAIT is the only legal action",
    )
    parser.add_argument("--defer-scalar-metrics", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--pool-capacity", type=int, default=8)
    parser.add_argument("--snapshot-interval", type=int, default=250_000)
    parser.add_argument("--snapshot-min-games", type=int, default=20)
    parser.add_argument("--snapshot-min-score", type=float, default=0.55)
    parser.add_argument("--role-league", action="store_true", help="30/50/20 current/PPO/BC league")
    parser.add_argument("--history-snapshot-interval", type=int, default=20_000)
    parser.add_argument("--history-seed-manifest", type=Path)
    parser.add_argument(
        "--league-diversity-manifest",
        type=Path,
        help="v2 frozen-opponent diversity manifest",
    )
    parser.add_argument("--frame-encoder-fp32", action="store_true")
    parser.add_argument("--opponent-cache-size", type=int, default=2)
    parser.add_argument("--training-opponent-cache-size", type=int,
                        help="Evict excess frozen opponents before PPO to preserve memory")
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--actor-batch-size", type=int, help="v2 Actor microbatch; does not set effective batch"
    )
    parser.add_argument(
        "--critic-batch-size", type=int, help="v2 Critic microbatch; effective batch fixed"
    )
    parser.add_argument("--grad-accumulation", type=int, default=16)
    parser.add_argument(
        "--actor-effective-batch-size",
        type=int,
        help="v2 Actor-only effective batch; Critic retains batch-size * grad-accumulation",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--resume-actor-learning-rate", type=float)
    parser.add_argument("--resume-critic-learning-rate", type=float)
    parser.add_argument(
        "--adaptive-actor-lr",
        action="store_true",
        help="v2: adapt next-update Actor LR within bounds using factorized KL",
    )
    parser.add_argument("--adaptive-actor-lr-min", type=float, default=2.5e-6)
    parser.add_argument(
        "--adaptive-actor-lr-max",
        type=float,
        help="Actor LR ceiling; defaults to the restored or initial Actor LR",
    )
    parser.add_argument(
        "--override-adaptive-actor-lr",
        action="store_true",
        help="Allow --adaptive-actor-lr-min/max overrides when resuming a checkpoint",
    )
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument(
        "--critic-gae-lambda",
        type=float,
        help="v2 Critic TD(lambda) target; defaults to the Actor GAE lambda",
    )
    parser.add_argument("--tower-shaping", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reserve-to-wait", action="store_true")
    parser.add_argument("--ppo-actor-fp32", action="store_true",
                        help="Match streaming and sequence learner policy arithmetic in FP32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--temporal-checkpoint-mode", choices=("block", "nested"), default="nested",
                        help="v3: block recomputes temporal layers; nested also recomputes KDA solve chunks for lower memory")
    parser.add_argument("--cache-state-frames", action="store_true",
                        help="v3: cache normalized rollout observations on training GPUs for one update")
    parser.add_argument("--full-episode-bptt", action="store_true",
                        help="v3: collect whole games under one policy version and backpropagate without temporal truncation")
    parser.add_argument("--army-reward", type=float, default=None)
    parser.add_argument('--army-reward-mode',choices=('uniform','elixir_potential'),default=None)
    parser.add_argument('--army-elixir-beta',type=float,default=None)
    parser.add_argument('--army-elixir-bar-factor', type=float, default=None)
    parser.add_argument('--army-elixir-config',type=Path,default=None)
    parser.add_argument("--focal-decks", default="")
    parser.add_argument("--focal-deck-prob", type=float, default=0.5)
    parser.add_argument("--sparky-wait-fraction", type=float, default=0.0,
                        help="v3: fraction of learner Sparky games eligible for one reserve window")
    parser.add_argument("--sparky-wait-max-steps", type=int, choices=(1, 2, 3), default=3,
                        help="v3: maximum 400 ms decisions in a Sparky reserve window")
    parser.add_argument("--sparky-play-fraction", type=float, default=0.0,
                        help="v3: disjoint fraction eligible for one forced Sparky card choice")
    parser.add_argument("--historical-cohort-size", type=int, default=2)
    parser.add_argument("--historical-cohort-updates", type=int, default=2)
    parser.add_argument("--princess-reward", type=float, default=0.15)
    parser.add_argument("--king-reward", type=float, default=0.20)
    parser.add_argument("--crown-reward", type=float, default=0.15)
    parser.add_argument("--overflow-penalty", type=float)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--critic-mode",
        choices=("auto", "independent", "shared"),
        default="auto",
        help="New runs use independent; resume infers the saved architecture",
    )
    parser.add_argument(
        "--critic-model",
        default="same",
        choices=(
            "same",
            "tiny",
            "small",
            "compact",
            "general",
            "large",
            "production",
            "general_300m",
        ),
    )
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-epochs", type=int, default=4)
    parser.add_argument(
        "--critic-warmup-updates",
        type=int,
        default=None,
        help="Value-fit rollout rounds before actor updates; new-run default 2",
    )
    parser.add_argument(
        "--critic-warmup-games",
        type=int,
        default=None,
        help="Complete games needed for warmup; new-run default 4",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--card-entropy-coef", type=float)
    parser.add_argument("--tile-entropy-coef", type=float)
    parser.add_argument("--reference-kl-coef", type=float, default=0.01)
    parser.add_argument(
        "--anchor-kl-coef",
        type=float,
        default=0.0,
        help="KL penalty coefficient toward the frozen BC anchor model",
    )
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--actor-choice-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--target-card-kl", type=float)
    parser.add_argument("--target-tile-kl", type=float)
    parser.add_argument(
        "--continue-after-kl-reject",
        action="store_true",
        help="v2: skip an over-limit Actor KL group and continue later groups instead of ending the epoch",
    )
    parser.add_argument("--max-initial-logp-error", type=float)
    parser.add_argument("--elixir-wait-bias", type=float)
    parser.add_argument("--disable-holding-prior", action="store_true",
                        help="Remove WAIT/slot tactical priors when resuming any checkpoint")
    parser.add_argument("--elixir-hold-min-cost", type=float)
    parser.add_argument("--holding-prior-version", type=int, choices=(1, 2))
    parser.add_argument("--holding-core-cards", nargs="+")
    parser.add_argument("--core-slot-bias", type=float)
    parser.add_argument("--current-opponent-fraction", type=float)
    parser.add_argument("--historical-pfsp-fraction", type=float)
    parser.add_argument("--forgotten-opponent-fraction", type=float)
    parser.add_argument("--historical-opponent-fraction", type=float,
                        help="v3 role-league historical frozen-policy fraction")
    parser.add_argument("--anchor-opponent-fraction", type=float,
                        help="v3 role-league fixed BC-anchor fraction")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--keep-periodic-checkpoints", type=int, default=2)
    parser.add_argument("--compact-host-memory", action="store_true",
                        help="Share immutable rollout arrays and release stale CPU mappings")
    parser.add_argument("--dual-rollout", action="store_true",
                        help="Sample learner Actor streams on both CUDA devices")
    parser.add_argument("--balanced-critic", action="store_true",
                        help="Use the Actor GPU for Critic data parallel after Actor updates")
    parser.add_argument("--amp-dtype", choices=("auto", "fp32", "fp16"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default=None,
                        help="v3 PPO autocast precision; V100 uses fp16")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--frame-microbatch", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--critic-device",
        help="v2/v3 Critic device; a different CUDA device enables concurrent optimization",
    )
    parser.add_argument(
        "--critic-peer-device",
        help="v3 optional peer CUDA device (e.g. cuda:2 NVLink) for data-parallel Critic optimization",
    )
    parser.add_argument(
        "--opponent-device",
        help="v3 optional CUDA device (e.g. cuda:1) to overlap frozen opponent rollout inference",
    )
    parser.add_argument(
        "--critic-effective-batch-size",
        type=int,
        help="v3 Critic target eligible samples per optimizer step (defaults to batch-size * grad-accumulation)",
    )
    parser.add_argument(
        "--tbptt-chunk",
        type=int,
        default=64,
        help="v3 truncated BPTT chunk length during Actor/Critic updates",
    )
    parser.add_argument(
        "--rollout-save-interval",
        type=int,
        default=0,
        help="v3: save compressed TrajectoryBuffer .npz rollouts every N updates (0 disables for peak speed)",
    )
    parser.add_argument(
        "--keep-rollouts",
        type=int,
        default=2,
        help="v3: number of recent compressed .npz rollouts to retain when --rollout-save-interval > 0",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from-bc", type=Path)
    parser.add_argument("--bc-anchor", type=Path,
                        help="v3 fixed BC opponent anchor (defaults to the v3 best.pt)")
    parser.add_argument("--decks", type=Path)
    parser.add_argument(
        "--expand-decks", type=Path, help="Explicit append-only v2 resume curriculum"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/selfplay"))
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument(
        "--min-training-games",
        type=int,
        help="v2: stop at a checkpoint after this many complete games in the saved counter scope",
    )
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="v3: save the single latest checkpoint every N updates")
    parser.add_argument("--checkpoint-first-update", action="store_true",
                        help="v3: also save the first update of this process")
    parser.add_argument("--critic-fp16-resume-scale-cap", type=float,
                        help="v3: cap restored Critic loss scale for a fresh FP16 comparison")
    parser.add_argument("--validation-interval", type=int, default=5600,
                        help="v3: run fixed validation every N updates (0 disables; default 5600)")
    parser.add_argument("--validation-pairs", type=int, default=5,
                        help="v3: paired games per validation opponent (5 gives 10 games)")
    parser.add_argument("--validation-argmax-lanes", type=int, default=0,
                        help="Batch deterministic validation across this many lanes (0 keeps the default evaluator)")
    parser.add_argument("--validation-decks", type=Path,
                        help="v3: fixed deck manifest for sparse validation")
    parser.add_argument(
        "--tensorboard-dir", type=Path,
        help="Write v3 PPO scalars to TensorBoard event files in this directory",
    )
    parser.add_argument('--awbc-sessions', type=Path, help='Optional finalized real-game replay source')
    parser.add_argument('--awbc-interval', type=int, default=50)
    parser.add_argument('--awbc-min-new-games', type=int, default=12)
    parser.add_argument('--awbc-batch-games', type=int, default=24)
    parser.add_argument('--awbc-max-reuse', type=int, default=2)
    parser.add_argument('--awbc-epochs', type=int, default=2)
    parser.add_argument('--awbc-value-epochs', type=int, default=20)
    parser.add_argument('--awbc-learning-rate', type=float, default=1e-6)
    parser.add_argument('--awbc-max-kl', type=float, default=.01)
    parser.add_argument('--require-awbc-smoke', action='store_true',
                        help='Fail unless AWBC actually optimizes Actor, then a subsequent PPO update succeeds')
    parser.add_argument('--no-backup-hash', action='store_true',
                        help='Async immutable snapshot copy with size-only completion checks')
    parser.add_argument('--expert-bc-dataset', type=Path)
    parser.add_argument('--ppo-only', action='store_true',
                        help='Skip BC and AWBC options saved in a resumed checkpoint')
    parser.add_argument('--expert-bc-plan-b', action='store_true',
                        help='Run weighted specialist BC with held-out and PPO health gates')
    parser.add_argument('--expert-bc-experimental', action='store_true')
    parser.add_argument('--expert-bc-interval', type=int, default=100)
    parser.add_argument('--expert-bc-sequences', type=int, default=8)
    parser.add_argument('--expert-bc-learning-rate', type=float, default=1e-6)
    parser.add_argument('--expert-bc-max-kl', type=float, default=.01)
    parser.add_argument('--macro-search', action='store_true',
                        help='Run universal 4x4 two-card sequence rollout search and gated distillation')
    parser.add_argument('--macro-search-interval', type=int, default=10)
    parser.add_argument('--macro-search-offset', type=int, default=0,
                        help='Update residue for macro search, to stagger it from other phases')
    parser.add_argument('--macro-search-candidates', type=int, default=2)
    parser.add_argument('--macro-search-win-margin', type=float, default=0.30)
    parser.add_argument('--macro-search-max-buffer', type=int, default=128)
    parser.add_argument('--macro-search-batch-episodes', type=int, default=8)
    parser.add_argument('--macro-search-learning-rate', type=float, default=1e-6)
    parser.add_argument('--macro-search-max-kl', type=float, default=0.005)
    parser.add_argument('--lookahead-distill', action='store_true',
                        help='Run online Critic-bootstrapped lookahead search and gated readout distillation')
    parser.add_argument('--lookahead-distill-interval', type=int, default=3)
    parser.add_argument('--lookahead-distill-offset', type=int, default=1)
    parser.add_argument('--lookahead-distill-games', type=int, default=3)
    parser.add_argument('--lookahead-distill-parallel-games', type=int, default=12)
    parser.add_argument('--lookahead-distill-parallel-searches', type=int, default=4)
    parser.add_argument('--lookahead-distill-threads', type=int, default=0)
    parser.add_argument('--lookahead-distill-max-buffer', type=int, default=192)
    parser.add_argument('--lookahead-distill-candidates', type=int, default=4)
    parser.add_argument('--lookahead-distill-rollouts', type=int, default=3)
    parser.add_argument('--lookahead-distill-horizon', type=int, default=10)
    parser.add_argument('--lookahead-distill-trigger-elixir', type=float, default=6.0)
    parser.add_argument('--lookahead-distill-max-searches', type=int, default=16)
    parser.add_argument('--lookahead-distill-batch-episodes', type=int, default=4)
    parser.add_argument('--lookahead-distill-steps', type=int, default=1)
    parser.add_argument('--lookahead-distill-margin', type=float, default=0.008)
    parser.add_argument('--lookahead-distill-learning-rate', type=float, default=1e-6)
    parser.add_argument('--lookahead-distill-max-kl', type=float, default=0.005)
    args = parser.parse_args()
    if (args.critic_fp16_resume_scale_cap is not None and
            not (np.isfinite(args.critic_fp16_resume_scale_cap) and args.critic_fp16_resume_scale_cap>=1)):
        parser.error("Critic FP16 resume scale cap must be finite and at least 1")
    if args.army_reward_mode=='elixir_potential' and not args.v3:
        parser.error('Elixir army shaping is supported only by persistent v3 PPO')
    if (args.expert_bc_interval < 1 or args.expert_bc_sequences < 1
            or not np.isfinite(args.expert_bc_learning_rate) or args.expert_bc_learning_rate <= 0
            or not np.isfinite(args.expert_bc_max_kl) or args.expert_bc_max_kl <= 0):
        parser.error('Invalid auxiliary BC settings')
    if args.expert_bc_dataset and not args.v3:
        parser.error('Auxiliary BC requires architecture-v3 PPO')
    if (args.macro_search_interval < 1 or args.macro_search_candidates < 1
            or not 0 <= args.macro_search_offset < args.macro_search_interval
            or not np.isfinite(args.macro_search_win_margin) or args.macro_search_win_margin < 0
            or not np.isfinite(args.macro_search_learning_rate) or not (0 < args.macro_search_learning_rate <= 1e-5)
            or not np.isfinite(args.macro_search_max_kl) or not (0 < args.macro_search_max_kl <= 0.005)):
        parser.error('Invalid macro-search settings')
    if args.macro_search and not args.v3:
        parser.error('--macro-search requires architecture-v3 PPO')
    if (args.lookahead_distill_interval < 1 or args.lookahead_distill_games < 1 or args.lookahead_distill_horizon < 2
            or args.lookahead_distill_parallel_games < 1 or args.lookahead_distill_parallel_searches < 1
            or args.lookahead_distill_candidates < 2 or args.lookahead_distill_rollouts < 1
            or args.lookahead_distill_max_searches < 1 or args.lookahead_distill_batch_episodes < 1
            or args.lookahead_distill_steps < 1
            or not 0 <= args.lookahead_distill_offset < args.lookahead_distill_interval
            or not np.isfinite(args.lookahead_distill_trigger_elixir) or not (1.0 <= args.lookahead_distill_trigger_elixir <= 10.0)
            or not np.isfinite(args.lookahead_distill_margin) or args.lookahead_distill_margin < 0
            or not np.isfinite(args.lookahead_distill_learning_rate) or not (0 < args.lookahead_distill_learning_rate <= 2e-5)
            or not np.isfinite(args.lookahead_distill_max_kl) or not (0 < args.lookahead_distill_max_kl <= 0.005)):
        parser.error('Invalid lookahead-distill settings')
    if args.lookahead_distill and not args.v3:
        parser.error('--lookahead-distill requires architecture-v3 PPO')
    if (min(args.awbc_interval,args.awbc_min_new_games,args.awbc_max_reuse,
            args.awbc_epochs,args.awbc_value_epochs)<1 or args.awbc_batch_games<12
            or args.awbc_min_new_games>args.awbc_batch_games-2
            or not np.isfinite(args.awbc_learning_rate) or args.awbc_learning_rate<=0
            or not np.isfinite(args.awbc_max_kl) or args.awbc_max_kl<=0):
        parser.error('Invalid AWBC playbuffer settings')
    if not 0 <= args.focal_deck_prob <= 1 or args.historical_cohort_updates < 1:
        parser.error("Invalid focal probability or historical cohort interval")
    if (not 0 <= args.sparky_wait_fraction <= 1 or not 0 <= args.sparky_play_fraction <= 1
            or args.sparky_wait_fraction + args.sparky_play_fraction > 1):
        parser.error("Sparky curriculum fractions must be nonnegative and sum to at most one")
    if (args.sparky_wait_fraction or args.sparky_play_fraction) and args.rollout_save_interval:
        parser.error("Sparky curriculum is incompatible with on-policy rollout export")
    if args.league_diversity_manifest is not None and not args.league_diversity_manifest.is_file():
        parser.error("League diversity manifest must be an existing file")
    if not np.isfinite(args.adaptive_actor_lr_min) or args.adaptive_actor_lr_min <= 0:
        parser.error("Adaptive Actor LR minimum must be finite and positive")
    if args.adaptive_actor_lr_max is not None and (
        not np.isfinite(args.adaptive_actor_lr_max)
        or args.adaptive_actor_lr_max < args.adaptive_actor_lr_min
    ):
        parser.error("Adaptive Actor LR maximum must be finite and at least the minimum")
    if args.training_opponent_cache_size is not None and args.training_opponent_cache_size < 1:
        parser.error("Training opponent cache size must be positive")
    if args.validation_interval < 0 or args.validation_pairs < 1:
        parser.error("Validation interval must be nonnegative and validation pairs must be positive")
    if not 0 <= args.validation_argmax_lanes <= 32:
        parser.error("Validation argmax lanes must be between 0 and 32")
    if args.validation_decks is not None and not args.validation_decks.is_file():
        parser.error("Validation deck manifest must be an existing file")
    if args.env_threads < 1:
        parser.error("Environment threads must be positive")
    for name in ("level_mixture", "tower_level_mixture"):
        values = getattr(args, name)
        if values is not None and any(value < 1 or value > 15 for value in values):
            parser.error(f"{name.replace('_', ' ')} values must be in [1, 15]")
    if args.tower_level_mixture is not None and args.level_mixture is not None and len(
        args.tower_level_mixture
    ) not in (1, len(args.level_mixture)):
        parser.error("--tower-level-mixture must contain one value or match --level-mixture length")
    if args.level_jitter_prob is not None and (
        not np.isfinite(args.level_jitter_prob) or not 0 <= args.level_jitter_prob <= 1
    ):
        parser.error("--level-jitter-prob must be finite and in [0, 1]")
    if args.level_jitter < 1 or args.level_jitter > 14:
        parser.error("--level-jitter must be in [1, 14]")
    for name in ("torch_threads", "torch_interop_threads"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error("Torch CPU thread counts must be positive")
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)
    import os
    os.environ["CRRL_AMP_DTYPE"] = args.amp_dtype
    reference = args.resume or args.init_from_bc
    saved = (
        torch.load(reference, map_location="cpu", weights_only=False, mmap=True)
        if reference
        else {}
    )
    args.v3 |= saved.get("model_architecture_version") == 3
    args.v2 |= (not args.v3) and saved.get("config", {}).get("representation_version", 1) == 2
    if not args.v2 and not args.v3 and (
        args.critic_device
        or args.critic_peer_device
        or args.opponent_device
        or args.amp_dtype == "fp16"
    ):
        parser.error("Separate devices and FP16 AMP require --v2 or --v3")
    if not args.v3 and (
        args.critic_peer_device
        or args.opponent_device
        or args.critic_effective_batch_size is not None
        or args.rollout_save_interval != 0
        or args.sparky_wait_fraction
        or args.sparky_play_fraction
    ):
        parser.error("Peer critic, opponent device offload, and v3 rollout options require --v3")
    if args.v3:
        args.precision = args.precision or ("fp16" if args.amp_dtype in ("auto", "fp16") else "fp32")
        if args.decks is None:
            args.decks = Path("configs/decks_ppo46.yaml")
        if (
            args.weight_decay < 0
            or args.frame_microbatch < 1
            or args.tbptt_chunk < 1
            or args.rollout_save_interval < 0
            or args.keep_rollouts < 0
        ):
            parser.error("Invalid v3 weight_decay, frame_microbatch, tbptt_chunk, or rollout interval")
    previous = saved.get("training_args", {})
    if args.army_reward_mode is None:
        args.army_reward_mode=saved.get('reward_config',{}).get('army_reward_mode','uniform')
    if args.army_reward is None:
        args.army_reward=(saved.get('reward_config',{}).get('army_potential_weight',0.)
                          if args.army_reward_mode=='uniform' else 0.)
    if args.army_elixir_beta is None:
        args.army_elixir_beta=saved.get('reward_config',{}).get('army_elixir_beta',.004)
    if args.army_elixir_bar_factor is None:
        args.army_elixir_bar_factor = saved.get('reward_config', {}).get('army_elixir_bar_factor', 0.)
    if args.army_elixir_config is None:
        args.army_elixir_config=Path(previous.get('army_elixir_config') or 'configs/army_elixir_values.yaml')
    if args.resume:
        import sys
        from cr_rl.training.awbc_playbuffer import restore_options
        if not (args.ppo_only or args.expert_bc_plan_b):
            restore_options(args,previous,sys.argv[1:])
        restore_league_options(args,saved,sys.argv[1:])
    if args.pool_capacity < 2 or min(args.history_snapshot_interval,args.snapshot_interval) < 1:
        parser.error('Invalid resumed league capacity or snapshot interval')
    if args.require_awbc_smoke and (
            not args.v3 or not args.awbc_sessions or args.awbc_interval != 1 or args.updates < 2):
        parser.error('--require-awbc-smoke requires v3, --awbc-sessions, --awbc-interval 1 and at least 2 updates')
    if args.awbc_sessions and not args.v3:
        parser.error('--awbc-sessions requires architecture-v3 PPO')
    if args.ppo_only and (args.expert_bc_dataset or args.awbc_sessions or args.macro_search or args.lookahead_distill):
        parser.error('--ppo-only cannot be combined with BC, AWBC, macro-search, or lookahead-distill')
    if args.expert_bc_plan_b and (not args.expert_bc_dataset or args.awbc_sessions or args.ppo_only):
        parser.error('--expert-bc-plan-b requires its expert dataset and an exclusive auxiliary phase')
    v2_or_v3 = bool(args.v2 or args.v3)
    defaults = {
        "gamma": saved.get("gamma", previous.get("gamma", 0.9995 if v2_or_v3 else 0.999)),
        "gae_lambda": saved.get(
            "gae_lambda", previous.get("gae_lambda", 0.98 if v2_or_v3 else 0.95)
        ),
        "critic_gae_lambda": saved.get("critic_gae_lambda", previous.get("critic_gae_lambda")),
        "actor_choice_only": saved.get("actor_update_config", {}).get("choice_only", args.v3),
        "target_card_kl": saved.get("actor_update_config", {}).get("target_card_kl"),
        "target_tile_kl": saved.get("actor_update_config", {}).get("target_tile_kl"),
        "max_initial_logp_error": saved.get("actor_update_config", {}).get(
            "max_initial_logp_error"
        ),
        "min_training_games": saved.get("min_training_games"),
        "tower_shaping": saved.get("reward_config", {}).get("enabled", v2_or_v3),
        "overflow_penalty": saved.get("reward_config", {}).get("overflow", 0.001 if v2_or_v3 else 0),
        "current_opponent_fraction": (
            0.5
            if args.v2
            else ((0.30 if args.role_league else 1.0) if args.v3 else 0.35)
        ),
        "historical_pfsp_fraction": (
            0.3 if args.v2 else ((0.50 if args.role_league else 0.0) if args.v3 else 0.5)
        ),
        "forgotten_opponent_fraction": (
            0.2 if args.v2 else ((0.20 if args.role_league else 0.0) if args.v3 else 0.15)
        ),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.v3:
        if args.historical_opponent_fraction is None:
            args.historical_opponent_fraction = 0.50 if args.role_league else 0.0
        if args.anchor_opponent_fraction is None:
            args.anchor_opponent_fraction = 0.20 if args.role_league else 0.0
    del saved
    if (
        min(
            args.updates,
            args.grad_accumulation,
            args.steps,
            args.envs,
            args.epochs,
            args.batch_size,
            args.critic_epochs,
        )
        < 1
    ):
        parser.error(
            "updates, grad-accumulation, steps, envs, epochs and batch-size must be positive"
        )
    if not v2_or_v3 and any(
        value is not None
        for value in (
            args.actor_batch_size,
            args.critic_batch_size,
            args.actor_effective_batch_size,
        )
    ):
        parser.error("Separate Actor/Critic batch overrides currently require --v2 or --v3")
    if args.min_training_games is not None and (not v2_or_v3 or args.min_training_games < 1):
        parser.error("--min-training-games requires v2 or v3 and a positive completed-game target")
    if any(
        v is not None for v in (args.resume_actor_learning_rate, args.resume_critic_learning_rate)
    ) and (not args.resume or not v2_or_v3):
        parser.error("Explicit resume LR transitions require a v2 or v3 checkpoint resume")
    critic_effective = (
        args.batch_size * args.grad_accumulation
        if args.critic_effective_batch_size is None
        else args.critic_effective_batch_size
    )
    actor_effective = (
        critic_effective
        if args.actor_effective_batch_size is None
        else args.actor_effective_batch_size
    )
    actor_microbatch = args.batch_size if args.actor_batch_size is None else args.actor_batch_size
    critic_microbatch = (
        args.batch_size if args.critic_batch_size is None else args.critic_batch_size
    )
    if min(critic_effective, actor_effective, actor_microbatch, critic_microbatch) < 1 or (
        not args.v3
        and (actor_effective % actor_microbatch or critic_effective % critic_microbatch)
    ):
        parser.error("Actor/Critic microbatches must divide their effective batch")
    if any(
        value is not None and value < 1
        for value in (args.critic_warmup_updates, args.critic_warmup_games)
    ):
        parser.error("Warmup rounds and games must be positive")
    if args.learning_rate <= 0 or args.critic_learning_rate <= 0 or args.target_kl <= 0:
        parser.error("Learning rates and target KL must be positive")
    entropy_pair = (args.card_entropy_coef, args.tile_entropy_coef)
    if any(c is not None for c in entropy_pair) and (
        not v2_or_v3
        or any(c is None or not np.isfinite(c) for c in entropy_pair)
        or args.card_entropy_coef < 0
        or (not args.v3 and args.tile_entropy_coef < 0)
    ):
        parser.error(
            "Split entropy requires v2 or v3, nonnegative card entropy, and finite tile entropy (nonnegative for v2)"
        )
    if (args.target_card_kl is None) != (args.target_tile_kl is None):
        parser.error("Provide both card and tile KL limits")
    if any(
        v is not None and (not np.isfinite(v) or v <= 0)
        for v in (args.target_card_kl, args.target_tile_kl, args.max_initial_logp_error)
    ):
        parser.error("Actor KL and initial log-prob limits must be finite and positive")
    if not np.isfinite(args.anchor_kl_coef) or args.anchor_kl_coef < 0:
        parser.error("--anchor-kl-coef must be finite and nonnegative")
    if not v2_or_v3 and (
        args.actor_choice_only
        or args.target_card_kl is not None
        or args.max_initial_logp_error is not None
    ):
        parser.error("Choice sampling and split KL require v2 or v3")
    if args.adaptive_actor_lr and (not v2_or_v3 or args.target_card_kl is None):
        parser.error("Adaptive Actor LR requires v2 or v3 and both Card/Tile KL limits")
    if args.league_diversity_manifest is not None and not v2_or_v3:
        parser.error("League diversity manifest requires v2 or v3")
    if args.elixir_wait_bias is not None and (
        not args.v2 or not np.isfinite(args.elixir_wait_bias) or args.elixir_wait_bias < 0
    ):
        parser.error("Elixir WAIT bias requires v2 and a finite nonnegative value")
    if args.elixir_hold_min_cost is not None and (
        not args.v2
        or not np.isfinite(args.elixir_hold_min_cost)
        or not 0 < args.elixir_hold_min_cost <= 10
    ):
        parser.error("Elixir hold minimum cost must be in (0, 10]")
    fractions = (
        (
            args.current_opponent_fraction,
            args.historical_opponent_fraction,
            args.anchor_opponent_fraction,
        )
        if args.v3
        else (
            args.current_opponent_fraction,
            args.historical_pfsp_fraction,
            args.forgotten_opponent_fraction,
        )
    )
    if min(fractions) < 0 or not np.isclose(sum(fractions), 1):
        parser.error("Opponent fractions must be nonnegative and sum to one")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("Invalid gamma/lambda")
    if args.critic_gae_lambda is not None and (
        not v2_or_v3
        or not np.isfinite(args.critic_gae_lambda)
        or not 0 <= args.critic_gae_lambda <= 1
    ):
        parser.error("Independent Critic GAE lambda requires v2 or v3 and a finite value in [0, 1]")
    if args.resume and args.init_from_bc:
        parser.error("Choose --resume or --init-from-bc")
    if args.expand_decks and (not args.resume or not v2_or_v3):
        parser.error("--expand-decks requires a v2 or v3 checkpoint resume")
    if args.forced_wait_fastpath and not v2_or_v3:
        parser.error("--forced-wait-fastpath requires v2 or v3")
    if args.v3:
        import os
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if args.device == "cuda":
                args.device = f"cuda:{local_rank}"
        from cr_rl.training.state_selfplay import run

        run(args)
        return
    if args.v2:
        gc.collect()
        from cr_rl.training.selfplay_v2 import run

        run(args)
        return
    initial_resume = args.resume
    base_seed = args.seed
    for update in range(args.updates):
        args.resume = initial_resume if update == 0 else args.out / "ppo_latest.pt"
        args.seed = base_seed + update
        if update > 0:
            args.init_from_bc = None
        train(args)


if __name__ == "__main__":
    main()
