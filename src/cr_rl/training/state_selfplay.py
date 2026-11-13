"""Architecture-v3 recurrent PPO with a current/history/BC role league.

Each lane trains one learner seat against a frozen or current opponent selected
from the role mixture.  Only learner-seat trajectories enter PPO; opponent
trajectories are never treated as on-policy data.
"""
from __future__ import annotations

import json
import copy
import shutil
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.advantages import compute_gae
from cr_rl.training.rewards import RewardConfig, shaped_rewards
from cr_rl.training.state_ppo import StatePPOConfig, state_ppo_update
from cr_rl.opponents.role_sampling import RoleMixture


def _snapshot_memory(memory, lane):
    """Clone one stream's recurrent memory as a fixed PPO sequence prefix."""
    def clone_slice(value):
        if torch.is_tensor(value):
            return value[lane:lane + 1].detach().clone()
        if isinstance(value, list):
            return [clone_slice(item) for item in value]
        if is_dataclass(value):
            values = {field.name: clone_slice(getattr(value, field.name))
                      for field in fields(value)}
            return type(value)(**values)
        return value
    return clone_slice(memory)


def _save(path, model, optimizer, scaler, update, args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "crrl-state-ppo-v1",
        "training_driver": "persistent-state-v3",
        "model_architecture_version": 3,
        "config": model.cfg.__dict__,
        "temporal_config": model.temporal_cfg.__dict__,
        "decision_ticks": model.decision_ticks,
        "learned_placement_definition": getattr(model, "learned_placement_definition", None),
        "model": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                  for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": {"update": update},
        "training_args": vars(args),
    }
    tmp = Path(path).with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _gae_sequence(rewards, values, ended, bootstrap, gamma, lam):
    reward = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1)
    value = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    end = torch.as_tensor(ended, dtype=torch.bool).reshape(-1, 1)
    next_values = torch.cat((value[1:], torch.tensor([[float(bootstrap)]])), dim=0)
    advantage, returns = compute_gae(
        reward, value, end, torch.tensor([float(bootstrap)]), gamma=gamma,
        gae_lambda=lam, next_values=next_values, episode_ends=end,
    )
    advantage = torch.nan_to_num(advantage, nan=0.0, posinf=1e3, neginf=-1e3)
    returns = torch.nan_to_num(returns, nan=0.0, posinf=1e3, neginf=-1e3)
    return advantage[:, 0].numpy(), returns[:, 0].numpy()


class StateSelfplay:
    def __init__(self, args):
        from cr_rl.envs.crforge import CrForgeVector

        self.args = args
        self.rank = int(__import__("os").environ.get("RANK", "0"))
        self.world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
        if self.world_size > 1 and not torch.distributed.is_initialized():
            raise RuntimeError("torchrun process group must be initialized before v3 PPO")
        if not args.init_from_bc and not args.resume:
            raise ValueError("v3 PPO requires --init-from-bc or --resume")
        source = args.resume or args.init_from_bc
        checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
        if checkpoint.get("model_architecture_version") != 3:
            raise ValueError("State PPO requires a v3 checkpoint")
        if checkpoint.get("decision_ticks") != 8:
            raise ValueError("State PPO requires the 0.4s / 8-tick decision grid")
        device_name = args.device
        if device_name == "cuda" and self.world_size > 1:
            device_name = f"cuda:{self.rank}"
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.model = state_model_from_checkpoint(checkpoint, self.device)
        self.model.eval()
        print(json.dumps({"phase": "v3_model_loaded", "device": str(self.device), "decision_ticks": 8}), flush=True)
        self.model.frame_microbatch = getattr(args, "frame_microbatch", 32)
        # Preserve the original league design: current policy, frozen historical
        # policies, and a fixed BC anchor.  The v3 BC source is architecture 3;
        # silently falling back to mirror self-play is forbidden.
        anchor_path = getattr(args, "bc_anchor", None)
        if anchor_path is None:
            anchor_path = checkpoint.get("training_args", {}).get("bc_anchor")
        if anchor_path is None:
            anchor_path = Path("artifacts/ppo_prep_20261113/best.pt")
        self.anchor_path = Path(anchor_path)
        if not self.anchor_path.is_file():
            raise FileNotFoundError(f"v3 league requires BC anchor: {self.anchor_path}")
        anchor_checkpoint = torch.load(self.anchor_path, map_location="cpu", mmap=True, weights_only=False)
        if anchor_checkpoint.get("model_architecture_version") != 3:
            raise ValueError("BC anchor must be an architecture-v3 checkpoint")
        self.anchor_model = state_model_from_checkpoint(anchor_checkpoint, self.device)
        self.anchor_model.frame_microbatch = self.model.frame_microbatch
        self.anchor_model.eval()
        for parameter in self.anchor_model.parameters():
            parameter.requires_grad_(False)
        historical_fraction = float(getattr(args, "historical_opponent_fraction", 0.50))
        anchor_fraction = float(getattr(args, "anchor_opponent_fraction", 0.20))
        self.role_mixture = RoleMixture(
            current=1.0 - historical_fraction - anchor_fraction,
            historical=historical_fraction,
            anchor=anchor_fraction,
        )
        self.history_models = {}
        self.history_paths = []
        self.history_scores = {}
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
            fused=self.device.type == "cuda",
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        if args.resume:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scaler"):
                self.scaler.load_state_dict(checkpoint["scaler"])
        self.update = int(checkpoint.get("progress", {}).get("update", 0)) if args.resume else 0
        self.rng = np.random.default_rng(args.seed + self.rank)
        self.env = CrForgeVector(args.envs, getattr(args, "env_threads", 1), independent=True,
                                 arena="Arena_L", decision_ticks=8)
        print(json.dumps({"phase": "v3_env_created", "envs": args.envs}), flush=True)
        self.env.set_autoreset(False)
        self.schema = self.env.vector.schema
        self.adapter = CrForgeObservationAdapter(self.schema, self.env.vector.vocabulary_size, entity_bucket=16)
        deck_path = Path(args.decks or "configs/decks_ppo46.yaml")
        self.league = yaml.safe_load(deck_path.read_text(encoding="utf-8"))
        self.deck_names = list(self.league["decks"])
        self.decks = [self.league["decks"][name] for name in self.deck_names]
        self.forms = [self.league.get("slots", {}).get(name, [0] * 8) for name in self.deck_names]
        self.rewards = RewardConfig(enabled=args.tower_shaping, princess=args.princess_reward,
                                    king=args.king_reward, crown=args.crown_reward,
                                    overflow=args.overflow_penalty)
        self.tracking = None
        self.tensorboard = None
        if getattr(args, "tensorboard_dir", None) and self.rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "TensorBoard logging was requested but tensorboard is not installed; "
                    "install the pinned training environment before starting v3 PPO"
                ) from exc
            self.tensorboard = SummaryWriter(str(args.tensorboard_dir))
            self.tensorboard.add_text("run/architecture", "architecture-v3 persistent-state PPO", 0)
            self.tensorboard.add_text("run/decks", str(deck_path), 0)
        if getattr(args, "wandb", False) and self.rank == 0:
            import wandb
            self.tracking = wandb.init(project="cr-rl-state-ppo-v3", config={
                "architecture": 3, "decision_ticks": 8, "precision": args.precision,
                "envs": args.envs, "steps": args.steps,
            }, resume="allow")
        self.states = None
        self.opponent_states = None
        self.learner_seats = None
        self.opponent_role = "anchor"
        self.opponent_model = self.anchor_model
        self.reset_all()

    def reset_all(self):
        seeds = [int(self.rng.integers(2**31 - 1)) for _ in range(self.args.envs)] if hasattr(self, "args") else [int(i) for i in range(self.env.count)]
        d0 = self.rng.integers(len(self.decks), size=self.env.count)
        d1 = self.rng.integers(len(self.decks), size=self.env.count)
        self.env.reset(seeds, [self.decks[i] for i in d0], [self.decks[i] for i in d1],
                       [self.forms[i] for i in d0], [self.forms[i] for i in d1])
        self.learner_seats = self.rng.integers(0, 2, size=self.env.count, dtype=np.int32)
        self.states = self.model.initial_state(self.env.count, device=self.device)
        self.opponent_states = self.opponent_model.initial_state(self.env.count, device=self.device)

    def _reset_lane(self, index):
        seed = int(self.rng.integers(2**31 - 1))
        d0, d1 = self.rng.integers(len(self.decks), size=2)
        self.env.vector.reset_env(index, seed, self.decks[d0], self.decks[d1], self.forms[d0], self.forms[d1])

    def _refresh_history(self):
        pool_dir = Path(self.args.out) / "opponents"
        pool_dir.mkdir(parents=True, exist_ok=True)
        score_path = pool_dir / "scores.json"
        if score_path.exists():
            try:
                self.history_scores.update(json.loads(score_path.read_text()))
            except (OSError, ValueError):
                pass
        paths = sorted(pool_dir.glob("history_*.pt"))[-4:]
        self.history_paths = paths

    def _select_opponent(self):
        """Select one frozen opponent for this rollout using role/PFSP buckets."""
        self._refresh_history()
        draw = float(self.rng.random())
        if draw < self.role_mixture.current:
            role, model = "current", self.model
            self.opponent_history_key = None
        elif draw < self.role_mixture.current + self.role_mixture.historical and self.history_paths:
            # PFSP over retained snapshots; absent matchup records use a neutral
            # score, while the uniform floor preserves older strategic niches.
            scores = np.asarray(
                [self.history_scores.get(str(path.resolve()), 0.5) for path in self.history_paths],
                dtype=np.float64,
            )
            weights = np.maximum(1.0 - scores, 1e-3)
            weights = 0.9 * weights / weights.sum() + 0.1 / len(weights)
            index = int(self.rng.choice(len(self.history_paths), p=weights))
            path = self.history_paths[index]
            key = str(path.resolve())
            if key not in self.history_models:
                # Keep one historical replica resident; current + anchor + one
                # history fit comfortably on a 16 GiB V100.
                self.history_models.clear()
                state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
                if state.get("model_architecture_version") != 3:
                    raise ValueError("Historical opponent must use architecture v3")
                frozen = state_model_from_checkpoint(state, self.device)
                frozen.frame_microbatch = self.model.frame_microbatch
                frozen.eval()
                for parameter in frozen.parameters():
                    parameter.requires_grad_(False)
                self.history_models[key] = frozen
            role, model = "historical", self.history_models[key]
            self.opponent_history_key = key
        else:
            role, model = "anchor", self.anchor_model
            self.opponent_history_key = None
        self.opponent_role = role
        self.opponent_model = model
        self.opponent_states = model.initial_state(self.env.count, device=self.device)

    def collect(self):
        self.model.eval()
        self.opponent_model.eval()
        self._select_opponent()
        print(json.dumps({"phase": "v3_rollout_start", "steps": self.args.steps,
                          "opponent_role": self.opponent_role}), flush=True)
        # One learner seat per lane; the other seat is always generated by the
        # selected frozen/current league opponent and is excluded from PPO data.
        lane_segments = [[[]] for _ in range(self.env.count)]
        history_results = []
        segment_states = [_snapshot_memory(self.states, lane)
                          for lane in range(self.env.count)]
        for _ in range(self.args.steps):
            raw = self.env.observations.copy()
            masks = self.env.masks.copy()
            rows = np.arange(self.env.count)
            learner_raw = raw[rows, self.learner_seats]
            opponent_raw = raw[rows, 1 - self.learner_seats]
            learner_masks = masks[rows, self.learner_seats]
            opponent_masks = masks[rows, 1 - self.learner_seats]
            l_header, l_entities, l_entity_mask = self.adapter(learner_raw)
            o_header, o_entities, o_entity_mask = self.adapter(opponent_raw)
            learner_legal = torch.as_tensor(learner_masks, device=self.device).bool()
            opponent_legal = torch.as_tensor(opponent_masks, device=self.device).bool()
            hand_index = np.asarray(self.schema.header["hand_card"])
            learner_hand = torch.as_tensor(learner_raw[:, hand_index], device=self.device)
            opponent_hand = torch.as_tensor(opponent_raw[:, hand_index], device=self.device)
            with torch.inference_mode():
                learner_logp, learner_values, self.states = self.model.step(
                    torch.as_tensor(l_header, device=self.device),
                    torch.as_tensor(l_entities, device=self.device),
                    torch.as_tensor(l_entity_mask, device=self.device), legal_action_mask=learner_legal,
                    state=self.states, hand_card_ids=learner_hand, compute_value=True,
                )
                opponent_logp, _, self.opponent_states = self.opponent_model.step(
                    torch.as_tensor(o_header, device=self.device),
                    torch.as_tensor(o_entities, device=self.device),
                    torch.as_tensor(o_entity_mask, device=self.device), legal_action_mask=opponent_legal,
                    state=self.opponent_states, hand_card_ids=opponent_hand, compute_value=False,
                )

                def sample(logp, legal):
                    probs = torch.nan_to_num(logp.float().exp(), nan=0.0, posinf=0.0, neginf=0.0)
                    probs = probs * legal.float()
                    denom = probs.sum(-1, keepdim=True)
                    count = legal.sum(-1, keepdim=True).clamp_min(1)
                    probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), legal.float() / count)
                    return torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()

                learner_actions = sample(learner_logp, learner_legal)
                opponent_actions = sample(opponent_logp, opponent_legal)
                actions = np.zeros((self.env.count, 2), dtype=np.int64)
                actions[rows, self.learner_seats] = learner_actions
                actions[rows, 1 - self.learner_seats] = opponent_actions
                old_logp = learner_logp.gather(-1, torch.as_tensor(learner_actions, device=self.device)[:, None]).squeeze(-1).cpu().numpy()
                old_values = learner_values.cpu().numpy()
            # Per-step flushed logging can saturate the training volume during long
            # rollouts.  Keep progress at update boundaries instead; detailed
            # metrics are already written to progress.jsonl/TensorBoard.
            self.env.step(actions)
            following = self.env.observations.copy()
            raw_reward = self.env.rewards.copy().astype(np.float32)
            done = self.env.dones.astype(bool).copy()
            infos = [self.env.vector.info(i) for i in range(self.env.count)]
            faulted = np.array([bool(info.get("fault", 0)) for info in infos])
            truncated = np.array([bool(info.get("truncated", 0)) for info in infos]) | faulted
            terminated = done & ~truncated
            shaped, _ = shaped_rewards(raw, following, actions, raw_reward, terminated,
                                       self.schema, self.args.gamma, self.rewards, faulted=faulted)
            for env in range(self.env.count):
                seat = int(self.learner_seats[env])
                lane_segments[env][-1].append({"raw": raw[env, seat].copy(), "mask": masks[env, seat].copy(),
                                    "action": int(actions[env, seat]), "old_logp": float(old_logp[env]),
                                    "value": float(old_values[env]), "reward": float(shaped[env, seat]),
                                    "ended": bool(done[env])})
            if done.any():
                reset_mask = torch.as_tensor(done, device=self.device)
                self.states = type(self.states)(self.states.temporal.reset(reset_mask), id(self.model), self.model.memory_generation)
                self.opponent_states = type(self.opponent_states)(self.opponent_states.temporal.reset(reset_mask), id(self.opponent_model), self.opponent_model.memory_generation)
                for env in np.flatnonzero(done):
                    if self.opponent_history_key is not None:
                        seat = int(self.learner_seats[int(env)])
                        result = float(np.clip((raw_reward[int(env), seat] + 1.0) * 0.5, 0.0, 1.0))
                        history_results.append(result)
                    self._reset_lane(int(env))
                    lane_segments[int(env)].append([])
                    segment_states[int(env)] = _snapshot_memory(self.states, int(env))
        # Bootstrap only unfinished segments from the post-transition state.
        bootstrap_values = np.zeros(self.env.count, dtype=np.float32)
        raw = self.env.observations.copy()
        learner_raw = raw[np.arange(self.env.count), self.learner_seats]
        header, entities, entity_mask = self.adapter(learner_raw)
        hand_index = np.asarray(self.schema.header["hand_card"])
        with torch.inference_mode():
            _, final_values, self.states = self.model.step(
                torch.as_tensor(header, device=self.device), torch.as_tensor(entities, device=self.device),
                torch.as_tensor(entity_mask, device=self.device),
                legal_action_mask=torch.as_tensor(self.env.masks[np.arange(self.env.count), self.learner_seats], device=self.device).bool(),
                state=self.states,
                hand_card_ids=torch.as_tensor(learner_raw[:, hand_index], device=self.device),
                compute_value=True,
            )
            bootstrap_values[:] = final_values.cpu().numpy()
        sequences = []
        for lane_index, segments in enumerate(lane_segments):
            for lane in segments:
                if not lane:
                    continue
                raw_rows = np.stack([x["raw"] for x in lane])
                h, e, m = self.adapter(raw_rows)
                bootstrap = 0.0 if lane[-1]["ended"] else float(bootstrap_values[lane_index])
                advantages, returns = _gae_sequence([x["reward"] for x in lane], [x["value"] for x in lane],
                                                      [x["ended"] for x in lane], bootstrap,
                                                      self.args.gamma, self.args.gae_lambda)
                sequences.append({"header": h, "entities": e, "entity_mask": m,
                                  "_raw_rows": raw_rows,
                                  "_rewards": np.asarray([x["reward"] for x in lane], dtype=np.float32),
                                  "legal_action_mask": np.stack([x["mask"] for x in lane]),
                                  "hand_card_ids": raw_rows[:, np.asarray(self.schema.header["hand_card"])],
                                  "actions": np.asarray([x["action"] for x in lane], dtype=np.int64),
                                  "initial_state": segment_states[lane_index],
                                  "old_logp": np.asarray([x["old_logp"] for x in lane], dtype=np.float32),
                                  "advantages": advantages, "returns": returns})
        # Decision-coverage diagnostics: these were present in the v2 trainer
        # and are useful for detecting the low-cost/WAIT bias.
        choice_rows = forced_affordable = high_elixir_rows = request_count = high_cost_requests = 0
        request_cost = 0.0
        for sequence in sequences:
            raw_rows = np.asarray(sequence["_raw_rows"] if "_raw_rows" in sequence else [])
            if raw_rows.size == 0:
                continue
            masks = np.asarray(sequence["legal_action_mask"])
            actions = np.asarray(sequence["actions"])
            eligible = np.ones(len(actions), dtype=bool)
            choice = masks.sum(-1) > 1
            choice_rows += int((eligible & choice).sum())
            available = raw_rows[:, self.schema.header["elixir"][0]] - raw_rows[:, self.schema.header["elixir_reserved"][0]]
            costs = raw_rows[:, list(self.schema.header["hand_cost"])]
            cards = raw_rows[:, list(self.schema.header["hand_card"])]
            pending = raw_rows[:, list(self.schema.header["hand_pending"])]
            affordable = (cards >= 0) & (pending == 0) & (costs * 10000 <= available[:, None])
            forced_affordable += int((~choice & affordable.any(-1)).sum())
            high_elixir_rows += int((available >= 50000).sum())
            play = (actions > 0) & (actions < 2305)
            slots = ((actions - 1) // 576).clip(0, 3)
            chosen_cost = np.take_along_axis(costs, slots[:, None], axis=1)[:, 0]
            request_count += int(play.sum())
            request_cost += float(chosen_cost[play].sum())
            high_cost_requests += int((play & (chosen_cost >= 5)).sum())
        self.last_rollout_stats = {
            "choice_rows": choice_rows,
            "forced_wait_with_affordable_card": forced_affordable,
            "available_elixir_ge5_rows": high_elixir_rows,
            "card_requests": request_count,
            "requested_card_cost_mean": request_cost / request_count if request_count else 0.0,
            "high_cost_requests": high_cost_requests,
        }
        if self.opponent_history_key is not None and history_results:
            previous = float(self.history_scores.get(self.opponent_history_key, 0.5))
            observed = float(np.mean(history_results))
            self.history_scores[self.opponent_history_key] = 0.9 * previous + 0.1 * observed
            if self.rank == 0:
                score_path = Path(self.args.out) / "opponents" / "scores.json"
                score_path.write_text(json.dumps(self.history_scores, indent=2), encoding="utf-8")
        return sequences

    def run(self):
        for update in range(self.update, self.args.updates):
            rollout_started = time.perf_counter()
            sequences = self.collect()
            rollout_seconds = time.perf_counter() - rollout_started
            metrics = {}
            ppo_started = time.perf_counter()
            for _ in range(self.args.epochs):
                metrics = state_ppo_update(
                    self.model, self.optimizer, sequences,
                    config=StatePPOConfig(clip_coef=self.args.clip_coef, value_coef=self.args.value_coef,
                                          entropy_coef=self.args.entropy_coef, max_grad_norm=self.args.max_grad_norm,
                                          precision=self.args.precision,
                                          max_initial_logp_error=self.args.max_initial_logp_error or 0.05),
                    scaler=self.scaler if self.args.precision == "fp16" else None)
            metrics["rollout_seconds"] = rollout_seconds
            metrics["ppo_seconds"] = time.perf_counter() - ppo_started
            # PPO changed weights; v3 memory generations are intentionally invalidated.
            # Re-seed memory at the next live observation rather than reusing stale KV state.
            self.states = self.model.initial_state(self.env.count, device=self.device)
            self.update = update + 1
            checkpoint = Path(self.args.out) / "ppo_latest.pt"
            should_save = (
                self.update % max(1, int(getattr(self.args, "checkpoint_interval", 10))) == 0
                or self.update >= self.args.updates
            )
            if should_save and self.rank == 0:
                _save(checkpoint, self.model, self.optimizer, self.scaler, self.update, self.args)
                # Retain a small set of frozen v3 snapshots for the historical
                # PFSP bucket.  The latest checkpoint remains the only live
                # learner file; old snapshots are bounded and pruned below.
                history_dir = Path(self.args.out) / "opponents"
                history_dir.mkdir(parents=True, exist_ok=True)
                history_path = history_dir / f"history_{self.update:08d}.pt"
                shutil.copy2(checkpoint, history_path)
                snapshots = sorted(history_dir.glob("history_*.pt"))
                for stale in snapshots[:-4]:
                    stale.unlink(missing_ok=True)
            if should_save and self.world_size > 1:
                torch.distributed.barrier()
            # Validation is deliberately sparse: a paired 20-game check every
            # validation_interval updates (10 games against the fixed BC anchor
            # and 10 against the newest historical snapshot).  It runs outside
            # PPO and its games never enter the training buffer.
            validation_interval = int(getattr(self.args, "validation_interval", 0) or 0)
            should_validate = (
                should_save and validation_interval > 0
                and self.update % validation_interval == 0
                and self.rank == 0
            )
            if should_validate:
                self._run_validation(checkpoint)
            if should_save and self.world_size > 1:
                torch.distributed.barrier()
            if should_save and self.args.backup_dir and self.rank == 0:
                destination = Path(self.args.backup_dir) / "ppo_latest.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                shutil.copy2(checkpoint, temporary)
                temporary.replace(destination)
            samples = sum(len(s["actions"]) for s in sequences)
            advantages = np.concatenate([s["advantages"] for s in sequences]) if sequences else np.empty(0)
            returns = np.concatenate([s["returns"] for s in sequences]) if sequences else np.empty(0)
            old_logp = np.concatenate([s["old_logp"] for s in sequences]) if sequences else np.empty(0)
            rewards = np.concatenate([s["_rewards"] for s in sequences]) if sequences else np.empty(0)
            event = {"phase": "state_ppo", "update": self.update,
                     "sequences": len(sequences), "samples": samples,
                     "samples_per_second": samples / max(float(metrics.get("rollout_seconds", 0.0)) +
                                                          float(metrics.get("ppo_seconds", 0.0)), 1e-9),
                     "adv_mean": float(np.mean(advantages)) if advantages.size else 0.0,
                     "adv_std": float(np.std(advantages)) if advantages.size else 0.0,
                     "return_mean": float(np.mean(returns)) if returns.size else 0.0,
                     "old_logp_mean": float(np.mean(old_logp)) if old_logp.size else 0.0,
                     "shaped_reward_mean": float(np.mean(rewards)) if rewards.size else 0.0,
                     "opponent_role": self.opponent_role,
                     **getattr(self, "last_rollout_stats", {}),
                     **metrics}
            if self.device.type == "cuda":
                event["cuda_allocated_gib"] = float(torch.cuda.memory_allocated(self.device) / 2**30)
                event["cuda_reserved_gib"] = float(torch.cuda.memory_reserved(self.device) / 2**30)
            if self.args.progress_jsonl and self.rank == 0:
                Path(self.args.progress_jsonl).parent.mkdir(parents=True, exist_ok=True)
                with Path(self.args.progress_jsonl).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
            if self.tracking:
                self.tracking.log({f"ppo/{key}": value for key, value in metrics.items()}, step=self.update)
            if self.tensorboard and self.rank == 0:
                for key, value in event.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        group = "rollout" if key in {"sequences", "samples", "samples_per_second",
                                                       "collect_seconds", "rollout_seconds", "ppo_seconds",
                                                       "adv_mean", "adv_std", "return_mean", "old_logp_mean",
                                                       "cuda_allocated_gib", "cuda_reserved_gib", "shaped_reward_mean",
                                                       "choice_rows", "forced_wait_with_affordable_card",
                                                       "available_elixir_ge5_rows", "card_requests",
                                                       "requested_card_cost_mean", "high_cost_requests"} else "ppo_update"
                        self.tensorboard.add_scalar(f"{group}/{key}", value, self.update)
                self.tensorboard.flush()
            if self.rank == 0:
                print(json.dumps(event), flush=True)
        self.env.close()
        if self.world_size > 1:
            torch.distributed.barrier()

    def _run_validation(self, checkpoint):
        """Run the fixed 20-game validation suite and publish scalars."""
        root = Path(__file__).resolve().parents[3]
        evaluator = root / "scripts" / "evaluate_state_policy.py"
        decks = Path(getattr(self.args, "validation_decks", None) or "configs/decks_eval.yaml")
        out_root = Path(self.args.out) / "validation" / f"update_{self.update:08d}"
        pairs = int(getattr(self.args, "validation_pairs", 5))
        out_root.mkdir(parents=True, exist_ok=True)
        results = {}
        history_paths = sorted((Path(self.args.out) / "opponents").glob("history_*.pt"))
        opponents = [("bc", self.anchor_path)]
        if history_paths:
            opponents.append(("history", history_paths[-1]))
        for label, opponent in opponents:
            target = out_root / f"vs_{label}"
            command = [sys.executable, str(evaluator), "--policy", str(checkpoint),
                       "--opponent", str(opponent), "--decks", str(decks),
                       "--pairs", str(pairs), "--envs", "2", "--device", "cuda:0",
                       "--precision", self.args.precision, "--out", str(target),
                       "--no-native-audit"]
            log_path = target.with_suffix(".log")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, env=dict(__import__("os").environ))
                payload = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))
                valid_games = int(payload.get("games", 0))
                wins = int(payload.get("wins", 0))
                results[label] = {"wins": wins, "games": valid_games,
                                  "win_rate": wins / valid_games if valid_games else 0.0,
                                  "invalid_games": int(payload.get("invalid_games", 0))}
            except Exception as exc:  # validation must never stop a long run
                (target / "validation_error.json").write_text(
                    json.dumps({"error": repr(exc)}, indent=2), encoding="utf-8")
                results[label] = {"wins": 0, "games": 0, "win_rate": 0.0,
                                  "invalid_games": -1}
        if self.tensorboard:
            for label, values in results.items():
                self.tensorboard.add_scalar(f"validation/{label}_win_rate", values["win_rate"], self.update)
                self.tensorboard.add_scalar(f"validation/{label}_valid_games", values["games"], self.update)
                self.tensorboard.add_scalar(f"validation/{label}_invalid_games", values["invalid_games"], self.update)
            self.tensorboard.flush()
        (out_root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "validation", "update": self.update, "results": results}), flush=True)


def run(args):
    trainer = StateSelfplay(args)
    trainer.args = args
    try:
        trainer.run()
    finally:
        trainer.env.close()
        if trainer.tracking:
            trainer.tracking.finish()
        if trainer.tensorboard:
            trainer.tensorboard.close()
        if trainer.world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
