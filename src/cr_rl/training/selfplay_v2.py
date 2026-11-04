"""Persistent v2 self-play: shared behavior seats, full-episode value warmup.

Only a process restart resets live simulator episodes. Every update keeps the
same vector, unfinished episode traces and histories; its behavior copy is frozen
until collection ends. Historical opponents stay fixed within each match.
"""

import copy
import json
import subprocess
import sys
import time
from collections import OrderedDict, deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from cr_rl.config import load_yaml
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.data.trajectory import TrajectoryBuffer
from cr_rl.models import ActorCritic, ActorPolicy, HistoryValueNetwork, ModelConfig
from cr_rl.opponents import OpponentEntry, OpponentPool
from cr_rl.opponents.admission import SnapshotAdmission
from cr_rl.opponents.role_sampling import FROZEN_VERSION_BASE, RoleMixture, historical_policy_id
from cr_rl.training.advantages import compute_gae
from cr_rl.training.episode_trace import EpisodeTrace, on_policy_seats
from cr_rl.training.inference import FrozenInference
from cr_rl.training.initialization import initialize_from_bc
from cr_rl.training.persistence import atomic_json, copy_verified, publish_update
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.resume_options import override_resume_rates
from cr_rl.training.rewards import RewardConfig, shaped_rewards
from cr_rl.training.separate import optimize_separate


class PersistentSelfplay:
    def __init__(self, args):
        from cr_rl.envs.crforge import CrForgeVector

        self.args, self.device = args, args.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        if not args.resume and (self.out / "ppo_latest.pt").exists():
            raise ValueError("Use a new v2 output directory or --resume")
        self.env = CrForgeVector(args.envs, min(args.envs, 8), independent=True)
        self.env.set_autoreset(False)
        self.schema = self.env.vector.schema
        self.adapter = CrForgeObservationAdapter(
            self.schema, self.env.vector.vocabulary_size, entity_bucket=16
        )
        source = args.resume or args.init_from_bc
        state = (
            torch.load(source, map_location="cpu", weights_only=False, mmap=True)
            if source
            else None
        )
        self.cfg = (
            ModelConfig(**state["config"])
            if state
            else (ModelConfig.preset(args.model).for_v2(self.schema))
        )
        if getattr(args, "frame_encoder_fp32", False):
            self.cfg = replace(self.cfg, frame_encoder_fp32=True)
        if self.cfg.representation_version != 2:
            raise ValueError("Persistent v2 training requires the clean-slate v2 representation")
        if self.cfg.header_fields != tuple(self.schema.header_fields) or (
            self.cfg.entity_fields != tuple(self.schema.entity_fields)
        ):
            raise ValueError("Saved v2 observation schema differs from the running simulator")
        torch.set_float32_matmul_precision("highest")
        shared = ActorCritic(self.cfg).to(self.device)
        self.initialization = None
        if args.init_from_bc:
            self.initialization = initialize_from_bc(
                shared, args.init_from_bc, self.cfg.__dict__, self.device
            )
        self.actor = ActorPolicy.from_shared(shared).eval()
        del shared
        critic_cfg = (
            self.cfg
            if args.critic_model == "same"
            else (ModelConfig.preset(args.critic_model).inherit_representation(self.cfg))
        )
        if args.resume:
            critic_cfg = ModelConfig(**state["critic_config"])
            if getattr(args, "frame_encoder_fp32", False):
                critic_cfg = replace(critic_cfg, frame_encoder_fp32=True)
        if critic_cfg.context_len != self.cfg.context_len:
            raise ValueError("Independent actor/critic history lengths must match")
        self.critic = HistoryValueNetwork(critic_cfg).to(self.device).eval()
        self.critic_initialization = self.critic.initialize_encoder(self.actor)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=args.critic_learning_rate
        )
        self.rewards = RewardConfig(
            enabled=args.tower_shaping,
            princess=args.princess_reward,
            king=args.king_reward,
            crown=args.crown_reward,
            overflow=args.overflow_penalty,
        )
        previous_warmup = state.get("critic_warmup_config", {}) if args.resume else {}
        self.warmup_config = {
            "rounds": args.critic_warmup_updates or previous_warmup.get("rounds", 2),
            "games": args.critic_warmup_games or previous_warmup.get("games", 4),
        }
        self.warmup = {"rounds": 0, "games": 0}
        self.update = self.samples = self.next_episode = 0
        pool_path = self.out / "opponent_pool.json"
        self.pool = (
            OpponentPool.load(pool_path)
            if args.resume
            else (OpponentPool(max_entries=args.pool_capacity, ema_decay=0.95))
        )
        self.admission = SnapshotAdmission(
            interval_samples=args.snapshot_interval,
            min_games=args.snapshot_min_games,
            min_score=args.snapshot_min_score,
        )
        self.resume_learning_rates = None
        if args.resume:
            if state.get("training_driver") != "persistent-v2":
                raise ValueError("Cannot resume a reset-per-update trainer as persistent v2")
            if state["reward_config"] != self.rewards.metadata() or state["gamma"] != args.gamma:
                raise ValueError("Resume reward/discount configuration mismatch")
            if state["critic_warmup_config"] != self.warmup_config:
                raise ValueError("Resume warmup configuration mismatch")
            self.actor.load_state_dict(state["model"], strict=True)
            self.critic.load_state_dict(state["critic"], strict=True)
            self.actor_optimizer.load_state_dict(state["optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.resume_learning_rates = override_resume_rates(
                self.actor_optimizer,
                self.critic_optimizer,
                getattr(args, "resume_actor_learning_rate", None),
                getattr(args, "resume_critic_learning_rate", None),
            )
            self.warmup, self.update, self.samples = (
                state["critic_warmup"],
                state["updates"],
                state["learner_samples_total"],
            )
            self.initialization = state["initialization"]
            self.next_episode = state["next_episode"]
            actual = {
                "backend_commit": self.env.metadata().get("commit"),
                "data_version": self.env.vector.data_version,
                "content_sha": self.env.vector.content_sha,
            }
            if any(state[key] != value for key, value in actual.items()):
                raise ValueError("Resume simulator provenance mismatch")
            self.admission = SnapshotAdmission(**state["admission"])
            self.rng.bit_generator.state = state["numpy_rng_state"]
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if self.device.startswith("cuda") and state.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng_state"]])
        elif args.init_from_bc:
            actual = {
                "backend_commit": self.env.metadata().get("commit"),
                "data_version": self.env.vector.data_version,
                "content_sha": self.env.vector.content_sha,
                "level": self.env.level,
                "decision_ticks": 5,
                "mask_version": 2,
            }
            if any(self.initialization[key] != value for key, value in actual.items()):
                raise ValueError("BC/simulator provenance mismatch")
        saved_mixture = state.get("role_mixture") if args.resume else None
        self.role_mixture = (
            RoleMixture(**saved_mixture)
            if saved_mixture
            else RoleMixture()
            if getattr(args, "role_league", False)
            else None
        )
        self.role_draws = dict(state.get("role_draws", {})) if args.resume else {}
        if self.role_mixture and self.admission.purpose != "history":
            # Explicit migration starts a fresh retention clock at the resumed sample count.
            self.admission = SnapshotAdmission(
                interval_samples=args.history_snapshot_interval,
                last_admitted_samples=self.samples,
                purpose="history",
            )
        self.league = (
            state["league_config"]
            if args.resume
            else (yaml.safe_load(args.decks.read_text()) if args.decks else load_yaml("decks.yaml"))
        )
        if args.resume and args.decks and yaml.safe_load(args.decks.read_text()) != self.league:
            raise ValueError("Resume deck pool mismatch")
        self.deck_names = list(self.league["decks"])
        self.decks = [self.league["decks"][name] for name in self.deck_names]
        self.forms = [self.league.get("slots", {}).get(name, [0] * 8) for name in self.deck_names]
        for deck in self.decks:
            if len(deck) != 8 or len(set(deck)) != 8:
                raise ValueError("Decks must have eight distinct cards")
            for card in deck:
                self.env.vector.card_id(card)
        self.deck_ids = np.zeros((args.envs, 2), dtype=np.int32)
        self.learner_seats = np.zeros(args.envs, dtype=np.int32)
        self.seeds = np.zeros(args.envs, dtype=np.int64)
        self.opponents = [None] * args.envs
        self.traces = [None] * args.envs
        self.histories = [deque(maxlen=self.cfg.context_len) for _ in range(args.envs * 2)]
        self.incremental_history = None
        if getattr(args, "incremental_inference", False):
            from cr_rl.models.incremental_frames import IncrementalFrameHistory

            self.incremental_history = IncrementalFrameHistory(
                args.envs * 2, self.cfg.context_len, self.schema.seat_ints
            )
        self.opponent_cache = OrderedDict()
        self.critic_inference = FrozenInference(self.critic, self.adapter, self.device)
        self.actor_inference = FrozenInference(self.actor, self.adapter, self.device)
        self.completed = []
        if not self.pool.entries:
            path = self.save_opponent(0)
            self.pool.add(OpponentEntry(0, str(path), 0, role="anchor"))
        if getattr(args, "history_seed_manifest", None):
            from cr_rl.opponents.bootstrap import seed_history

            if not self.role_mixture:
                raise ValueError("History seeding requires --role-league")
            seed_history(
                self.pool,
                args.history_seed_manifest,
                self.out,
                self.cfg.__dict__,
                {
                    "backend_commit": self.env.metadata().get("commit"),
                    "data_version": self.env.vector.data_version,
                    "content_sha": self.env.vector.content_sha,
                },
            )
        for env in range(args.envs):
            self.reset_match(env)
        self.tracking = self.telemetry = None
        if getattr(args, "wandb", False):
            import wandb

            from cr_rl.training.telemetry import TrainingTelemetry

            run_file = self.out / "wandb_run.json"
            options = (
                {"id": json.loads(run_file.read_text())["id"], "resume": "allow"}
                if run_file.exists()
                else {}
            )
            self.tracking = wandb.init(
                project="cr-rl",
                name=self.out.name,
                job_type="ppo-v2",
                config={
                    "model": self.cfg.__dict__,
                    "reward": self.rewards.metadata(),
                    "driver": "persistent-v2",
                },
                **options,
            )
            atomic_json(run_file, {"id": self.tracking.id, "url": self.tracking.url})
            self.telemetry = TrainingTelemetry(self.tracking, args.steps * args.envs)

    def save_opponent(self, policy_id):
        target = self.out / f"opponent_{policy_id:06d}.pt"
        temporary = target.with_suffix(".tmp.pt")
        torch.save(
            {
                "model": self.actor.state_dict(),
                "config": self.cfg.__dict__,
                "policy_only": True,
                "policy_id": policy_id,
                "model_architecture_version": 2,
            },
            temporary,
        )
        temporary.replace(target)
        return target

    def candidate_gate(self, candidate):
        """Check the exact new checkpoint in paired complete games before admission."""
        evaluation = self.out / "candidate_gates" / f"update_{self.update:06d}"
        evaluation.mkdir(parents=True, exist_ok=True)
        league_path = self.out / "league.yaml"
        league_path.write_text(yaml.safe_dump(self.league), encoding="utf-8")
        anchor = next(
            (entry for entry in self.pool.entries if entry.role == "anchor"), self.pool.entries[0]
        )
        script = Path(__file__).resolve().parents[3] / "scripts/evaluate_policy.py"
        self.emit(phase="candidate_gate", candidate_policy_id=self.update)
        with (evaluation / "evaluation.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--policy",
                    str(candidate),
                    "--opponent",
                    anchor.checkpoint,
                    "--decks",
                    str(league_path),
                    "--out",
                    str(evaluation),
                    "--pairs",
                    "1",
                    "--frame-cache",
                    "--device",
                    self.device,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
            )
        report_path = evaluation / "evaluation.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        passed = (
            result.returncode == 0 and report.get("games") == 2 and not report.get("invalid_games")
        )
        atomic_json(
            evaluation / "gate.json",
            {
                "passed": passed,
                "exit_code": result.returncode,
                "scope": "Numerical, legal, complete paired games; not a strength claim",
            },
        )
        return passed

    def emit(self, **event):
        event = {
            "time_unix": time.time(),
            "update": self.update + 1,
            "learner_samples_total": self.samples,
            **event,
        }
        path = getattr(self.args, "progress_jsonl", None) or self.out / "progress.jsonl"
        with Path(path).open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if self.telemetry:
            self.telemetry.log(event)

    def reset_match(self, env):
        self.deck_ids[env] = self.rng.integers(len(self.decks), size=2)
        self.learner_seats[env] = self.rng.integers(2)
        self.seeds[env] = self.rng.integers(1, 2**31 - 1)
        own, enemy = (
            self.deck_ids[env, self.learner_seats[env]],
            self.deck_ids[env, 1 - self.learner_seats[env]],
        )
        if self.role_mixture:
            opponent, requested, actual = self.role_mixture.sample(
                self.pool, self.rng, self.deck_names[own], self.deck_names[enemy], self.update
            )
            key = f"{requested}->{actual}"
            self.role_draws[key] = self.role_draws.get(key, 0) + 1
        else:
            opponent = self.sample_legacy_opponent(own, enemy)
        self.opponents[env] = opponent
        d0, d1 = self.deck_ids[env]
        self.env.vector.reset_env(
            env,
            int(self.seeds[env]),
            self.decks[d0],
            self.decks[d1],
            self.forms[d0],
            self.forms[d1],
        )
        self.traces[env] = EpisodeTrace(self.next_episode, int(self.seeds[env]))
        self.next_episode += 1
        self.histories[env * 2].clear()
        self.histories[env * 2 + 1].clear()

    def sample_legacy_opponent(self, own, enemy):
        draw = self.rng.random()
        history = sorted(self.pool.entries, key=lambda entry: entry.generation)
        if draw < self.args.current_opponent_fraction or not history:
            opponent = None
        else:
            candidates = (
                history[-4:]
                if draw < (self.args.current_opponent_fraction + self.args.historical_pfsp_fraction)
                else (history[:-4] or history)
            )
            scores = np.array(
                [
                    self.pool.score_for(
                        entry, self.deck_names[own], self.deck_names[enemy], self.update
                    )
                    for entry in candidates
                ]
            )
            weights = np.maximum(1 - scores, 1e-3)
            weights = 0.9 * weights / weights.sum() + 0.1 / len(weights)
            opponent = candidates[int(self.rng.choice(len(candidates), p=weights))]
        return opponent

    def windows(self, raw, append=False):
        windows = np.zeros(
            (self.args.envs * 2, self.cfg.context_len, self.schema.seat_ints), dtype=np.int32
        )
        time_mask = np.zeros(windows.shape[:2], dtype=bool)
        for i, frame in enumerate(raw.reshape(-1, self.schema.seat_ints)):
            sequence = [*self.histories[i], frame][-self.cfg.context_len :]
            windows[i, -len(sequence) :] = sequence
            time_mask[i, -len(sequence) :] = True
            if append:
                self.histories[i].append(frame.copy())
        return windows, time_mask

    def historical_inference(self, entry):
        if entry.policy_id not in self.opponent_cache:
            while len(self.opponent_cache) >= self.args.opponent_cache_size:
                self.opponent_cache.popitem(last=False)
            state = torch.load(entry.checkpoint, map_location="cpu", weights_only=False, mmap=True)
            model = ActorPolicy(ModelConfig(**state["config"])).to(self.device).eval()
            model.load_state_dict(state["model"], strict=True)
            model.requires_grad_(False)
            self.opponent_cache[entry.policy_id] = FrozenInference(model, self.adapter, self.device)
        self.opponent_cache.move_to_end(entry.policy_id)
        return self.opponent_cache[entry.policy_id]

    def inference_history(self, raw, append=False):
        if self.incremental_history is None:
            return self.windows(raw, append=append)
        raw = raw.reshape(-1, self.schema.seat_ints)
        episodes = np.repeat([trace.episode_id for trace in self.traces], 2)
        if append:
            self.incremental_history.append(raw, episodes)
            return self.incremental_history.request()
        return self.incremental_history.request(following=raw, episodes=episodes)

    @staticmethod
    def infer(inference, history, rows=None, masks=None, sample=False):
        if not isinstance(history, tuple):
            selected = history if rows is None else history.select(rows)
            method = inference.sample_incremental if sample else inference.evaluate_incremental
            return method(selected, masks)
        windows, time_mask = history
        if rows is not None:
            windows, time_mask = windows[rows], time_mask[rows]
        method = inference.sample if sample else inference.evaluate
        return method(windows, time_mask, masks)

    def close(self):
        self.env.close()
        if self.tracking:
            self.tracking.finish()

    def collect(self):
        args = self.args
        if not 0 <= self.update < FROZEN_VERSION_BASE:
            raise ValueError("Current behavior version overlaps frozen policy namespace")
        snapshots, references, completed = [], [], []
        starts = [int(self.env.vector.info(i)["episode_step"]) for i in range(args.envs)]
        collected = 0
        began = time.perf_counter()
        for step in range(args.steps):
            raw, masks = self.env.observations.copy(), self.env.masks.copy()
            history = self.inference_history(raw, append=True)
            versions = np.full((args.envs, 2), self.update, dtype=np.int32)
            for env, entry in enumerate(self.opponents):
                if entry is not None:
                    # Stored legacy IDs (including BC=0) are not behavior versions.
                    versions[env, 1 - self.learner_seats[env]] = historical_policy_id(
                        entry.generation
                    )
            learners = on_policy_seats(versions, self.update, self.learner_seats)
            actions = np.zeros((args.envs, 2), dtype=np.int32)
            logp = np.zeros((args.envs, 2), dtype=np.float32)
            actor_indices = np.flatnonzero(learners.reshape(-1))
            selected, selected_logp, stats = self.infer(
                self.actor_inference,
                history,
                actor_indices,
                masks.reshape(-1, self.schema.actions)[actor_indices],
                sample=True,
            )
            actions.reshape(-1)[actor_indices] = selected
            logp.reshape(-1)[actor_indices] = selected_logp
            for policy_id in sorted({e.policy_id for e in self.opponents if e is not None}):
                indices = np.array(
                    [
                        env * 2 + 1 - self.learner_seats[env]
                        for env, e in enumerate(self.opponents)
                        if e is not None and e.policy_id == policy_id
                    ]
                )
                entry = next(
                    entry for entry in self.opponents if entry and entry.policy_id == policy_id
                )
                inference = self.historical_inference(entry)
                selected, selected_logp, _ = self.infer(
                    inference,
                    history,
                    indices,
                    masks.reshape(-1, self.schema.actions)[indices],
                    sample=True,
                )
                actions.reshape(-1)[indices], logp.reshape(-1)[indices] = selected, selected_logp
            values = self.infer(self.critic_inference, history).cpu().numpy().reshape(args.envs, 2)
            if not np.take_along_axis(masks, actions[..., None], -1).all():
                raise RuntimeError("v2 policy sampled an illegal action")
            self.env.step(actions)
            following, next_masks = self.env.observations.copy(), self.env.masks.copy()
            raw_reward, done = self.env.rewards.copy(), self.env.dones.astype(bool).copy()
            infos = [self.env.vector.info(i) for i in range(args.envs)]
            faults = np.array([int(info["fault"]) for info in infos], dtype=np.int16)
            truncated = done & (
                np.array([bool(info["truncated"]) for info in infos]) | (faults != 0)
            )
            terminated = done & ~truncated
            training_reward, components = shaped_rewards(
                raw,
                following,
                actions,
                raw_reward,
                terminated,
                self.schema,
                args.gamma,
                self.rewards,
                faulted=faults != 0,
            )
            final_values = np.zeros((args.envs, 2), dtype=np.float32)
            if (truncated & (faults == 0)).any():
                following_history = self.inference_history(following)
                final_values = (
                    self.infer(self.critic_inference, following_history)
                    .cpu()
                    .numpy()
                    .reshape(args.envs, 2)
                )
            episode_ids = np.array([trace.episode_id for trace in self.traces], dtype=np.int64)
            episode_start = np.array(
                [len(trace.observations) == 0 for trace in self.traces], dtype=np.uint8
            )
            for env, trace in enumerate(self.traces):
                index = trace.append(raw[env], masks[env], training_reward[env], learners[env])
                references.extend((trace, index, seat) for seat in range(2))
            snapshots.append(
                {
                    "obs": raw,
                    "masks": masks,
                    "actions": actions,
                    "rewards": raw_reward,
                    "training_rewards": training_reward,
                    "next_obs": following,
                    "next_masks": next_masks,
                    "dones": done.astype(np.uint8),
                    "terminated": terminated.astype(np.uint8),
                    "truncated": truncated.astype(np.uint8),
                    "old_log_probs": logp,
                    "old_values": values,
                    "learner_mask": learners,
                    "policy_ids": versions,
                    "policy_versions": versions.copy(),
                    "opponent_ids": versions[:, ::-1].copy(),
                    "episode_ids": episode_ids,
                    "episode_start": episode_start,
                    "episode_steps": np.array([info["episode_step"] for info in infos]),
                    "ticks": np.array([info["tick"] for info in infos]),
                    "winner": np.array(
                        [infos[i]["winner"] if terminated[i] else -1 for i in range(args.envs)]
                    ),
                    "end_reason": np.where(
                        faults, 3, np.where(truncated, 2, terminated.astype(int))
                    ),
                    "fault_codes": faults,
                    "seeds": self.seeds.copy(),
                    "action_consumed": np.ones((args.envs, 2), dtype=np.uint8),
                    "action_refusals": following[..., self.schema.header["last_refusal"][0]],
                    "run_refusals": following[..., self.schema.header["last_run_refusal"][0]],
                    "_final_values": final_values,
                    "_components": components,
                }
            )
            collected += int(learners.sum())
            for env in np.flatnonzero(done):
                trace, seat, entry = (
                    self.traces[env],
                    int(self.learner_seats[env]),
                    self.opponents[env],
                )
                trace.finish(terminated=terminated[env], faulted=faults[env] != 0)
                if trace.completed:
                    completed.append(trace)
                    if entry is not None:
                        d0, d1 = self.deck_ids[env, seat], self.deck_ids[env, 1 - seat]
                        self.pool.record(
                            entry.policy_id,
                            float(raw_reward[env, seat]),
                            self.deck_names[d0],
                            self.deck_names[d1],
                            self.update,
                        )
                        self.admission.record(float(raw_reward[env, seat]))
                self.emit(
                    phase="episode",
                    reward=float(raw_reward[env, seat]),
                    episode_length=int(infos[env]["episode_step"]),
                    fault=int(faults[env]),
                    truncated=int(truncated[env]),
                    learner_samples_total=self.samples + collected,
                )
                self.reset_match(env)
            if step % 64 == 0 or step == args.steps - 1:
                self.emit(
                    phase="policy",
                    learner_samples_total=self.samples + collected,
                    play_fraction=float((actions[learners.astype(bool)] != 0).mean()),
                    **stats,
                )
                self.emit(
                    phase="rollout",
                    rollout_step=step + 1,
                    rollout_total=args.steps,
                    rollout_samples=collected,
                    seconds=time.perf_counter() - began,
                    learner_samples_total=self.samples + collected,
                )
        following_history = self.inference_history(self.env.observations.copy())
        last = (
            self.infer(self.critic_inference, following_history).cpu().numpy().reshape(args.envs, 2)
        )
        next_values = np.empty((args.steps, args.envs, 2), dtype=np.float32)
        if len(snapshots) > 1:
            next_values[:-1] = np.stack([row["old_values"] for row in snapshots[1:]])
        next_values[-1] = last
        for index, row in enumerate(snapshots):
            next_values[index, row["terminated"].astype(bool)] = 0
            trunc = row["truncated"].astype(bool)
            next_values[index, trunc] = row["_final_values"][trunc]
            validity = np.array(
                [references[index * args.envs * 2 + env * 2][0].valid for env in range(args.envs)],
                dtype=np.uint8,
            )
            row["valid"] = np.repeat(validity[:, None], 2, axis=1)
            row["bootstrap_values"], row["bootstrap_valid"] = (
                next_values[index].copy(),
                row["valid"].copy(),
            )
        return snapshots, references, completed, next_values, starts

    def optimize(self, snapshots, refs, completed, next_values):
        args = self.args
        actor_enabled = all(self.warmup[key] >= self.warmup_config[key] for key in self.warmup)
        reward = np.stack([row["training_rewards"] for row in snapshots])
        values = np.stack([row["old_values"] for row in snapshots])
        terminated = np.stack([row["terminated"] for row in snapshots])
        ends = np.stack([row["dones"] for row in snapshots])
        shape = (args.steps, args.envs * 2)
        advantage, returns = compute_gae(
            torch.as_tensor(reward.reshape(shape), device=self.device),
            torch.as_tensor(values.reshape(shape), device=self.device),
            torch.as_tensor(
                np.repeat(terminated[..., None], 2, -1).reshape(shape), device=self.device
            ),
            torch.as_tensor(next_values[-1].reshape(-1), device=self.device),
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            next_values=torch.as_tensor(next_values.reshape(shape), device=self.device),
            episode_ends=torch.as_tensor(
                np.repeat(ends[..., None], 2, -1).reshape(shape), device=self.device
            ),
        )
        eligible = (
            np.stack([row["learner_mask"] & row["valid"] for row in snapshots])
            .reshape(-1)
            .astype(bool)
        )
        actor_indices = np.flatnonzero(eligible)
        if not len(actor_indices):
            raise RuntimeError("No valid on-policy samples in the collected rollout")
        self.samples += len(actor_indices)
        advantage = advantage.flatten()
        selected = advantage[torch.as_tensor(actor_indices, device=self.device)]
        advantage = (advantage - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
        targets = returns.flatten()
        critic_indices = actor_indices
        old_logp = np.stack([row["old_log_probs"] for row in snapshots]).reshape(-1)
        actions = np.stack([row["actions"] for row in snapshots]).reshape(-1)
        if not actor_enabled:
            refs, mc_targets = [], []
            for trace in completed:
                mc = trace.monte_carlo_returns(args.gamma)
                for index, learners in enumerate(trace.learners):
                    for seat in np.flatnonzero(learners):
                        refs.append((trace, index, int(seat)))
                        mc_targets.append(mc[index, seat])
            critic_indices = np.arange(len(refs), dtype=np.int64)
            actor_indices = np.zeros(0, dtype=np.int64)
            targets = torch.as_tensor(mc_targets, dtype=torch.float32, device=self.device)
            advantage = torch.zeros_like(targets)
            old_logp = np.zeros(len(refs), dtype=np.float32)
            actions = np.zeros(len(refs), dtype=np.int64)

        preparation_started = time.perf_counter()
        make_batch = (
            PreparedHistoryBatches(refs, self.cfg.context_len, self.adapter, self.device)
            if refs
            else None
        )
        preparation_metadata = make_batch.metadata if make_batch else {"bytes": 0, "samples": 0}
        self.emit(phase="inputs_prepared", **preparation_metadata)
        preparation_seconds = time.perf_counter() - preparation_started

        behavior = copy.deepcopy(self.actor).eval() if actor_enabled else self.actor
        if actor_enabled:
            for parameter in behavior.parameters():
                parameter.requires_grad_(False)
        optimization_started = time.perf_counter()
        result = optimize_separate(
            self.actor,
            self.critic,
            behavior,
            self.actor_optimizer,
            self.critic_optimizer,
            make_batch,
            actor_indices,
            critic_indices,
            torch.as_tensor(old_logp, device=self.device),
            torch.as_tensor(actions, device=self.device),
            advantage,
            targets,
            rng=self.rng,
            device=self.device,
            batch_size=args.batch_size,
            accumulation=args.grad_accumulation,
            actor_epochs=args.epochs,
            critic_epochs=args.critic_epochs,
            clip_coef=args.clip_coef,
            entropy_coef=args.entropy_coef,
            reference_kl_coef=args.reference_kl_coef,
            target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm,
            actor_enabled=actor_enabled,
            emit=self.emit,
            cache_reference=getattr(args, "cache_reference", False),
            defer_scalar_metrics=getattr(args, "defer_scalar_metrics", False),
            actor_batch_size=getattr(args, "actor_batch_size", None),
            critic_batch_size=getattr(args, "critic_batch_size", None),
        )
        result["metrics"]["input_preparation_seconds"] = preparation_seconds
        result["metrics"]["optimization_seconds"] = time.perf_counter() - optimization_started
        result["metrics"]["prepared_input_gib"] = preparation_metadata["bytes"] / 2**30
        if not actor_enabled and result["critic_steps"]:
            self.warmup["rounds"] += 1
            self.warmup["games"] += len(completed)
        return result, actor_enabled, int(eligible.sum())

    def checkpoint(self):
        state = {
            "training_driver": "persistent-v2",
            "model_architecture_version": 2,
            "training_architecture_version": 4,
            "policy_only": True,
            "config": self.cfg.__dict__,
            "critic_config": self.critic.cfg.__dict__,
            "model": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.update,
            "learner_samples_total": self.samples,
            "initialization": self.initialization,
            "critic_initialization": self.critic_initialization,
            "critic_warmup": self.warmup,
            "critic_warmup_config": self.warmup_config,
            "reward_config": self.rewards.metadata(),
            "gamma": self.args.gamma,
            "gae_lambda": self.args.gae_lambda,
            "league_config": self.league,
            "admission": self.admission.state_dict(),
            "resume_learning_rates": self.resume_learning_rates,
            "role_mixture": self.role_mixture.state_dict() if self.role_mixture else None,
            "role_draws": self.role_draws,
            "next_episode": self.next_episode,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if self.device.startswith("cuda")
            else None,
            "numpy_rng_state": self.rng.bit_generator.state,
            "backend_commit": self.env.metadata().get("commit"),
            "data_version": self.env.vector.data_version,
            "content_sha": self.env.vector.content_sha,
            "resume_semantics": "continuous_simulator_between_updates_reset_on_process_restart",
            "unfinished_episode_steps": [len(trace.observations) for trace in self.traces],
        }
        temporary = self.out / "ppo_latest.tmp.pt"
        torch.save(state, temporary)
        temporary.replace(self.out / "ppo_latest.pt")
        export = self.out / "actor_export.tmp.pt"
        torch.save(
            {
                key: state[key]
                for key in (
                    "model",
                    "config",
                    "model_architecture_version",
                    "policy_only",
                    "updates",
                    "backend_commit",
                    "data_version",
                    "content_sha",
                )
            },
            export,
        )
        export.replace(self.out / "actor_export.pt")

    def run_update(self):
        started = time.perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        snapshots, refs, completed, next_values, starts = self.collect()
        collected_at = time.perf_counter()
        result, actor_enabled, valid_samples = self.optimize(
            snapshots, refs, completed, next_values
        )
        optimized_at = time.perf_counter()
        self.update += 1
        admission_started = time.perf_counter()
        admission_seconds = 0.0
        if result["actor_steps"] and self.admission.ready(self.samples):
            snapshot_id = historical_policy_id(self.update)
            path = self.save_opponent(snapshot_id)
            if self.candidate_gate(path):
                self.pool.add(
                    OpponentEntry(
                        snapshot_id,
                        str(path.resolve()),
                        self.update,
                        deck_id="multi_deck",
                        role="historical" if self.role_mixture else "main",
                    )
                )
                self.admission.admitted(self.samples)
            else:
                self.admission.last_failed_candidate_samples = self.samples
                path.unlink()
            admission_seconds = time.perf_counter() - admission_started
        evicted = self.pool.prune(
            [entry.policy_id for entry in self.opponents if entry is not None]
        )
        if evicted:
            path = self.out / "archived_opponents.jsonl"
            with path.open("a") as stream:
                for entry in evicted:
                    stream.write(json.dumps(entry.__dict__) + "\n")
        self.pool.save(self.out / "opponent_pool.json")
        buffer = TrajectoryBuffer(
            {
                **self.env.metadata(),
                "training_driver": "persistent-v2",
                "model_config": self.cfg.__dict__,
                "gamma": self.args.gamma,
                "gae_lambda": self.args.gae_lambda,
                "training_reward_config": self.rewards.metadata(),
                "behaviour_temperature": 1.0,
            }
        )
        for row in snapshots:
            buffer.append(**{key: value for key, value in row.items() if not key.startswith("_")})
        rollout = buffer.save(self.out / "rollouts" / f"rollout_{self.update:06d}.npz")
        self.checkpoint()
        saved_at = time.perf_counter()
        terminal_results = np.concatenate(
            [row["rewards"][row["terminated"].astype(bool)].reshape(-1) for row in snapshots]
        )
        summary = {
            "update": self.update,
            "learner_samples_total": self.samples,
            "valid_learner_samples": valid_samples,
            "actor_enabled": actor_enabled,
            "optimizer_steps": result["actor_steps"],
            "critic_optimizer_steps": result["critic_steps"],
            "actor_kl_stopped": result["actor_kl_stopped"],
            "critic_warmup": self.warmup.copy(),
            "metrics": result["metrics"],
            "reference_cache": result.get("reference_cache", {}),
            "optimization_batches": {
                "effective": self.args.batch_size * self.args.grad_accumulation,
                "actor_microbatch": getattr(self.args, "actor_batch_size", None)
                or self.args.batch_size,
                "critic_microbatch": getattr(self.args, "critic_batch_size", None)
                or self.args.batch_size,
            },
            "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
            if self.device.startswith("cuda")
            else None,
            "terminal_games": int(sum(row["terminated"].sum() for row in snapshots)),
            "terminal_seat_wins": int((terminal_results > 0).sum()),
            "faulted_samples": int(sum((row["valid"] == 0).sum() for row in snapshots)),
            "learner_reward_sum": float(
                sum(
                    (row["training_rewards"] * row["learner_mask"] * row["valid"]).sum()
                    for row in snapshots
                )
            ),
            "learner_raw_reward_sum": float(
                sum(
                    (row["rewards"] * row["learner_mask"] * row["valid"]).sum() for row in snapshots
                )
            ),
            "reward_component_sums": {
                name: float(
                    sum(
                        (row["_components"][name] * row["learner_mask"] * row["valid"]).sum()
                        for row in snapshots
                    )
                )
                for name in ("hp", "crown", "overflow")
            },
            "episode_steps_at_update_start": starts,
            "episode_steps_at_update_end": [
                int(self.env.vector.info(i)["episode_step"]) for i in range(self.args.envs)
            ],
            "initial_logp_max_error": max(result["initial_logp_errors"], default=0),
            "opponent_pool_size": len(self.pool.entries),
            "role_mixture": self.role_mixture.state_dict() if self.role_mixture else None,
            "role_draws": dict(self.role_draws),
            "learning_rates": {
                "actor": [float(g["lr"]) for g in self.actor_optimizer.param_groups],
                "critic": [float(g["lr"]) for g in self.critic_optimizer.param_groups],
            },
            "actor_cached_frames_encoded_total": (
                self.actor_inference.incremental_cache.encoded_frames
                if self.actor_inference.incremental_cache is not None
                else self.actor_inference.cache.encoded_frames
            ),
            "inference_mode": "incremental_frames"
            if self.incremental_history
            else "legacy_windows",
            "rollout": str(rollout),
            "checkpoint": str(self.out / "ppo_latest.pt"),
            "resume_semantics": "continuous_simulator_between_updates_reset_on_process_restart",
        }
        atomic_json(self.out / "summary.json", summary)
        with (self.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(summary) + "\n")
        if self.telemetry:
            self.telemetry.log_summary(summary)
        publication_started = time.perf_counter()
        if getattr(self.args, "backup_dir", None):
            if getattr(self.args, "incremental_backup", False):
                from cr_rl.training.incremental_backup import publish_incremental

                publish_incremental(self.out, self.args.backup_dir, self.update)
            else:
                publish_update(self.out, self.args.backup_dir, self.update)
            for name in ("archived_opponents.jsonl",):
                if (self.out / name).exists():
                    copy_verified(self.out / name, self.args.backup_dir / name)
        published_at = time.perf_counter()
        timing = dict(
            update=self.update,
            learner_samples_total=self.samples,
            valid_learner_samples=valid_samples,
            rollout_seconds=collected_at - started,
            optimization_seconds=optimized_at - collected_at,
            admission_seconds=admission_seconds,
            serialization_seconds=saved_at - optimized_at - admission_seconds,
            prepublication_logging_seconds=publication_started - saved_at,
            backup_seconds=published_at - publication_started,
            update_wall_seconds=published_at - started,
            effective_samples_per_second=valid_samples / (published_at - started),
            incremental_backup=getattr(self.args, "incremental_backup", False),
            incremental_inference=self.incremental_history is not None,
            cache_reference=getattr(self.args, "cache_reference", False),
            defer_scalar_metrics=getattr(self.args, "defer_scalar_metrics", False),
        )
        timing_path = self.out / "timings" / f"update_{self.update:06d}.json"
        atomic_json(timing_path, timing)
        if getattr(self.args, "backup_dir", None):
            copy_verified(timing_path, self.args.backup_dir / "timings" / timing_path.name)
        print(json.dumps(summary), flush=True)
        return summary


def run(args):
    if args.batch_size * args.grad_accumulation < 64:
        raise ValueError("v2 requires an effective PPO optimizer batch of at least 64")
    if args.critic_mode == "shared":
        raise ValueError("v2 uses an independent critic")
    runner = PersistentSelfplay(args)
    try:
        for _ in range(args.updates):
            runner.run_update()
    finally:
        runner.close()
