"""Persistent v2 self-play: shared behavior seats, full-episode value warmup.

Only a process restart resets live simulator episodes. Every update keeps the
same vector, unfinished episode traces and histories; its behavior copy is frozen
until collection ends. Historical opponents stay fixed within each match.
"""

import copy
import gc
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
from cr_rl.opponents.role_sampling import (
    FROZEN_VERSION_BASE,
    RoleMixture,
    frozen_policy_version,
    next_history_snapshot_id,
)
from cr_rl.training.adaptive_kl import restore_actor_lr_controller
from cr_rl.training.advantages import compute_actor_critic_targets
from cr_rl.training.deck_expansion import expand_deck_pool
from cr_rl.training.episode_trace import EpisodeTrace, on_policy_seats
from cr_rl.training.fp16 import configure_optimizer_amp
from cr_rl.training.game_budget import TrainingGameCounter
from cr_rl.training.inference import FrozenInference
from cr_rl.training.initialization import initialize_from_bc
from cr_rl.training.persistence import (
    atomic_json,
    copy_verified,
    file_sha256,
    prune_backup,
    prune_local_opponents,
    publish_update,
)
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.resume_options import override_resume_rates
from cr_rl.training.rewards import RewardConfig, shaped_rewards
from cr_rl.training.sample_selection import actor_critic_rows, normalize_actor_advantages
from cr_rl.training.separate import optimize_separate


class PersistentSelfplay:
    def __init__(self, args):
        from cr_rl.envs.crforge import CrForgeVector

        self.args, self.device = args, args.device
        if getattr(args, "async_publication", False) and (
            not getattr(args, "backup_dir", None) or not getattr(args, "incremental_backup", False)
        ):
            raise ValueError("Async publication requires an incremental backup directory")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.critic_device = getattr(args, "critic_device", None) or self.device
        if torch.device(self.critic_device) != torch.device(self.device) and not (
            self.device.startswith("cuda") and self.critic_device.startswith("cuda")
        ):
            raise ValueError("Concurrent Actor/Critic requires two CUDA devices")
        torch.manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        if not args.resume and (self.out / "ppo_latest.pt").exists():
            raise ValueError("Use a new v2 output directory or --resume")
        self.env = CrForgeVector(
            args.envs,
            getattr(args, "env_threads", 1),
            independent=True,
            level_mixture=getattr(args, "level_mixture", None),
            tower_level_mixture=getattr(args, "tower_level_mixture", None),
            level_jitter_prob=getattr(args, "level_jitter_prob", None),
            level_jitter=getattr(args, "level_jitter", 2),
            level_jitter_seed=args.seed,
        )
        self.env.set_autoreset(False)
        self.level_profile = self.env.level_profile
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
        self.level_profile_migrated = bool(
            args.resume
            and state is not None
            and state.get("level_profile") is None
            and getattr(args, "allow_level_profile_migration", False)
        )
        self.cfg = (
            ModelConfig(**state["config"])
            if state
            else (ModelConfig.preset(args.model).for_v2(self.schema))
        )
        if getattr(args, "frame_encoder_fp32", False):
            self.cfg = replace(self.cfg, frame_encoder_fp32=True)
        if getattr(args, "disable_holding_prior", False):
            if any(getattr(args, name, None) is not None for name in (
                "elixir_wait_bias", "elixir_hold_min_cost", "core_slot_bias",
                "holding_prior_version", "holding_core_cards",
            )):
                raise ValueError("--disable-holding-prior conflicts with holding prior overrides")
            self.cfg = self.cfg.without_holding_prior()
        for name in ("elixir_wait_bias", "elixir_hold_min_cost", "core_slot_bias"):
            value = getattr(args, name, None)
            if value is not None:
                self.cfg = replace(self.cfg, **{name: value})
        prior_version = getattr(args, "holding_prior_version", None)
        core_cards = getattr(args, "holding_core_cards", None)
        if prior_version == 2 or core_cards:
            self.cfg = replace(
                self.cfg,
                holding_prior_version=prior_version or self.cfg.holding_prior_version,
                core_card_ids=tuple(self.env.vector.card_id(name) for name in core_cards)
                if core_cards else self.cfg.core_card_ids,
                tower_card_ids=tuple(self.env.vector.card_id(name)
                                     for name in ("KingTower", "PrincessTower")),
            )
        elif prior_version is not None:
            self.cfg = replace(self.cfg, holding_prior_version=prior_version)
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
        self.critic = HistoryValueNetwork(critic_cfg).to(self.critic_device).eval()
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
            from cr_rl.training.host_memory import detach_optimizer_counters

            detach_optimizer_counters(self.actor_optimizer)
            detach_optimizer_counters(self.critic_optimizer)
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
            saved_level_profile = state.get("level_profile")
            if saved_level_profile is None and (
                getattr(args, "level_mixture", None)
                or getattr(args, "tower_level_mixture", None)
                or getattr(args, "level_jitter_prob", None) is not None
            ) and not getattr(args, "allow_level_profile_migration", False):
                raise ValueError(
                    "This checkpoint predates level profiles; pass --allow-level-profile-migration for an explicit curriculum migration"
                )
            if saved_level_profile is not None and saved_level_profile != self.level_profile:
                raise ValueError("Resume level profile differs from the saved run")
            if saved_level_profile is None and (
                len(set(self.level_profile["card_levels"])) > 1
                or len(set(self.level_profile["tower_levels"])) > 1
            ) and not getattr(args, "allow_level_profile_migration", False):
                raise ValueError(
                    "An older checkpoint has no level profile; start a fresh level-mixture pilot"
                )
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
        configure_optimizer_amp(self.actor_optimizer, self.device,
                                state.get("actor_amp_scaler") if args.resume else None)
        configure_optimizer_amp(self.critic_optimizer, self.critic_device,
                                state.get("critic_amp_scaler") if args.resume else None)
        self.game_counter = TrainingGameCounter(
            state.get("training_game_counter") if args.resume else None,
            origin_update=self.update,
        )
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
        self.deck_pool_transitions = (
            list(state.get("deck_pool_transitions", [])) if args.resume else []
        )
        self.actor_lr_controller = restore_actor_lr_controller(
            self.actor_optimizer,
            state.get("adaptive_actor_lr") if args.resume else None,
            enabled=getattr(args, "adaptive_actor_lr", False),
            min_lr=getattr(args, "adaptive_actor_lr_min", 2.5e-6),
            max_lr=getattr(args, "adaptive_actor_lr_max", None),
        )
        diversity_manifest = getattr(args, "league_diversity_manifest", None)
        if diversity_manifest:
            manifest = json.loads(Path(diversity_manifest).read_text())
            if manifest["evaluated_against_version"] != self.update:
                raise ValueError(
                    "A new expert manifest must be evaluated against the resumed version"
                )
            candidate = Path(manifest.get("candidate_checkpoint", self.out / "actor_export.pt"))
            if not candidate.is_absolute():
                candidate = Path(diversity_manifest).resolve().parent / candidate
            if file_sha256(candidate) != manifest["candidate_sha256"]:
                raise ValueError("Expert evidence differs from the frozen learner export")
            self.pool.configure_diverse_retention(
                diversity_manifest, learner_version=self.update, verify_files=True
            )
        # The resume checkpoint contains CPU copies of the actor, critic and
        # both optimizer states. They are loaded into the live CUDA modules
        # above; retaining the deserialized dict would waste several GiB of
        # host RAM for the entire run.
        del state
        gc.collect()
        expansion_path = getattr(args, "expand_decks", None)
        requested = yaml.safe_load(expansion_path.read_text()) if expansion_path else None
        if args.resume and args.decks:
            supplied = yaml.safe_load(args.decks.read_text())
            if supplied != self.league and supplied != requested:
                raise ValueError("Resume deck pool mismatch")
        if expansion_path:
            if not args.resume:
                raise ValueError("Deck expansion requires a checkpoint resume")
            self.league, transition = expand_deck_pool(self.league, requested)
            transition.update(
                source_generation=self.update,
                source_learner_samples=self.samples,
                requested_file_sha256=file_sha256(expansion_path),
            )
            self.deck_pool_transitions.append(transition)
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
        self.critic_inference = FrozenInference(self.critic, self.adapter, self.critic_device)
        self.actor_inference = FrozenInference(
            self.actor,
            self.adapter,
            self.device,
            forced_wait_fastpath=getattr(args, "forced_wait_fastpath", False),
        )
        self._critic_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="critic-inference")
            if torch.device(self.critic_device) != torch.device(self.device) else None
        )
        self.critic_peer = None
        if getattr(args, "balanced_critic", False):
            from cr_rl.training.peer_critic import PeerCritic
            self.critic_peer = PeerCritic(self.critic, self.device)
        if getattr(args, "compact_host_memory", False):
            from cr_rl.training.host_memory import release_host_memory

            release_host_memory()
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
        self._publisher = None
        self.last_publication = None
        if getattr(args, "async_publication", False):
            from cr_rl.training.async_publication import BoundedAsyncPublisher

            self._publisher = BoundedAsyncPublisher(self.out, Path(args.backup_dir))
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
        if self.device.startswith("cuda"):
            # Parent is idle during this subprocess; return unused reservations
            # so its two frozen policies can allocate their own CUDA context.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        self.emit(phase="candidate_gate", candidate_policy_id=self.update)
        eval_command = [
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
        ]
        # The gate has only two lanes. Pass the realized profile rather than
        # re-sampling jitter with a different lane count/seed.
        eval_command.extend(
            ["--level-mixture", *[str(value) for value in self.level_profile["card_levels"][:2]]]
        )
        eval_command.extend(
            ["--tower-level-mixture", *[str(value) for value in self.level_profile["tower_levels"][:2]]]
        )
        with (evaluation / "evaluation.log").open("w") as log:
            result = subprocess.run(
                eval_command,
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
            self.opponent_cache[entry.policy_id] = FrozenInference(
                model,
                self.adapter,
                self.device,
                forced_wait_fastpath=getattr(self.args, "forced_wait_fastpath", False),
            )
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
    def infer(inference, history, rows=None, masks=None, sample=False, collect_stats=True):
        if not isinstance(history, tuple):
            selected = history if rows is None else history.select(rows)
            method = inference.sample_incremental if sample else inference.evaluate_incremental
            return (
                method(selected, masks, collect_stats=collect_stats)
                if sample
                else method(selected, masks)
            )
        windows, time_mask = history
        if rows is not None:
            windows, time_mask = windows[rows], time_mask[rows]
        method = inference.sample if sample else inference.evaluate
        return (
            method(windows, time_mask, masks, collect_stats=collect_stats)
            if sample
            else method(windows, time_mask, masks)
        )

    def close(self):
        peer = getattr(self, "critic_peer", None)
        if peer:
            peer.close()
        executor = getattr(self, "_critic_executor", None)
        if executor:
            executor.shutdown(wait=True)
        try:
            self.wait_for_publication()
        finally:
            try:
                if self._publisher is not None:
                    self._publisher.close()
            finally:
                self.env.close()
                if self.tracking:
                    self.tracking.finish()

    def wait_for_publication(self):
        if self._publisher is not None:
            return self._publisher.wait()
        return None

    def collect(self):
        args = self.args
        if not 0 <= self.update < FROZEN_VERSION_BASE:
            raise ValueError("Current behavior version overlaps frozen policy namespace")
        snapshots, references, completed = [], [], []
        starts = [int(self.env.vector.info(i)["episode_step"]) for i in range(args.envs)]
        from cr_rl.training.rollout_peer import actor_replica, actor_rows, merge_stats

        dual_rollout = getattr(args, "dual_rollout", False)
        if dual_rollout and self._critic_executor is None:
            raise ValueError("Dual rollout requires separate Actor/Critic CUDA devices")
        peer_actor = (
            actor_replica(self.actor, self.adapter, self.critic_device,
                          getattr(args, "forced_wait_fastpath", False))
            if dual_rollout else None
        )
        self.rollout_parallel = {"enabled": dual_rollout, "primary_actor_rows": 0,
                                 "peer_actor_rows": 0, "env_threads": self.env.vector.threads}
        collected = 0
        began = time.perf_counter()
        done = np.ones(args.envs, dtype=bool)
        following = next_masks = None
        for step in range(args.steps):
            if step % 64 == 0 and self._publisher is not None:
                self._publisher.check()
            compact = getattr(args, "compact_host_memory", False)
            if compact and step and not done.any():
                raw, masks = following, next_masks
            else:
                raw, masks = self.env.observations.copy(), self.env.masks.copy()
            history = self.inference_history(raw, append=True)
            versions = np.full((args.envs, 2), self.update, dtype=np.int32)
            for env, entry in enumerate(self.opponents):
                if entry is not None:
                    # Stored legacy IDs (including BC=0) are not behavior versions.
                    versions[env, 1 - self.learner_seats[env]] = frozen_policy_version(entry)
            learners = on_policy_seats(versions, self.update, self.learner_seats)
            actions = np.zeros((args.envs, 2), dtype=np.int32)
            logp = np.zeros((args.envs, 2), dtype=np.float32)
            actor_indices = np.flatnonzero(learners.reshape(-1))
            own_indices, peer_indices = actor_rows(actor_indices, dual_rollout)
            current_masks = masks.reshape(-1, self.schema.actions)
            collect_stats = step % 64 == 0 or step == args.steps - 1

            def critic_values(
                history=history, peer_indices=peer_indices, current_masks=current_masks,
                collect_stats=collect_stats,
            ):
                context = (torch.cuda.device(self.critic_device)
                           if self._critic_executor else nullcontext())
                with context:
                    peer_result = (
                        self.infer(peer_actor, history, peer_indices,
                                   current_masks[peer_indices], sample=True,
                                   collect_stats=collect_stats)
                        if len(peer_indices) else None
                    )
                    return self.infer(self.critic_inference, history).cpu().numpy(), peer_result

            value_future = (
                self._critic_executor.submit(critic_values) if self._critic_executor else None
            )
            selected, selected_logp, stats = self.infer(
                self.actor_inference,
                history,
                own_indices,
                current_masks[own_indices],
                sample=True,
                collect_stats=collect_stats,
            )
            actions.reshape(-1)[own_indices] = selected
            logp.reshape(-1)[own_indices] = selected_logp
            self.rollout_parallel["primary_actor_rows"] += len(own_indices)
            self.rollout_parallel["peer_actor_rows"] += len(peer_indices)
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
                    collect_stats=False,
                )
                actions.reshape(-1)[indices], logp.reshape(-1)[indices] = selected, selected_logp
            values, peer_result = value_future.result() if value_future else critic_values()
            values = values.reshape(args.envs, 2)
            if peer_result is not None:
                selected, selected_logp, peer_stats = peer_result
                actions.reshape(-1)[peer_indices] = selected
                logp.reshape(-1)[peer_indices] = selected_logp
                stats = merge_stats(stats, len(own_indices), peer_stats, len(peer_indices))
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
                index = trace.append(raw[env], masks[env], training_reward[env], learners[env],
                                     copy=not compact)
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
        if peer_actor is not None:
            self.rollout_parallel["peer_actor_frames_encoded"] = (
                peer_actor.incremental_cache.encoded_frames
                if peer_actor.incremental_cache is not None else peer_actor.cache.encoded_frames
            )
        return snapshots, references, completed, next_values, starts

    def optimize(self, snapshots, refs, completed, next_values):
        from cr_rl.training.rollout_peer import trim_opponent_cache

        args = self.args
        removed = trim_opponent_cache(
            self.opponent_cache, getattr(args, "training_opponent_cache_size", None)
        )
        if hasattr(self, "rollout_parallel"):
            self.rollout_parallel["opponent_models_evicted_for_training"] = removed
            self.rollout_parallel["opponent_models_retained_for_training"] = (
                len(self.opponent_cache)
            )
        actor_enabled = all(self.warmup[key] >= self.warmup_config[key] for key in self.warmup)
        reward = np.stack([row["training_rewards"] for row in snapshots])
        values = np.stack([row["old_values"] for row in snapshots])
        terminated = np.stack([row["terminated"] for row in snapshots])
        ends = np.stack([row["dones"] for row in snapshots])
        shape = (args.steps, args.envs * 2)
        advantage, returns = compute_actor_critic_targets(
            torch.as_tensor(reward.reshape(shape), device=self.device),
            torch.as_tensor(values.reshape(shape), device=self.device),
            torch.as_tensor(
                np.repeat(terminated[..., None], 2, -1).reshape(shape), device=self.device
            ),
            torch.as_tensor(next_values[-1].reshape(-1), device=self.device),
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            critic_gae_lambda=getattr(args, "critic_gae_lambda", None),
            next_values=torch.as_tensor(next_values.reshape(shape), device=self.device),
            episode_ends=torch.as_tensor(
                np.repeat(ends[..., None], 2, -1).reshape(shape), device=self.device
            ),
        )
        eligible = np.stack([row["learner_mask"] & row["valid"] for row in snapshots]).astype(bool)
        actor_indices, critic_indices = actor_critic_rows(
            eligible,
            np.stack([row["masks"] for row in snapshots]),
            choice_only=getattr(args, "actor_choice_only", False),
        )
        if not len(critic_indices):
            raise RuntimeError("No valid on-policy samples in the collected rollout")
        self.samples += len(critic_indices)
        advantage = normalize_actor_advantages(advantage.flatten(), actor_indices)
        targets = returns.flatten()
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
            PreparedHistoryBatches(
                refs, self.cfg.context_len, self.adapter, self.device,
                cuda_budget_gib=(0 if getattr(args, "stream_training_batches", False) else 2),
                deduplicate_frames=getattr(args, "deduplicate_training_frames", False),
            )
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
            critic_device=self.critic_device,
            critic_peer=self.critic_peer,
            batch_size=args.batch_size,
            accumulation=args.grad_accumulation,
            actor_epochs=args.epochs,
            critic_epochs=args.critic_epochs,
            clip_coef=args.clip_coef,
            entropy_coef=args.entropy_coef,
            card_entropy_coef=getattr(args, "card_entropy_coef", None),
            tile_entropy_coef=getattr(args, "tile_entropy_coef", None),
            continue_after_kl_reject=getattr(args, "continue_after_kl_reject", False),
            reference_kl_coef=args.reference_kl_coef,
            target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm,
            actor_enabled=actor_enabled,
            emit=self.emit,
            cache_reference=getattr(args, "cache_reference", False),
            defer_scalar_metrics=getattr(args, "defer_scalar_metrics", False),
            actor_batch_size=getattr(args, "actor_batch_size", None),
            critic_batch_size=getattr(args, "critic_batch_size", None),
            actor_effective_batch_size=getattr(args, "actor_effective_batch_size", None),
            target_card_kl=getattr(args, "target_card_kl", None),
            target_tile_kl=getattr(args, "target_tile_kl", None),
            max_initial_logp_error=getattr(args, "max_initial_logp_error", None),
            actor_lr_controller=self.actor_lr_controller,
        )
        result["actor_training_samples"] = len(actor_indices) if actor_enabled else 0
        result["critic_training_samples"] = len(critic_indices)
        result["metrics"]["input_preparation_seconds"] = preparation_seconds
        result["metrics"]["optimization_seconds"] = time.perf_counter() - optimization_started
        result["metrics"]["prepared_input_gib"] = preparation_metadata["bytes"] / 2**30
        for name in ("frames_requested", "frames_encoded"):
            result["metrics"][name] = preparation_metadata.get(name, 0)
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
            "adaptive_actor_lr": (
                self.actor_lr_controller.state_dict() if self.actor_lr_controller else None
            ),
            "amp_dtype": os.environ.get("CRRL_AMP_DTYPE", "auto"),
            "actor_device": self.device,
            "critic_device": self.critic_device,
            "actor_amp_scaler": self.actor_optimizer.amp_scaler.state_dict(),
            "critic_amp_scaler": self.critic_optimizer.amp_scaler.state_dict(),
            "updates": self.update,
            "learner_samples_total": self.samples,
            "training_game_counter": self.game_counter.state_dict(),
            "min_training_games": getattr(self.args, "min_training_games", None),
            "initialization": self.initialization,
            "critic_initialization": self.critic_initialization,
            "critic_warmup": self.warmup,
            "critic_warmup_config": self.warmup_config,
            "reward_config": self.rewards.metadata(),
            "entropy_config": {
                "joint_coef": self.args.entropy_coef,
                "card_coef": getattr(self.args, "card_entropy_coef", None),
                "tile_coef": getattr(self.args, "tile_entropy_coef", None),
                "tile_card_mass_detached": (
                    getattr(self.args, "card_entropy_coef", None) is not None
                ),
            },
            "gamma": self.args.gamma,
            "gae_lambda": self.args.gae_lambda,
            "critic_gae_lambda": getattr(self.args, "critic_gae_lambda", None),
            "actor_update_config": {
                "choice_only": getattr(self.args, "actor_choice_only", False),
                "target_card_kl": getattr(self.args, "target_card_kl", None),
                "target_tile_kl": getattr(self.args, "target_tile_kl", None),
                "max_initial_logp_error": getattr(self.args, "max_initial_logp_error", None),
            },
            "execution_config": {
                name: bool(getattr(self.args, name, False))
                for name in ("async_publication", "cache_reference", "defer_scalar_metrics",
                             "deduplicate_training_frames", "balanced_critic", "dual_rollout")
            },
            "league_config": self.league,
            "deck_pool_transitions": self.deck_pool_transitions,
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
            "level_profile": self.level_profile,
            "level_profile_migrated": self.level_profile_migrated,
            "resume_semantics": "continuous_simulator_between_updates_reset_on_process_restart",
            "unfinished_episode_steps": [len(trace.observations) for trace in self.traces],
        }
        temporary = self.out / "ppo_latest.tmp.pt"
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Retain immutable local restore points without another multi-GB copy.
        checkpoints = self.out / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        restore = checkpoints / f"update_{self.update:06d}.pt"
        restore_temporary = restore.with_suffix(".pt.tmp")
        restore_temporary.unlink(missing_ok=True)
        os.link(temporary, restore_temporary)
        restore_temporary.replace(restore)
        restore_pool = json.loads((self.out / "opponent_pool.json").read_text())
        for entry in restore_pool["entries"]:
            source = Path(entry["checkpoint"])
            if not source.is_absolute():
                entry["checkpoint"] = str((self.out / source).resolve())
        atomic_json(checkpoints / f"update_{self.update:06d}.pool.json", restore_pool)
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
                    "level_profile",
                )
            },
            export,
        )
        export.replace(self.out / "actor_export.pt")

    def run_update(self):
        started = time.perf_counter()
        if self.device.startswith("cuda"):
            for device in {self.device, self.critic_device}:
                torch.cuda.reset_peak_memory_stats(device)
        snapshots, refs, completed, next_values, starts = self.collect()
        collected_at = time.perf_counter()
        result, actor_enabled, valid_samples = self.optimize(
            snapshots, refs, completed, next_values
        )
        optimized_at = time.perf_counter()
        self.update += 1
        completed_training_games = self.game_counter.record(snapshots)
        admission_started = time.perf_counter()
        admission_seconds = 0.0
        if result["actor_steps"] and self.admission.ready(self.samples):
            snapshot_id = next_history_snapshot_id(self.pool.entries, self.update)
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
            [entry.policy_id for entry in self.opponents if entry is not None],
            learner_version=self.update,
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
                "level_profile": self.level_profile,
                "model_config": self.cfg.__dict__,
                "gamma": self.args.gamma,
                "gae_lambda": self.args.gae_lambda,
                "critic_gae_lambda": getattr(self.args, "critic_gae_lambda", None),
                "actor_update_config": {
                    "choice_only": getattr(self.args, "actor_choice_only", False),
                    "target_card_kl": getattr(self.args, "target_card_kl", None),
                    "target_tile_kl": getattr(self.args, "target_tile_kl", None),
                },
                "training_reward_config": self.rewards.metadata(),
                "entropy_config": {
                    "joint_coef": self.args.entropy_coef,
                    "card_coef": getattr(self.args, "card_entropy_coef", None),
                    "tile_coef": getattr(self.args, "tile_entropy_coef", None),
                    "tile_card_mass_detached": (
                        getattr(self.args, "card_entropy_coef", None) is not None
                    ),
                },
                "execution_config": {
                    name: bool(getattr(self.args, name, False))
                    for name in ("async_publication", "cache_reference", "defer_scalar_metrics",
                                 "deduplicate_training_frames", "balanced_critic", "dual_rollout")
                },
                "behaviour_temperature": 1.0,
            }
        )
        for row in snapshots:
            buffer.append(**{key: value for key, value in row.items() if not key.startswith("_")})
        rollout = buffer.save(self.out / "rollouts" / f"rollout_{self.update:06d}.npz")
        self.checkpoint()
        retention = prune_backup(
            self.out, self.update,
            recent=getattr(self.args, "keep_checkpoints", 2),
            max_periodic=getattr(self.args, "keep_periodic_checkpoints", 2),
        )
        retention.update(prune_local_opponents(
            self.out,
            active=[entry.checkpoint for entry in self.opponents if entry is not None],
        ))
        self.emit(update=self.update, phase="storage_retention", **retention)
        saved_at = time.perf_counter()
        terminal_results = np.concatenate(
            [row["rewards"][row["terminated"].astype(bool)].reshape(-1) for row in snapshots]
        )
        choice_rows = forced_affordable = elixir_five = request_count = core_requests = 0
        request_cost = 0.0
        for row in snapshots:
            eligible = (row["learner_mask"] & row["valid"]).astype(bool)
            choice = row["masks"].sum(-1) > 1
            choice_rows += int((eligible & choice).sum())
            raw = row["obs"]
            available = (
                raw[..., self.schema.header["elixir"][0]]
                - raw[..., self.schema.header["elixir_reserved"][0]]
            )
            costs = raw[..., list(self.schema.header["hand_cost"])]
            cards = raw[..., list(self.schema.header["hand_card"])]
            pending = raw[..., list(self.schema.header["hand_pending"])]
            affordable = (cards >= 0) & (pending == 0) & (costs * 10000 <= available[..., None])
            forced_affordable += int((eligible & ~choice & affordable.any(-1)).sum())
            elixir_five += int((eligible & (available >= 50000)).sum())
            actions = row["actions"]
            play = eligible & (actions > 0) & (actions < 2305)
            slots = ((actions - 1) // 576).clip(0, 3)
            chosen_cost = np.take_along_axis(costs, slots[..., None], -1)[..., 0]
            request_count += int(play.sum())
            request_cost += float(chosen_cost[play].sum())
            core_requests += int((play & (chosen_cost >= 5)).sum())
        summary = {
            "update": self.update,
            "level_profile": self.level_profile,
            "level_profile_migrated": self.level_profile_migrated,
            "learner_samples_total": self.samples,
            "valid_learner_samples": valid_samples,
            "actor_enabled": actor_enabled,
            "actor_training_samples": result["actor_training_samples"],
            "critic_training_samples": result["critic_training_samples"],
            "actor_choice_only": getattr(self.args, "actor_choice_only", False),
            "completed_training_games": completed_training_games,
            "completed_training_games_total": self.game_counter.count,
            "training_game_counter_origin_update": self.game_counter.origin_update,
            "decision_coverage": {
                "choice_rows": choice_rows,
                "forced_wait_rows": valid_samples - choice_rows,
                "forced_wait_with_affordable_unqueued_card": forced_affordable,
                "available_elixir_at_least_five_rows": elixir_five,
                "card_requests": request_count,
                "requested_card_cost_mean": request_cost / request_count if request_count else None,
                "at_least_five_cost_requests": core_requests,
                "scope": "On-policy valid requests; native replay confirms successful placements",
            },
            "optimizer_steps": result["actor_steps"],
            "critic_optimizer_steps": result["critic_steps"],
            "actor_kl_stopped": result["actor_kl_stopped"],
            "critic_warmup": self.warmup.copy(),
            "metrics": result["metrics"],
            "critic_parallel": result.get("critic_parallel", {}),
            "rollout_parallel": getattr(self, "rollout_parallel", {}),
            "sample_utilization": result["sample_utilization"],
            "actor_lr_schedule": result.get("actor_lr_schedule", {"enabled": False}),
            "rejected_policy_metrics": result["rejected_policy_metrics"],
            "reference_cache": result.get("reference_cache", {}),
            "optimization_batches": {
                "effective": self.args.batch_size * self.args.grad_accumulation,
                "actor_effective": getattr(self.args, "actor_effective_batch_size", None)
                or self.args.batch_size * self.args.grad_accumulation,
                "critic_effective": self.args.batch_size * self.args.grad_accumulation,
                "actor_microbatch": getattr(self.args, "actor_batch_size", None)
                or self.args.batch_size,
                "critic_microbatch": getattr(self.args, "critic_batch_size", None)
                or self.args.batch_size,
            },
            "seconds": time.perf_counter() - started,
            "devices": {"actor": self.device, "critic": self.critic_device},
            "amp_dtype": os.environ.get("CRRL_AMP_DTYPE", "auto"),
            "peak_cuda_per_device_gib": {
                str(device): torch.cuda.max_memory_allocated(device) / 2**30
                for device in {self.device, self.critic_device}
            } if self.device.startswith("cuda") else {},
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
            "opponent_retention": self.pool.diversity_metadata(self.update),
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
            "forced_wait_fastpath": self.actor_inference.forced_wait_fastpath,
            "actor_forced_wait_rows_total": self.actor_inference.forced_wait_rows,
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
        timing_base = dict(
            update=self.update,
            learner_samples_total=self.samples,
            valid_learner_samples=valid_samples,
            rollout_seconds=collected_at - started,
            optimization_seconds=optimized_at - collected_at,
            admission_seconds=admission_seconds,
            serialization_seconds=saved_at - optimized_at - admission_seconds,
            prepublication_logging_seconds=publication_started - saved_at,
            incremental_backup=getattr(self.args, "incremental_backup", False),
            incremental_inference=self.incremental_history is not None,
            cache_reference=getattr(self.args, "cache_reference", False),
            defer_scalar_metrics=getattr(self.args, "defer_scalar_metrics", False),
            deduplicate_training_frames=getattr(self.args, "deduplicate_training_frames", False),
            async_publication=self._publisher is not None,
        )
        # Freeze generation values; the next collection may run while this callback publishes.
        timing_path = self.out / "timings" / f"update_{self.update:06d}.json"
        backup_dir = Path(self.args.backup_dir) if getattr(self.args, "backup_dir", None) else None
        publication_callback = getattr(self, "_publication_callback", None)

        def finish_timing(manifest, backup_seconds, snapshot_seconds=0.0, backpressure_seconds=0.0):
            elapsed = time.perf_counter() - started
            foreground = publication_started - started + snapshot_seconds + backpressure_seconds
            timing = dict(
                timing_base, backup_seconds=backup_seconds, update_wall_seconds=elapsed,
                effective_samples_per_second=valid_samples / elapsed,
                foreground_update_seconds=foreground,
                snapshot_seconds=snapshot_seconds,
                publication_backpressure_seconds=backpressure_seconds,
                timing_scope="Publication latency overlaps next update when async_publication=true",
            )
            atomic_json(timing_path, timing)
            if backup_dir is not None:
                copy_verified(timing_path, backup_dir / "timings" / timing_path.name)
            if publication_callback is not None:
                publication_callback(manifest)

        publication_interval = max(1, int(getattr(self.args, "publication_interval", 1)))
        should_publish = (
            self.update % publication_interval == 0
            or self.update >= int(getattr(self.args, "updates", self.update))
        )
        if self._publisher is not None and should_publish:
            self.last_publication = self._publisher.submit(self.update, after_publish=finish_timing)
        elif self._publisher is not None:
            self._publisher.check()
            finish_timing(None, 0.0)
        else:
            if backup_dir is not None and should_publish:
                if getattr(self.args, "incremental_backup", False):
                    from cr_rl.training.incremental_backup import publish_incremental

                    publish_incremental(self.out, backup_dir, self.update)
                else:
                    publish_update(self.out, backup_dir, self.update)
                for name in ("archived_opponents.jsonl",):
                    if (self.out / name).exists():
                        copy_verified(self.out / name, backup_dir / name)
            finish_timing(None, time.perf_counter() - publication_started)
        print(json.dumps(summary), flush=True)
        return summary


def run(args):
    effective = args.batch_size * args.grad_accumulation
    actor_effective = getattr(args, "actor_effective_batch_size", None)
    if min(effective, effective if actor_effective is None else actor_effective) < 64:
        raise ValueError("v2 requires Actor and Critic effective batches of at least 64")
    if args.critic_mode == "shared":
        raise ValueError("v2 uses an independent critic")
    runner = PersistentSelfplay(args)
    try:
        for _ in range(args.updates):
            target = getattr(args, "min_training_games", None)
            if target is not None and runner.game_counter.count >= target:
                break
            runner.run_update()
            if getattr(args, "compact_host_memory", False):
                from cr_rl.training.host_memory import release_host_memory

                for trace in runner.traces:
                    trace.detach_views()
                release_host_memory()
    finally:
        runner.close()
