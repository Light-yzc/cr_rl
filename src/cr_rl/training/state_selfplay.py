"""Architecture-v3 recurrent self-play PPO driver.

This driver owns the v3 state memory from simulator reset through PPO update.
It intentionally keeps both seats on the current policy for the first v3 path;
historical opponents can be added later as the same StateActorCritic replica.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.advantages import compute_gae
from cr_rl.training.rewards import RewardConfig, shaped_rewards
from cr_rl.training.state_ppo import StatePPOConfig, state_ppo_update


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
    return advantage[:, 0].numpy(), returns[:, 0].numpy()


class StateSelfplay:
    def __init__(self, args):
        from cr_rl.envs.crforge import CrForgeVector

        self.args = args
        if not args.init_from_bc and not args.resume:
            raise ValueError("v3 PPO requires --init-from-bc or --resume")
        source = args.resume or args.init_from_bc
        checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
        if checkpoint.get("model_architecture_version") != 3:
            raise ValueError("State PPO requires a v3 checkpoint")
        if checkpoint.get("decision_ticks") != 8:
            raise ValueError("State PPO requires the 0.4s / 8-tick decision grid")
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.model = state_model_from_checkpoint(checkpoint, self.device)
        self.model.eval()
        print(json.dumps({"phase": "v3_model_loaded", "device": str(self.device), "decision_ticks": 8}), flush=True)
        self.model.frame_microbatch = getattr(args, "frame_microbatch", 32)
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
        self.rng = np.random.default_rng(args.seed)
        self.env = CrForgeVector(args.envs, getattr(args, "env_threads", 1), independent=True,
                                 arena="Arena_L", decision_ticks=8)
        print(json.dumps({"phase": "v3_env_created", "envs": args.envs}), flush=True)
        self.env.set_autoreset(False)
        self.schema = self.env.vector.schema
        self.adapter = CrForgeObservationAdapter(self.schema, self.env.vector.vocabulary_size, entity_bucket=16)
        deck_path = Path(args.decks or "configs/decks.yaml")
        self.league = yaml.safe_load(deck_path.read_text(encoding="utf-8"))
        self.deck_names = list(self.league["decks"])
        self.decks = [self.league["decks"][name] for name in self.deck_names]
        self.forms = [self.league.get("slots", {}).get(name, [0] * 8) for name in self.deck_names]
        self.rewards = RewardConfig(enabled=args.tower_shaping, princess=args.princess_reward,
                                    king=args.king_reward, crown=args.crown_reward,
                                    overflow=args.overflow_penalty)
        self.tracking = None
        if getattr(args, "wandb", False):
            import wandb
            self.tracking = wandb.init(project="cr-rl-state-ppo-v3", config={
                "architecture": 3, "decision_ticks": 8, "precision": args.precision,
                "envs": args.envs, "steps": args.steps,
            }, resume="allow")
        self.states = None
        self.reset_all()

    def reset_all(self):
        seeds = [int(self.rng.integers(2**31 - 1)) for _ in range(self.args.envs)] if hasattr(self, "args") else [int(i) for i in range(self.env.count)]
        d0 = self.rng.integers(len(self.decks), size=self.env.count)
        d1 = self.rng.integers(len(self.decks), size=self.env.count)
        self.env.reset(seeds, [self.decks[i] for i in d0], [self.decks[i] for i in d1],
                       [self.forms[i] for i in d0], [self.forms[i] for i in d1])
        self.states = self.model.initial_state(self.env.count * 2, device=self.device)

    def _reset_lane(self, index):
        seed = int(self.rng.integers(2**31 - 1))
        d0, d1 = self.rng.integers(len(self.decks), size=2)
        self.env.vector.reset_env(index, seed, self.decks[d0], self.decks[d1], self.forms[d0], self.forms[d1])

    def collect(self):
        self.model.eval()
        print(json.dumps({"phase": "v3_rollout_start", "steps": self.args.steps}), flush=True)
        lane_segments = [[[]] for _ in range(self.env.count * 2)]
        active = np.ones(self.env.count, dtype=bool)
        for _ in range(self.args.steps):
            raw = self.env.observations.copy()
            masks = self.env.masks.copy()
            header, entities, entity_mask = self.adapter(raw.reshape(-1, self.schema.seat_ints))
            legal = torch.as_tensor(masks.reshape(-1, self.schema.actions), device=self.device).bool()
            hand_index = np.asarray(self.schema.header["hand_card"])
            hand = torch.as_tensor(raw.reshape(-1, self.schema.seat_ints)[:, hand_index], device=self.device)
            with torch.inference_mode():
                logp, values, self.states = self.model.step(
                    torch.as_tensor(header, device=self.device),
                    torch.as_tensor(entities, device=self.device),
                    torch.as_tensor(entity_mask, device=self.device), legal_action_mask=legal,
                    state=self.states, hand_card_ids=hand, compute_value=True,
                )
                probs = logp.float().exp()
                actions = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy().reshape(self.env.count, 2)
                old_logp = logp.gather(-1, torch.as_tensor(actions.reshape(-1), device=self.device)[:, None]).squeeze(-1).cpu().numpy().reshape(self.env.count, 2)
                old_values = values.cpu().numpy().reshape(self.env.count, 2)
            print(json.dumps({"phase": "v3_rollout_step", "actions": int(actions.size)}), flush=True)
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
                for seat in (0, 1):
                    lane = env * 2 + seat
                    lane_segments[lane][-1].append({"raw": raw[env, seat].copy(), "mask": masks[env, seat].copy(),
                                        "action": int(actions[env, seat]), "old_logp": float(old_logp[env, seat]),
                                        "value": float(old_values[env, seat]), "reward": float(shaped[env, seat]),
                                        "ended": bool(done[env])})
            if done.any():
                reset_mask = np.repeat(done, 2)
                self.states = type(self.states)(self.states.temporal.reset(torch.as_tensor(reset_mask, device=self.device)), id(self.model), self.model.memory_generation)
                for env in np.flatnonzero(done):
                    self._reset_lane(int(env))
                    lane_segments[int(env) * 2].append([])
                    lane_segments[int(env) * 2 + 1].append([])
                    active[env] = False
        # Bootstrap only unfinished segments from the post-transition state.
        bootstrap_values = np.zeros(self.env.count * 2, dtype=np.float32)
        raw = self.env.observations.copy()
        header, entities, entity_mask = self.adapter(raw.reshape(-1, self.schema.seat_ints))
        hand_index = np.asarray(self.schema.header["hand_card"])
        with torch.inference_mode():
            _, final_values, self.states = self.model.step(
                torch.as_tensor(header, device=self.device),
                torch.as_tensor(entities, device=self.device),
                torch.as_tensor(entity_mask, device=self.device),
                legal_action_mask=torch.as_tensor(self.env.masks.reshape(-1, self.schema.actions), device=self.device).bool(),
                state=self.states,
                hand_card_ids=torch.as_tensor(raw.reshape(-1, self.schema.seat_ints)[:, hand_index], device=self.device),
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
                                  "legal_action_mask": np.stack([x["mask"] for x in lane]),
                                  "hand_card_ids": raw_rows[:, np.asarray(self.schema.header["hand_card"])],
                                  "actions": np.asarray([x["action"] for x in lane], dtype=np.int64),
                                  "old_logp": np.asarray([x["old_logp"] for x in lane], dtype=np.float32),
                                  "advantages": advantages, "returns": returns})
        return sequences

    def run(self):
        for update in range(self.update, self.args.updates):
            sequences = self.collect()
            metrics = {}
            for _ in range(self.args.epochs):
                metrics = state_ppo_update(
                    self.model, self.optimizer, sequences,
                    config=StatePPOConfig(clip_coef=self.args.clip_coef, value_coef=self.args.value_coef,
                                          entropy_coef=self.args.entropy_coef, max_grad_norm=self.args.max_grad_norm,
                                          precision=self.args.precision,
                                          max_initial_logp_error=self.args.max_initial_logp_error or 0.05),
                    scaler=self.scaler if self.args.precision == "fp16" else None)
            # PPO changed weights; v3 memory generations are intentionally invalidated.
            # Re-seed memory at the next live observation rather than reusing stale KV state.
            self.states = self.model.initial_state(self.env.count * 2, device=self.device)
            self.update = update + 1
            checkpoint = Path(self.args.out) / "ppo_latest.pt"
            _save(checkpoint, self.model, self.optimizer, self.scaler, self.update, self.args)
            if self.args.backup_dir:
                destination = Path(self.args.backup_dir) / "ppo_latest.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                shutil.copy2(checkpoint, temporary)
                temporary.replace(destination)
            event = {"phase": "state_ppo", "update": self.update,
                     "sequences": len(sequences), "samples": sum(len(s["actions"]) for s in sequences), **metrics}
            if self.args.progress_jsonl:
                Path(self.args.progress_jsonl).parent.mkdir(parents=True, exist_ok=True)
                with Path(self.args.progress_jsonl).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
            if self.tracking:
                self.tracking.log({f"ppo/{key}": value for key, value in metrics.items()}, step=self.update)
            print(json.dumps(event), flush=True)
        self.env.close()


def run(args):
    trainer = StateSelfplay(args)
    trainer.args = args
    try:
        trainer.run()
    finally:
        trainer.env.close()
        if trainer.tracking:
            trainer.tracking.finish()
