"""Versioned, episode-state actor and critic using KDA and full-history grouped attention."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .policy import ActorCritic, HistoryEncoder, ModelConfig, spatial_features
from .state_temporal import GQAState, KDAState, StateTemporalConfig, StateTemporalCore, TemporalState


@dataclass
class PolicyMemory:
    temporal: object
    owner: int
    generation: int


class _RecurrentStateMixin:
    """Shared explicit recurrent memory management and chunked sequence encoding."""

    architecture_version = 3

    def _init_recurrent(self, temporal_cfg: StateTemporalConfig, *, decision_ticks: int, frame_microbatch: int) -> None:
        self.temporal_cfg = temporal_cfg
        self.decision_ticks = int(decision_ticks)
        self.frame_microbatch = int(frame_microbatch)
        self.memory_generation = 0

    def checkpoint_config(self):
        definition = {
            "model_architecture_version": self.architecture_version,
            "config": asdict(self.cfg),
            "temporal_config": asdict(self.temporal_cfg),
            "decision_ticks": self.decision_ticks,
        }
        if hasattr(self, "learned_placement"):
            definition["learned_placement_definition"] = self.learned_placement.get_extra_state()
        return definition

    def initial_state(self, batch_size=1, device=None, dtype=None):
        reference = self.to_temporal.weight
        device = torch.device(device or reference.device)
        if dtype is None:
            dtype = (
                torch.get_autocast_dtype(device.type)
                if torch.is_autocast_enabled(device.type)
                else reference.dtype
            )
        temporal = self.temporal.init_state(batch_size, device, dtype)
        return PolicyMemory(temporal, id(self), self.memory_generation)

    def detach_state(self, state):
        if state is None:
            return None
        self._check_memory(state)
        temporal = state.temporal.detach()
        return PolicyMemory(temporal, id(self), self.memory_generation)

    def mark_weights_updated(self):
        """Call after every optimizer update or weight reload; old caches expire."""
        self.memory_generation += 1

    def _check_memory(self, state):
        if state.owner != id(self) or state.generation != self.memory_generation:
            raise ValueError("Memory belongs to a different policy or stale weight version")

    def forward(self, *args, **kwargs):
        raise RuntimeError("Architecture-v3 requires forward_state()/step() with explicit episode memory")

    def _activation_forward(self, function, *args, **kwargs):
        if self.training and torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
            return activation_checkpoint(function, *args, use_reentrant=False, **kwargs)
        return function(*args, **kwargs)

    def _encode_state_sequence(self, header, entities, entity_mask, *, state=None, time_mask=None):
        single = header.ndim == 2
        if single:
            header = header[None]
            entities = entities[None]
            entity_mask = entity_mask[None]
            time_mask = None if time_mask is None else time_mask[None]
        if header.ndim != 3 or entities.ndim != 4:
            raise ValueError("Expected chronological [T,...] or [B,T,...] frames")
        batch, steps, count, features = entities.shape
        if state is None:
            state = self.initial_state(batch, entities.device)
        self._check_memory(state)
        flat_h = header.reshape(batch * steps, -1)
        flat_e = entities.reshape(batch * steps, count, features)
        flat_m = entity_mask.reshape(batch * steps, count)
        real_frames = None if time_mask is None else time_mask.reshape(-1).bool()
        encoder_mask = flat_m
        if real_frames is not None:
            # Eval attention may emit NaN for a completely masked row. Give
            # only synthetic padding a dummy entity, then zero its outputs.
            # NaN * 0 would otherwise poison a real prefix in the KV cache.
            first_entity = torch.arange(count, device=flat_m.device) == 0
            encoder_mask = flat_m.bool() | (~real_frames[:, None] & first_entity[None])
        summaries, tokens = [], []
        for start in range(0, len(flat_h), self.frame_microbatch):
            summary, token = self._activation_forward(self.encode_frames,
                flat_h[start : start + self.frame_microbatch],
                flat_e[start : start + self.frame_microbatch],
                encoder_mask[start : start + self.frame_microbatch],
            )
            if real_frames is not None:
                padding = ~real_frames[start : start + self.frame_microbatch]
                summary = summary.masked_fill(padding[:, None], 0)
                token = token.masked_fill(padding[:, None, None], 0)
            summaries.append(summary)
            tokens.append(token)
        summary = torch.cat(summaries).reshape(batch, steps, -1)
        token = torch.cat(tokens)
        self.temporal.gradient_checkpointing = bool(getattr(self, "gradient_checkpointing", False))
        self.temporal.checkpoint_kda_chunks = getattr(self, "temporal_checkpoint_mode", "nested") == "nested"
        temporal, next_temporal = self.temporal.prefill(
            self.to_temporal(summary), state=state.temporal, time_mask=time_mask
        )
        context = temporal.reshape(batch * steps, -1)
        memory = PolicyMemory(next_temporal, id(self), self.memory_generation)
        return context, token, flat_e, flat_m, memory, batch, steps, single, real_frames


class StateActorCritic(_RecurrentStateMixin, ActorCritic):
    """Reuse the entity/placement representation with a new causal temporal core.

    An explicit state belongs to exactly one frozen weight version and stream
    batch. All real frames, including forced WAITs, must advance this state.
    """

    architecture_version = 3

    def __init__(self, cfg: ModelConfig, temporal_cfg: StateTemporalConfig, *,
                 decision_ticks=8, frame_microbatch=8):
        if decision_ticks < 1 or frame_microbatch < 1:
            raise ValueError("Positive decision interval and frame batch required")
        # Both the isolated BC source tree and current production tree are
        # supported; only the latter has tactical-prior config fields.
        if hasattr(cfg,"without_holding_prior"):
            cfg = cfg.without_holding_prior()
        cfg = replace(cfg, temporal_dim=temporal_cfg.width,
                      temporal_heads=temporal_cfg.heads,
                      temporal_layers=len(temporal_cfg.pattern))
        super().__init__(cfg, temporal_factory=lambda: StateTemporalCore(temporal_cfg))
        self._init_recurrent(temporal_cfg, decision_ticks=decision_ticks, frame_microbatch=frame_microbatch)
        self.sparse_placement = os.environ.get("CRRL_SPARSE_PLACEMENT") == "1"

    def forward_state(self, header, entities, entity_mask, legal_action_mask=None, **kwargs):
        # PPO compares streaming and whole-episode forwards. Volta FP16 kernels
        # round these differently as batch/time shapes change; keep the learner
        # distribution consistent without changing critic or opponent precision.
        if getattr(self, "ppo_actor_fp32", False):
            with torch.autocast(header.device.type, enabled=False):
                return self._forward_state(header, entities, entity_mask, legal_action_mask, **kwargs)
        return self._forward_state(header, entities, entity_mask, legal_action_mask, **kwargs)

    def _forward_state(self, header, entities, entity_mask, legal_action_mask=None, *,
                      state=None, hand_card_ids=None, deck_card_ids=None,
                      enemy_seen_card_ids=None, time_mask=None, compute_value=True,
                      placement_mask=None, placement_slots=None):
        context, token, flat_e, flat_m, memory, batch, steps, single, real_frames = (
            self._encode_state_sequence(
                header, entities, entity_mask, state=state, time_mask=time_mask
            )
        )
        if single:
            legal_action_mask = None if legal_action_mask is None else legal_action_mask[None]
            hand_card_ids = None if hand_card_ids is None else hand_card_ids[None]
            placement_mask = None if placement_mask is None else placement_mask[None]
            placement_slots = None if placement_slots is None else placement_slots[None]
        if hand_card_ids is None:
            hand_card_ids = torch.full((batch, steps, 4), -1, dtype=torch.long,
                                       device=header.device)
        hands = hand_card_ids.reshape(batch * steps, 4)
        card_features = self.card_embedding((hands.long()+1).clamp(0,self.cfg.vocab_size))
        card_features = card_features + self.card_slot_embedding
        context_slots = context[:, None].expand(-1,4,-1)
        slots = self.slot_actor(torch.cat((context_slots,card_features),-1)).squeeze(-1)
        wait = self.wait_actor(context) + self.wait_hand_residual(header.reshape(batch*steps, -1), slots, hands)
        cards = torch.cat((wait,slots,self.ability_actor(context)),-1)
        legal = (torch.ones(batch*steps,self.cfg.action_count,dtype=torch.bool,device=header.device)
                 if legal_action_mask is None else legal_action_mask.reshape(batch*steps,-1).bool())
        # A WAIT/ability BC target has no placement loss. Computing its 2304
        # conditional locations wastes most of continuous-trajectory training.
        # The card mass is exactly independent of these normalized tile logits.
        if placement_mask is not None and not self.training:
            raise ValueError("Selective placement is a supervised-training optimization only")
        if placement_slots is not None and (not self.training or placement_mask is None or not hasattr(self, 'learned_placement')):
            raise ValueError('GT placement slots require learned-placement supervised training')
        legal_slots = None
        if placement_mask is not None:
            selected = placement_mask.reshape(-1).bool().nonzero().flatten()
            if hasattr(self, "learned_placement") and real_frames is not None:
                selected = selected[real_frames[selected]]
        elif legal_action_mask is not None:
            # Forced-WAIT / ability-only rows mask out all 2304 tile logits in
            # _joint_log_probs(), so skipping placement preserves exact logits and grads.
            tile_legal = legal[:, 1:2305]
            if self.sparse_placement and hasattr(self, "learned_placement"):
                legal_slots = tile_legal.reshape(len(legal), 4, 576).any(-1)
                has_tile = legal_slots.any(-1)
            else:
                has_tile = tile_legal.any(-1)
            if real_frames is not None:
                has_tile = has_tile & real_frames
            selected = has_tile.nonzero().flatten()
        else:
            selected = torch.arange(len(context), device=context.device)
            if hasattr(self, "learned_placement") and real_frames is not None:
                selected = selected[real_frames[selected]]
        tiles = context.new_zeros(len(context),4,576,dtype=torch.float32)
        for start in range(0, len(selected), self.frame_microbatch):
            rows = selected[start:start+self.frame_microbatch]
            if placement_slots is not None:
                chosen_slots = placement_slots.reshape(-1)[rows].long()
                logits = self.learned_placement.forward_gt(context[rows], token[rows], flat_m[rows],
                    card_features[rows], chosen_slots, entity_xy=flat_e[rows, :, 4:6], card_ids=hands[rows])
                partial = logits.new_zeros(len(rows), 4, 576).scatter(1,
                    chosen_slots[:, None, None].expand(-1, 1, 576), logits[:, None])
                tiles = tiles.index_copy(0, rows, partial)
                continue
            elif hasattr(self, "learned_placement"):
                placement_kwargs = {} if legal_slots is None else {"legal_slots": legal_slots[rows]}
                logits = self._activation_forward(self.learned_placement, context[rows], token[rows], flat_m[rows], card_features[rows],
                    entity_xy=flat_e[rows, :, 4:6], card_ids=hands[rows], **placement_kwargs)
            else:
                logits = self._activation_forward(self.placement_logits,context[rows],token[rows],flat_m[rows],card_features[rows])
            tiles = tiles.index_copy(0,rows,logits.float())
        logp = self._joint_log_probs(cards,tiles,legal, hand_present_mask=(hands >= 0) if self.cfg.reserve_to_wait else None).reshape(batch,steps,-1)
        value = (self.critic(context).squeeze(-1).reshape(batch,steps) if compute_value
                 else context.new_zeros(batch,steps))
        return (logp[0],value[0],memory) if single else (logp,value,memory)

    def step(self, header, entities, entity_mask, legal_action_mask=None, *, state=None,
             hand_card_ids=None, compute_value=False):
        output,value,state = self.forward_state(header[:,None],entities[:,None],
            entity_mask[:,None],None if legal_action_mask is None else legal_action_mask[:,None],
            state=state,hand_card_ids=None if hand_card_ids is None else hand_card_ids[:,None],
            compute_value=compute_value)
        return output[:,0],value[:,0],state

    def migrate_bc_representation(self, source):
        """Copy matching entity/action tensors, never old temporal weights/moments."""
        target = self.state_dict()
        compatible = {name:value for name,value in source.items()
                      if not name.startswith(("temporal.","position"))
                      and name in target and target[name].shape == value.shape}
        self.load_state_dict(compatible,strict=False)
        self.mark_weights_updated()
        return {"copied_tensors":len(compatible),
                "copied_parameters":sum(value.numel() for value in compatible.values()),
                "new_temporal_core":True,"initial_policy_equivalent":False,
                "optimizer_migrated":False}


class StateValueNetwork(_RecurrentStateMixin, HistoryEncoder):
    """Independent architecture-v3 recurrent critic; shares no parameters with Actor."""

    architecture_version = 3

    def __init__(self, cfg: ModelConfig, temporal_cfg: StateTemporalConfig, *,
                 decision_ticks=8, frame_microbatch=8):
        if decision_ticks < 1 or frame_microbatch < 1:
            raise ValueError("Positive decision interval and frame batch required")
        if hasattr(cfg, "without_holding_prior"):
            cfg = cfg.without_holding_prior()
        cfg = replace(
            cfg,
            temporal_dim=temporal_cfg.width,
            temporal_heads=temporal_cfg.heads,
            temporal_layers=len(temporal_cfg.pattern),
        )
        super().__init__(cfg, temporal_factory=lambda: StateTemporalCore(temporal_cfg))
        self._init_recurrent(temporal_cfg, decision_ticks=decision_ticks, frame_microbatch=frame_microbatch)
        self.critic = nn.Sequential(
            nn.Linear(cfg.temporal_dim, cfg.temporal_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.temporal_dim // 2, 1),
        )

    def initialize_encoder(self, actor: StateActorCritic) -> dict:
        source = actor.state_dict()
        own = self.state_dict()
        copied = {
            k: source[k]
            for k in own
            if not k.startswith("critic.") and k in source and source[k].shape == own[k].shape
        }
        self.load_state_dict(copied, strict=False)
        self.mark_weights_updated()
        return {
            "encoder_tensors_copied": len(copied),
            "encoder_parameters_copied": sum(v.numel() for v in copied.values()),
            "value_head_initialized_from_bc": False,
        }

    def forward_state(self, header, entities, entity_mask, *, state=None, time_mask=None):
        context, _, _, _, memory, batch, steps, single, _ = self._encode_state_sequence(
            header, entities, entity_mask, state=state, time_mask=time_mask
        )
        value = self.critic(context).squeeze(-1).float().reshape(batch, steps)
        return (value[0], memory) if single else (value, memory)

    def step(self, header, entities, entity_mask, *, state=None):
        value, state = self.forward_state(
            header[:, None], entities[:, None], entity_mask[:, None], state=state
        )
        return value[:, 0], state


def state_model_from_checkpoint(checkpoint, device="cpu"):
    if checkpoint.get("model_architecture_version") != 3:
        raise ValueError("Expected architecture-v3 episode-state checkpoint")
    cfg = ModelConfig(**checkpoint["config"])
    temporal_cfg = StateTemporalConfig(**checkpoint["temporal_config"])
    with torch.device("meta"):
        model = StateActorCritic(cfg,temporal_cfg,
                                decision_ticks=checkpoint.get("decision_ticks",8))
    model.to_empty(device=device)
    grid = torch.arange(576,device=device)
    xy = torch.stack(((grid%18+.5)/18,(grid//18+.5)/32),-1)
    model.tile_grid_features = spatial_features(xy,cfg.entity_dim)
    if hasattr(model.frame,"flag_bits"):
        model.frame.flag_bits = torch.arange(16,device=device)
    placement_definition = checkpoint.get("learned_placement_definition")
    if placement_definition is not None:
        from .learned_placement import PlacementAdapter
        # Versioned construction happens before strict restoration. Adapter
        # parameters are small; the 183M trunk remains meta -> target-device.
        model.add_module("learned_placement", PlacementAdapter(model, placement_definition).to(device))
        model.learned_placement_definition = dict(placement_definition)
    state_dict = checkpoint.get("actor") if "model" not in checkpoint and "actor" in checkpoint else checkpoint["model"]
    # Migrate only the additive, zero-output gate; all other tensors remain strict.
    state_dict = dict(state_dict)
    if not any(k.startswith("wait_hand_gate.") for k in state_dict):
        model.wait_hand_gate[0].reset_parameters()
        nn.init.zeros_(model.wait_hand_gate[-1].weight)
        nn.init.zeros_(model.wait_hand_gate[-1].bias)
        state_dict.update({k: v for k, v in model.state_dict().items() if k.startswith("wait_hand_gate.")})
    model.load_state_dict(state_dict, strict=True)
    return model


def state_value_model_from_checkpoint(checkpoint, device="cpu"):
    if checkpoint.get("model_architecture_version") != 3:
        raise ValueError("Expected architecture-v3 episode-state checkpoint")
    cfg = ModelConfig(**checkpoint["critic_config"])
    temporal_cfg = StateTemporalConfig(**checkpoint.get("critic_temporal_config", checkpoint["temporal_config"]))
    with torch.device("meta"):
        critic = StateValueNetwork(
            cfg,
            temporal_cfg,
            decision_ticks=checkpoint.get("decision_ticks", 8),
        )
    critic.to_empty(device=device)
    if hasattr(critic.frame, "flag_bits"):
        critic.frame.flag_bits = torch.arange(16, device=device)
    critic.load_state_dict(checkpoint["critic"], strict=True)
    return critic


def clone_memory(memory: PolicyMemory | None, *, device=None, detach: bool = True, trim_gqa: bool = False) -> PolicyMemory | None:
    """Copy a PolicyMemory (optionally moving device and trimming unused GQA slack)."""
    if memory is None:
        return None
    temporal = memory.temporal
    pos = temporal.positions
    if detach:
        pos = pos.detach()
    pos_out = pos.to(device=device).clone() if device is not None else pos.clone()
    pos_max = int(pos.max().item()) if pos.numel() else 0
    used = max(1, pos_max) if trim_gqa and pos.numel() else None
    layers = []
    for layer in temporal.layers:
        if isinstance(layer, KDAState):
            mat = layer.matrix.detach() if detach else layer.matrix
            mat = mat.to(device=device).clone() if device is not None else mat.clone()
            layers.append(KDAState(matrix=mat))
        else:
            key, val = layer.key, layer.value
            if used is not None and key.shape[2] > used:
                key = key[:, :, :used]
                val = val[:, :, :used]
            if detach:
                key, val = key.detach(), val.detach()
            key = key.to(device=device).clone() if device is not None else key.clone()
            val = val.to(device=device).clone() if device is not None else val.clone()
            layers.append(GQAState(key=key, value=val))
    bound = min(temporal.length_bound, pos_max) if trim_gqa and pos.numel() else max(temporal.length_bound, pos_max if trim_gqa else 0)
    if not trim_gqa:
        bound = temporal.length_bound
    return PolicyMemory(
        TemporalState(tuple(layers), pos_out, bound),
        memory.owner,
        memory.generation,
    )


def slice_memory(memory: PolicyMemory, indices, *, device=None, detach: bool = True, trim_gqa: bool = False, position_bound: int | None = None) -> PolicyMemory:
    """Gather a batch subset from a PolicyMemory along axis 0."""
    temporal = memory.temporal
    idx = torch.as_tensor(indices, dtype=torch.long, device=temporal.positions.device)
    pos = temporal.positions[idx]
    if detach:
        pos = pos.detach()
    # Untrimmed slices already have a safe CPU bound. For trimmed per-lane
    # slices callers can transfer all positions once, rather than synchronizing
    # CUDA once per lane on every physical decision.
    pos_max = (int(position_bound) if position_bound is not None else
               int(pos.max().item()) if pos.numel() else 0) if trim_gqa else 0
    used = max(1, pos_max) if trim_gqa and pos.numel() else None
    pos_out = pos.to(device=device) if device is not None else pos.clone()
    layers = []
    for layer in temporal.layers:
        if isinstance(layer, KDAState):
            mat = layer.matrix[idx]
            if detach:
                mat = mat.detach()
            layers.append(KDAState(matrix=mat.to(device=device) if device is not None else mat.clone()))
        else:
            # Trim before advanced indexing: indexing first materializes the
            # entire allocated KV capacity just to discard its unused suffix.
            key = layer.key[:, :, :used][idx] if used is not None else layer.key[idx]
            val = layer.value[:, :, :used][idx] if used is not None else layer.value[idx]
            if detach:
                key, val = key.detach(), val.detach()
            layers.append(
                GQAState(
                    key=key.to(device=device) if device is not None else key.clone(),
                    value=val.to(device=device) if device is not None else val.clone(),
                )
            )
    bound = pos_max if trim_gqa else temporal.length_bound
    return PolicyMemory(
        TemporalState(tuple(layers), pos_out, bound),
        memory.owner,
        memory.generation,
    )


def reserve_step_memory(memory: PolicyMemory, minimum_capacity: int = 128) -> PolicyMemory:
    """Reserve a geometric KV bucket without synchronizing GPU positions.

    Per-lane storage can remain trimmed. Reserving before the batched step avoids
    reallocating to twice the exact prefix and reading positions back every step.
    All existing matrices/positions remain unchanged.
    """
    capacity = max(1, int(minimum_capacity))
    while capacity < memory.temporal.length_bound + 1:
        capacity *= 2
    layers = []
    changed = False
    for layer in memory.temporal.layers:
        if isinstance(layer, GQAState) and layer.key.shape[2] < capacity:
            pad = capacity - layer.key.shape[2]
            layer = GQAState(F.pad(layer.key, (0, 0, 0, pad)), F.pad(layer.value, (0, 0, 0, pad)))
            changed = True
        layers.append(layer)
    return (PolicyMemory(TemporalState(layers, memory.temporal.positions, memory.temporal.length_bound),
                         memory.owner, memory.generation) if changed else memory)


def stack_memories(memories: list[PolicyMemory], *, device=None) -> PolicyMemory:
    """Concatenate a list of single/batch PolicyMemory objects along axis 0."""
    if not memories:
        raise ValueError("Cannot stack empty memory list")
    first = memories[0]
    pos_list = [m.temporal.positions.to(device=device) if device is not None else m.temporal.positions for m in memories]
    positions = torch.cat(pos_list, dim=0)
    max_bound = max(int(m.temporal.length_bound) for m in memories)
    num_layers = len(first.temporal.layers)
    stacked_layers = []
    for layer_idx in range(num_layers):
        sample_layer = first.temporal.layers[layer_idx]
        if isinstance(sample_layer, KDAState):
            mats = [
                (m.temporal.layers[layer_idx].matrix.to(device=device) if device is not None else m.temporal.layers[layer_idx].matrix)
                for m in memories
            ]
            stacked_layers.append(KDAState(matrix=torch.cat(mats, dim=0)))
        else:
            max_cap = max(m.temporal.layers[layer_idx].key.shape[2] for m in memories)
            keys, vals = [], []
            for m in memories:
                layer = m.temporal.layers[layer_idx]
                k = layer.key.to(device=device) if device is not None else layer.key
                v = layer.value.to(device=device) if device is not None else layer.value
                if k.shape[2] < max_cap:
                    pad = max_cap - k.shape[2]
                    k = F.pad(k, (0, 0, 0, pad))
                    v = F.pad(v, (0, 0, 0, pad))
                keys.append(k)
                vals.append(v)
            stacked_layers.append(GQAState(key=torch.cat(keys, dim=0), value=torch.cat(vals, dim=0)))
    return PolicyMemory(
        TemporalState(tuple(stacked_layers), positions, max_bound),
        first.owner,
        first.generation,
    )


def rebind_memory(memory: PolicyMemory | None, model) -> PolicyMemory | None:
    """Bind a detached memory (e.g., rollout initial state) to `model`'s current identity/gen."""
    if memory is None:
        return None
    return PolicyMemory(memory.temporal.detach(), id(model), model.memory_generation)
