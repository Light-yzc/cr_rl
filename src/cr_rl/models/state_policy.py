"""Versioned, episode-state actor using KDA and full-history grouped attention."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch

from .policy import ActorCritic, ModelConfig, spatial_features
from .state_temporal import StateTemporalConfig, StateTemporalCore


@dataclass
class PolicyMemory:
    temporal: object
    owner: int
    generation: int


class StateActorCritic(ActorCritic):
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
        self.temporal_cfg = temporal_cfg
        self.decision_ticks = int(decision_ticks)
        self.frame_microbatch = int(frame_microbatch)
        self.memory_generation = 0

    def checkpoint_config(self):
        definition = {"model_architecture_version": self.architecture_version,
                "config": asdict(self.cfg), "temporal_config": asdict(self.temporal_cfg),
                "decision_ticks": self.decision_ticks}
        if hasattr(self, "learned_placement"):
            definition["learned_placement_definition"] = self.learned_placement.get_extra_state()
        return definition

    def initial_state(self, batch_size=1, device=None, dtype=None):
        reference = self.to_temporal.weight
        device = torch.device(device or reference.device)
        if dtype is None:
            dtype = (torch.get_autocast_dtype(device.type)
                     if torch.is_autocast_enabled(device.type) else reference.dtype)
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

    def forward_state(self, header, entities, entity_mask, legal_action_mask=None, *,
                      state=None, hand_card_ids=None, deck_card_ids=None,
                      enemy_seen_card_ids=None, time_mask=None, compute_value=True,
                      placement_mask=None, placement_slots=None):
        # A single episode/seat [T,...], or a batch of episode chunks [B,T,...].
        single = header.ndim == 2
        if single:
            header, entities, entity_mask = header[None], entities[None], entity_mask[None]
            legal_action_mask = None if legal_action_mask is None else legal_action_mask[None]
            hand_card_ids = None if hand_card_ids is None else hand_card_ids[None]
            time_mask = None if time_mask is None else time_mask[None]
            placement_mask = None if placement_mask is None else placement_mask[None]
            placement_slots = None if placement_slots is None else placement_slots[None]
        if header.ndim != 3 or entities.ndim != 4:
            raise ValueError("Expected chronological [T,...] or [B,T,...] frames")
        batch, steps, count, features = entities.shape
        if state is None:
            state = self.initial_state(batch, entities.device)
        self._check_memory(state)
        flat_h = header.reshape(batch * steps, -1)
        flat_e = entities.reshape(batch * steps, count, features)
        flat_m = entity_mask.reshape(batch * steps, count)
        real_frames = (None if time_mask is None else time_mask.reshape(-1).bool())
        encoder_mask = flat_m
        if real_frames is not None:
            # Eval attention may emit NaN for a completely masked row. Give
            # only synthetic padding a dummy entity, then zero its outputs.
            # NaN * 0 would otherwise poison a real prefix in the KV cache.
            first_entity = torch.arange(count, device=flat_m.device) == 0
            encoder_mask = flat_m.bool() | (~real_frames[:, None] & first_entity[None])
        summaries, tokens = [], []
        for start in range(0, len(flat_h), self.frame_microbatch):
            summary, token = self.encode_frames(flat_h[start:start+self.frame_microbatch],
                flat_e[start:start+self.frame_microbatch], encoder_mask[start:start+self.frame_microbatch])
            if real_frames is not None:
                padding = ~real_frames[start:start+self.frame_microbatch]
                summary = summary.masked_fill(padding[:, None], 0)
                token = token.masked_fill(padding[:, None, None], 0)
            summaries.append(summary); tokens.append(token)
        summary = torch.cat(summaries).reshape(batch, steps, -1)
        token = torch.cat(tokens)
        temporal, next_temporal = self.temporal.prefill(
            self.to_temporal(summary), state=state.temporal, time_mask=time_mask)
        context = temporal.reshape(batch * steps, -1)
        if hand_card_ids is None:
            hand_card_ids = torch.full((batch, steps, 4), -1, dtype=torch.long,
                                       device=header.device)
        hands = hand_card_ids.reshape(batch * steps, 4)
        card_features = self.card_embedding((hands.long()+1).clamp(0,self.cfg.vocab_size))
        card_features = card_features + self.card_slot_embedding
        context_slots = context[:, None].expand(-1,4,-1)
        slots = self.slot_actor(torch.cat((context_slots,card_features),-1)).squeeze(-1)
        cards = torch.cat((self.wait_actor(context),slots,self.ability_actor(context)),-1)
        # A WAIT/ability BC target has no placement loss. Computing its 2304
        # conditional locations wastes most of continuous-trajectory training.
        # The card mass is exactly independent of these normalized tile logits.
        if placement_mask is not None and not self.training:
            raise ValueError("Selective placement is a supervised-training optimization only")
        if placement_slots is not None and (not self.training or placement_mask is None or not hasattr(self, 'learned_placement')):
            raise ValueError('GT placement slots require learned-placement supervised training')
        selected = (torch.arange(len(context),device=context.device) if placement_mask is None
                    else placement_mask.reshape(-1).bool().nonzero().flatten())
        # Finished parallel streams carry empty padded frames. The learned
        # spatial decoder requires real entities, so skip only those padding
        # rows; every active inference frame still evaluates all four cards.
        if hasattr(self, "learned_placement") and time_mask is not None:
            selected = selected[time_mask.reshape(-1).bool()[selected]]
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
                logits = self.learned_placement(context[rows], token[rows], flat_m[rows], card_features[rows],
                    entity_xy=flat_e[rows, :, 4:6], card_ids=hands[rows])
            else:
                logits = self.placement_logits(context[rows],token[rows],flat_m[rows],card_features[rows])
            tiles = tiles.index_copy(0,rows,logits.float())
        legal = (torch.ones(batch*steps,self.cfg.action_count,dtype=torch.bool,device=header.device)
                 if legal_action_mask is None else legal_action_mask.reshape(batch*steps,-1))
        logp = self._joint_log_probs(cards,tiles,legal).reshape(batch,steps,-1)
        value = (self.critic(context).squeeze(-1).reshape(batch,steps) if compute_value
                 else context.new_zeros(batch,steps))
        memory = PolicyMemory(next_temporal,id(self),self.memory_generation)
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
    model.load_state_dict(checkpoint["model"],strict=True)
    return model
