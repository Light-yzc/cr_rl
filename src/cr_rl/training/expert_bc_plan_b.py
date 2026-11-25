"""Balanced, transactional readout BC for the Sparky Plan B snapshot."""

import copy
import math

import numpy as np
import torch

from cr_rl.training.expert_bc import ExpertBC
from cr_rl.training.live_awbc import amp, freeze_actor_readouts


PLAN = 'sparky_b'
COHORTS = ('sparky', 'general')
MAX_KL = 0.005
MAX_WAIT_DRIFT = 0.005
TILE_COUNT = 576
PLACEMENT_END = 1 + 4 * TILE_COUNT


def frame_weights(episode, schema, sparky_id):
    """Weight only decisions with a usable expert label and a legal choice."""
    raw = episode['raw_obs']
    actions = episode['actions']
    keep = np.asarray(episode['actor_keep'], dtype=bool)
    hand = raw[:, schema.header['hand_card']]
    elixir = raw[:, schema.header['elixir'][0]].astype(np.int64)
    reserved = raw[:, schema.header['elixir_reserved'][0]].astype(np.int64)
    available = elixir - reserved
    scale = int(schema.scales['elixir_scale'])
    if scale <= 0:
        raise ValueError('Plan B requires a positive elixir scale')
    zone = ((hand == sparky_id).any(axis=-1) & (elixir >= 0) & (reserved >= 0)
            & (10 * available >= 48 * scale) & (available < 6 * scale))
    weights = np.where(actions > 0, 1., np.where(zone, .5, .06))
    deploy_rows = np.flatnonzero((actions > 0) & (actions < PLACEMENT_END))
    deploy_slots = (actions[deploy_rows] - 1) // TILE_COUNT
    weights[deploy_rows[hand[deploy_rows, deploy_slots] == sparky_id]] = 1.5
    return np.where(keep, weights, 0.).astype(np.float32)


def _sequence(actor, adapter, episode, device, precision, *, chunk=32):
    """Advance the complete recurrent prefix in bounded chunks, including WAIT."""
    state = None
    hands = list(adapter.schema.header['hand_card'])
    for start in range(0, len(episode['actions']), chunk):
        span = slice(start, start + chunk)
        raw = episode['raw_obs'][None, span]
        header, entities, mask = adapter(raw)
        tensor = lambda value: torch.as_tensor(value, device=device)
        with amp(device, precision):
            logp, _, state = actor.forward_state(
                tensor(header), tensor(entities), tensor(mask), tensor(episode['legal_masks'][None, span]),
                hand_card_ids=tensor(raw[..., hands]), state=state, compute_value=False,
            )
        state = actor.detach_state(state)
        yield span, logp[0].float()


def _aux_episode_batch_size(device, default_bs: int) -> int:
    dev = torch.device(device)
    if dev.type == 'cuda':
        vram_gib = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        if vram_gib < 18.0:
            return 1 if default_bs <= 6 else 2
        if vram_gib < 32.0:
            return min(default_bs, 2 if default_bs <= 6 else 4)
    return default_bs


def _batched_episodes(actor, adapter, episodes, device, precision, *, batch_size=16):
    """Run causal full-episode forwards in padded batches without cross-episode leakage."""
    if not episodes:
        return
    hands_idx = list(adapter.schema.header['hand_card'])
    act_dim = int(episodes[0]['legal_masks'].shape[-1])
    seat_ints = int(episodes[0]['raw_obs'].shape[-1])
    for b_start in range(0, len(episodes), batch_size):
        group = episodes[b_start : b_start + batch_size]
        b_cnt = len(group)
        lengths = [len(ep['actions']) for ep in group]
        max_len = max(lengths, default=0)
        if max_len == 0:
            continue
        raw_b = np.zeros((b_cnt, max_len, seat_ints), dtype=np.int32)
        leg_b = np.zeros((b_cnt, max_len, act_dim), dtype=bool)
        leg_b[:, :, 0] = True
        for pos, (ep, n_len) in enumerate(zip(group, lengths)):
            raw_b[pos, :n_len] = ep['raw_obs']
            leg_b[pos, :n_len] = ep['legal_masks']
        header, entities, mask = adapter(raw_b)
        tensor = lambda value: torch.as_tensor(value, device=device)
        with amp(device, precision):
            logp, _, _ = actor.forward_state(
                tensor(header), tensor(entities), tensor(mask), tensor(leg_b),
                hand_card_ids=tensor(raw_b[..., hands_idx]), state=None, compute_value=False,
            )
        out_logp = logp.float()
        del logp
        yield b_start, group, lengths, out_logp
        del out_logp


@torch.no_grad()
def _score(actor, adapter, episodes, device, precision, *, weights=None, reference=None, capture=False):
    """Evaluate observed choices with frame weights, or capture/compare a PPO reference."""
    nll = weighted_nll = kl = wait = weight_sum = 0.
    frames = deploy_frames = correct_cards = 0
    probabilities = {}
    for b_start, group, lengths, logp_b in _batched_episodes(
        actor, adapter, episodes, device, precision, batch_size=_aux_episode_batch_size(device, 16)
    ):
        for pos, (episode, n_len) in enumerate(zip(group, lengths)):
            index = b_start + pos
            keep = np.asarray(episode['actor_keep'][:n_len], dtype=bool)
            if not keep.any():
                continue
            current = logp_b[pos, :n_len][torch.as_tensor(keep, device=device)]
            actions = torch.as_tensor(episode['actions'][:n_len][keep], device=device)
            losses = -current.gather(1, actions[:, None]).squeeze(1)
            nll += float(losses.sum())
            if weights is not None:
                sample_weights = torch.as_tensor(weights[index][:n_len][keep], device=device)
                weighted_nll += float((sample_weights * losses).sum())
                weight_sum += float(sample_weights.sum())
                deploy = (actions > 0) & (actions < PLACEMENT_END)
                chosen = (actions[deploy] - 1) // TILE_COUNT
                card_logp = current[deploy, 1:PLACEMENT_END].reshape(-1, 4, TILE_COUNT).logsumexp(-1)
                correct_cards += int((card_logp.argmax(-1) == chosen).sum())
                deploy_frames += len(chosen)
            wait += float(current[:, 0].exp().sum())
            frames += len(actions)
            if capture:
                probabilities[index] = current.exp().cpu()
            if reference is not None:
                old = reference[index].to(device)
                term = torch.where(old > 0, old * (old.clamp_min(1e-30).log() - current), 0.)
                kl += float(term.sum())
        del logp_b
    if not frames or (weights is not None and weight_sum <= 0):
        raise ValueError('Plan B evaluation requires legal choice frames')
    result = dict(unweighted_nll=nll / frames, reference_forward_kl=kl / frames,
                  mean_p_wait=wait / frames, choice_frames=frames)
    if weights is not None:
        if not deploy_frames:
            raise ValueError('Plan B held-out validation requires deployment labels')
        result.update(weighted_nll=weighted_nll / weight_sum,
                      deploy_card_top1_acc=correct_cards / deploy_frames, deploy_frames=deploy_frames)
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError('Nonfinite Plan B evaluation')
    return result, probabilities


def _train_accumulated(actor, adapter, episodes, weights, optimizer, scaler, device, precision):
    optimizer.zero_grad(set_to_none=True)
    n_eps = len(episodes)
    for ep in episodes:
        if not int(np.count_nonzero(ep['actor_keep'])):
            raise ValueError('Plan B training requires usable choice frames')
    for b_start, group, lengths, logp_b in _batched_episodes(
        actor, adapter, episodes, device, precision, batch_size=_aux_episode_batch_size(device, 6)
    ):
        batch_loss = None
        for pos, (episode, n_len) in enumerate(zip(group, lengths)):
            weight = weights[b_start + pos][:n_len]
            denominator = int(np.count_nonzero(episode['actor_keep'][:n_len])) * n_eps
            selected = weight > 0
            if not selected.any():
                continue
            active = torch.as_tensor(selected, device=device)
            actions = torch.as_tensor(episode['actions'][:n_len][selected], device=device)
            row_weights = torch.as_tensor(weight[selected], device=device)
            ep_loss = -(logp_b[pos, :n_len][active].gather(1, actions[:, None]).squeeze(1) * row_weights).sum() / denominator
            batch_loss = ep_loss if batch_loss is None else (batch_loss + ep_loss)
        if batch_loss is not None:
            if not torch.isfinite(batch_loss):
                raise FloatingPointError('Nonfinite Plan B training loss')
            scaler.scale(batch_loss).backward()
        del logp_b, batch_loss
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in actor.parameters() if parameter.requires_grad], 1.
    )
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite Plan B readout gradients')
    scaler.step(optimizer)
    scaler.update()
    actor.mark_weights_updated()


class PlanBExpertBC(ExpertBC):
    """Use the standard ExpertBC loader with balanced data and fresh optimization."""

    bc_plan = PLAN

    def __init__(self, dataset, actor, *, seed=20261117, state=None, experimental=False):
        super().__init__(dataset, actor, seed=seed, experimental=experimental)
        if self.metadata.get('bc_plan') != PLAN or not self.validation_enabled or self.focus_card is not None:
            raise ValueError('Plan B requires a mixed sparky_b snapshot with validation enabled')
        vocab = self.metadata['vocabulary']
        self.sparky_id = (vocab['names'] if isinstance(vocab, dict) else vocab).index('ZapMachine')
        self.train_by_cohort = self._cohorts(self.train)
        self.validation_by_cohort = self._cohorts(self.validation)
        self.samplers = {cohort: dict(order=[], cursor=0) for cohort in COHORTS}
        if state:
            if (state.get('bc_plan') != PLAN or state['snapshot_id'] != self.snapshot_id
                    or state['train'] != self.train):
                raise ValueError('The Plan B snapshot changed across resume')
            self.rng.bit_generator.state = state['rng']
            self.samplers = copy.deepcopy(state['samplers'])
            self.last_update = state['last_update']
            self.phase_count = state['phase_count']
            self.optimized_sequences = state['optimized_sequences']

    def _cohorts(self, descriptors):
        grouped = {cohort: [] for cohort in COHORTS}
        for index, seat in descriptors:
            shard = self.manifest['shards'][index]
            cohort = shard.get('cohort')
            replay = self.metadata['replays'][shard['replay_offset']]
            cards = replay['decks'][seat]
            if (cohort not in COHORTS or shard.get('selected_seats') != [seat]
                    or ('ZapMachine' in cards) != (cohort == 'sparky')):
                raise ValueError('Plan B cohort disagrees with the selected seat or deck')
            grouped[cohort].append((index, seat))
        if len(grouped['sparky']) != len(grouped['general']) or len(grouped['sparky']) < 6:
            raise ValueError('Plan B needs equal Sparky and general cohorts with at least six games each')
        return grouped

    def state_dict(self):
        return dict(bc_plan=PLAN, snapshot_id=self.snapshot_id, train=self.train,
                    rng=self.rng.bit_generator.state, samplers=copy.deepcopy(self.samplers),
                    last_update=self.last_update, phase_count=self.phase_count,
                    optimized_sequences=self.optimized_sequences)

    def draw(self, count):
        if count % 2 or not 0 < count <= 2 * min(map(len, self.train_by_cohort.values())):
            raise ValueError('Plan B draw requires equal cohorts and a supported sequence count')
        chosen = []
        for cohort in COHORTS:
            population = self.train_by_cohort[cohort]
            sampler = self.samplers[cohort]
            distinct = set()
            while len(distinct) < count // 2:
                if sampler['cursor'] == len(sampler['order']):
                    sampler['order'] = self.rng.permutation(len(population)).tolist()
                    sampler['cursor'] = 0
                descriptor = population[sampler['order'][sampler['cursor']]]
                sampler['cursor'] += 1
                if descriptor not in distinct:
                    chosen.append(descriptor)
                    distinct.add(descriptor)
        half = count // 2
        return [item for pair in zip(chosen[:half], chosen[half:], strict=True) for item in pair]

    def run_phase(self, actor, adapter, probes, *, update, device, precision='fp32',
                  sequences=12, lr=1e-6, max_kl=MAX_KL, placement=False):
        if not probes or not any(np.any(episode['actor_keep']) for episode in probes):
            return dict(ran=False, reason='no_ppo_choice_probes')
        if update <= self.last_update:
            return dict(ran=False, reason='already_processed')
        if placement or sequences != 12:
            raise ValueError('Plan B requires readout-only BC with 12 training sequences')
        if not 0 < lr <= 1e-6 or not 0 < max_kl <= MAX_KL:
            raise ValueError('Plan B requires a positive learning rate <= 1e-6 and KL cap <= 0.005')
        descriptors = self.draw(sequences)
        validation_descriptors = self.validation
        train = self.load(descriptors)
        holdout = self.load(validation_descriptors)
        train_weights = [frame_weights(ep, adapter.schema, self.sparky_id) for ep in train]
        holdout_weights = [frame_weights(ep, adapter.schema, self.sparky_id) for ep in holdout]
        parameters = dict(actor.named_parameters())
        original_requires = {name: p.requires_grad for name, p in parameters.items()}
        mode = actor.training
        originals = {}
        steps = 0
        baseline = validation = probe_before = probe_after = None
        reasons = []
        try:
            names = freeze_actor_readouts(actor)
            if hasattr(actor, 'learned_placement'):
                for lp_name, lp_param in actor.named_parameters():
                    if lp_name.startswith('learned_placement.') and not lp_name.endswith('.anchor') and lp_name not in names:
                        lp_param.requires_grad_(True)
                        names.append(lp_name)
            originals = {name: parameters[name].detach().clone() for name in names}
            head_params = [parameters[n] for n in names if not n.startswith('learned_placement.')]
            tile_params = [parameters[n] for n in names if n.startswith('learned_placement.')]
            opt_groups = [{'params': head_params, 'lr': lr}]
            if tile_params:
                opt_groups.append({'params': tile_params, 'lr': lr * 0.2})
            optimizer = torch.optim.AdamW(opt_groups, weight_decay=0.)
            scaler = torch.amp.GradScaler('cuda', enabled=torch.device(device).type == 'cuda', init_scale=128.)
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
            actor.eval()
            baseline, _ = _score(actor, adapter, holdout, device, precision, weights=holdout_weights)
            probe_before, reference = _score(actor, adapter, probes, device, precision, capture=True)
            for start in range(0, sequences, 6):
                _train_accumulated(actor, adapter, train[start:start + 6],
                                   train_weights[start:start + 6], optimizer, scaler, device, precision)
                steps += 1
            validation, _ = _score(actor, adapter, holdout, device, precision, weights=holdout_weights)
            probe_after, _ = _score(actor, adapter, probes, device, precision, reference=reference)
            nll_improved = (
                validation['unweighted_nll'] < baseline['unweighted_nll']
                or (
                    'weighted_nll' in validation
                    and 'weighted_nll' in baseline
                    and validation['weighted_nll'] < baseline['weighted_nll'] - 1e-5
                )
            )
            if not nll_improved:
                reasons.append('heldout_unweighted_nll_not_improved')
            if probe_after['reference_forward_kl'] > max_kl:
                reasons.append('ppo_forward_kl_limit')
            if probe_after['mean_p_wait'] - probe_before['mean_p_wait'] > MAX_WAIT_DRIFT:
                reasons.append('mean_p_wait_drift')
        except FloatingPointError:
            reasons.append('nonfinite_candidate')
        except BaseException:
            self._restore(actor, parameters, originals)
            raise
        finally:
            for name, parameter in parameters.items():
                parameter.requires_grad_(original_requires[name])
                parameter.grad = None
            actor.train(mode)
        accepted = not reasons and steps == 2
        invalidated = bool(steps)
        if not accepted and originals:
            self._restore(actor, parameters, originals)
            invalidated = True
        self.last_update = int(update)
        self.phase_count += 1
        self.optimized_sequences += steps * 6
        before_wait = probe_before['mean_p_wait'] if probe_before is not None else None
        after_wait = probe_after['mean_p_wait'] if probe_after is not None else None
        return dict(
            bc_plan=PLAN, ran=True, accepted=accepted, rejected_reasons=reasons,
            actor_attempted=bool(steps), memory_invalidated=invalidated,
            changed_parameters=list(originals) if accepted else [],
            snapshot_id=self.snapshot_id, sequence_descriptors=descriptors,
            validation_enabled=True, validation_descriptors=validation_descriptors,
            validation_sequences=len(validation_descriptors),
            baseline=baseline, validation=validation,
            ppo_forward_kl=probe_after['reference_forward_kl'] if probe_after is not None else None,
            mean_p_wait_before=before_wait, mean_p_wait_after=after_wait,
            mean_p_wait_delta=None if after_wait is None else after_wait - before_wait,
            wait_prob_delta=None if after_wait is None else after_wait - before_wait,
            optimizer_steps=steps, trained_sequences=steps * 6,
            phase_count=self.phase_count, ppo_critic_updated=False, backbone_updated=False,
        )

    @staticmethod
    @torch.no_grad()
    def _restore(actor, parameters, originals):
        for name, original in originals.items():
            parameters[name].copy_(original)
        if originals:
            actor.mark_weights_updated()
