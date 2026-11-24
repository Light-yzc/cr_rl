"""Small, transactional expert-BC phases between complete PPO updates.

Snapshots are copied to SSD before use. No file hashes or whole-pool array scans
are performed. Original match identity keeps both seats in one split. Initial
experiments adapt readouts only; PPO still trains its normal parameters.
"""
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import torch

from cr_rl.training.live_awbc import freeze_actor_readouts, reference_pass, train_actor_epoch


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


class ExpertBC:
    def __init__(self, dataset, actor, *, seed=20261117, state=None, experimental=False):
        self.root = Path(dataset).resolve()
        if self.root.is_file():
            self.root = self.root.parent
        if str(self.root).startswith('/content/drive/'):
            raise ValueError('Expert BC arrays must be staged on local SSD, not Drive')
        self.manifest = json.loads((self.root/'dataset.manifest.json').read_text())
        self.metadata = json.loads((self.root/self.manifest['metadata']).read_text())
        self.splits = json.loads((self.root/'split_registry.json').read_text())['assignments']
        self.snapshot_id = self.manifest.get('snapshot_id')
        if not self.snapshot_id:
            raise ValueError('A frozen, named snapshot is required')
        if not self.metadata.get('training_ready') and not experimental:
            raise ValueError('Candidate BC requires explicit --expert-bc-experimental')
        if self.metadata.get('decision_ticks') != 8:
            raise ValueError('Expert BC requires native 8-tick history')
        schema = self.metadata['schema']
        if (tuple(schema['header_fields']) != tuple(actor.cfg.header_fields)
                or tuple(schema['entity_fields']) != tuple(actor.cfg.entity_fields)):
            raise ValueError('BC/model observation field order differs')
        vocab = self.metadata['vocabulary']
        names = vocab['names'] if isinstance(vocab,dict) else vocab
        if len(names) != actor.cfg.vocab_size:
            raise ValueError('BC/model vocabulary size differs')
        self.train = []
        self.validation = []
        self.focus_card = self.metadata.get('focus_card')
        self.validation_enabled = self.metadata.get('validation_enabled', True)
        if self.validation_enabled is not True and (self.validation_enabled is not False or self.focus_card is None):
            raise ValueError('Disabling BC validation requires an explicit focused snapshot')
        identities = set()
        for index, shard in enumerate(self.manifest['shards']):
            record = self.metadata['replays'][shard['replay_offset']]
            identity = record['canonical_game_signature']
            if identity in identities:
                raise ValueError('Duplicate whole game in frozen snapshot')
            identities.add(identity)
            partition = self.splits.get(identity)
            if partition not in ('train','validation','heldout'):
                raise ValueError('Missing game-level partition')
            if not self.validation_enabled and partition != 'train':
                raise ValueError('Full-data BC snapshots must assign every game to train')
            seats = shard.get('selected_seats', [0, 1])
            if (not isinstance(seats, list) or not seats
                    or any(type(seat) is not int or seat not in (0, 1) for seat in seats)
                    or len(seats) != len(set(seats))):
                raise ValueError('Invalid BC seat selection')
            if self.focus_card is not None:
                decks = record.get('decks', ())
                matching = [i for i, deck in enumerate(decks) if self.focus_card in deck]
                if self.focus_card not in names or len(decks) != 2 or seats != matching:
                    raise ValueError('BC seat selection disagrees with the focus card deck')
            target = self.train if partition == 'train' else self.validation if partition == 'validation' else None
            if target is not None:
                target.extend((index, seat) for seat in seats)
        if len(self.train) < 8 or (self.validation_enabled and len(self.validation) < 4):
            raise ValueError('BC snapshot needs >=8 train sequences and, when enabled, >=4 validation sequences')
        self.rng = np.random.default_rng(seed)
        self.order = []
        self.cursor = 0
        self.epoch = 0
        self.last_update = -1
        self.phase_count = 0
        self.optimized_sequences = 0
        self.optimizer_state = None
        self.scaler_state = None
        if state:
            if state['snapshot_id'] != self.snapshot_id or state['train'] != self.train:
                raise ValueError('BC snapshot changed across PPO resume; start a separate experiment')
            self.rng.bit_generator.state = state['rng']
            for name in ('order','cursor','epoch','last_update','phase_count','optimized_sequences',
                         'optimizer_state','scaler_state'):
                setattr(self, name, copy.deepcopy(state[name]))

    def state_dict(self):
        return dict(snapshot_id=self.snapshot_id, train=self.train,
                    rng=self.rng.bit_generator.state, **{k:getattr(self,k) for k in (
                        'order','cursor','epoch','last_update','phase_count','optimized_sequences',
                        'optimizer_state','scaler_state')})

    def draw(self, count):
        chosen = []
        count = min(count, len(self.train))
        if not self.validation_enabled and self.cursor < len(self.order):
            count = min(count, len(self.order) - self.cursor)
        while len(chosen) < count:
            if self.cursor >= len(self.order):
                self.order = list(map(int, self.rng.permutation(len(self.train))))
                self.cursor = 0
                self.epoch += 1
            item = self.train[self.order[self.cursor]]
            self.cursor += 1
            if item not in chosen:
                chosen.append(item)
        return chosen

    def load(self, descriptors):
        episodes = []
        grouped = {}
        for index, seat in descriptors:
            grouped.setdefault(index, []).append(seat)
        for index, seats in grouped.items():
            record = self.manifest['shards'][index]
            path = (self.root/record['path']).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('BC array path escapes snapshot')
            with np.load(path, allow_pickle=False) as arrays:
                required = ['observations','actions','masks','ticks','flags','seats','replay_ids']
                values = {k:arrays[k] for k in required}
            n = len(values['actions'])
            if any(len(a) != n for a in values.values()):
                raise ValueError('BC array row counts disagree')
            for seat in seats:
                rows = np.flatnonzero(values['seats'] == seat)
                if not len(rows):
                    raise ValueError('Missing seat sequence')
                rows = rows[np.argsort(values['ticks'][rows], kind='stable')]
                ticks = values['ticks'][rows]
                if ticks[0] != 0 or (len(ticks)>1 and not np.all(np.diff(ticks)==8)):
                    raise ValueError('Incomplete/non-native BC recurrent history')
                raw = values['observations'][rows]
                masks = values['masks'][rows].astype(bool)
                actions = values['actions'][rows].astype(np.int64)
                if (raw.shape != (len(rows),self.metadata['schema']['seat_ints'])
                        or masks.shape != (len(rows),2307) or not masks.any(-1).all()
                        or ((actions<0)|(actions>=2307)).any()):
                    raise ValueError('Invalid BC observation/action shapes')
                legal = masks[np.arange(len(rows)),actions]
                if not legal.all():
                    raise ValueError('Illegal target in strict BC snapshot')
                keep = (values['flags'][rows] == 0) & (masks.sum(-1)>1)
                if not keep.any():
                    raise ValueError('BC seat has no usable choice labels')
                episodes.append(dict(raw_obs=raw, legal_masks=masks, actions=actions,
                    actor_keep=keep, descriptor=(index,seat),
                    game_identity=self.metadata['replays'][record['replay_offset']]['canonical_game_signature']))
        # Restore requested sampling order after grouping file reads.
        by_key = {ep['descriptor']:ep for ep in episodes}
        return [by_key[tuple(key)] for key in descriptors]

    def run_phase(self, actor, adapter, probes, *, update, device, precision='fp32',
                  sequences=8, validation_sequences=4, lr=1e-6, max_kl=.01):
        if not probes:
            return dict(ran=False, reason='no_ppo_choice_probes')
        if update <= self.last_update:
            return dict(ran=False, reason='already_processed')
        descriptors = self.draw(sequences)
        holdout = self.load(self.validation[:validation_sequences]) if self.validation_enabled else None
        old_requires = {name:p.requires_grad for name,p in actor.named_parameters()}
        old_mode = actor.training
        names = freeze_actor_readouts(actor)
        original = {name:p.detach().cpu().clone() for name,p in actor.named_parameters() if name in names}
        opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=lr, weight_decay=0.)
        if self.optimizer_state:
            opt.load_state_dict(copy.deepcopy(self.optimizer_state))
            for group in opt.param_groups:
                group['lr'] = lr
        scaler = torch.amp.GradScaler('cuda', enabled=str(device).startswith('cuda'), init_scale=128.)
        if self.scaler_state:
            scaler.load_state_dict(copy.deepcopy(self.scaler_state))
        baseline_opt = cpu_copy(opt.state_dict())
        baseline_scaler = cpu_copy(scaler.state_dict())
        def restore():
            with torch.no_grad():
                for name,p in actor.named_parameters():
                    if name in original:
                        p.copy_(original[name].to(p.device))
            actor.mark_weights_updated()
        try:
            baseline = validation = None
            if holdout is not None:
                baseline = reference_pass(actor,adapter,holdout,list(range(len(holdout))),
                    device=device,precision=precision,capture=True)
            reference_pass(actor,adapter,probes,list(range(len(probes))),
                device=device,precision=precision,capture=True)
            batch_size = len(descriptors) if self.validation_enabled else 4
            optimizer_steps = optimized_sequences = 0
            for start in range(0, len(descriptors), max(1, batch_size)):
                episodes = self.load(descriptors[start:start + batch_size])
                weights = {i:np.ones(len(ep['actions']),dtype=np.float32) for i,ep in enumerate(episodes)}
                metrics = train_actor_epoch(actor,adapter,episodes,list(range(len(episodes))),weights,opt,
                    device=device,precision=precision,seed=int(update)+start,scaler=scaler)
                if not np.isfinite(metrics['episode_mean_weighted_nll']):
                    raise ValueError('Nonfinite BC training loss')
                optimizer_steps += metrics['optimizer_steps']
                optimized_sequences += len(metrics['optimized_indices'])
            if holdout is not None:
                validation = reference_pass(actor,adapter,holdout,list(range(len(holdout))),
                    device=device,precision=precision)
            probe = reference_pass(actor,adapter,probes,list(range(len(probes))),
                device=device,precision=precision)
            scores = [*(baseline or {}).values(), *(validation or {}).values(), *probe.values()]
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite BC candidate/validation metrics')
            accepted = bool(optimizer_steps and probe['reference_forward_kl'] <= max_kl)
            if validation is not None:
                accepted &= (validation['reference_forward_kl'] <= max_kl
                             and validation['unweighted_nll'] < baseline['unweighted_nll'] - 1e-5)
            if not accepted:
                restore()
            self.optimizer_state = cpu_copy(opt.state_dict()) if accepted else baseline_opt
            self.scaler_state = cpu_copy(scaler.state_dict()) if accepted else baseline_scaler
            self.last_update = int(update)
            self.phase_count += 1
            self.optimized_sequences += optimized_sequences
            return dict(ran=True, accepted=accepted, actor_attempted=bool(optimizer_steps),
                memory_invalidated=True, changed_parameters=names if accepted else [],
                snapshot_id=self.snapshot_id, sequence_descriptors=descriptors,
                **({'focus_card': self.focus_card} if self.focus_card is not None else {}),
                validation_descriptors=self.validation[:validation_sequences],
                baseline=baseline, validation=validation, validation_enabled=self.validation_enabled,
                ppo_forward_kl=probe['reference_forward_kl'],
                optimizer_steps=optimizer_steps, trained_sequences=optimized_sequences, phase_count=self.phase_count,
                ppo_critic_updated=False, backbone_updated=False,
                expert_skill_verified=False, experimental=True)
        except BaseException:
            restore()
            raise
        finally:
            for name,p in actor.named_parameters():
                p.requires_grad_(old_requires[name])
                p.grad = None
            actor.train(old_mode)
