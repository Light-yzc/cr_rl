"""Train-only card-location initialization plus a complete learned head residual.

This isolated diagnostic candidate starts from log(card tile mass + alpha),
not the old head's function. It is a training-initialization candidate, NOT a
capacity-only comparison, tactical rule, WAIT/holding bias, or high-cost-card
protection mechanism. The original full placement head can subsequently learn
state-conditioned corrections; the card-location table is also trainable.

The caller owns legal masking and conditional log-softmax. No validation or
heldout labels are read or used to fit a prior here. Use the predeclared alpha
arms .01 and 1.; do not choose an alpha by inspecting validation performance.

Run ``python scripts/state_bc_card_location_probe.py --self-test`` for tiny CPU
checks. Optionally provide ``--train-prior PATH`` to check its initialization.
There is no GPU, trainer, checkpoint loader, or weight export entry point.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


RAW_FIELDS = ('context', 'entities', 'entity_mask', 'card_features', 'slots')
CARD_LOCATION_FIELDS = RAW_FIELDS + ('card_ids',)
TILES = 18 * 32
PREDECLARED_ALPHAS = (.01, 1.)


def load_train_prior(source):
    """Load only the already training-fitted mass/count artifact.

    Partition provenance must be established by the producer and frozen
    experiment protocol. Aggregate validation-count metadata in that artifact
    is ignored; there is no fitting or validation-label input in this module.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        payload = path.read_bytes()
        prior = json.loads(payload)
        info = {'source_path': str(path), 'source_sha256': hashlib.sha256(payload).hexdigest()}
    elif isinstance(source, Mapping):
        prior = source
        info = {'source_path': None, 'source_sha256': None}
    else:
        raise TypeError('train_prior must be a frozen train-prior JSON path or mapping')
    if not isinstance(prior.get('mass'), Mapping) or not isinstance(prior.get('card_counts'), Mapping):
        raise ValueError('Training prior requires card_counts and mass mappings')
    return prior, info


class CardLocationPriorResidual(nn.Module):
    """Trainable card identity x tile logits plus the complete old-head residual.

    ``original_head`` accepts RAW_FIELDS and returns [B,576] GT-slot logits.
    Its ``tile_actor[-1]`` must be the original one-output Linear layer. The
    wrapper takes ownership without deep-copying and zeros ONLY that layer's
    weight/bias. Every other old weight is retained exactly; construct an
    independent original head for each arm. Requires-grad flags are preserved,
    so an outer probe can explicitly enable all parameters after wrapping.

    Raw card IDs are vocabulary indices 0..vocab_size-1, not hand slots and not
    ID+1 embedding indices. Unknown -1 has its own uniform initial row. A seen
    card's initial logits are exactly log(train mass + alpha); unseen cards
    have constant logits and thus uniform legal conditional probabilities.

    The first backward learns the old final projection and location table.
    Earlier old-head layers receive gradients after that projection becomes
    nonzero. There is no additional zero gate which would block this route.
    """
    probe_input_fields = CARD_LOCATION_FIELDS

    def __init__(self, original_head, train_prior, *, vocab_size=1466, alpha=.01):
        super().__init__()
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError('vocab_size must be a positive integer')
        if alpha not in PREDECLARED_ALPHAS:
            raise ValueError('Use the predeclared independent alpha arms .01 or 1.')
        if not isinstance(getattr(original_head, 'tile_actor', None), nn.Sequential):
            raise ValueError('Original full placement head must expose tile_actor Sequential')
        final = original_head.tile_actor[-1]
        if not isinstance(final, nn.Linear) or final.out_features != 1:
            raise ValueError('Only an original one-output final Linear layer may be zero-initialized')
        if final.weight.device.type == 'meta':
            raise ValueError('Materialize the independent original head before initializing this candidate')
        prior, info = load_train_prior(train_prior)
        self.original = original_head
        self.vocab_size, self.alpha = vocab_size, float(alpha)
        self.card_location_bias = nn.Embedding(vocab_size + 1, TILES,
            device=final.weight.device, dtype=torch.float32)
        with torch.no_grad():
            self.card_location_bias.weight.fill_(self.alpha)
            for raw_id, mass in prior['mass'].items():
                card_id = int(raw_id)
                if not 0 <= card_id < vocab_size:
                    raise ValueError('Training prior card ID lies outside the declared vocabulary')
                values = torch.as_tensor(mass, dtype=torch.float32, device=final.weight.device)
                if values.shape != (TILES,) or not torch.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Each training-card mass must have 576 finite nonnegative entries')
                count = prior['card_counts'].get(str(raw_id), prior['card_counts'].get(card_id))
                if count is None or not isinstance(count, (int, float)) or count <= 0:
                    raise ValueError('Every mass row must have a positive training card count')
                if not torch.isclose(values.sum(), values.new_tensor(float(count)), rtol=1e-4, atol=1e-4):
                    raise ValueError('Training tile target mass does not sum to its card count')
                self.card_location_bias.weight[card_id + 1].copy_(values + self.alpha)
            # Unknown and unseen rows remain log(alpha), an exactly constant
            # logit vector; legal softmax supplies their uniform distribution.
            self.card_location_bias.weight.log_()
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()
        self.prior_info = {**info, 'fit_games': prior.get('fit_games'),
            'sigma': prior.get('sigma'), 'cards_in_training_prior': len(prior['mass']),
            'training_ordinary_plays': sum(prior['card_counts'].values())}

    def added_parameter_count(self):
        return self.card_location_bias.weight.numel()

    def initialization_info(self):
        return {**self.prior_info, 'alpha': self.alpha, 'vocab_size': self.vocab_size,
            'added_parameters': self.added_parameter_count(),
            'original_function_preserved': False,
            'original_final_projection_zeroed': True,
            'interpretation': 'Train-data initialization candidate; not a capacity-only or tactical-prior comparison'}

    def prior_logits(self, card_ids):
        if card_ids.ndim != 1 or card_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('GT raw card_ids must be an integral tensor [B]')
        if card_ids.device != self.card_location_bias.weight.device:
            raise ValueError('Card IDs and the location table must be on the same device')
        if ((card_ids < -1) | (card_ids >= self.vocab_size)).any():
            raise ValueError('Raw card ID is outside [-1,vocab_size); do not pass hand slots or ID+1 indices')
        return self.card_location_bias(card_ids.long() + 1)

    def residual_logits(self, context, entities, entity_mask, card_features, slots):
        residual = self.original(context=context, entities=entities, entity_mask=entity_mask,
            card_features=card_features, slots=slots)
        if residual.shape != (len(context), TILES):
            raise ValueError('Complete original head must return [B,576] GT-slot logits')
        return residual

    def forward(self, context, entities, entity_mask, card_features, slots, card_ids):
        if card_ids.shape != (len(context),):
            raise ValueError('Provide one actual GT card identity for every action row')
        residual = self.residual_logits(context, entities, entity_mask, card_features, slots)
        return self.prior_logits(card_ids) + residual.float()


class _FixtureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_projection = nn.Linear(4, 8)
        self.tile_actor = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 6), nn.GELU(), nn.Linear(6, 1))
        self.register_buffer('grid', torch.randn(TILES, 8))

    def forward(self, context, entities, entity_mask, card_features, slots):
        card = card_features.gather(1, slots[:, None, None].expand(-1, 1, 8)).squeeze(1)
        token = torch.where(entity_mask[..., None], entities, 0.).mean(1)
        feature = self.grid[None] + (self.context_projection(context) + card + token)[:, None]
        return self.tile_actor(feature).squeeze(-1)


def cpu_self_test(train_prior=None, *, vocab_size=1466):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261111)
            mass = [0.] * TILES
            mass[5], mass[77] = 3., 1.
            prior = {'mass': {'2': mass}, 'card_counts': {'2': 4}, 'fit_games': 1, 'sigma': 1.}
            fixture = _FixtureHead()
            old = {name: value.detach().clone() for name, value in fixture.state_dict().items()}
            model = CardLocationPriorResidual(fixture, prior, vocab_size=5, alpha=.01)
            retained = [name for name in old if not name.startswith('tile_actor.3.')]
            assert all(torch.equal(value, model.original.state_dict()[name]) for name, value in old.items()
                if name in retained)
            assert torch.count_nonzero(model.original.tile_actor[-1].weight) == 0
            assert torch.count_nonzero(model.original.tile_actor[-1].bias) == 0
            raw = {'context': torch.randn(3, 4), 'entities': torch.randn(3, 5, 8),
                'entity_mask': torch.ones(3, 5, dtype=torch.bool),
                'card_features': torch.randn(3, 4, 8), 'slots': torch.tensor([0, 3, 1]),
                'card_ids': torch.tensor([2, 4, -1])}
            initial = model(**raw)
            residual = model.residual_logits(**{key: raw[key] for key in RAW_FIELDS})
            assert torch.count_nonzero(residual) == 0
            torch.testing.assert_close(initial[0], (torch.tensor(mass) + .01).log(), rtol=0, atol=0)
            assert initial[1].unique().numel() == initial[2].unique().numel() == 1
            legal = torch.rand(3, TILES) > .35
            legal[:, 5] = True
            actual = initial.masked_fill(~legal, -torch.inf).softmax(-1)
            expected = torch.stack((torch.tensor(mass) + .01, torch.full((TILES,), .01), torch.full((TILES,), .01)))
            expected *= legal
            expected /= expected.sum(-1, keepdim=True)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            assert actual[~legal].sum() == 0

            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            target = torch.tensor([5, 12, 90])
            nn.functional.cross_entropy(initial, target).backward()
            assert model.original.tile_actor[-1].weight.grad.abs().sum() > 0
            assert model.card_location_bias.weight.grad.abs().sum() > 0
            assert model.original.context_projection.weight.grad.abs().sum() == 0
            assert model.original.tile_actor[1].weight.grad.abs().sum() == 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(**raw), target).backward()
            assert model.original.context_projection.weight.grad.abs().sum() > 0
            assert model.original.tile_actor[1].weight.grad.abs().sum() > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

            for invalid in (torch.tensor([-2]), torch.tensor([5]), torch.tensor([2.])):
                try:
                    model.prior_logits(invalid)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid raw card identity was accepted')

            source_checks = []
            if train_prior is not None:
                source, _ = load_train_prior(train_prior)
                ids = torch.tensor([int(key) for key in source['mass']])
                for alpha in PREDECLARED_ALPHAS:
                    candidate = CardLocationPriorResidual(_FixtureHead(), train_prior,
                        vocab_size=vocab_size, alpha=alpha)
                    logits = candidate.prior_logits(ids)
                    legal = torch.rand(len(ids), TILES) > .35
                    legal[:, 0] = True
                    expected = torch.stack([torch.tensor(source['mass'][str(int(card))]) + alpha for card in ids])
                    expected *= legal
                    expected /= expected.sum(-1, keepdim=True)
                    actual = logits.masked_fill(~legal, -torch.inf).softmax(-1)
                    error = float((actual - expected).abs().max())
                    assert error < 1e-6
                    source_checks.append({'alpha': alpha, 'cards': len(ids),
                        'legal_probability_max_error': error, **candidate.initialization_info()})
            return {'passed': True, 'device': 'cpu', 'initial_residual_exactly_zero': True,
                'other_original_weights_retained_exactly': True, 'initialization_matches_legal_prior': True,
                'unseen_and_unknown_initially_uniform': True,
                'first_step_last_layer_and_bias_gradients': True,
                'second_step_earlier_head_gradients': True, 'finite_gradients': True,
                'production_vocab_added_parameters': (vocab_size + 1) * TILES,
                'real_prior_checks': source_checks,
                'limitation': 'CPU functional/algebra checks only; no GPU/full-model quality or speed result'}
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--train-prior', type=Path)
    parser.add_argument('--vocab-size', type=int, default=1466)
    args = parser.parse_args()
    if not args.self_test:
        parser.error('Only --self-test is exposed; this candidate does not train/export checkpoints')
    print(json.dumps(cpu_self_test(args.train_prior, vocab_size=args.vocab_size), allow_nan=False))
