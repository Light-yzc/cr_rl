"""Deferred scalar readback must not change episode gradients or supervision."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import copy

import pytest
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.training.state_bc import EpisodeProfiler, ReplaySequenceDataset, run_state_episode

spec = importlib.util.spec_from_file_location('speed_state_fixtures', Path(__file__).with_name('test_state_bc.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class CaptureSGD(torch.optim.SGD):
    def step(self, closure=None):
        self.captured = [p.grad.detach().clone() if p.grad is not None else None
                         for group in self.param_groups for p in group['params']]
        return super().step(closure)


@pytest.mark.parametrize('timing', ['natural', 'balanced'])
@pytest.mark.parametrize('chunk_size', [1, 2, 4])
def test_deferred_episode_metrics_weights_and_gradients_match_baseline(tmp_path, timing, chunk_size):
    manifest, _ = fixtures.synthetic_dataset(tmp_path / 'data')
    dataset = ReplaySequenceDataset(manifest, tmp_path / 'scratch')
    episodes = dataset.episodes({0: 'train', 1: 'validation', 2: 'heldout'}, 'train')
    try:
        episode = next(episodes)
        adapter = CrForgeObservationAdapter(dataset.schema, vocabulary_size=16)
        baseline, deferred = fixtures.TinyStatePolicy(), fixtures.TinyStatePolicy()
        original_optimizer = CaptureSGD(baseline.parameters(), lr=.05)
        deferred_optimizer = CaptureSGD(deferred.parameters(), lr=.05)
        options = dict(chunk_size=chunk_size, sampled_action_prior=6 / 11, timing_objective=timing,
                       max_grad_norm=1e6)
        a = run_state_episode(baseline, episode, adapter, optimizer=original_optimizer, **options)
        b = run_state_episode(deferred, episode, adapter, optimizer=deferred_optimizer, defer_metrics=True, **options)
        for key in a.keys() - {'seconds', 'rss_mib'}:
            assert b[key] == pytest.approx(a[key], abs=2e-7), key
        for left, right in zip(original_optimizer.captured, deferred_optimizer.captured):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(baseline.weight, deferred.weight, rtol=0, atol=0)
        assert baseline.seen == deferred.seen and baseline.generation == deferred.generation == 1
    finally:
        episodes.close()


def test_deferred_metrics_remove_per_chunk_scalar_float_transfers(tmp_path, monkeypatch):
    manifest, _ = fixtures.synthetic_dataset(tmp_path / 'data')
    dataset = ReplaySequenceDataset(manifest, tmp_path / 'scratch')
    episodes = dataset.episodes({0: 'train'}, 'train')
    original_float = torch.Tensor.__float__
    calls = []
    def observed_float(value):
        calls.append(value.numel())
        return original_float(value)
    monkeypatch.setattr(torch.Tensor, '__float__', observed_float)
    try:
        episode = next(episodes)
        adapter = CrForgeObservationAdapter(dataset.schema, vocabulary_size=16)
        run_state_episode(fixtures.TinyStatePolicy(), episode, adapter, chunk_size=2, sampled_action_prior=6 / 11)
        old = len(calls)
        calls.clear()
        report = run_state_episode(fixtures.TinyStatePolicy(), episode, adapter, chunk_size=2,
            sampled_action_prior=6 / 11, defer_metrics=True)
        assert old >= 20 and len(calls) == 0
        assert report['rows'] == 5 and report['play_rows'] == 2
    finally:
        episodes.close()


@pytest.mark.parametrize('defer', [False, True])
def test_metric_optimization_keeps_forward_finite_guard(tmp_path, defer):
    manifest, _ = fixtures.synthetic_dataset(tmp_path / 'data')
    dataset = ReplaySequenceDataset(manifest, tmp_path / 'scratch')
    episodes = dataset.episodes({0: 'train'}, 'train')
    class BadPolicy(fixtures.TinyStatePolicy):
        def forward_state(self, *args, **kwargs):
            logp, value, state = super().forward_state(*args, **kwargs)
            return logp * float('nan'), value, state
    try:
        episode = next(episodes)
        with pytest.raises(FloatingPointError, match='policy output'):
            run_state_episode(BadPolicy(), episode,
                CrForgeObservationAdapter(dataset.schema, vocabulary_size=16),
                chunk_size=2, sampled_action_prior=6 / 11, defer_metrics=defer)
    finally:
        episodes.close()


def test_profile_is_opt_in_and_reports_cpu_component_boundaries(tmp_path):
    manifest, _ = fixtures.synthetic_dataset(tmp_path / 'data')
    dataset = ReplaySequenceDataset(manifest, tmp_path / 'scratch')
    episodes = dataset.episodes({0: 'train'}, 'train')
    try:
        episode = next(episodes)
        model = fixtures.TinyStatePolicy()
        profiler = EpisodeProfiler('cpu')
        result = run_state_episode(model, episode,
            CrForgeObservationAdapter(dataset.schema, vocabulary_size=16),
            optimizer=CaptureSGD(model.parameters(), lr=0), chunk_size=2,
            sampled_action_prior=6 / 11, defer_metrics=True, profile=profiler)
        timing = profiler.resolve()
        assert set(timing) >= {'adapter_h2d', 'forward_loss', 'backward', 'scalar_metrics', 'optimizer', 'final_metrics'}
        assert timing['forward_loss']['calls'] == 3 and timing['final_metrics']['calls'] == 1
        assert all(item['cpu_wall_seconds'] >= 0 and item['cuda_interval_seconds'] is None for item in timing.values())
        assert result['optimizer_stepped']
    finally:
        episodes.close()


def test_profile_candidate_does_not_mutate_shared_source_adam_step(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/profile_state_bc.py'
    spec = importlib.util.spec_from_file_location('speed_profile_cli_test', path)
    profiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiler)
    source = torch.nn.Linear(2, 2)
    old_optimizer = torch.optim.AdamW(source.parameters(), lr=3e-5)
    source(torch.ones(1, 2)).sum().backward()
    old_optimizer.step()
    checkpoint = {'optimizer': old_optimizer.state_dict(), 'scaler': None}
    before = {key: values['step'].clone() for key, values in checkpoint['optimizer']['state'].items()}
    monkeypatch.setattr(profiler, 'state_model_from_checkpoint', lambda state, device: copy.deepcopy(source))
    args = SimpleNamespace(seed=123, device='cpu', precision='fp32')
    model, optimizer, _ = profiler.build_case(checkpoint, {}, profiler.CASES['baseline'], args)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    for key, expected in before.items():
        torch.testing.assert_close(checkpoint['optimizer']['state'][key]['step'], expected, rtol=0, atol=0)
    assert all(group['lr'] == 0 for group in optimizer.param_groups)
