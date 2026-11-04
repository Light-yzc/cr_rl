"""Reproduce the dated L4 Critic benchmark with training paused at update 2."""

import gc
import json
import os
import time
from pathlib import Path

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import HistoryValueNetwork, ModelConfig
from cr_rl.training.episode_trace import EpisodeTrace
from cr_rl.training.inference import history_tensors
from cr_rl.training.prepared_history import PreparedHistoryBatches
from cr_rl.training.separate import autocast_for, forward_inputs

r = Path("/content/cr-rl-lab/artifacts/121m_ppo_pilot_20261104")
torch.set_num_threads(4)
torch.set_float32_matmul_precision("highest")
state = torch.load(r / "ppo_latest.pt", map_location="cpu", weights_only=False, mmap=True)
assert state["updates"] == 2
critic = HistoryValueNetwork(ModelConfig(**state["critic_config"])).cuda()
critic.load_state_dict(state["critic"])
critic.train()
env = CrForgeVector(1, 1, independent=True)
adapter = CrForgeObservationAdapter(env.vector.schema, env.vector.vocabulary_size, entity_bucket=16)
z = dict(np.load(r / "rollouts/rollout_000001.npz"))
traces = {}
refs = []
for t in range(len(z["actions"])):
    for lane in range(2):
        key = lane, int(z["episode_ids"][t, lane].reshape(-1)[0])
        trace = traces.setdefault(key, EpisodeTrace(len(traces), 1))
        pos = trace.append(z["obs"][t, lane], z["masks"][t, lane], z["training_rewards"][t, lane])
        for seat in range(2):
            refs.append((trace, pos, seat))
T = critic.cfg.context_len


def legacy(indices):
    windows = []
    times = []
    legal = []
    for index in indices:
        trace, p, seat = refs[int(index)]
        w, t = trace.window(p, seat, T)
        windows.append(w)
        times.append(t)
        legal.append(trace.masks[p][seat])
    return history_tensors(np.stack(windows), np.stack(times), np.stack(legal), adapter, "cuda")


prepared = PreparedHistoryBatches(refs, T, adapter, "cuda")
print("PREPARED", json.dumps(prepared.metadata), flush=True)
indices = np.random.default_rng(17).choice(len(refs), 64, replace=False)
# Exact input equality is independent of model precision and microbatch size.
for a, b in zip(legacy(indices[:4]), prepared(indices[:4]), strict=True):
    torch.testing.assert_close(a, b, atol=0, rtol=0)
results = []
for name, batch in [
    ("legacy", 4),
    ("prepared", 4),
    ("prepared", 8),
    ("prepared", 16),
    ("prepared", 32),
    ("prepared", 64),
]:
    fn = legacy if name == "legacy" else prepared
    try:
        timings = []
        preps = []
        for _repeat in range(3):
            critic.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            prep = 0
            for start in range(0, 64, batch):
                p = time.perf_counter()
                tensors = fn(indices[start : start + batch])
                prep += time.perf_counter() - p
                with autocast_for("cuda"):
                    predictions = forward_inputs(critic, tensors).float()
                    loss = 0.5 * predictions.square().sum() / 64
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite benchmark")
                loss.backward()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - began)
            preps.append(prep)
        row = dict(
            mode=name,
            microbatch=batch,
            effective_batch=64,
            seconds_per_64=float(np.median(timings[1:])),
            input_seconds_per_64=float(np.median(preps[1:])),
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    except torch.cuda.OutOfMemoryError:
        critic.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"mode": name, "microbatch": batch, "oom": True}), flush=True)
        break
report = {
    "source_update": state["updates"],
    "device": torch.cuda.get_device_name(),
    "cpu_threads": torch.get_num_threads(),
    "prepared": prepared.metadata,
    "results": results,
    "input_equivalence": "bit exact for representative real histories",
    "note": "Critic forward/backward benchmark; not end-to-end rollout or Actor speed",
}
(r / "efficiency_benchmark.json").write_text(json.dumps(report, indent=2))
print("BENCHMARK_DONE", flush=True)
env.close()
