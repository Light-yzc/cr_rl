"""Compare V100 Critic microbatches on identical recorded 128-frame histories.

This is an isolated performance/numeric probe, never a published training update.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import HistoryValueNetwork, ModelConfig
from cr_rl.training.fp16 import backward_loss, configure_optimizer_amp
from cr_rl.training.inference import history_tensors
from cr_rl.training.separate import _optimizer_step, autocast_for, forward_inputs, move_inputs

os.environ["CRRL_AMP_DTYPE"] = "fp16"
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]
run = root / "artifacts/local_dual_v100"
output = root / "artifacts/local_dual_v100_checks"
state = torch.load(run / "ppo_latest.pt", map_location="cpu", mmap=True, weights_only=False)
cfg = ModelConfig(**state["critic_config"])
env = CrForgeVector(1, 1, independent=True)
adapter = CrForgeObservationAdapter(env.vector.schema, env.vector.vocabulary_size, entity_bucket=16)
rollout = np.load(run / "rollouts" / f"rollout_{state['updates']:06d}.npz")
obs, masks = rollout["obs"], rollout["masks"]
eligible = rollout["learner_mask"].astype(bool) & rollout["valid"].astype(bool)
episode_steps = rollout["episode_steps"]
entity_index = adapter.schema.header["entity_count"][0]
rows = []
for t, lane, seat in np.argwhere(eligible):
    if t >= cfg.context_len - 1 and episode_steps[t, lane] >= cfg.context_len - 1:
        rows.append((int(obs[t, lane, seat, entity_index]), int(t), int(lane), int(seat)))
selected = sorted(rows, reverse=True)[:64]
assert len(selected) == 64
windows = np.stack(
    [obs[t - cfg.context_len + 1 : t + 1, lane, seat] for _, t, lane, seat in selected]
)
legal = np.stack([masks[t, lane, seat] for _, t, lane, seat in selected])
tensors = history_tensors(windows, np.ones((64, cfg.context_len), bool), legal, adapter, "cuda:0")
old = rollout["old_values"]
targets = torch.as_tensor(
    [old[t, lane, seat] + 0.01 for _, t, lane, seat in selected], device="cuda:1"
)
env.close()
model = HistoryValueNetwork(cfg).to("cuda:1")
results = []
baseline = None
for batch in (8, 16):
    model.load_state_dict(state["critic"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(state["critic_optimizer"])
    configure_optimizer_amp(optimizer, "cuda:1", state.get("critic_amp_scaler"))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    model.train()
    # Warm kernels and the scaler before timing each candidate.
    warm = move_inputs(tuple(t[:batch] for t in tensors), "cuda:1")
    with autocast_for("cuda:1"):
        warm_value = forward_inputs(model, warm).float()
        warm_loss = 0.5 * (warm_value - targets[:batch]).square().mean()
    backward_loss(warm_loss * batch, optimizer)
    optimizer.zero_grad(set_to_none=True)
    del warm, warm_value, warm_loss
    torch.cuda.synchronize(1)
    torch.cuda.reset_peak_memory_stats(1)
    began = time.perf_counter()
    try:
        model.train()
        for _ in range(2):
            for left in range(0, 64, batch):
                inputs = move_inputs(tuple(t[left : left + batch] for t in tensors), "cuda:1")
                with autocast_for("cuda:1"):
                    predictions = forward_inputs(model, inputs).float()
                    loss = 0.5 * (predictions - targets[left : left + batch]).square().mean()
                backward_loss(loss * len(predictions), optimizer)
            _optimizer_step(model, optimizer, 64, 0.5)
        torch.cuda.synchronize(1)
        elapsed = time.perf_counter() - began
        trained = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if baseline is None:
            baseline = trained
        maximum = max(float((trained[key] - baseline[key]).abs().max()) for key in trained)
        result = {
            "batch": batch,
            "accumulation": 64 // batch,
            "seconds": elapsed,
            "samples_per_second": 128 / elapsed,
            "max_parameter_diff_vs_8": maximum,
            "peak_critic_gib": torch.cuda.max_memory_allocated(1) / 2**30,
            "numeric_gate": maximum <= 1e-5,
        }
    except torch.cuda.OutOfMemoryError:
        result = {"batch": batch, "error": "CUDA OOM"}
        optimizer.zero_grad(set_to_none=True)
    results.append(result)
    print(json.dumps(result), flush=True)
    if "error" in result:
        break
report = {
    "source_update": state["updates"],
    "history": cfg.context_len,
    "max_current_entities": max(r[0] for r in selected),
    "effective_batch": 64,
    "results": results,
    "scope": "Isolated identical-input Critic probe; not a PPO update",
}
(output / "critic_microbatches.json").write_text(json.dumps(report, indent=2))
