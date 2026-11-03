# 119M versus 309M general-model capacity probe

Date: 2026-11-03. Recommendation for the first formal learning baseline: use the existing `production` preset (119,541,766 parameters); retain `general_300m` (309,233,670 parameters) as a direct capacity comparison. Both fit and update on L4. There is no measured policy-strength advantage for either size yet.

## Measured results

NVIDIA L4, PyTorch 2.11.0+cu130, FP32 weights with BF16 autocast, full 128-frame histories, 200 entity slots. Three models remain on the GPU: trainable learner, frozen behavior policy and one frozen opponent. Each case uses two warmup iterations and eight timed iterations with CUDA synchronization. The common window comes from a previously saved real simulator trajectory, stays within one episode and ends with 577 legal actions. PPO uses synthetic advantages and value targets to exercise forward, backward, gradient clipping and AdamW; it is not an online learning run.

| Preset | Parameters | Microbatch | Two-seat policy forwards, mean ms | Update, mean ms | Update samples/s | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `production` | 119,541,766 | 1 | 52.29 | 155.42 | 6.43 | 3.76 | 4.15 |
| `general_300m` | 309,233,670 | 1 | 63.22 | 231.28 | 4.32 | 7.98 | 8.51 |
| `production` | 119,541,766 | 4 | 52.70 | 400.70 | 9.98 | 6.75 | 7.19 |
| `general_300m` | 309,233,670 | 4 | 62.23 | 484.80 | 8.25 | 10.98 | 11.65 |

Both models produced finite losses and gradients, changed weights and completed optimizer steps in both microbatch settings. Unit tests passed 16/16. Raw evidence is in `artifacts/capacity_20261103/capacity_results.json` and the four individual result JSON files. The initial WAIT-only exploratory probes are excluded from this table.

At microbatch 4, the 309M model takes about 21% longer per optimizer step and uses about 4.22 GiB more allocated memory. Its parameter count is 2.59 times larger. It is therefore a feasible experiment on L4, rather than an architecture limited to forward-only tests. At microbatch 1, its update penalty is larger, approximately 49%.

## Selection decision and limits

The first formal 119M baseline leaves more room for extra frozen historical opponents, environment batching and repeated seeds, with higher measured update throughput. The next comparison should train 119M and 309M on the same deck distribution and held-out opponents, comparing both equal sample budgets and equal wall-clock budgets. Compare role-swapped win rates and per-deck outcomes before committing a long run to 309M. This decision does not set 119M as the final capacity ceiling; a 309M main model and later 90–120M distilled model remain options if the larger policy earns its cost.

These numbers exclude simulator stepping, CPU observation adaptation, host/device transfers, rollout storage and checkpoint writes. The two-seat timing measures one forward per frozen policy, on batch 1. Update throughput measures replay-window optimization with resident GPU inputs and a reference-policy forward; it is not simulator samples per second or complete PPO-update throughput. Peak allocation is the PyTorch allocator measurement, not total driver memory. Larger opponent populations consume additional GPU memory. This L4 BF16 result does not establish T4 or V100 performance. For the mixed-precision distinction, see the [official PyTorch AMP documentation](https://docs.pytorch.org/docs/2.15/amp.html): the FP16 training path normally needs gradient scaling; the existing trainer uses BF16 only on supported GPUs and otherwise FP32.

## Model configuration and commands

Both candidates share a 4-layer, width-256 entity encoder. `production` uses a 16-layer, width-768 temporal trunk with 12 heads; `general_300m` uses 24 layers, width 1024 and 16 heads. Both keep a 128-frame window. The new preset is available to the self-play, BC, legacy PPO and model-check entry points. Existing defaults and checkpoints remain unchanged.

After Linux bootstrap, start a fresh baseline run with:

```bash
uv run python scripts/train_selfplay.py --model production --device cuda \
  --envs 2 --steps 1536 --epochs 2 --batch-size 4 --grad-accumulation 4 \
  --updates 10 --out artifacts/general_119m
```

For the 309M comparison, use `--model general_300m` and a separate output directory. A 34M checkpoint cannot be resumed as either larger model; architecture dimensions differ. Do not initialize the new runs from the previous smoke checkpoints.

To reproduce the capacity probe on a prepared CUDA machine with an existing rollout:

```bash
uv run python scripts/benchmark_capacity.py --model general_300m --batch-size 4 \
  --rollout artifacts/linux_final/linux_validation/tiny_resumed/rollouts/rollout_000001.npz \
  --out artifacts/capacity_309m_b4.json
```
