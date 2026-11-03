# BC and PPO audit follow-up

The matched 8,192-draw placement trial finished from parent optimizer step 2,176.
Parent, dataset, training order and validation-row hashes matched. The combined
residual/FP32/placement-LR fix was selected and the choice was verified on Drive.

| Same held-out trial rows | Control | Fixed |
| --- | ---: | ---: |
| Hard tile CE | 5.5589 | 5.2576 |
| Exact tile accuracy | 0.208% | 3.750% |
| Tile L1 distance | 16.446 | 13.115 |
| Within 2 tiles | 3.542% | 12.083% |
| Conditional card-slot accuracy | 41.042% | 40.625% |

These are imitation metrics. The fixed decoder's own initial L1 was 12.046, so
the final L1 improves over the control, not over its own initial state. Neither
this trial nor a small head-overfit experiment proves better battle win rate.

Full BC now continues under `119m_all_bc_tilefix_20261103`, using a new prefix-safe
dataset. Each replay is physically cut before the first omitted-command decision
window; old broad suffix data is archived and excluded by default. Canonical
replay deduplication is enabled for new stage snapshots, with whole-replay stable
train/validation membership. Prefixes cannot guarantee matching human states:
levels, simulation version and game-mode mechanics can still differ.

The new training configuration enables shared deterministic XY Fourier features
for entities and tile queries, legal-mask Gaussian tile supervision (sigma=.75),
separate ability sampling, and importance weights for natural WAIT/play/ability
priors. Natural-distribution validation is reported alongside balanced action
validation. A marginal action rate close to the human rate does not establish
correct action timing; WAIT argmax accuracy alone is especially misleading.

Live inspection on 2026-11-03 found stage 4 training with all three placement
flags enabled and distinct 3e-5 backbone / 3e-4 placement learning rates. The
producer had published 3,248 usable prefixes (526,222 frame rows, 8,958 valid
plays). These counts are an intermediate snapshot, not completion of the 11,592
candidate replay pool or of the upstream repositories.

The independent PPO optimizer's KL guard uses sample-weighted accumulated-batch
statistics, and the Colab command now actually requests accumulation 16 at
microbatch 4 (effective batch 64), matching its recorded configuration. An actor
KL stop does not stop the independent critic's separate fitting loop.

Tower-potential shaping is an opt-in experiment; see
[the exact formula and terminal handling](REWARD_SHAPING_20261103.md).

Remaining before claiming a fully repaired formal PPO pipeline: persistent
simulator episodes across update boundaries, partial-episode warmup handling,
opponent snapshot admission/capacity/EMA, discrete entity embeddings and their
checkpoint migration, same-policy two-seat trajectory reuse, and measured frame
encoder caching. No formal PPO strength result is claimed by these BC changes.
