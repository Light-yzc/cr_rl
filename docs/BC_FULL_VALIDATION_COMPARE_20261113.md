# BC full validation comparison — 2026-11-13

All three frozen models were evaluated on the same 426 validation games, both seats: 852 episodes and 273,230 supervised rows. The previous full BC checkpoint has the same dataset fingerprint and assigns all 426 games to validation; none overlap its training partition.

| Model | Loss ↓ | Timing loss ↓ | Conditional slot loss ↓ | Position loss ↓ | Slot accuracy ↑ | Tile distance ↓ | Act recall ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous full BC (`experiment_v04_16`, step 9887) | 6.1766 | 0.4141 | 1.1501 | 4.6124 | 43.47% | 6.8726 | 6.63% |
| Current stage `best.pt` | 5.7022 | 0.4003 | 1.0932 | 4.2088 | 49.00% | 5.2669 | 11.40% |
| Current stage `latest.pt` loaded for this evaluation | 7.3175 | 0.4819 | 1.7241 | 5.1114 | 43.75% | 5.4852 | 18.52% |

Latest versus best: loss +28.3%, position loss +21.4%, conditional slot loss +57.7%, slot accuracy -5.25 percentage points, tile distance +4.14%. Act recall improves by 7.13 percentage points. This is deterioration in held-out imitation quality, while action recall improves; these measurements do not establish a decline in battle win rate.

The new placement stage did improve over the previous full BC at its best checkpoint. Continuing this training configuration no longer improves held-out likelihood. Keep training paused; preserve both latest and best before any restart. Best should be the next comparison baseline rather than automatically selecting latest. Do not infer a unique cause (capacity, label noise, confidence miscalibration or distribution shift) from this comparison alone.

Protocol: chronological recurrent teacher-forced evaluation; no optimizer; FP16 autocast with the model's existing FP32 branches; chunk size 32, frame microbatch 512; equal weight per episode-seat, matching the historical validation aggregation. All models use the current stage's objective settings and its saved sampled action prior (0.05115493846). This prior differs slightly from the validation pool's natural prior (0.05153533653), so historical sampled-validation numbers are reference values rather than an exactly identical protocol.

Operational correction: initial Colab kernel IDLE status was incorrectly reported as training paused. Background training was subsequently found active. SIGSTOP was then sent to the supervisor, controller and trainer (PIDs 13213, 13214, 13269); trainer state T was verified. Evaluation weights were already loaded into independent GPU model instances and were not updated by background training. The evaluated latest snapshot is the one loaded at evaluation start, not necessarily the final checkpoint subsequently written by the background process. No automatic training resume was requested or performed.

Evidence:

- `artifacts/latest_bc_local_20261112/full_validation_compare.jsonl`
- `artifacts/latest_bc_local_20261112/previous_full_split_audit.json`
- Evaluation scripts in the same artifact directory: `full_valid_remote.py`, `eval_previous_full.py`.
