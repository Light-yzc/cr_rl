# Colab longrun cache recovery, 2026-11-17

The single remaining runtime is `cr-train-l4-1117` (L4). CPU preparation and the surplus GPU runtime were stopped after data staging. Training uses `/content/crrl_training/runs/longrun`, with Drive backup under `cr_rl_ppo/training/longrun`. W&B is disabled.

## Recovery and loss of unsaved updates

The previous trainer completed U949 but stalled collecting the next update. Python stack inspection found the main thread waiting in `IndependentBattles.step_lanes`, with one native battle lane active indefinitely. SIGINT could not complete while the native call remained blocked. An attempted debugger-assisted checkpoint rescue failed; the process was subsequently terminated. U931–U949 from that process were not saved and must not be reported as preserved.

The durable checkpoint was inspected on Colab with CPU mmap and confirmed to contain **U930**. The new controller PID is 49100 and trainer PID is 49101. They resumed that checkpoint and started collecting U931. The original U929 input checkpoint was not overwritten.

## Cache correction

Both `--opponent-cache-size` and `--training-opponent-cache-size` are now 4, previously 1. The historical pool can require multiple opponent models in the same rollout. Cache size 1 repeatedly evicted and deserialized them: old U949 spent 503 seconds in rollout model work, versus roughly 53–74 seconds in ordinary rounds. This diagnosis does not establish that the cache caused the native simulator hang.

The shared command builder, prepared local and Drive config, and durable code ZIP were updated. The local command-construction test passed. New launch arguments were read back and confirmed as 4/4. Early resumed sampling showed approximately 10 GiB VRAM usage and GPU utilization 77–86%; these are instantaneous observations, not a completed throughput comparison.

## Preserved pipeline

- Target update: 10000; 8 environments, frame microbatch 64, actor/critic batch size 2.
- AWBC: 80 labeled games staged on local SSD; phase interval 50 updates.
- BC: 2842 non-held-out candidate games (2552 train / 290 validation); phase interval 100 updates. These are experimental filtered candidates, not certified expert demonstrations.
- Reward mode remains uniform. The new elixir reward is not enabled by this restart.
- Checkpoint interval remains 25, with a checkpoint on the first resumed update.

The native simulator hang is not fixed by the cache correction. A subsequent recurrence requires backend diagnosis; no native fix or automatic hang recovery is claimed here.
