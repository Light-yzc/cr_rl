# Colab GPU validation

Historical architecture-v1 PPO/forward smoke record. For the current architecture-v2 Linux self-play acceptance, commands and measured results, see [LINUX_TRAINING.md](LINUX_TRAINING.md). The current launcher runs multi-update self-play and checkpoint-resume validation instead of the legacy PPO/production-forward test described below.

Date: 2026-11-02. Runtime: Google Colab L4, CUDA enabled, approximately 22 GiB VRAM.

## Verified

- The local `colab` CLI authenticated and provisioned an L4 runtime.
- Uploaded the project source archive through `colab upload`.
- Installed OpenJDK 21, cloned crforge at `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`, built the JVM environment and generated game tables 16.635.3.
- Installed the project and simulator Python packages in the Colab runtime.
- `cr_rl.cli doctor` completed successfully against the Colab simulator installation.
- Ran a 128-step PPO rollout with 2 vector environments / 4 seats, 2 optimization epochs, on CUDA. One update processed 512 seat decisions in 11.24 seconds on the tiny 770,374-parameter preset. The reported joint policy entropy was 0.268 and approximate KL was 0.000066, confirming that this longer smoke included nontrivial legal action choices and an optimizer update. This is a pipeline smoke, not evidence of a stronger policy.
- Instantiated the production model on L4: 119,148,550 trainable parameters. A forward pass with batch 1, sequence length 8, and 200 entity slots returned action logits `(1, 2307)` and value `(1,)`; measured peak allocated GPU memory was approximately 0.485 GiB for that inference-only test.
- Downloaded PPO checkpoints and summary artifacts through `colab download`.

## Caveats

- The Colab run is intentionally short and uses the tiny model for PPO optimization. The production-sized model was checked for instantiation and forward compatibility only, not optimizer-step memory or sustained throughput.
- The simulator run did not reach terminal match outcomes; mean reward was zero. Do not interpret the smoke metrics as a win-rate result.
- Colab runtimes are ephemeral. The current launcher downloads artifacts to the local ignored `artifacts/` directory. For long runs, add Google Drive persistence or periodic remote artifact downloads before scaling up.
- The first CLI attempt used the default 30-second execution timeout and lost the session reply after the training script completed. The launcher now uses a 900-second timeout. The successful run used a fresh L4 session.
