# Linux training and CUDA acceptance

Date: 2026-11-02. The primary training entry is `scripts/train_selfplay.py`.

## Measured acceptance result

The final clean-session run passed on NVIDIA L4, Linux 6.6.159 / glibc 2.39, Python 3.13.16, PyTorch 2.11.0+cu130 and Java 21. Evidence is in `artifacts/linux_final/linux_validation/validation.json`, with the source archive and downloaded result archive in `artifacts/linux_final/`.

| Run | Learner samples | Optimizer steps | Normal terminal games | Nonzero learner rewards | Seconds | Peak allocated CUDA GiB |
|---|---:|---:|---:|---:|---:|---:|
| Tiny update 1 | 2600 | 82 | 2 | 2 | 57.38 | 0.206 |
| Tiny update 2 | 2600 | 82 | 2 | 2 | 62.19 | 0.209 |
| Tiny relocated resume, update 3 | 64 | 2 | 0 | 0 | 3.72 | 0.206 |
| General update 1, full 128-frame history | 132 | 33 | 0 | 0 | 21.21 | 1.270 |
| General update 2, full 128-frame history | 132 | 33 | 0 | 0 | 18.96 | 1.272 |

All five saved rollouts passed the independent audits; all samples were valid, with no simulator faults or time-limit truncations in these runs. Initial recomputed action log probabilities differed by at most 5.96e-8. The test suite passed 16/16, Ruff passed, and both shell launchers passed `bash -n` on Linux. The general run's full-history coverage was checked from the actual optimization minibatches, not inferred from the configured context length. These timings include rollout, optimization and checkpoint/trajectory writes but exclude runtime provisioning, simulator setup, compression and transfer.

## Setup and commands

Install Git, Java 21 and uv, then run:

```bash
bash scripts/bootstrap_linux.sh
uv run python scripts/validate_training.py --device cuda
```

The bootstrap uses Python 3.12 and `uv.lock`, builds the simulator revision and tables specified by `configs/backends.yaml`, and runs the simulator doctor. It can discover the Linux JDK through the resolved `java` executable without requiring `JAVA_HOME`. Use `bash` to invoke shell scripts after transferring source from Windows.

Colab already supplies a CUDA-enabled PyTorch environment. From Windows, Linux or macOS:

```bash
python scripts/colab_launch.py --gpu L4 --out artifacts/my_linux_acceptance
# T4 is an alternative allocation option; its performance has not been measured here.
```

This provisions a fresh runtime and runs the same acceptance suite after installing the pinned JVM simulator. The Colab worker runs in a separate process and writes a completion status file. The launcher checks that file and the downloaded validation report, since a successful Colab CLI exit alone does not establish that a Python cell succeeded. It downloads logs, checkpoints and trajectories, then releases the runtime. Logs from failed training are also archived when the results directory exists. An authentication or installation failure remains visible in the job log.

Result bundles are split into 16 MiB chunks and checked against per-chunk SHA-256 hashes before assembly and extraction. The launcher uses an available OpenSSH executable through `colab ssh --proxy-mode`, or up to four Colab contents-API requests when SSH is unavailable. SSH downloads are serial: this runtime rejects a second active SSH connection with HTTP 429. Each download has a timeout and up to three attempts. Verified archive-prefix chunks can be retained across interrupted transfers. The final acceptance bundle is 652,729,165 bytes; large contents-API responses stalled during retrieval, so the completed retrieval uses checksummed chunks through the CLI's SSH bridge.

## Acceptance requirements

`scripts/validate_training.py` performs:

1. Unit tests and real-simulator doctor checks, including Linux discovery with `JAVA_HOME` unset.
2. Two consecutive tiny updates: two environments, 1300 decisions per environment, 2600 learner samples per update. The suite requires normal terminal games and nonzero learner rewards.
3. A move of the whole run directory, followed by an explicit checkpoint resume against historical opponents. It verifies increasing optimizer steps, changed model weights, and update numbers 1, 2, 3.
4. Two updates of the 34,168,454-parameter `general` model, using 132 decisions, microbatch 1 and gradient accumulation 4. It asserts that optimization actually sees a populated 128-frame history. The acceptance run raises the KL stopping threshold to 100 to cover every minibatch; this is a coverage test, not the recommended training setting. Normal training retains the default threshold 0.02.
5. Independent NPZ audits of both seats' legal actions, finite behavior values/log probabilities, exactly one learner seat, valid bootstrap targets, terminal/truncation separation, zero-sum rewards, and episode boundaries after resets.

Each training update reports the initial recomputed action-log-probability error, nonzero learner rewards, maximum history length used in optimization, optimizer steps and peak allocated CUDA memory. Loss and gradient non-finiteness abort training.

## Repairs made for Linux/CUDA

- Linux no longer falls through to macOS `/usr/libexec/java_home`.
- BF16 value predictions are converted to FP32 before NumPy conversion. Action distributions use FP32 in both collection and optimization.
- CUDA RNG byte tensors are moved back to CPU before restoration; loading the complete checkpoint with `map_location='cuda'` previously broke update 2.
- Opponent pool files store local checkpoint paths relative to the run directory, so transferring the run does not retain obsolete machine-specific paths.
- Resume checks the actual simulator Git revision and the game-data version/hash. Existing output checkpoints require explicit `--resume` rather than accidental overwrite.

## Resuming and scaling

```bash
uv run python scripts/train_selfplay.py --model general --device cuda \
  --envs 2 --steps 1536 --epochs 2 --batch-size 1 --grad-accumulation 8 \
  --updates 100 --out artifacts/general_run
uv run python scripts/train_selfplay.py --model general --device cuda \
  --envs 2 --steps 1536 --epochs 2 --batch-size 1 --grad-accumulation 8 \
  --updates 100 --out artifacts/general_run --resume artifacts/general_run/ppo_latest.pt
```

Keep the whole output directory: optimizer checkpoint, opponent pool, opponent snapshots, metrics and rollouts. Resume restores training state; simulator episodes restart. Environments also restart at update boundaries, so a very short rollout can miss all terminal rewards. Snapshots and rollout files currently accumulate once per update, and each selected historical opponent is copied to the GPU. Check disk space and memory as the population grows. Long Colab runs still need periodic external backups; the acceptance launcher retrieves results only at job completion.

The current CUDA acceptance uses the PyTorch version preinstalled on Colab rather than replacing it with the newer version in `uv.lock`. Treat results as specific to the recorded runtime. The separate locked Python 3.12 environment is not covered by those measurements. T4 runs use FP32 because this trainer only enables mixed precision on BF16-capable devices.

Passing acceptance establishes a working training pipeline. It does not establish improved win rate, complete live-game mechanic parity, or a long-run performance baseline. Full-game terminal validation uses tiny; the general run tests history and optimizer compatibility. Frozen evaluation, pool-size limits and long-run general-model learning remain follow-up work.
