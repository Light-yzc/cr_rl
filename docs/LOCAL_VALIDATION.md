# Local validation log

Date: 2026-11-02. Machine: Apple M4 MacBook Air, 10 CPU cores, 16 GB unified memory, macOS ARM64.

## Environment setup

- Installed Homebrew OpenJDK 21.0.13 (kept alongside the existing Java 17).
- Created `.venv` with Python 3.12.16 and installed PyTorch 2.15.0, NumPy 2.5.4, JPype 1.7.1, Gymnasium 1.4.0 and PettingZoo 1.27.0.
- Cloned crforge at `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`.
- Built `:env:installDist` and game tables 16.635.3 from the explicitly selected official asset CDN.
- Confirmed table content SHA `dae6c6cc7e9126338da0c606e47e48a9ef80462c`.
- Confirmed all eight configured card rows exist. Sparky's internal card row is `ZapMachine` (its table image is `image/chr/sparky_dl.png`); using the display label `Sparky` as an internal row name fails deck validation.
- Confirmed actual costs from the pinned tables: Giant 5, ZapMachine 6, MiniPekka 4, Musketeer 4, Minions 3, Knight 3, Zap 2, Arrows 3. Total 30, mean 3.75.
- Simulator schema: v4, 224 header ints, 34 ints/entity, 200 entity slots, 2 seats, 2307 actions (WAIT + 4 x 18 x 32 tile actions + 2 ability actions).

## Smoke results

- `cr_rl.cli doctor`: passed; JVM loaded, reset accepted the deck, observation shape `(1, 2, 7024)`, mask shape `(1, 2, 2307)`.
- `cr_rl.cli smoke --episodes 4`: passed 40 WAIT decisions per environment; no simulator faults.
- Random-legal-action smoke: 4 mirrored-deck environments x 300 decisions, no refusal/engine/environment faults. The rollout actually deployed all eight configured cards, including `ZapMachine` (Sparky) once, Giant once, and the other six cards multiple times. This validates basic card deployment/cycle integration, not the detailed Sparky-vs-counter mechanic suite.
- Trajectory collection: saved 128 vector decisions x 4 environments x 2 seats = 1,024 seat transitions to `artifacts/random_rollout.npz` (643 KB compressed) with a JSON sidecar containing simulator/schema/vocabulary metadata; no simulator faults. The local artifact is git-ignored by design.
- Tiny model forward on MPS: passed; 663,945 trainable parameters; logits `(2, 8, 2307)`, values `(2, 8)`.
- Production model instantiation/parameter count: 119,541,766 trainable parameters after adding explicit deck/public-seen-card conditioning, matching the requested ~120M target. This is a parameter-count check, not a production training benchmark.
- Headless simulator throughput (200 WAIT decisions per environment, 8 JVM worker threads maximum): 1 env 3,040 env-decisions/s; 4 envs 12,048/s; 8 envs 11,242/s; 16 envs 19,039/s; 32 envs 27,246/s. No simulator faults. These are short, WAIT-only engine measurements, not policy-in-the-loop training throughput; 32 envs is the best measured point so far, so test 32 and 64 again with actual action diversity and a policy forward pass before choosing rollout parallelism.
- `pytest`: 6 tests passed; Ruff checks passed.

## Still not validated

- No BC training on human trajectories yet.
- No PPO learning curve or self-play strength claim.
- No dedicated Sparky scenario suite has been run against official-client reference traces. The current engine has Sparky's `ZapMachine` row and relevant charge/load/recoil tests, but the full plan's matchups still require a dedicated acceptance pass.
- Throughput figures above are WAIT-only microbenchmarks; policy-in-the-loop throughput remains unmeasured.
- No client memory reader, Android emulator, or real-device action executor is implemented.
