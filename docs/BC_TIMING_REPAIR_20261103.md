# BC timing repair — 2026-11-03

> 最新执行记录：[全周期 BC 校准与32局正式换边评估](CYCLE_BC_CALIBRATION_20261103.md)。已完成数据源切换、自然校准和原生成功支付审计；对 early237 为8–8，对 conditional896 为11–5，PPO 交接尚未通过，BC 已恢复。下方旧交接中的 cmd 计数、184秒加时判断、统一18–28张/局目标均需按最新证据修正。

The user approved three changes after the step-1280 policy lost all four GUI
games. Native replay reconstruction showed 45 successful plays for that policy
versus 104 for the early policy. In game 3 it spent about 42.5 seconds at full
elixir while WAIT and a legal card play were available; its last successful play
preceded the loss by 39.2 seconds. Conditional placement improvement alone did
not establish better play.

## Implemented objective and sampling

`hierarchical_bc_loss(..., conditional_card_loss=True)` derives three terms from
the existing joint action distribution, preserving the model and PPO interfaces:

1. Balanced ACT/WAIT timing cross-entropy.
2. Conditional choice among four cards and two ability actions, averaged only
   over ACT labels. This term receives no natural-prior importance multiplier.
3. Conditional placement loss on PLAY labels, with legal Gaussian targets.

New training uses `--bc-objective conditional --no-card-prior-correction`.
It records timing and conditional card losses separately. The tile coefficient
is 1.0, compared with 0.2 in the previous run, so total loss and weighted
`global_loss` are **not numerically comparable across those two objectives**.
Compare component NLLs, decision metrics and paired complete games instead.

`--wait-margin-before 4 --wait-margin-after 2` excludes WAIT supervision within
1 second before and 0.5 seconds after an ACT event, within the same replay and
seat. It does not remove observations from histories, positive labels, or WAIT
labels from validation. Exclusion uses physical ticks, not concatenated row
indices. The initial production pilot excluded 60,180 WAIT candidates.

## Omission budget and audit

`--max-omissions 2` is supported by the importer and shard builder. Both the
reconstruction loop and prefix cutoff must use this budget. The third missed
command cuts both trajectories at its decision window. A five-tick neighborhood
of earlier misses is excluded from supervision for both seats. Strict budget 0
is retained for comparison; allowing two misses does not guarantee state fidelity.

The initial implementation changed the cutoff but missed an existing early
`break` in the replay loop. Matched-command audit exposed that error: only
5,949 versus 5,935 commands were retained on 2,036 matched records. That data
producer was stopped and its partial output marked superseded/failed. Its
training run `121m_conditional_bc_20261103` is a two-change pilot, not evidence
of expanded data. Its step-128 checkpoint was preserved before switching.

Corrected data lives in `bc_omission2_20261103_r2`. Records explicitly mark
`omission_budget_enforced_in_loop`; the staged trainer rejects budgeted records
without this marker. A real-backend audit of the first 176 reconstructed games
verified 164 third-omission cutoffs and continuation after the first miss. On
96 games also present in the strict data, retained commands increased from
381 to 1,199 (3.147x). This is a matched small-batch command count, not a claim
that the full dataset or legal training labels have already grown threefold.

## New production workflow

Run: `121m_conditional_bc_20261103_r2`.
Initial stage W&B: `https://wandb.ai/weidemingdeguge-test/cr-rl/runs/2pqh5so5`.

- Initialize from the preserved conditional pilot step 128, including optimizer
  state. That pilot itself initialized from the original v2 epoch-5 weights with
  a fresh optimizer when the BC objective changed.
- Prefer corrected expanded replays; retain all available strict records as
  fallback until their replacements are reconstructed. Freeze each dataset
  stage; incorporate new data only at stage boundaries.
- First stage: 6,037 raw records, 343 duplicate records excluded, 5,694 supervised
  games. Old recognized match identities retain their original train/validation/
  held-out assignment through `split_registry.json`.
- Batch 16, accumulation 6, fused optimizer, backbone LR 3e-5, tile LR 3e-4.
- Every 32 optimizer steps: fixed small-cohort validation. Every 128 steps and
  epoch boundary: verified Drive checkpoint. Data, logs, source and initial
  checkpoint are cloud-backed; large artifacts remain on Colab/Drive.
- While the source producer builds the full candidate pool, train one pass per
  frozen stage. Train up to eight epochs on the final complete snapshot.

The full candidate pool is still being reconstructed. The original automatic
PPO handoff was stopped. This workflow publishes readiness for evaluation; it
does not automatically promote a model based only on imitation loss.

## Verification and remaining work

The unit suite passed 76 tests, including conditional-gradient robustness under
WAIT imbalance, forced-WAIT finite gradients, omission boundaries and WAIT-margin
isolation across seats/matches. Real-data tiny training completed eight optimizer
steps, saved checkpoints and emitted quick validation under the new objective.
Corrected omission-loop behavior is additionally verified by the real replay
audit described above.

Still required: observe production convergence on expanded data, calibrate the
deployment WAIT bias using complete-game experiments, and compare full-elixir
waiting, play counts and paired outcomes against the frozen baselines. Balanced
training is not by itself a guarantee against either underplaying or overplaying.

## First complete-game timing calibration

The stage-1 corrected-data checkpoint reached optimizer step 444 (including its
128-step conditional pilot parent). Stage 2 then continued training on an expanded
snapshot: 6,415 supervised games, 1,731,842 total rows, 20,306 training card plays
and 138 training abilities. Snapshot counts are not final corpus counts.

`evaluate_policy.py --wait-logit-bias` adjusts only the loaded model's WAIT bias;
source checkpoints remain untouched. A new regression test passed on Linux and
checks that the bias shifts WAIT/ACT log odds exactly, preserves all conditional
ACT probabilities and other parameters, and rejects non-finite bias values.

The same step-444 checkpoint played the frozen epoch-1 BC actor on one fixed
held-out deck pairing with both seat assignments:

| WAIT logit bias | Wins-losses | Successful plays per game | Full-elixir WAIT with a legal card |
| --- | --- | --- | --- |
| 0 | 2-0 | 30, 28 | 0 s, 0 s |
| +2.5 | 1-1 | 29, 21 | 0 s, 0 s |

All four matches completed without simulator faults, truncation or illegal
actions. Native-replay reconstruction supplied play counts and full-elixir WAIT
durations. These four games are diagnostic only: they do not establish stable
strength or select a globally optimal bias. In particular, a fixed +2.5 correction
is not supported by this small experiment. The default remains unchanged.

Evaluation temporarily suspended the live training process and resumed it in a
`finally` block; its running process state was verified afterwards. Reports,
actor export and original replays were copied to Drive under
`runs/121m_conditional_bc_20261103_r2/calibration`. Only 16 small reports/replay
files were downloaded locally. The local GUI archive is
`artifacts/v2_conditional_calibration_20261103/all_matches.jsonl`; its first two
matches show bias 0 and its last two bias +2.5, with model identities in the
panel/window titles.

## Expanded paired evaluation and stage transition

A later step-896 checkpoint was evaluated with three held-out deck pairings and
both policy-seat assignments (six complete games per opponent):

| Opponent | Wins-losses | Invalid games |
| --- | --- | --- |
| Conditional BC step 444 | 3-3 | 0 |
| Early BC step 237 | 2-4 | 0 |

Native replay audits found zero seconds of full-elixir WAIT while a legal card
play existed in all 12 games. New-model play counts were 26-45 against step 444
and 18-45 against step 237. This supports resolution of the observed full-elixir
stall on this cohort, but does **not** establish improved multi-deck strength.
The earlier two-game 2-0 result was too small to generalize.

The evaluation actor, individual original replays, labeled GUI archive, results
and action audits are backed up under the run's `expanded_evaluation` directory.
Only small reports/replays were downloaded to the local workspace. The default
GUI launcher now selects the combined 12-game step-896 archive.

Two additional engineering fixes passed the complete Linux unit suite (80 tests):

- Automatic inference caches now hold all parallel rolling windows plus their
  next frames. Six 128-frame lanes need more than the former fixed capacity of
  512; the regression test verifies exactly six newly encoded frames per shift.
  Explicit user capacity limits are preserved. No GPU speed multiplier is claimed
  from the two different-opponent evaluation timings.
- Each frozen training stage persists its epoch target. Producer completion can
  no longer change that target during resume or let an unchanged final manifest
  skip the requested final epochs. The controller was replaced only after stage 2
  completed and its step-915 final checkpoint was verified in Drive.

Stage 3 resumed that optimizer on 5,328 training games, 32,864 card-play labels
and 138 ability labels. The full staged snapshot has 2,902,770 rows; these remain
partial-corpus counts while source processing continues. Its W&B run is
https://wandb.ai/weidemingdeguge-test/cr-rl/runs/uvf789tk.

A partial-source coverage audit of 5,154 reconstructed games also found an
important remaining limitation: among truncated games, median cutoffs were
29.0 seconds for Cochon and 31.75 seconds for IL_Replay. Respectively 48 and 252
truncated games reached at least 120 seconds; 50 and 110 games had no cutoff.
Allowing two omissions increased label availability but has not made the corpus
uniformly cover late-game play. Source input supplies no verified opening hand;
the importer currently uses a seeded simulator deal. Initial-hand mismatch is a
specific remaining reconstruction hypothesis to investigate, rather than assuming
all early truncations result from small combat-mechanic differences.


## Root-Cause Diagnosis of Step-896 Degradation (Why 3-3 vs Step 444 & 2-4 vs Step 237)

Cross-referencing `artifacts/v2_conditional_expanded_eval_20261103/action_audit.json`,
the paired `evaluation.json` files, and `stage_0003/bc.baseline.json` (step 915)
identified three concrete reasons why step 896 did not outperform step 444 or
early step 237:

1. **ACT/WAIT Prior Over-Correction at Deployment (`wait_logit_bias = 0.0` under 1:1 Balanced Sampling)**:
   - Under `--bc-objective conditional --no-card-prior-correction`, the training
     sampler feeds roughly 50% `ACT` and 50% `WAIT` rows (`label_nonwait_fraction = 0.52`),
     whereas natural 4 Hz trajectories contain only `2.93%` `ACT` and `97.07%` `WAIT`.
   - As optimizer steps increased from 444 to 896–915 without prior offset, the
     model's `ACT/WAIT` head converged toward the 50:50 training prior: on
     `bc_quick_natural`, `policy_nonwait_probability` reached `42%–50%` per 250ms tick,
     `wait_accuracy` fell to `43.1%`, and `expected_false_actions_per_wait_second`
     rose to `1.93`.
   - In closed-loop matches (`match_002`, `match_004`), step 896 executed **45 card plays
     in 184 seconds** (1 play every 4.08s vs 24–25 plays for `early_step237`),
     dumping affordable low-cost cards immediately instead of banking elixir for
     high-cost core units or reactive defense.

2. **Early-Cutoff Training Bias in Stages 1–2 (Pre-`opening_cycle` Corpus)**:
   - Step 896 was trained on `bc_omission2_20261103_r2` (without `--infer-opening-cycle`),
     where median truncated game length was only **29.0s–31.75s** (`~7.57` valid plays/replay).
   - Meanwhile, all 12 evaluation games ran **130s–184s** (`tick 2601–3681`, deep into
     Double/Triple Elixir overtime). Training longer on first-30s opening prefixes
     overfitted opening tempo while degrading late-game defensive pacing.

3. **Deck-Matchup Dominance on Pairs 0 & 1**:
   - In all 8 games across Pairs 0 and 1 (against both `conditional_step444` and
     `early_step237`), the winning side was 100% determined by deck seat (`seat 0`
     won all 4 games of Pair 0; `seat 1` won all 4 games of Pair 1), yielding a
     mandatory `2-2` split. The entire `3-3` vs `2-4` difference came down to a
     single 1-crown timeout tiebreaker (`tick 3681`, `184.05s`) in Pair 2 (`match_005`),
     where step 896's over-playing (45 plays vs 25) lost elixir tempo.

---

## Execution Plan for Next Agent (Full-Cycle Data Switch + Prior Calibration)

### Task 1: Promote Opening-Cycle Reconstructed Shards (`bc_cycle_20261103`) to Primary BC Source
- **Context**: `src/cr_rl/data/opening_cycle.py` (`--infer-opening-cycle`) is already
  validated (`opening_cycle_trial_20261103` increased plays 5.34x; `opening_cycle_il_trial_20261103`
  increased plays 6.20x). On Colab (`crrl-v2-20261103`), producer `bc_cycle_20261103`
  is live and yielding **~37.35 valid plays/replay** and **~1,038 rows/replay** (vs
  7.57 plays/replay in `bc_omission2_20261103_r2`).
- **Actions**:
  1. Stop or finish the superseded non-cycle producer (`bc_omission2_20261103_r2`, PID 71141)
     so `bc_cycle_20261103` gets full CPU throughput for `crforge` reconstruction.
  2. Update `scripts/run_all_bc.py` validation guard to accept `infer_opening_cycle: true`
     shards alongside `bounded_omission_prefix` (`max_omissions <= 2` and `omission_budget_enforced_in_loop`).
  3. Switch the staged BC controller (`jobs.json` / `sources` list in `run_all_bc.py`) at the
     clean Stage 3 checkpoint boundary so Stage 4+ freezes `bc_cycle_20261103` first (highest
     priority in `freeze_datasets(..., deduplicate=True)`), falling back to `bc_omission2_20261103_r2`
     only for replay signatures not yet reconstructed by `bc_cycle_20261103`.
  4. Verify in the frozen Stage 4+ metadata that `canonical_game_signature` deduplication
     replaces short non-cycle prefixes with full-match `bc_cycle_20261103` trajectories,
     and log the median cutoff tick / late-game (`>= 120s`) replay count.

### Task 2: Calibrate Deployment & PPO-Init `WAIT` Logit Bias (Fix Over-Playing)
- **Context**: Training with 1:1 balanced `ACT/WAIT` (`p_train ≈ 0.50`) while evaluating
  in a `p_nat ≈ 0.029` environment with `wait_logit_bias = 0.0` causes 45 plays/game
  (spamming cards every 4s as soon as elixir hits 2–3).
- **Actions**:
  1. Implement an offline natural-validation calibration helper (or extend `evaluate_policy.py`
     / `train_bc.py`) that computes on `bc_quick_natural` (or full natural validation rows)
     the exact `wait_logit_bias` $\Delta b_{\text{wait}}$ such that:
     - Mean predicted non-wait probability matches the empirical human rate (`~2.9%–3.5%` per tick), OR
     - `expected_false_actions_per_wait_second` is bounded within `[0.15, 0.35]` while preserving
       `play_recall` on action windows and `0.0s` full-elixir stall.
  2. Sweep `wait_logit_bias` across a fine grid (e.g. `0.0, +1.0, +1.5, +2.0, +2.5, +3.0`)
     in closed-loop paired games on the full-cycle checkpoint, auditing:
     - `successful_plays` per game (target human-like `18–28` plays per 3-minute game, not `45` spam plays and not `<10` stall plays),
     - `full_elixir_wait_legal_seconds` (must remain near `0.0s`),
     - Average elixir at card play (verify the policy can save 5–7 elixir for core tanks/Sparky/win-conditions),
     - Win-loss-draw across paired held-out decks.

### Task 3: Expand Paired Multi-Deck Evaluation Cohort & Handoff to v2 PPO
- **Context**: A 3-pair (6-game) cohort is dominated by deck matchup counters on Pairs 0 and 1.
- **Actions**:
  1. Expand closed-loop evaluation to at least **8–10 held-out deck pairings (16–20 games with
     seat swap)**, reporting both overall W-L-D and **seat-neutral pair outcomes** (pairs where
     one model wins both seats `2-0` vs `1-1` deck-counter splits).
  2. Once a `bc_cycle_20261103`-trained checkpoint + calibrated `wait_logit_bias` demonstrates:
     - Normal play cadence (`18–28` plays/game, `~0s` full-elixir stall),
     - Improved late-game (`>120s`) coverage and validation metrics (`play_slot_accuracy`, `play_tile_within_2`),
     - Positive seat-neutral score against both `early_step237` and `conditional_step896`,
     export the calibrated actor as the official initialization checkpoint for **v2 PPO Self-Play**
     (with Independent Critic warmup, v2 zero-sum tower HP + crown + overflow reward shaping,
     and $\gamma=0.9995, \lambda=0.98$).
