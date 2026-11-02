# Self-play algorithm and model design

Date: 2026-11-02. This is the implementation decision record for the general multi-deck agent.

## 1. Problem formulation

Treat a match as a two-player, approximately zero-sum, partially observable stochastic game. Each side acts every five 50 ms simulator ticks. The actor must use only its seat-relative observation; training may use both seats' trajectories and other privileged fields for critic diagnostics, but must not leak the opponent's private hand/elixir into the actor.

The objective is not merely to beat the latest checkpoint. Decks and tactics are non-transitive, so the training target is a robust population strategy that performs against a mixture of old policies, current policies, scripted archetypes and held-out opponents.

### Why the new trainer uses a frozen opponent

The previous `train_ppo.py` sampled both seats with the same shared policy and optimized both seats' own rewards in one batch. Shared weights do **not** imply that the two sampled policy-gradient terms always cancel: the two seats generally have different histories and actions. Exact cancellation is possible in special symmetric samples, but it is not a universal property of parameter sharing. The more general issue is that simultaneous updates change both sides of the game while collecting data, coupling the two player objectives and making it harder to interpret an update as a unilateral response improvement.

The new runner therefore samples one learner seat per match, freezes the opposing policy for the entire rollout/update, and optimizes only the learner seat's actions. This is an explicit approximate-best-response step against a fixed league member, not a claim that shared-policy self-play is mathematically impossible. Learner seat 0/1 is randomized to preserve side symmetry.

## 2. Literature and design choices

| Work | Transferable idea | Decision for this project |
|---|---|---|
| Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, Nature 2019, https://doi.org/10.1038/s41586-019-1724-z | League of frozen competitors, PFSP matchmaking, exploiters, replay-derived initialization and continual payoff evaluation address strategy cycles and forgetting. | Use a frozen checkpoint pool and PFSP first. Add a dedicated exploiter only after the main loop and payoff matrix work. |
| Lanctot et al., *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning*, 2017, https://arxiv.org/abs/1711.00832 | PSRO alternates empirical-game meta-strategy computation with approximate best-response learning; independent RL can overfit to the current opponent. | Maintain a payoff matrix and use regret-matching meta strategies as an evaluation/selection tool. |
| Balduzzi et al., *Open-ended learning in symmetric zero-sum games*, ICML 2019, https://proceedings.mlr.press/v97/balduzzi19a.html | Non-transitive games need population-aware objectives; rectified Nash response uses game-theoretic niching. | Never rank policies only by latest-checkpoint win rate. Retain diverse strategic niches. |
| McAleer et al., *Pipeline PSRO*, NeurIPS 2020, https://proceedings.neurips.cc/paper/2020/hash/e9bcd1b063077573285ae1a41025f5dc-Abstract.html | Parallel best-response workers can expand a policy population without serializing every PSRO iteration. | Future scale-out: parallelize independent response workers after the single-worker algorithm is correct. |
| Huang et al., *A Robust and Opponent-Aware League Training Method for StarCraft II*, NeurIPS 2023, https://papers.nips.cc/paper_files/paper/2023/hash/94796017d01c5a171bdac520c199d9ed-Abstract-Conference.html | Goal-conditioned exploiters and opponent modeling improve league coverage. | Add opponent archetype conditioning / exploiters as a second-stage ablation, not as a first implementation dependency. |
| Fan & Farina, *GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning*, 2026, https://arxiv.org/abs/2605.19235 | GAE carries variance from sampled future actions in stochastic equilibrium play. Q-boosting replaces sampled future TD residuals with centralized-Q Expected-SARSA(λ) expectations; VRPO retains PPO's clipped actor objective. | Keep masked GAE as the correctness baseline. Add a centralized Q critic and Q-boosting only after stable rollouts and a trusted trajectory schema exist. |
| Sokota et al., *Scalable decision-making for games of imperfect information*, Nature 2026, https://doi.org/10.1038/s41586-026-11036-y | Ataraxos combines self-play, a belief model and search; dynamically damps learning by coordinating regularization and update size; uses action-value filtering in its move learner. | Adopt scheduled entropy regularization, PPO clipping, gradient clipping, explicit KL diagnostics and later hidden-information belief modelling. Do not copy Stratego's phase-specific architecture or search blindly. |
| Schulman et al., *Proximal Policy Optimization Algorithms*, 2017, https://arxiv.org/abs/1707.06347 | On-policy rollouts, clipped ratios and multiple minibatch epochs provide a simple trust-region-like update. | PPO remains the initial actor optimizer and the baseline for Q-boosting ablations. |

### Important nuance: VRPO is a second-stage upgrade

For a centralized action-value critic, the expected state value is

\[
V_i(s)=\sum_{a'}\pi_i(a'\mid o')Q_i(s,a'),
\]

and the Expected-SARSA residual is

\[
\delta^+_{i,t}=r_{i,t}+\gamma V_i(s_{t+1})-Q_i(s_t,a_t).
\]

Q-boosting constructs an advantage from \(Q_i(s_t,a_t)-V_i(s_t)\) plus a discounted \((\gamma\lambda)\)-trace of these residuals. It removes future sampled-action noise when the Q critic is accurate, but replaces it with critic approximation error. The 2,307-action codec is too large to add a naive dense centralized Q head before validating factorization and memory. First establish a clean on-policy baseline; then compare GAE, one-step Q-boosting and multi-step Q-boosting under equal compute.

## 3. Opponent sampling and population objective

The first league implementation stores immutable policy snapshots and uses prioritized fictitious self-play:

\[
p_j \propto (1-\widehat w_j)^\alpha,
\qquad p=(1-\eta)\,p_{\rm PFSP}+\eta\,\mathrm{Uniform}.
\]

Here \(\widehat w_j\) is the learner's smoothed score against opponent \(j\), \(\alpha\) controls hardness emphasis and \(\eta\) preserves a nonzero chance of revisiting every niche. A result is scored from the learner's perspective: win +1, draw 0, loss -1. The current implementation uses a Beta-smoothed score and a 10% uniform floor.

For evaluation, estimate the payoff matrix \(A_{ij}\) using role-swapped games and common random seeds. The included regret-matching solver returns average row/column meta-strategies and an approximate exploitability diagnostic. Do not confuse PFSP training probabilities with a Nash equilibrium: PFSP is a practical curriculum; the payoff-matrix meta strategy is the game-theoretic evaluation layer.

## 4. Trajectory schema v3

Each time slice preserves explicit `[time, environment, seat]` axes and stores:

- pre-step and post-step observations and legal masks for both seats;
- both requested actions, both action log-probabilities and both value estimates;
- raw rewards for both seats and separately stored training rewards, plus environment-level `terminated`, `truncated` and `done`;
- `valid` and `learner_mask`, so a simulator fault can invalidate a match without corrupting other environments;
- `policy_ids`, `opponent_ids`, `policy_versions`, `episode_ids`, episode-start markers, episode step/tick, and the actual episode seed;
- explicit winner, end-reason and simulator-fault codes, distinguishing a completed match, time-limit truncation and engine fault;
- whether the submitted action was consumed, action/run refusal codes, final bootstrap values and their validity masks;
- run metadata: simulator commit, game-data version/hash, schema, deck IDs, card/tower levels and model config.

The learner mask marks exactly one seat per match. The other seat's transition is retained for audits and future centralized-critic training, but is not treated as on-policy actor data. A truncated time-limit transition bootstraps from the final pre-reset observation and stops the GAE trace. A simulator fault invalidates that match's samples. Autoreset is disabled in the trainer; the final observation is captured before `reset_env` and the two seat histories are cleared.

## 5. Model capacity decision

| Preset | Entity encoder | Temporal trunk | Exact parameters | Role |
|---|---|---|---:|---|
| `small` | 2x128 | 4x256 | 4.40M | Existing small baseline |
| `compact` | 2x192 | 8x384 | 16.60M | Capacity lower bound |
| `general` | 2x192 | 10x512 | 34.17M | Recommended starting model |
| `large` | 2x192 | 14x512 | 46.78M | Scale-up ablation |
| `production` | 4x256 | 16x768 | 119.54M | Retain as an upper-bound experiment, not default |

Recommendation: start with **34M parameters**, compare 16.60M and 46.78M at matched environment decisions and wall-clock. The earlier 119M design has only passed forward compatibility; no evidence shows it improves game strength. The 34M model provides substantially more cross-deck capacity than the 4.40M baseline while preserving room on a 22 GiB L4 for optimizer states, frozen opponents, a frozen behaviour/reference copy, sequence activations and rollout batches. Exact feasible batch size must still be measured with backward/optimizer steps; parameter count alone is not a memory benchmark.

The 128-frame context is 32 seconds at 4 Hz. Long-term deck-cycle knowledge remains in the structured public card history. If tests show tactical memory beyond this window matters, compare a recurrent state or compressed memory before simply increasing context length and quadratic attention cost.

The actor now also receives categorical embeddings for its own eight-card deck and the publicly observed enemy-card set, in addition to the four hand-slot embeddings. This avoids relying solely on normalized numeric card IDs for cross-deck identity. The enemy embedding only uses cards already exposed in the acting player's observation; it is not a hidden-deck leak.

## 6. Implemented now and staged next

Implemented in this revision:

1. `scripts/train_selfplay.py`: one learner seat versus a frozen sampled opponent, randomized learner side, multiple validated deck archetypes, explicit terminal/truncation handling, fault filtering, exact-window PPO recomputation and reference-policy KL regularization.
2. `src/cr_rl/opponents/league.py`: PFSP sampling, smoothed score updates, checkpoint metadata and zero-sum regret-matching meta solver.
3. `TrajectoryBuffer` schema v3: two-seat data with policy provenance, learner masks, raw/training rewards, refusal diagnostics and bootstrap values. Legacy collectors are explicitly marked schema v1 / incomplete provenance rather than being silently presented as PPO-ready data.
4. General model presets and Colab/local smoke coverage are being added to the validation log.

Next research/code gates:

- evaluate full matches and fill the cross-play payoff matrix before claiming a stronger policy;
- add a separate exploiter response worker and opponent-conditioned deck/task sampling;
- add a centralized Q critic and compare VRPO/Q-boosting with GAE;
- add a hidden-card belief head only after validating that it improves held-out opponent/deck prediction and decision quality;
- benchmark production 128-frame backward/optimizer-step memory and throughput before considering 119M.
