# Colab lifetime and PPO metric interpretation

Updated 2026-11-03. Run: `119m_20261103_pilot01`, W&B ID `av8ajonn`.

Final verification: five updates, 15,360 learner transitions, 17 normal terminal games, six learner wins and eleven losses, 23 optimizer steps. All saved trajectories passed independent audits. Five checkpoint generations were published to Drive; every file referenced by the final manifest passed size/SHA-256 checks and the latest model/optimizer state loaded successfully. W&B server history contains the new learning metrics and its Files listing includes all five diagnostic files. The actual update runtime totaled approximately 1197 seconds, excluding setup and Drive backups. This short test does not verify survival past four hours or playing strength. The user's next formal training requirement is BC first; these randomly initialized PPO checkpoints are pipeline-test outputs.

## What was repaired

The initial W&B logger emitted every progress number as a chart. The first rollout lasts several minutes before its first PPO update, so these charts looked busy without answering whether the policy was learning. `TrainingTelemetry` now separates progress metadata from actual learning observations. Sampling metrics are emitted every 64 decisions, episode results at each game boundary, and optimizer diagnostics after gradient-accumulation steps. A manual workspace groups useful charts and explicitly sets learner-sample or optimizer-step axes. Old data can recover action rates, episode outcomes and sampled surprisal from saved trajectories; historical full-distribution entropy and gradient norms cannot be reconstructed from those files and are not fabricated.

The first three updates finished using the old trainer process. The logging supervisor was replaced while retaining their checkpoints, optimizer and run ID; the fourth and fifth updates use the new instrumentation. All metric code has unit coverage and a real CUDA smoke. Original trainer defaults were retained.

## Colab lifetime

The [official FAQ](https://research.google.com/colaboratory/faq.html) distinguishes idle reclamation from maximum VM lifetime, and states that resource availability and limits vary. Pro+ advertises up to 24 hours of continuous execution with sufficient compute units, not a universal 24-hour guarantee for all plans. A browser disconnect alone does not establish VM loss. A Drive `Transport endpoint is not connected` error may be a storage-mount failure rather than the GPU VM being deleted.

The [current CLI session documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md) states that the keep-alive daemon was removed and liveness is maintained through active kernel executions/interactions. Older indexed CLI pages still describe a daemon, so they cannot be used as evidence of current behavior. Runtime proxy tokens are refreshed for new commands, but long-lived connections do not refresh their initial authentication mid-connection. Credential/transport failure and VM reclamation are different conditions.

This trial uses a bounded foreground kernel task to monitor real training progress, completed checkpoints and failures. The heavy trainer remains a detached process to survive a client-side connection loss. The monitor stops on job completion/failure and does not create fake mouse clicks or dummy GPU work. This reduces exposure to idle classification but does not guarantee runtime survival. No four-hour survival claim has been verified in this session.

Durability is independent of liveness:

- Train on local VM disk; copy logs about once per minute.
- After each complete update, copy checkpoint, every referenced opponent snapshot, raw two-seat trajectories and pool metadata to `MyDrive/cr_rl/runs/119m_20261103_pilot01/`.
- Verify file sizes and SHA-256 readbacks before publishing the new `latest.json` generation.
- Retain older completed checkpoint generations so interrupted publication cannot replace the last complete generation.
- Restore a backup into a fresh local directory and load the real optimizer/model state to test recoverability. This passed for this trial.

A forced VM deletion can still lose the current uncompleted update. Full simulator-tick state is not saved; restored training starts new games. Automatic reallocation after VM loss is not implemented, and a new VM may require new Drive authorization. More predictable multi-day runs need a compute service with appropriate resource guarantees; they require explicit storage integration instead of assuming Colab's Drive mount is available.

## Metric definitions

`selfplay/episode_return`: terminal learner reward, +1 win, -1 loss, 0 draw. Mean return averages completed valid games; it is approximately win fraction minus loss fraction. `selfplay/win_rate` counts wins divided by completed valid games, including draws in the denominator. Faulted/time-limit games are excluded. The population and deck matchups change, so this is a training-opponent diagnostic, not held-out playing strength. Small numbers of games are highly noisy.

`policy/play_fraction`: fraction of learner decisions whose action is not WAIT. `policy/wait_when_play_legal`: fraction selecting WAIT among decisions with at least one legal card-placement action. The latter distinguishes excessive waiting from forced waiting. Neither metric should be maximized blindly: preserving elixir is sometimes correct. Interpret persistent high WAIT alongside episode outcomes, deck pairing and replay behavior.

`policy/entropy`: mean of `-sum(p(a) log p(a))` over the masked joint action distribution, in nats. Only legal actions carry probability. `entropy_when_play_legal` conditions on card play being possible; forced-WAIT states have zero entropy and depress the unconditional average. More legal tiles naturally allow more entropy. High entropy means uncertainty, not strong play; a rapid collapse before good game outcomes is concerning. `max_action_probability` approaches one for highly concentrated policies. Historical `sampled_surprisal = mean(-log p(sampled_action))` is a sample-based entropy estimator, not the exact entropy of every saved policy distribution.

`value/advantage_*`: statistics before advantage standardization. The GAE TD residual is `delta = reward + gamma * next_value - value`, and the advantage is its exponentially weighted backward accumulation, with gamma 0.999 and lambda 0.95. Terminal states do not bootstrap; time-limit truncation bootstraps from the final observation but stops the trace at that episode boundary. Actor advantages are standardized before PPO, so normalized advantage mean near zero is expected and does not prove zero learning signal.

`value/return_*`: statistics of the GAE target `R = A + V_old`. This is a bootstrapped critic target and may be nonzero before a game finishes; it is not measured game reward. `value/explained_variance = 1 - Var(R - V_old) / Var(R)` measures how much target variation the behavior critic explains. One is ideal, zero means no variance explained, negative is worse. Near-constant target variance makes the ratio undefined/unstable; this implementation omits it in that case. It is measured before the current optimization, not an end-of-update held-out score.

`ppo/policy_loss`: with `r = exp(logp_new - logp_old)`, this implementation minimizes `mean(max(-A*r, -A*clip(r, 0.85, 1.15)))`. It favors actions with positive advantages while restricting harmful ratio changes. The sign and magnitude depend on the current batch and advantages; a more negative actor loss across different rollouts does not directly imply a stronger policy.

`ppo/value_loss = 0.5 * mean((V_new - R)^2)`. This is critic fitting error. Lower is useful only alongside target variance and explained variance: constantly predicting zero in sparse-reward states can produce a small loss without knowing how to win.

`ppo/approx_kl = mean((r - 1) - log(r))` estimates old-to-new policy change on the sampled actions. `ppo/reference_kl = mean(sum(p_new * (logp_new - logp_behavior)))` measures the full masked reverse KL on recomputed minibatch windows; it is also a regularization term. They have different directions/estimators and need not match. The frozen reference is this rollout's behavior policy, not the opposing policy or a permanently fixed teacher. The current early-stop threshold is sampled approximate KL 0.02, checked per microbatch. A tiny minibatch can produce a noisy spike; the per-update average can be below the threshold even though one batch triggered stopping. Fewer optimizer steps than requested may therefore be intentional.

`ppo/clip_fraction`: fraction with `abs(r - 1) > 0.15`. Zero at initialization is normal. Persistently high clipping plus KL spikes suggests the update is moving too far or the data are poorly matched; persistent near-zero clipping is not by itself proof of healthy learning.

`ppo/total_loss = policy_loss + 0.5*value_loss - entropy_coef*entropy + reference_kl_coef*reference_kl`. Initial coefficients are entropy 0.005 and reference KL 0.01; both decay with update count as `(1 + previous_updates)^(-0.25)`. Because these weights and targets change, the total is an optimization objective, not a globally comparable performance score.

`ppo/grad_norm` is the total gradient L2 norm after microbatch-accumulation averaging and before clipping. The threshold is 0.5. Raw values above 0.5 indicate clipping is active, not that clipped gradients exceed the limit. Non-finite losses or gradients abort the trainer. Compare spikes with KL, critic errors and terminal events. `learning_rate` records the actual optimizer setting (currently 2e-5).

`performance/rollout_samples_per_second` uses collected learner transitions divided by rollout elapsed time, with one learner seat per environment. It is distinct from replay optimizer throughput. Update time includes simulator sampling, optimization and local artifact writes but excludes Drive publication. `peak_cuda_gib` is PyTorch peak allocated memory for the update, not total VRAM reported by the driver.

Learner-sample axes count decisions collected; PPO axes count optimizer steps, not microbatches or W&B logging calls. The current effective optimization batch is 4 microbatch samples times accumulation 4 = 16, except a shorter final/early-stop group. Self-play metrics use current training opponents; fixed held-out, role-swapped evaluation remains necessary before claiming learned strength.

The objective and KL early-stop rationale are consistent with the [PPO documentation and reference implementation](https://spinningup.openai.com/en/latest/algorithms/ppo.html); exact coefficients, clipping conventions and timing above are taken from this repository's implementation.
