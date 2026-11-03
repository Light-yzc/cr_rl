# Tower-potential reward trial

The initial experimental potential is:

`Phi = .15 * (own princess HP fractions - enemy princess HP fractions)`
`    + .20 * (own king HP fraction - enemy king HP fraction)`
`    + .15 * (own crowns - enemy crowns)`

Fractions are clipped to [0, 1], with zero for a missing/destroyed tower whose
maximum HP is zero. The potential is antisymmetric between the two seat views.
Training receives `raw_terminal_reward + gamma * Phi(next) - Phi(current)`.
True terminal states have zero next potential. Time-limit truncations retain the
final observation's potential and bootstrap its shaped value; simulator faults
receive no shaping and their episode samples remain invalidated. Computation
uses the copied final observation, never the newly reset arena.

Raw +/-1/0 results remain separate for opponent-pool records, win rates and saved
trajectory `rewards`. The `training_rewards` array, GAE and independent critic
targets use the shaped reward. Checkpoints and trajectory metadata record the
reward configuration. Resume rejects a changed configuration or gamma because
the saved critic was trained on a different target.

Enable a fresh experiment with:

```bash
python scripts/train_selfplay.py --model production --init-from-bc BC.pt \
  --critic-mode independent --tower-shaping --gamma 0.9995 --gae-lambda 0.98 \
  --batch-size 4 --grad-accumulation 16 --out artifacts/ppo_tower_trial
```

This is an experimental invocation, not an instruction to skip critic warmup or
the remaining PPO readiness checks. The Colab wrapper forwards the same options.
Legacy runs retain sparse rewards and gamma=.999/lambda=.95 by default.

`--overflow-penalty .001` is an explicit optional experiment. It penalizes WAIT
at >=9.9 elixir per decision. Its default is zero: it is neither potential-based
nor zero-sum and changes the objective. It is inactive without `--tower-shaping`.
No automatic reward-weight annealing is implemented; changing a state potential
over training time requires further care and changes the critic target.

For gamma=.9995/lambda=.98, 1/(1-gamma*lambda) is about 48.8 decisions (12.2s),
whereas the exponential half-life is about 33.5 decisions (8.4s). These describe
the decay of TD residuals in GAE, not a hard credit-assignment horizon. Accurate
bootstrapping can carry information over much longer intervals.

With zero terminal potential, the discounted shaping return telescopes to
`-Phi(initial)` for a complete episode. Intermediate gains are cancelled at the
terminal boundary; a losing policy cannot keep a positive HP bonus to outweigh
the loss. This differs from adding permanent .30-per-tower bonuses. With a
perfect critic, this potential transformation leaves the exact advantage
unchanged. Faster learning or better win rates are hypotheses, not guarantees.
Do not interpret the shaped value as a win probability.

Theory: Ng, Harada and Russell, *Policy invariance under reward transformations*
(1999), https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf.

The first real-simulator acceptance completed a 737-decision game with nonzero
HP and crown potentials. Seat rewards remained antisymmetric and the discounted
shaping sum cancelled to within 8e-9. The CPU PPO acceptance exposed an existing
left-padding inference NaN in the history Transformer; dummy queries now have a
visible dummy key while valid queries retain the original causal/padding mask.
No parameters or checkpoint keys change. A regression checks finite inference,
invariance to padded input values, train/eval agreement and finite gradients.

Acceptance completed on Colab Linux with the tiny model on CPU (not a 119M
strength experiment): 53 unit tests passed; a 1,536-step rollout completed two
fault-free games, kept the actor frozen, and performed 21 critic optimizer steps.
Restoring that checkpoint then completed one actor and one critic optimizer step
on a new 32-step rollout. Recomputed shaped rewards matched the saved trajectory
arrays. Reports, trajectories, logs, checkpoints and implementation files were
copied with verification to
`MyDrive/cr_rl/runs/reward_shaping_acceptance_20261103`.
