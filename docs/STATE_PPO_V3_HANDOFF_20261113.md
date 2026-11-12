# State PPO v3 handoff

## 接手先读：当前暂停，目标未完成

用户目标是「把整个 PPO infra 都改成适配 v3」，明确禁止把 v3 权重转换为 v2。2026-11-13 用户要求停下给另一个项目释放资源，目标状态已设为 paused；随后仅要求写交接文档。不要自动恢复训练、启动 GPU 或继续跑实验，等用户恢复任务。

此前关于“基础设施已接通”的汇报只代表入口和短 smoke 有产物，不代表生产 PPO 已完成。下文列出尚未解决的正确性问题。**不能直接按现有默认 1024/1536 steps 开长训。**

## 本地文件和环境

- 仓库：`G:/Code/cr_rl`，分支 `codex/training-game-budget`。改动已本地 commit，未确认 push；接手以 `git status` 和文件为准。
- Python：`F:/yzc/Miniconda/envs/cs285/python.exe`，已有 Torch/CUDA，不要重装或下载 Torch。
- 本地 GPU：RTX 4090；本地 RAM 16GB，所有实验、缓存和下载放 G 盘，避免一次加载完整数据集/优化器副本。之后迁移 Linux V100，需要 FP16，不可依赖 BF16。
- Java 21：`F:/JDK`；模拟器 Python：`.cache/vendor/crforge/python`；JAR 已在 `.cache/vendor/crforge/env/build/install/env/lib`。
- `PYTHONPATH` 需包含 `src` 和模拟器 Python 路径。JPype1、Gymnasium、PettingZoo 已在上述 conda 环境补齐。
- 2026-11-13 停止时检查无遗留本地 Python 训练/测试进程，GPU compute list 无这些测试。Colab 曾有单个 L4，BC 后台 PID 13213/13214/13269 被 SIGSTOP；不代表远端 runtime 已退还，远端现状需要重新查询，不能靠 PID/别名推断存活。

### 初始 BC 权重

`artifacts/ppo_prep_20261113/best.pt`，743,483,587 bytes；SHA256 `f60c0ac3f395285fed6313ea2575bd7029c55977b4d75def7c0b6da8fb6a25e`。

本地实读：`crrl-state-bc-weights-v1`，architecture 3，decision_ticks 8，progress.optimizer_steps 5112，含 `card_prior_cnn_001`。这是新架构当前阶段的 best，不是旧版 BC；weights-only，不含可恢复 BC 的 Adam。原始 BC 权重不得覆盖。

Full validation 对比见 `docs/BC_FULL_VALIDATION_COMPARE_20261113.md`。426 局、852 seats 上 best loss 5.7022，latest 7.3175，旧 `experiment_v04_16` loss 6.1766；这些是离线 BC 指标，不是实战胜率。12k/14k/18k 权重未在已扫描目录找到，不可声称已评估。

## 当前代码、证据及其局限

| 文件 | 当前用途 / 状态 |
|---|---|
| `src/cr_rl/training/state_ppo.py` | v3 序列 PPO 更新原型；尚缺生产级边界、分批和信赖域处理 |
| `src/cr_rl/training/state_selfplay.py` | v3 模拟器 collector 原型；仅 current-vs-current，共享 actor/value，不具备旧 league/独立 critic 的完整功能 |
| `scripts/train_selfplay.py` | 新增 `--v3` 和从 checkpoint 自动识别路由；旧 v2 参数仍存在，有些被 v3 静默忽略，需要审计 |
| `scripts/run_colab_training.py` | v3 启动分支原型；备份/恢复/更新预算语义需修正验证 |
| `src/cr_rl/training/inference.py` | 添加 StateInference；插入位置可能改变原 FrozenInference 方法归属，必须核查类边界 |
| `loc_transfer/run_v100_v3.sh` | 单 GPU FP16 启动脚本原型，不是双 V100 DDP/DP 实现 |
| `scripts/smoke_state_ppo.py` | 随机输入数值 smoke；复用 old_logp 的多轮调用不是新 rollout |
| `scripts/verify_state_ppo_run.py` | 只检查 checkpoint 字段/Adam 非空等结构，不证明学习语义正确 |

已观测证据：

- 本地随机输入 CUDA FP16 2 次更新、FP32 1 次更新得到 finite loss 和 ratio 约 1；这不证明 Volta 硬件可用。
- 真 simulator smoke 仅 `envs=1,steps=1,epochs=1`，双方 forced WAIT，仅 2 样本。第一版熵错误为 46100；已修为不钳高零概率的 legal support 熵，重跑 entropy=0。
- `artifacts/ppo_prep_20261113/real_smoke.stdout.log`、`.stderr.log`：update 1；`artifacts/ppo_v3_smoke_real/ppo_latest.pt`，2,217,019,892 bytes。
- `resume_smoke.stdout.log`、`.stderr.log`：恢复 update 1 后写出 update 2；`artifacts/ppo_v3_smoke_resume/ppo_latest.pt`。未观测 traceback；未持久记录 shell exitcode。
- `verify_state_ppo_run.py --checkpoint artifacts/ppo_v3_smoke_real/ppo_latest.pt` 输出 format=crrl-state-ppo-v1、arch=3、ticks=8、update=1、optimizer_parameters=232。
- smoke 无实际出牌、无终局/截断、无长序列、无非平凡 policy loss，因此**不能作为整套 PPO 正确性的验收**。

## 接手必须优先修正的问题

1. **循环上下文不一致**：优化 `_sequence_forward` 每段从零 memory 开始；collector 在 rollout 头可能沿用之前的 live memory，update 后又直接清零但不重建历史。必须设计按 episode 保存原始前缀、用当前权重 burn-in 重建 state，再只对 rollout 段取 loss；不能截断整局全局注意力或复用旧权重 KV。当前 done 分段也需要完整 reset/前缀测试。
2. **终止和截断语义错误**：当前 `_gae_sequence` 将 ended 同时当 true terminal 和 trace boundary；截断未用 final pre-reset observation bootstrap，fault 未排除整条相关训练轨迹。补 terminated/truncated/fault/bootstrap_valid，并复用已有 `compute_actor_critic_targets` 正确掩码。最后一次 bootstrap 当前会推进 live state，需改为只读评估，避免下一次重复输入同一帧。
3. **多 epoch 门禁错误**：`max_initial_logp_error` 在每次更新都对 rollout old_logp 比较，第二 epoch 合法参数变化也会被当错误。只在首个 optimizer step 前验证行为一致性，之后用 KL/clip fraction/trust region 限制，不放宽门禁掩盖 state 错位。
4. **内存/梯度规模**：当前把所有完整序列图存列表再一次 backward，无 TBPTT/微批，长局必然高 RAM/VRAM。需按时间 chunk、前缀 burn-in、梯度累积释放图；全局 GQA/KDA 前向上下文必须保留。按有效样本数加权，避免当前短段和长段等权造成无意偏置。
5. **精度和 AMP**：collector 的 model.step 还没包与 optimizer 一致的 FP16 autocast；初始 memory dtype 也需一致。先测 step-vs-prefill/微批 logp parity。不要用“finite”声称 AMP 未 overflow：当前未记录 scaler 是否 skipped，mark_weights_updated 无条件递增，梯度 Inf/skip 没正确处理。
6. **优化机制缺失**：actor choice-only、card/tile 分离 entropy/KL、reference KL、target KL early stop、critic warmup/独立 critic 均未接入；用户此前批准的机制不能被 shared current-only 原型静默替代。新 value head 的训练质量也没有验收。
7. **league 和评估缺失**：当前所有对局都是 current-current，旧 BC/history/frozen opponents、PFSP、对局积分/快照池、46 train + 5 holdout 与定期 paired evaluation 未接完整。已有 `scripts/evaluate_state_policy.py` 支持 v3，但尚未接调度。Frozen 对手不能误进 on-policy actor 样本。
8. **保存恢复**：checkpoint 目前将整个 model state 转 CPU，Adam 还含 CUDA tensors；需要低 RAM 流式/分步保存方案和原子校验发布。缺 RNG、league、game count、reward/deck/schema/precision provenance，resume 未严格检查。当前 `run` 的 updates 是绝对终点，Colab launcher 却传 `updates-completed`，需统一。`--resume-drive` 在 v3 launcher 未实现，进度/summary 与 monitor 也不完整。
9. **CLI/迁移**：v3 device=auto 未解析；`--v2` 与自动 v3 可能同时为真；很多 v2-only 参数会接受但忽略。分离共用/版本专用配置并明确报错。V100 launcher 需加 resume、PYTHONPATH/Java/backend 检查、CPU RAM 预算；双 V100 多卡加速尚未实现与验证。不要启动付费新 runtime 来绕过本地问题。
10. **停止和生命周期**：构造失败时 env/tracking 清理、signal graceful checkpoint、重复 close、后台进程 exitcode 等需健全。smoke launcher 在 ignored artifacts，后续应整理为仓库脚本。

## 恢复任务后的验收顺序

先用 tiny v3 配置/可控假环境做真实语义测试（不能只测试代码自身输出）：episode 中途跨 update、done reset、truncation bootstrap、fault exclusion、forced WAIT 仍推进 memory、首次 ratio 一致、多 epoch 参数真实更新、AMP skip、短长段加权、resume RNG/预算/配置匹配。然后 4090 上用真实 best 做含出牌和完整终局的多 update smoke，并固定对手评测。最后在 Linux V100 实测 FP16/磁盘/RAM、保存恢复与多卡；在硬件验证前只称代码路径可迁移。

用户恢复前保持暂停，不标记目标 complete。

## 早期实现说明（仅供定位，以以上审计为准）

The v3 recurrent PPO update prototype is implemented in `src/cr_rl/training/state_ppo.py`.
Each PPO sequence starts from a fresh `PolicyMemory`, advances frames in order,
and recomputes log-probabilities and values before applying the clipped ratio
loss. The model retains the 8-tick (0.4 second) configuration and learned
placement head; correct episode-memory continuity is not yet implemented.

Local smoke:

```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/smoke_state_ppo.py `
  --checkpoint artifacts/ppo_prep_20261113/best.pt `
  --device cuda --precision fp16 --steps 2
```

The smoke passed on the local CUDA GPU with finite losses, ratios near one, and
finite outputs. AMP overflow/skip was not explicitly instrumented. V100 is intended to use the same `--precision fp16` path, but has not been tested. Do not use
BF16 on V100. Keep recurrent matrices in FP32 and use CUDA autocast FP16 for
the forward/loss path, as the v3 model already does.

`src/cr_rl/training/state_selfplay.py` is the v3 native rollout driver. It
records chronological raw frames, masks, actions, old log-probabilities, values,
rewards and episode boundaries for both seats, then calls `state_ppo_update`.
`scripts/train_selfplay.py --v3` routes directly to it, and
`scripts/run_colab_training.py --v3` plus `loc_transfer/run_v100_v3.sh` provide
the Colab/V100 launch paths. Checkpoint backup, progress JSONL and optional
W&B logging are enabled on the v3 driver.

The real crforge smoke completed one simulator rollout and one PPO update on
the local CUDA GPU. It produced two v3 sequences, two samples, a finite loss,
and `artifacts/ppo_v3_smoke_real/ppo_latest.pt`. Forced-WAIT entropy was zero
for that smoke, as expected; the synthetic smoke covers nonzero categorical
entropy. Illegal flat actions are excluded from the entropy calculation.

The legacy `selfplay_v2.py` path remains available only for architecture-v2
checkpoints. It is never selected for a v3 checkpoint.
