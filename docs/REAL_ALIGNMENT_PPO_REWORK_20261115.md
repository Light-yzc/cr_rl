# 实机对齐与 PPO 自博弈改造（2026-11-15）

基线：`artifacts/ppo_v3_v100/ppo_latest.pt`，architecture-v3，update 697，带 `card_prior_cnn_001` learned placement。继续该 PPO 权重与独立 Critic；不是重新从早期 BC 启动。固定 BC 锚点仍是 `artifacts/ppo_prep_20261113/best.pt`。

## 1. Reserve-to-WAIT 与 checkpoint 迁移

`ModelConfig.reserve_to_wait` 作为可保存的版本开关。旧 checkpoint 默认关闭，保留历史对手和 BC 的原分布；训练启动器显式开启学习者的新语义。

非空手牌全部参与卡槽 softmax。无法执行的槽位质量用 FP32 logsumexp 合并到 WAIT；最终对全部非法动作统一填 `-1e9`。不增加任何非法动作，不更改 2307 动作编码。现有 factorized entropy / KL 基于最终分布，继续成立。

新增 `wait_hand_gate`：`[4 个 detached 槽位分数, 4 个费用缺口, 4 个够费标记] → 12→32→1`。可用圣水扣除已预留量，未知费用/空槽不产生够费特征。最后一层零初始化。

**零初始化只保证新增旁路的初始输出等价；Reserve-to-WAIT 的启用本身会改变分布。** 迁移后必须重新采样，不能复用迁移前的 old_logp/轨迹。旧 checkpoint 仅允许补齐这四个新增参数，其余权重严格加载。

优化器迁移按参数名称插入空的 gate moments，保留原 Actor、learned-placement 的 Adam moments。不能简单往参数 ID 队尾追加：learned-placement 在 gate 后注册，会造成 moments 错配。加载前检查 moments 形状。

CPU 回归验证旧权重补 gate 的输出误差 <=1e-6；实际 FP16 rollout/recompute 存在浮点误差，因此单独检查 PPO 初始 logp，限制 0.02，没有宣称 FP16 能达到 1e-4。

## 2. 实机适配与出牌

### 观察字段

根据仓库实际 `Schema.java` / `LadderEnv.enemyRows`，而非交接单中的过期索引：

- `H_ENEMY_PLAYS=116`，8 行×9 字段：card、cast、form、经过 ticks、座位坐标 x/y、kind、flags、point 经过 ticks。
- 索引 4 是 `H_CROWNS`，没有 `H_OVERTIME_LEFT`；加时剩余使用已有 `H_SECTION_LEFT_MS`，按当前 section 的长度与已过时间计算，保留 `H_OVERTIME` 标记。
- 敌方记录以 played_count 的增长为门控，从敌方实际牌库的主体单位映射卡牌，成组召唤去重，排除塔/普通子召唤物。无法判断的记录保持未知，不能把每个新实体都当作一次出牌。
- 每个 eid 保存首次观察时间，根据本地表 `DeployTime` 推算剩余部署时间。首次中途接入不把已有部队重新标成部署中；一旦观测到移动就结束推算部署。兜底状态依次为部署 4、移动 1、有目标攻击 2、站立 0。
- `NullsEntity` 支持可选的 state/deploy_ms/attack_ms，验证过的直接值优先于推算。现有反编译材料未确立可靠读取偏移，**没有虚构内存字段**，也没有增加 ADB 交换。
- **攻击倒计时仍可能未知。** 单凭位置和 target 指针不能可靠确定攻击周期相位，因此不伪造精确攻击计时器。兜底状态也不能区分所有眩晕、冲刺、变形等状态；当前是减少缺失，不是实现精确服务器状态重建。

### 推理门控

`src/cr_rl/client/action_selection.py` 为可测试的部署专用选择器；`run_nulls_policy.py` / PowerShell 启动器接入：

- play threshold 0.45，slot threshold 0.28；>=9.5 费降为 0.15 / 0.10。
- 对可执行卡牌概率质量做门控，选最大概率槽位；落点默认 argmax，可用 `--tile-temperature` 开启低温采样。
- 当前触控标定不支持英雄技能，技能概率不计入可执行出牌质量，避免把技能误当手牌点击。
- 伤害/控制法术按交接单半径检查同一己方坐标系内的敌方部队/建筑；塔仅在 <=450 HP 时独立计为目标。空场改成 WAIT，日志记录 `spell_vetoed` / `selection_reason`。
- 这些护栏只用于实机部署。PPO 继续从训练分布采样，避免把启发式动作伪装为 on-policy 数据。

已有录制 `artifacts/nulls_live/play_20261115_214246` 重放：420 帧，部队/塔状态统计 standing 38、moving 932、attacking 2856、deploying 135；无未知单位状态。新门控 33 次出牌，原录制 37 次。小电/箭雨各自 576 个空场落点全部被 veto。原录制重放未触发法术 veto，不能据此宣称已提高实战胜率。报告：`artifacts/ppo_alignment_smoke/client_replay_report.json`。

## 3. PPO 数据与对手

- current 对局双侧均保存独立 segment、动作、old_logp、初始 memory 快照、奖励、bootstrap 与终止标记。
- 持续对局在权重更新后重放前缀，分别重建两侧 memory；行为策略/BC KL 的另一侧前缀也按批烧入，不拿主侧 memory 代用。
- fault 让同局两侧无效；时间截断用各自最终观察 bootstrap；终局两侧都不 bootstrap。完整局数依然按物理对局去重，避免双边收集把局数翻倍。
- 回归检查 current 双侧收集且 frozen 泳道不重复帧；GPU smoke 逐段检查 tick 严格间隔 8。
- 角色比例 current/history/anchor = 0.60/0.25/0.15。保留 PFSP 历史池容量 20 和固定 BC 锚点。
- 活跃历史 cohort 上限 2，每 2 个 update 尝试轮换一个；旧对手在对局结束前保留并逐渐退出分配，不中途更换。每步 frozen forward 最多 2 个历史组+1 个 BC 组。rollout/training cache 为 4/3。
- 学习者主侧 50% 从 `giant_sparky_v1,user_giant_sparky_pump_plain,hog_cycle_v1` 抽样，剩余从原完整卡池抽样；另一侧始终从完整池抽样。保留 `decks_augmented.yaml` 对 51 个基础卡组的形态扩展及其抽样权重。焦点抽样并不保证其他卡组强度不下降，需要继续观察分卡组指标。

## 4. Reward 与反传/吞吐

- 新增 `--army-reward=0.015`，对存活非塔 character/building 的 `0.3+0.7*hp/max_hp` 计算己敌差，再除 10。
- 使用交接单允许的**统一单位权重**，不把一张群体牌的完整费用重复计算到每个子单位上。
- 严格使用 `gamma*Phi(next)-Phi(current)`；终局 next potential 为 0，截断保持 potential，fault 不 shaping。回归验证折扣望远镜求和与零和性质。原 overflow 惩罚仍是独立、非势能项。
- TBPTT 从 32 提升到 128（51.2 秒），GQA 继续保存完整历史；这不是无限长度反向传播，窗口前 memory 仍 detach。
- 空间 encoder 与落点头开启 non-reentrant activation checkpointing，仅训练且启用梯度时重计算。CPU 验证输出/梯度一致；rollout 不重计算。
- 两个 V100 同时参与 rollout、Actor/Critic 更新，通过现有 NVLink 副本同步。4090 不参与训练。
- gamma=0.9995，Actor/Critic GAE=0.985/0.992，reference KL=0.02，anchor KL=0.002，card/tile entropy=0.003/0.0005。
- 保留 KL 拒绝机制和 adaptive LR；不为通过 smoke 放宽概率或 KL 检查。

## 5. 保存、磁盘与监控

- 修复恢复训练后的结束判定：绝对目标轮数=起始 update+本次 updates，避免 update 已超过剩余轮数后每轮保存。
- 默认绝对目标 250000 updates，完整 checkpoint 每 800 updates；正常结束和显式停止保留最终恢复点。
- 稀疏保存时按实际存在的文件保留最近两个恢复点，另保留一个周期点。原有历史池与恢复点引用的对手文件继续受保护。
- 本次删除 u512/u576 两份旧恢复点及关联 pool manifest，共 8,752,090,387 字节；保留 u640/u696/u697 与 best/latest。
- 可选读取缓存仅使用 Intel 数据分区 `/run/media/regen/本地磁盘/cr_rl_active/frozen_opponent_cache_v3`，有容量与剩余空间限制。没有写 Windows 系统分区或 GALAX。
- 保持 validation 关闭，继续记录 `train_traj/*`。新增 `reward/army_sum`、双侧样本数、frozen 组数、各 GPU 显存峰值、初始 logp 误差和溢出计数到 TensorBoard。

## 验证与启动

相关测试命令：

```bash
CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=2 .venv-v100/bin/python -m pytest tests/ \
  -k 'policy or state_policy or ppo or reward or nulls_adapter or reserve_wait' -q
.venv-v100/bin/python -m pytest tests/test_local_retention.py -q
```

最终 119 项相关回归全部通过（含历史 cohort 轮换、双侧轨迹、稀疏保留策略、低磁盘空间回收、WAIT 特征和优化器迁移）。`scripts/smoke_state_alignment.py` 复用生产 CLI，限定两次 update，检查帧间隔和样本数，不保存训练 checkpoint、不发布 league 快照。调优阶段的失败尝试没有覆盖原 u697。

最终 GPU smoke、参数与启动状态见下方追加记录。

### 最终完整长度 smoke

36 环境，384 决策步/rollout，Actor/Critic 微批 4/4，Actor/Critic 有效批次 512/2048，TBPTT 128，空间与落点激活重计算。语义迁移首次 Actor LR 取 `min(原 LR, 1.5e-6)`；后续恢复保留 adaptive LR，范围 1e-6..5e-6，不在每次重启强制重置 LR。

| 指标 | U698 | U699 |
| --- | ---: | ---: |
| 单侧基准样本 | 13824 | 13824 |
| 实际双侧样本 | 21738 | 21177 |
| 增收副侧样本 | 7914 | 7353 |
| 每轮秒数 | 110.05 | 118.69 |
| 采样阶段秒数（含前缀重建） | 26.68 | 38.13 |
| 有效样本/秒 | 197.53 | 178.43 |
| 初始 logp 最大误差 | 0.00310 | 0.00479 |
| V100#1 显存峰值 GiB | 8.24 | 8.59 |
| V100#2 显存峰值 GiB | 7.89 | 8.12 |
| Actor overflow / KL stop | 0 / false | 0 / false |
| Critic overflow | 1 | 0 |
| frozen 模型推理组数最大值 | 3 | 3 |

主进程最大 RSS 5.56 GiB。Critic 的一次 GradScaler 溢出跳过发生于第一轮，第二轮没有；未出现非有限权重或无效模拟器样本。

两轮合计 42915 样本 / 228.73 秒，约 **187.6 样本/秒**。旧 U697 为 9216 / 70.61 秒，约 130.5 样本/秒；有效吞吐提高约 **44%**，同时反传窗口由 32 增至 128。对局长度、帧数和 choice 比例不同，这不是严格同轨迹硬件基准，也不能据此证明胜率提升或所有负载下达到硬件上限。

完整证据：`artifacts/ppo_alignment_smoke/full_smoke.log`、`verification_summary.json`、`smoke_complete.json`。调优时未开启落点重计算的 4/4 微批曾达到 14.68 GiB 并触发一次 KL stop，该配置未用于长训。

历史快照 GC 现在在 pool 淘汰后执行，保护当前对局与现存恢复点引用。若快照 admission 前可用空间不足 9 GiB，先取消可选旧周期点、保留最近实际恢复点，再回收无引用对手，避免等待第 800 轮保存时才处理积累的快照。

### 长训启动

2026-11-15 晚，systemd 用户服务 `crrl-v100-v3.service` 已从原 U697 启动，PID 启动时为 39510。绝对目标 U250000，本次计划 249303 个 updates；36×384，微批 4/4，有效批次 512/2048，保留上述全部最终参数。`NRestarts=0` 的运行状态与后续实际 update 以服务和日志为准。

```bash
systemctl --user status crrl-v100-v3.service --no-pager
# 训练日志
tail -n 2 artifacts/ppo_v3_v100/train.log
```

TensorBoard 已作为 `crrl-tensorboard.service` 启动，地址 http://127.0.0.1:6006/ ，数据目录为本仓库 `artifacts/ppo_v3_v100/tensorboard`。训练胜率看 `train_traj/*`，出牌/高费牌比例看 `decision/*`，吞吐、显存和副侧样本看 `performance/*`，兵力势能看 `reward/army_sum`。

实际长训首轮 U698 已完成：21939 有效样本，196.48 样本/秒，副侧增收 8115 样本，V100 峰值 8.27 / 7.90 GiB，初始 logp 最大误差 0.00379，Actor KL stop=false。服务无自动重启。后续指标见训练日志与 TensorBoard。

## 后续：整局 BPTT

随后增加了真正的整局采样/反传模式，详见 [FULL_EPISODE_BPTT_20261115.md](FULL_EPISODE_BPTT_20261115.md)。该模式下 `tbptt_chunk=128` 仅用于无梯度评估/参考缓存，不截断 Actor/Critic 的训练梯度；旧版 128 步截断模式仍可作为可配置回退。
