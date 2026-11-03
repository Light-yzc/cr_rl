# 多卡组人类 BC → 自博弈 PPO（v2 修订版）

主入口：`scripts/run_bc_ppo_pipeline.py`（全量 BC 分片流式入口：`scripts/run_all_bc.py`）。必须先挂载 Drive；`--start` 通过 SSH 标准输入接收仅含 `WANDB_API_KEY` 的 JSON，密钥只进入进程环境。

## 数据来源与适用边界（含 v2 失步前缀截断与去重）

- 人类动作：[VanguardX101/IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay)，固定 revision `059d43a02138a34b1b3009cc2acc7630fb99a638`，以及本地清洗源 `cochon_full_1100`。逐分片校验上游 manifest 中的 SHA-256。
- 当前热门组合：[RoyaleAPI Popular Decks](https://royaleapi.com/decks/popular)，2026-11-03 的 Normal Battle / 1 day / rating 快照，18 个成功读取的完整组合，保留原始 URL 和进化/英雄形式。见 `configs/meta_sources_20261103.json`。另从扫描的人类回放中按实际出现频率提取 32 套组合。它们的来源与统计时间不同，不把历史频率当成当前天梯使用率。
- `collect_training_sources.py` 保留原始 Parquet、校验和与候选池；`validate_league.py` 用固定版本模拟器对每套卡组重置并执行最多 120 个合法动作决策。通过短测不代表所有技能与对局交互已经穷举验证。
- `import_il_replay.py` 与 `build_bc_shards.py`：
  - **严格模式（Strict）**：要求至少 20 个成功出牌事件、事件执行覆盖率至少 90%、最多跳过 4 个事件。
  - **宽泛模式（Broad）与 v2 前缀截断（Prefix Cutoff）**：由于观测是在固定 `crforge` 版本与统一等级 11 下开环重放人类动作得到的近似状态，一旦发生多次动作漏判（`omitted_commands`），后续模拟器战场状态将与人类真实对局彻底分叉。虽然 flag `65536` 会屏蔽失败命令附近 5 tick 的监督，但分叉后的整局后续帧仍会产生严重的错位噪声。**v2 规范要求对开环失步回放执行前缀截断（Prefix Cutoff）**——仅保留首次严重失步之前的高保真前缀帧用于 BC 监督，并在合并多目录分片（如 `reconstructed_bc_expanded` 与 `broad_bc`）时按回放唯一键去重。
- 保留被标记的观测帧维持历史时间间隔，仅过滤监督目标；决策间隔为 5 tick = 250ms。大规模训练统一使用 `ShardedBCDataset` 分片流式加载。

## 数据划分、BC 损失与先验校准（v2）

候选先按“八张卡 + 形式”去重，合并来源。按规范键哈希生成留出候选后，同一八张卡的其他形式也整体归入留出池；重复来源名或形式差异不会把同一八张卡组合漏回训练池。BC 同样排除包含留出池八张卡组合的整场回放。剩余回放按整场划分训练/验证，同场双方及历史窗口不会跨划分。

模型使用与 PPO 相同的 preset、128 帧上下文、词表、手牌、自身卡组及敌方已公开卡牌输入：

1. **损失函数与落点头升级（v2）**：
   - 标签目标为动作类型交叉熵 + `position_weight` × 实际出牌时的条件落点损失。
   - **Tile Head v2**：开启 `--upgrade-tile-head`（`tile_query_residual=True, tile_head_fp32=True`）、注入 $18 \times 32$ 网格 2D 坐标编码，并支持为主干与落点头设置分组学习率（如 `--learning-rate 3e-5 --tile-learning-rate 3e-4`）以及合法格邻域 $\sigma \approx 1.0$ 的 **2D 高斯软标签平滑**，解决 v1 中 `position_loss` 停滞在 5.44 的问题。
2. **过采样先验校准（Logit Prior Calibration）与技能分层采样**：
   - 自然回放中出牌帧占比仅约 $p_{\text{nat}} \approx 2.2\%$，训练期采用 50% 出牌 / 50% WAIT 过采样（`--all-plays --play-fraction 0.5`）会使出牌 logit 产生约 $\Delta \text{logit} = \ln\frac{0.5}{0.5} - \ln\frac{0.022}{0.978} \approx +3.79$ 的先验偏移，导致未校准模型在等待帧出现 $>40\%$ 的误出牌率。
   - **v2 修正**：(a) 在 `bc_sampling.py` 中对英雄技能（`action >= 2305`）单独分层采样，消除 `ability_accuracy = 0.0`；(b) 在推理评估及迁移至 PPO 时对 `WAIT` logit 施加先验偏置校准（或在训练 CE 中加入 logit offset），恢复符合人类圣水节奏的等待/出牌边界。

W&B 记录训练 loss、梯度范数、实际优化器步数和验证动作指标。当前正确的指标版本为 **3**，完整评估使用留出回放中全部合法出牌标签和抽样的 WAIT 标签（同时记录自然分布下的误出牌率）：

- `play_recall`：用四个出牌槽位的总概率与 WAIT 概率比较，统计真实出牌时预测要出牌的比例。
- `wait_accuracy`：同一二分类规则下，真实等待时预测等待的比例。
- `balanced_wait_play_accuracy`：上述两者的平均。一直 WAIT 得分约 0.5。
- `play_slot_accuracy`：已知要出牌时，仅在四个手牌槽位之间选最大概率的准确率，不再把 WAIT 混入这个比较。
- `play_tile_accuracy`：给定真实卡牌槽位，预测网格与标注完全相同的比例；同时记录网格曼哈顿距离（`play_tile_mean_l1_distance`）和距离不超过两格的比例（`play_tile_within_2`）。

每个 epoch/stage 原子保存完整模型、优化器、随机数状态、数据哈希和划分，Drive 写入后读取校验 SHA-256，再推进 `latest.json`。

## 迁移、稠密奖励塑形与自博弈 PPO（v2）

`train_selfplay.py --init-from-bc bc.pt --decks league_train.yaml --critic-mode independent`：

1. 要求 `stage=bc`、架构版本 2，严格比较模型配置，包含上下文长度、词表大小及 `tile_query_residual` / `tile_head_fp32` 标志。
2. 默认启用**独立 Transformer Critic**（`training_architecture_version=3`，见 [`INDEPENDENT_CRITIC.md`](INDEPENDENT_CRITIC.md)）：Actor 加载完整 BC 策略权重（剥离共享 value head），独立 Critic 从 BC 编码器同构权重热启动，并在冻结 Actor 的情况下先完成完整对局预热（Critic Warmup）。
3. **奖励函数 v2（零和塔血/破塔塑形 + 满费溢出惩罚）**：
   - 为解决纯终局奖励在 3~5 分钟对局（720~1200 步）中信用分配过短的问题，v2 在终局胜负 $r_t^{\text{terminal}} \in \{-1, 0, +1\}$ 基础上，直接从 `crforge` Header 提取实时塔血与皇冠变化：
     $$r_t = r_t^{\text{terminal}} + \big[\Phi_{\text{hp}}(s_t) - \Phi_{\text{hp}}(s_{t-1})\big] + w_{\text{crown}} \big[\Delta \text{crowns}_t - \Delta \text{crowns\_enemy}_t\big] + r_t^{\text{overflow}}$$
   - 默认参数：公主塔满血权重 $w_{\text{princess}} = 0.15$，国王塔满血权重 $w_{\text{king}} = 0.20$，破塔皇冠权重 $w_{\text{crown}} = 0.15$，10 费满溢仍选择 `WAIT` 的微惩罚 $r_t^{\text{overflow}} = -0.001$/step。
   - **GAE 视野拉长**：将折扣因子与 GAE 参数升级为 $\gamma = 0.9995, \lambda = 0.98$（有效视野从 ~4.9 秒扩展至 ~12.2 秒）。
4. **PPO 优化稳定性与常驻环境修复（v2）**：
   - **累积步级 KL 早停**：将有效批大小提升至 $\ge 64$（如 `batch_size=4~8, grad_accumulation=16~32`），并在完整梯度累积步（或滑动窗口）上判定 `target_kl`，彻底修复 v1 因单个 4-sample microbatch 噪声导致第 4 步早停、丢弃 98% 采样数据的问题。
   - **跨 Update 模拟器常驻**：单一训练进程连续运行 `--updates N`，不在每个 update 边界重启子进程重置正在进行的比赛。

完整流程先执行 tiny 的 BC/BC 续训/迁移/PPO 续训验收，再训练 119M。完成后复测四份 epoch checkpoint，用全部留出出牌标签选择平衡指标最好的完整 checkpoint；它必须好于初始化基线，才会自动进入 PPO。最后一轮单独保留供续训，不覆盖。这只是模仿改善门槛，不是专家水平或游戏强度门槛。

PPO 使用本轮冻结采样策略和历史冻结对手，只有 learner 席位反向传播；本轮行为策略同时作为 KL 参考。每局双方独立抽卡组，允许镜像。固定后端的 `reset_env` 只能保留原卡组，因此通过每个环境一个单场 `VectorBattle`，使用公开 `reset` API 换卡组，保证其他正在进行的对局不被重置。代价是当前 CPU 模拟串行步进，扩并发时需要重新衡量吞吐。

PPO 的日志、原始轨迹、对手池和完整 checkpoint 每轮同步 Drive；每分钟同步活动日志。恢复时重建新对局，不是精确恢复模拟器中途状态。留出卡组不进入 PPO 训练；强度与泛化仍需要固定对手、种子、换边的独立评估，不能以滚动自博弈胜率替代。

新 checkpoint 内保存完整卡组配置。`--resume` 未显式指定卡组时使用保存的配置；显式指定不一致卡组池会拒绝。旧 checkpoint 可从运行目录的 `run_config.json` 恢复卡组，否则需要明确传入 `--decks`。恢复不会因为当前默认池变化而悄悄换卡组。最终验收还会从 Drive 恢复 119M 到新目录，在 GPU 上实际增加一个独立的短更新；该检查不计入原训练 run 的样本或轮次。

## 在 Linux 手动执行

```bash
python scripts/collect_training_sources.py --out artifacts/sources --shards 2
python scripts/validate_league.py artifacts/sources/league_candidates.yaml --out artifacts/league.yaml
python scripts/import_il_replay.py artifacts/sources/replays.parquet --limit 40 \
  --source-revision 059d43a02138a34b1b3009cc2acc7630fb99a638 --out artifacts/human_bc
python scripts/train_bc.py artifacts/human_bc --model production --device cuda \
  --batch-size 4 --grad-accumulation 4 --epochs 4 --samples-per-epoch 2048 \
  --learning-rate 3e-5 --holdout-decks artifacts/league_eval.yaml --out artifacts/bc.pt
python scripts/train_selfplay.py --model production --device cuda \
  --init-from-bc artifacts/bc.pt --decks artifacts/league_train.yaml \
  --updates 3 --envs 2 --steps 1536 --out artifacts/ppo_from_bc
```

Colab 以前台内核任务监视真实流水线、检查心跳；不承诺绕过平台寿命或资源限额。可靠性来自周期保存、哈希验证和恢复测试。
