# 电磁炮价值发现与安全 BC 混训改造计划 (2026-11-21)

## 1. 背景与事故复盘

在 `u=1543..1563` 的首轮在线电车 BC 试验中（`--expert-bc-interval 1 --expert-bc-sequences 48 --no-expert-bc-placement`，数据集 `crrl_sparky_bc_all_train_20261120` 共 632 局纯电车局，`validation_enabled=False`）：
- **现象**：仅 7 轮后（`u=1550`），确定性 `argmax` 验证从 `u=1500/1525` 的 `bc 68% (+1.16) / hist 58% (+0.36)` 断崖跌至 **`bc 1/50 (2%, -2.28) / hist 0/50 (0%, -2.82)`**，出牌间隔从 `7.1s–7.5s` 暴涨至 **`35.1s–41.1s`**（几乎全程死选 `WAIT`）。至 `u=1563`，即使在训练期随机采样下，对 BC 锚点胜率也从 `~58%` 跌至 **`42.7% (47/110)`**，策略熵 `policy_entropy` 从 `1.14` 跌至 **`0.398`**，`anchor_kl` 从 `0.50` 暴涨至 **`2.27`**。
- **三大根因**：
  1. **样本极度不平衡 + 单一卡组冲毁共享头**：数据集有效监督帧中 **`WAIT (0)` 占 `94.3%`、真实出牌仅占 `5.7%`（16.5 : 1）**，且 100% 为电车卡组；冻结 PPO backbone 后只训全 51 套卡组共享的 `wait_actor / wait_hand_gate / slot_actor`，用全帧等权 `w=1.0` 的纯 NLL 连锤，把全局 `P(WAIT)` 推过 `argmax` 临界点并破坏了其余 50 套牌的出牌校准。
  2. **更新强度碾压 PPO + 跨轮动量滚雪球**：每轮 PPO 后执行 48 次独立 `optimizer.step()` 并跨轮保留 AdamW `optimizer_state`，21 轮累计打入 **1,008 步 BC**，而单轮 `ppo_forward_kl` 仅 `0.0004–0.0013`（低于单轮 `0.01` 门禁），导致 21 轮全部放行、单向累积崩塌。
  3. **RL 侧本身存在两处压制电磁炮的短视偏差**：
     - `--gae-lambda 0.985`（每拍 0.4s，$\gamma=0.9995$）下 Actor GAE 权重半衰期仅 **17.8 秒**，而“国王塔后沉底电磁炮 → 走到桥头顶巨人 → 推进拆塔”完整回报周期需 **23–25 秒**（前 15 秒空虚期落入高权重区，25 秒后拆塔落入 0.35 低权重区）；
     - `elixir_potential`（`army_elixir.py` + `rewards.py`）**只把已下到场上的兵算进势能 $\Phi(s)$，圣水条内未花掉的圣水势能为 $0$**。在 $\lambda < 1$ 的 GAE 下，模型在 `5.0–5.6` 费立刻拍下 5 费巨人/亡灵大军能马上拿到 `+5 × 0.004 = +0.020` 势能跳变，而选择 `WAIT` 攒到 6 费出电磁炮在等待帧上势能增量为 `0`，形成“到 5 费立刻花光”的短视激励。

---

## 2. 改造内容（四步走，可直接落地到当前 Colab 运行环境）

### Step 0：切 BC 前先做干净 PPO 冷备（必做）
当前回滚主线 `/content/crrl_ppo_rollback_u1380` 使用 `--keep-checkpoints 1`。在重新启用 `expert_bc` 之前：
1. 将当前干净的纯 PPO 运行目录（至少包含 `ppo_latest.pt`、`actor_export.pt`、`opponent_pool.json`、`summary.json`）**单独完整复制一份冷备**到 Drive：
   - `/content/drive/MyDrive/cr_rl_ppo/training/pre_sparky_bc_v2_backup/`
2. 同时把 `--keep-periodic-checkpoints` 从 `0` 改为 `2`（保留最近两个周期快照），确保任何时候都能一键回滚到最近的健康 PPO 节点。

---

### Step 1：拉长 GAE 信用视野 + 补充圣水守恒势能（让 RL 自身看懂“留费与沉底电磁炮”的价值）

#### 1.1 拉长 Actor GAE 半衰期（零风险超参）
- 在 `train_selfplay.py` 启动参数中，将：
  - `--gae-lambda 0.985` 调整为 **`--gae-lambda 0.990`**（与 `--critic-gae-lambda 0.992` 接近）。
- **效果**：Actor GAE 半衰期从 **17.8 秒（44 拍）** 拉长到 **26.3 秒（66 拍）**，完整覆盖“沉底电磁炮 → 14 秒到桥头顶巨人 → 25 秒拆塔”的全周期。

#### 1.2 在 `elixir_potential` 中加入手牌可用圣水守恒项（消除“5 费立刻花光”短视偏置）
- 修改位置：`src/cr_rl/training/rewards.py` 的 `shaped_rewards()`（约 `L110-L118`）。
- 当 `config.army_reward_mode == 'elixir_potential'` 时，`army_values` 传入的是 `(previous_board_diff, subsequent_board_diff)`（单位为圣水费差）。
- 从 `current` 和 `following`（形状 `(..., 2, obs_dim)` 或 `(..., obs_dim)`）读取双方圣水条可用圣水差 `bar_diff`（单位同样换算为圣水费：`elixir / schema.scales['elixir_scale']`），按系数 **`bar_factor = 0.75`** 并入总资源势能：
  $$\Phi_{\text{resource}}(s) = \beta_{\text{army}} \cdot \Big(\text{board\_elixir\_diff}(s) + 0.75 \cdot \text{bar\_elixir\_diff}(s)\Big)$$
  - 若 `current` 形状为 `(lanes, 2, obs_dim)`：
    ```python
    elixir_idx = schema.header["elixir"][0]
    scale = float(schema.scales["elixir_scale"])
    cur_elix = current[..., elixir_idx].astype(np.float64) / scale
    nxt_elix = following[..., elixir_idx].astype(np.float64) / scale
    # cur_elix[..., 0] - cur_elix[..., 1] 为 seat 0 的圣水条费差
    cur_bar_diff = np.stack([cur_elix[..., 0] - cur_elix[..., 1], cur_elix[..., 1] - cur_elix[..., 0]], axis=-1)
    nxt_bar_diff = np.stack([nxt_elix[..., 0] - nxt_elix[..., 1], nxt_elix[..., 1] - nxt_elix[..., 0]], axis=-1)
    ```
  - 然后将 `army_values` 的 `previous` 与 `subsequent` 加上 `0.75 * cur_bar_diff` 与 `0.75 * nxt_bar_diff` 再乘 `config.army_elixir_beta`（保留 `--allow-reward-migration`）。
- **效果**：
  - 在 `4.8 → 6.0` 费等待回费时，圣水条增长每拍都直接产生正势能；
  - 在 `5.0` 费拍下 5 费牌时，5 费从圣水条（`-0.75 × 5`）转移到场上兵力（`+1.0 × 5`），瞬时势能跳变从 `+5.0` 降为 `+1.25`（削减 **75%** 的虚假早出牌贿赂），同时满 10 费溢费仍受 `overflow_penalty` 惩罚且场上兵力仍略高于死憋手牌。

---

### Step 2：方案 B —— 按帧重加权与混合防遗忘的电车 BC

修改 `/content/drive/MyDrive/cr_rl_ppo/preparation/sparky_bc_20261120/expert_bc_specialist.py`（及对应训练配置）：

#### 2.1 按动作与“电车留费窗口”分档赋权（替代原有的全 `1.0` 权重）
在 `run_phase()` 构造每条 episode 的帧权重 `weights[i]` 时（仅对 `keep` 帧有效）：
1. 从 `ep['raw_obs']` 读取：
   - `hand_card`（4 个手牌槽位）
   - `available_elixir = (elixir - elixir_reserved) / elixir_scale`
   - `sparky_id = vocab.index('ZapMachine')`
   - `sparky_in_hand = (hand_card == sparky_id).any(axis=-1)`
   - `sparky_wait_zone = sparky_in_hand & (available_elixir >= 4.8) & (available_elixir < 6.0)`
2. 按帧设置权重 `w`：
   - **真实出牌/技能帧（`actions > 0`）**：
     - 普通出牌帧：**`w = 1.0`**
     - 若该帧打出的手牌槽位正好是 `ZapMachine`（即 `1 + slot * 576 <= action < 1 + (slot + 1) * 576` 且 `hand_card[slot] == sparky_id`）：**`w = 1.5`**（温和强化“够 6 费且合适时选电磁炮”）
   - **电车留费区 WAIT 帧（`actions == 0` 且 `sparky_wait_zone == True`）**：**`w = 0.5`**（精准教模型“手里有电磁炮且 4.8–5.99 费时多等一拍”）
   - **其余普通 WAIT 帧（`actions == 0` 且 `sparky_wait_zone == False`）**：**`w = 0.06`**（原数据 `deploy:wait = 5.7%:94.3%`，乘 `0.06` 后把普通 WAIT 与出牌帧的总梯度质量拉回 `~1:1`，彻底消除 16.5:1 的 WAIT 压迫）

#### 2.2 数据集 50/50 混合 + 开启验证集
- 不要只用 100% 纯电车局训共享 readout：从 `/content/drive/MyDrive/cr_rl_ppo/preparation/ssd_snapshots/quality_bc_full_20261117_073038_52shards.zip` 中抽取等量（~600 局）非电车高质量人类局，与 632 局电车局按 **50% 电车 + 50% 通用高质量局** 打包为新的 SSD 快照（或双源轮流抽样 6 局电车 + 6 局通用），保护其余 50 套卡组的 readout 分布。
- **开启 `validation_enabled = True`**：划出 10–15% 对局作为 `validation` split，启用 `validation['unweighted_nll'] < baseline['unweighted_nll']` 门禁（注：验证集同时统计加权 NLL `weighted_nll` 与选牌准确率 `deploy_card_top1_acc`）。

#### 2.3 降频、合步与清空跨轮动量
- **频率与序列数**：
  - `--expert-bc-interval 5`（每 **5** 个 PPO update 执行 1 次 BC，绝不用 `1`）
  - `--expert-bc-sequences 12`（每次抽 12 条序列，绝不用 `48`）
- **合并梯度步数（关键）**：
  - 修改 `train_actor_epoch` 调用或 batch 更新逻辑：12 条序列通过梯度累加合并为 **最多 2 次 `optimizer.step()`**（每 6 条序列 `step()` 一次，而不是每条序列 1 次 step），`lr = 1e-6`。
- **禁止跨轮持久化 BC 动量**：
  - 每次 `run_phase()` 均新建干净的 `AdamW`（不再加载 `self.optimizer_state`），防止动量跨轮朝同一方向滚雪球。

---

### Step 3：三条硬门禁与自动熔断（写入 `expert_bc_specialist.py`）

1. **收紧单轮 KL 门禁**：
   - `max_kl = 0.005`（原为 `0.01`），超过即 `restore()` 回滚本次 BC。
2. **新增 `P(WAIT)` 漂移门禁（防 `argmax` 憋牌崩塌）**：
   - 在 `reference_pass(..., probes, capture=True)` 时顺带记录 PPO 探针帧上的平均 `WAIT` 概率 `mean_p_wait_before = mean(exp(lp[..., 0])[keep])`；
   - BC 更新后在 `probes` 上复测 `mean_p_wait_after`；
   - **硬门槛**：若 `mean_p_wait_after - mean_p_wait_before > +0.005`，立即判定 `accepted = False` 并调用 `restore()`，同时将 `mean_p_wait_before`、`mean_p_wait_after`、`wait_prob_delta` 记入 `expert_bc` 返回字典。
3. **全局健康熔断（Circuit Breaker）**：
   - 在 `state_selfplay.py` 进入 `bc.run_phase()` 前检查最新 PPO 指标：
     - 若 `policy_entropy < 0.85` **或** `anchor_kl > 0.80` **或** 最近一次 `argmax` 验证 `mean_card_request_interval_seconds > 11.0`：
     - 直接跳过本次 BC（返回 `{'ran': False, 'reason': 'health_circuit_breaker'}`），由纯 PPO 自行恢复。

---

## 3. 上线后的监控与止损红线（每 30 分钟巡检）

| 状态 | 判定条件 | 动作 |
| :--- | :--- | :--- |
| **🟢 绿灯（正常推进）** | `policy_entropy >= 0.85`、`anchor_kl <= 0.80`、`wait_prob_delta <= +0.005`、训练对 Anchor 胜率 `>= 52%`、`argmax` 出牌间隔 `6.5s ~ 10.5s` | 保持运行，观察每 25 轮 `argmax` 胜率与电车出牌率 |
| **🟡 黄灯（轻微偏置）** | `policy_entropy` 滑入 `0.75 ~ 0.85`，或 `anchor_kl` 升至 `0.80 ~ 1.00`，或 `argmax` 出牌间隔升至 `10.5s ~ 12.5s` | 将普通 WAIT 帧权重从 `0.06` 下调至 `0.02`（或 `0.0`，退回纯选牌 BC），`--expert-bc-interval` 拉大至 `10` |
| **🔴 红灯（立即回滚）** | `policy_entropy < 0.70`，或 `argmax` 出牌间隔 `> 14.0s`，或 `argmax` vs BC 胜率跌破 `50%` | 立即关闭 `--expert-bc`，从 Step 0 冷备的 `pre_sparky_bc_v2_backup` 恢复纯 PPO |
