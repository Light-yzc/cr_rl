# BC100 目标与优化审计（2026-11-10）

当前有缓慢的训练集拟合，但固定验证集改善很弱，出牌时机的阈值表现还在明显漂移。现有证据不足以把问题归为模型容量不足，也不能断言继续相同训练一定没有收益。应先完成小规模、能证伪的本地诊断，再决定改变训练目标、更新组织或时间模型。

本次只读生产实现与已有快照；没有更改运行中的 L4、训练参数或生产代码。CPU 探针只使用 G 盘、已有 Python/Torch、极小张量；没有载入完整 checkpoint 或把大数据装入 RAM。当前阶段是 BC：PPO reward、GAE、KL 和熵奖励不是此处 BC loss 的组成部分。

## 1. 快照范围与趋势

本地 JSONL 到 step 1762；更新的 W&B 快照到 step 1916、epoch 5 第 260 个座位更新。二者重叠字段数值一致。验证最新为 step 1822，不应把它说成 step 1916 的同步验证结果。一次重复 step 610 的验证记录保留在原证据中。

每个完整 epoch 是 175 场训练对局的 350 个座位更新；epoch 5 尚未完成，因此不能直接把它的均值当完整 epoch 对比。

| epoch | 座位更新 | 总 loss | timing | 条件 card NLL | tile CE | slot accuracy | norm 中位数 | norm 最大值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 350 | 6.70851 | 0.43664 | 1.22333 | 5.04854 | 43.33% | 4.028 | 1398.49 |
| 3 | 350 | 6.67613 | 0.43986 | 1.22710 | 5.00917 | 43.08% | 4.546 | 160.62 |
| 4 | 350 | 6.62930 | 0.43529 | 1.19922 | 4.99479 | 45.75% | 3.992 | 140.93 |
| 5（未完） | 260 | 6.57882 | 0.43742 | 1.17950 | 4.96189 | 47.47% | 4.237 | 38554.56 |

对相同的 350 个座位配对，epoch 2 → 4 平均 loss 下降 0.07921，中位下降 0.06309，81.71% 的座位下降。因此“没有任何学习”与日志不符。

| 验证 step | 总 loss | timing | 条件 card NLL | tile CE | ACT recall | WAIT accuracy | slot accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 606 | 6.75144 | 0.46566 | 1.20816 | 5.07762 | 18.50% | 93.07% | 44.78% |
| 956 | 6.80967 | 0.45802 | 1.21630 | 5.13535 | 21.99% | 91.72% | 44.88% |
| 960 | 6.73183 | 0.44111 | 1.21396 | 5.07676 | 5.18% | 98.69% | 44.79% |
| 1306 | 6.74738 | 0.45291 | 1.23367 | 5.06080 | 0.32% | 99.99% | 46.33% |
| 1656 | 6.65884 | 0.43986 | 1.18234 | 5.03663 | 5.40% | 98.71% | 45.07% |
| 1822 | 6.72164 | 0.45248 | 1.21727 | 5.05189 | 3.96% | 99.34% | 46.97% |

相对 606，最新总验证 loss 仅下降约 0.441%，条件 card NLL 略差，tile 略好。956 → 960 仅四次更新就使 ACT recall 从 21.99% 变为 5.18%，提示时机分布/阈值不稳定；它不能单独证明随机采样出牌率下降到 5.18%，因为此指标用的是 `p(ACT) > p(WAIT)` 的确定性阈值。

## 2. 损失归一化没有把 card/tile 除以海量 WAIT

实现位置：`src/cr_rl/training/bc.py:124-170,191-205`；`src/cr_rl/training/state_bc.py:364-404`。

对整座位 N 个监督行、A 个非 WAIT 动作（含技能）、P 个普通出牌，当前 `position_coef=1` 的目标为：

```text
L = Σ[timing_weight × timing_NLL] / N
  + Σ[conditional_card_NLL on ACT] / A
  + Σ[Gaussian_tile_CE on ordinary PLAY] / P
```

分母来自完整座位，chunk 内计算的是可相加的贡献。每个 chunk backward 后累积梯度，没有在座位末尾再除一次 chunk 数。条件 card/tile 分别按 A/P 归一化，不能套用“约 94% WAIT 把全部 card/tile 梯度稀释掉”的解释。

与按全部数据行等权的目标相比，这个目标会给每个座位相同总权重。epoch 2 每座位普通出牌数 min/median/max 为 1/28/89；1 张牌座位的单张牌名义 loss 权重是 89 张牌座位单张牌的 89 倍。实际参数变化还受梯度方向、clip 和 Adam 矩估计影响，不能直接由此换算学习率。

## 3. timing prior 的含义与不能下的结论

epoch 2 共 180506 行、11752 个 ACT，按行 ACT 比例 q=0.065106；当前目标 prior=0.20 对 timing 给 ACT 权重 3.07192、WAIT 权重 0.855712。它只调整 timing，不改条件 card/tile 梯度。

由于训练是座位等权，而 q 是全 epoch 行统计，忽略状态的最优常数 `p(ACT)` 实测数学值约 0.193831，略偏离 0.20。这是口径差异，但偏差很小，没有证据把它当 plateau 主因。

`H(0.2)=0.5004` 只是忽略状态的常数分类器参照，并非 timing loss 的不可约下界；状态可分时硬标签 BCE 可以趋近 0。合法 mask、forced WAIT 和座位权重也改变朴素常数基线。prior=0.20 不是强制每场出牌 20%，也不能直接把阈值 recall 当成动作采样频率。

数据 Agent 补算同一固定 16 座位的合法参照：choice 状态用常数 p(ACT)=0.2，forced WAIT 必须 p(ACT)=0，验证 timing CE 为 0.475818；若解析优化这个单一 choice 常数，p*=0.217425，CE=0.474975。step 1822 模型 timing=0.452478，只比后者改善约 0.02250 nats。这里按生产验证使用全验证集 q=0.0699737；不能拿未经合法性处理的 H(0.2) 夸大进步。

## 4. sigma=1 的 tile loss 存在软标签下界，但离下界仍远

实现把 Gaussian 目标乘上所选卡牌的合法落点 mask，再归一化。每个样本有：

```text
tile_CE = H(masked_Gaussian_target) + KL(target || predicted_tile_distribution)
```

下界随边界、卡牌合法区域和标签坐标变化，不能统一拿 0 或二维中央 Gaussian 的 2.838 作整个验证集下界。

数据审计对真实固定验证集合 games `[5,13,15,23,33,37,54,58]` 双座位、583 张普通牌，使用与 trainer 相同的座位等权口径得到：

| 指标 | 真实基线 | step 1822 模型 | 含义 |
|---|---:|---:|---|
| 合法非 WAIT 类别均匀预测 NLL | 1.267470 | 1.217271 | 仅改善 0.050199 nats |
| 合法普通 slot 均匀预测 accuracy | 32.413% | 46.970% | 有 slot 排序能力，但概率拟合改善小 |
| 合法 tile 均匀预测 CE | 5.569354 | 5.051888 | 改善 0.517465 nats |
| sigma=1 目标自身熵 | 2.464013 | 5.051888 | 仍有平均 target KL 2.587876 nats |

因此 loss 不必趋 0，但也不能用软标签下界解释 tile CE 停在 5.05。

补充的廉价基线只按训练集 11248 个动作拟合 118 张卡各自的落点频率，然后按当前真实合法 mask 重新归一化，不使用实体或时序输入。事前固定两档平滑 alpha=0.01/1，在同一 fixed16 集合得到 tile CE 4.87417/5.03105。当前模型尚未超过这两个基线，说明场景条件信息的利用仍弱；不等于证明容量不够，也不把较好一档当验证后挑出的部署参数。

基线来源：`artifacts/bc_diagnostic_20261110/supervision/bc100/mask_baselines.json` 与 `card_tile_prior.json`。旧 128 状态诊断 cohort 的 64 张普通牌目标熵 2.52158 仅作为独立计算检查，不冒充完整固定验证下界。

## 5. 单座位更新与 TBPTT 是明确训练约束，尚非已证根因

`state_bc.py:357-358,373-444,459-482` 在整座位开始前清空梯度，顺序读取该座位，积累所有 chunk 梯度，结束后才更新一次权重。`train_state_bc.py:484-513` 对下一座位继续。这避开了“同一个 live memory 中途改权重导致缓存过期”的错误。

每场两个座位各更新一次。每次更新只看一条高度相关的局内序列，epoch 2 中位仅 30 个 ACT。与跨多局混合梯度相比，它可能有较大的梯度方差和顺序漂移，但必须通过同数据、同初始权重、同更新预算的对照验证。

当前 `chunk_size=32`、决策间隔 0.4 秒，每 12.8 秒 detach 一次 state。它保留前向记忆，但后续标签不会反传到上一 chunk 早期的编码/写入参数路径。前向全历史不代表整局 BPTT。

CPU 使用生产 `StateTemporalCore`（width 8、KDA+GQA、8 tokens、chunk 2）实际比较：

| 探针 | 全 BPTT | 每 chunk detach |
|---|---:|---:|
| 最后一帧 loss 对前 6 帧输入的梯度范数 | 0.168666 | 0 |
| 对最后 chunk 输入的梯度范数 | 0.820574 | 0.820574 |
| 前向输出最大差（两者比较） | 2.384e-7 | 2.384e-7 |

两者最终 memory position 均为 8。此约束可能妨碍学习跨 12.8 秒的记忆写入，但早期状态仍能通过固定 memory 参与后续查询；不能说模型完全看不到更早历史。

## 6. 梯度、FP16 与调度：有异常，不是已证明“炸了”

1310 条 W&B 更新全部 preclip norm > 1；日常中位约 4，而不是极端 1398/38554。`max_grad_norm=1` 给出中位 clip 系数约 0.22–0.25，但不能说 Adam 实际 LR 也等比例降到四分之一：Adam 的一阶/二阶矩会部分抵消共同尺度。CPU 首次 AdamW step 中，[3,4] 梯度与 clip 成范数 1 后，两个参数的变化相同，均约 3e-5。变化的 clip 系数会影响长期矩历史，仍需测量实际 update/weight 范数。

最大峰值 step 1911（epoch 5）norm=38554.5625，loss=6.49733，LR≈2.99485e-5。数据 Agent 按当前 seed/order 定位到 replay 253 seat 0（source row 259），该座位没有 hand/tile 修复。这个有限峰值需要逐模块和输入定位；正常 loss 不能排除梯度病态，但也不能据此宣称 FP16 已 NaN。

进一步输入审计在该座位发现近永久 buff 的原始毫秒值 999999899，经 adapter 后达 99999.99，14 个实体时刻超过 FP16 最大有限值 65504；另一个 buff 槽最大约 100。当前 frame encoder 用 FP32 可避免这一特征直接在入口转 FP16 时变 inf，但特征尺度仍可能影响条件数。它是值得优先做冻结权重梯度对照的具体线索；同局共现不等于已证明导致该峰或 plateau。输入证据：`artifacts/bc_diagnostic_20261110/supervision/bc100/spike_inputs.json`。

本地可见 1156 条 JSONL 更新均 `amp_overflow_skipped=False`、scale=32；更新的 W&B 没有完整 overflow 字段，不能把这个“全部无 overflow”范围扩大到 step 1916。

调度 anchor=606，warmup=350；step 1916 已经过了 warmup，当前 LR=2.99479557e-5，相对峰值 3e-5 只下降约 0.17%。因此“还在热身”或“cosine 已把 LR 衰减过低”都不适用。长达 100 epoch 的 cosine 也不是对当前早期 plateau 的有力解释。

### 已证实但未在真实日志触发的 robustness 边界错误

`state_bc.py:460-478` 先 unscale，随后计算/clip aggregate norm；若 norm 非有限，仍调用 `scaler.step`，假定 scaler 一定跳过。

CPU 反例：每个梯度元素有限（1e20），但 FP32 聚合 norm 溢出 inf；unscale 的 found_inf 已记录为 0。clip 后 `scaler.step` 实际执行 AdamW step，step 计数变为 1，参数因 weight decay 变化约 0.001。生产分支却会报告 overflow、不给成功更新计数，也不调用 mark_weights_updated。

这是可复现的实现边界错误，后续应修为明确拒绝 nonfinite aggregate norm 并保证 scaler/optimizer/schedule/counters 一致；该反例不是当前日志已发生事故的证据，本审计没有修改生产代码。

## 7. 本地验收顺序与决策条件

1. 对同一小训练 cohort 先做真实输入/标签一致性审查及固定样本拟合，保留独立 control。记录三项 loss、Gaussian excess KL、mask-aware slot 基线、p(ACT) 分布与 PR；固定验证不能只看总 loss。
2. 对 step 1911 对应座位做有界复现，统计模块级 preclip norm、参数更新范数、FP16 scale 与非有限计数。若无法复现原 checkpoint/顺序，明确它只是邻近 checkpoint 压测。
3. 若小 cohort 能拟合而 control/固定验证不改善，优先检查单局更新噪声、监督时机一致性和泛化；可比较 2–4 个座位累积一次更新，按 ACT/PLAY 总数明确归一化。每座位都必须重建 state，不能把别人的 live memory 混用。对照需固定样本量和算力预算。
4. 只有当长历史相关任务表现依赖 TBPTT 时，再比较 chunk 32/64/128，衡量 wall time、显存、excess KL 与时机 PR。不能凭“全历史”三个字决定改成 GTrXL，也不能保证更长 chunk 必然更快收敛。
5. 若小 cohort 本身不能显著胜过合法均匀/按卡落点先验，先定位输入表达和梯度流；在该门槛通过前直接加容量、改多项 LR/prior/clip 会失去诊断能力。

这些是可证伪的验收实验，不是已经验证有效的生产方案。当前证据支持优先检查输入/监督、时机校准与更新组织，尚不支持直接更换大架构。

## 8. 追加：加权 timing 后验与在线直接采样的口径

精确调用链已确认：

- `train_state_bc.py:83,331-334` 明确配置并保存 target prior=0.20，`state_bc.py:393-394` 将训练实际全行 q 传入 timing loss。
- `bc.py:153-168` 用 wACT=r/q、wWAIT=(1-r)/(1-q) 加权 BCE。
- `state_policy.py:115-131` 合成 WAIT/slot/ability logits，经 `policy.py:600-623` 的合法 card softmax 与条件 tile softmax 构造 joint。
- `evaluate_state_policy.py:79,170-173` 明确没有应用 calibration，直接 `exp(logp)` 后 multinomial 采样。checkpoint calibration 只出现在 info 中，未改变概率。

在模型能逐状态拟合加权 BCE 最优解的理想情况下，原分布自然 posterior p 与训练加权 posterior p_w 满足：

```text
logit(p_w) = logit(p) + log(wACT / wWAIT)
logit(p) = logit(p_w) + logit(q_train) - logit(r)
```

当前 q_train=0.065105869、r=0.20，ACT/WAIT odds 乘数为 3.5898996，理论逆变换 d=-1.278124233。可以给全部非 WAIT joint logp（包括技能）加 d，再在完整动作维归一化；等价于 WAIT logit 增加 1.278124233。此操作保持 p(card,tile,ability | ACT) 不变，合法 forced WAIT 仍为 1。

| 模型加权 p(ACT) | 理论还原 p(ACT) |
|---:|---:|
| 0.20 | 0.065106 |
| 0.2045 | 0.066824 |
| 0.2962 | 0.104932 |
| 0.50 | 0.217870 |

必须使用训练 loss 实际 q，不能因为只在 choice 状态采样就擅自替换为 q_choice=0.0737029；forced WAIT 是 posterior=0 的特例，本来就不受逆变换影响。若人为换成 choice prior，会定义另一个与实际加权目标不同的变换。

CPU 直接调用生产 joint composer 检查，逆 posterior 数学误差 5.55e-17，joint ACT 质量公式误差 5.36e-7，合法非 WAIT 条件分布误差 4.77e-7，非法动作概率保持 0。

这个现象能说明“确定性 0.5 阈值 recall 很低”与“随机采样经常出牌”可以同时发生：p(ACT)=0.2 从未过 0.5 阈值，但在固定状态、0.4 秒重复抽样的假想情形下，均值约每 2 秒出一次。真实局面与合法性会随决策变化，不能直接用该几何分布估计整场行为。

这不等于已经找到 plateau 的原因，也不等于应立即部署 d=-1.278。target=0.20 本来就是明确指定的成本目标，直接按加权分布服务可以是有意设计，而非无意遗漏。若目标是恢复训练示范的自然概率，才有上述口径不一致；如果目标是提高 ACT 代价权重，则训练定义本身没有算错。当前模型拟合不足、不同人类策略混合、标签时机与线上状态漂移，都可能使理论逆变换失准。

合适的下一步是冻结相同权重和相同种子，用未经训练选择的公式变换做短期推理对照，记录真实出牌间隔、费用、满费等待与非法计数，不能把它写成已验证的收敛加速方案。单调 logit 变换不改变 AUROC/AP 排序，也不可能凭校准本身提高模型表达力；它只改变采样概率和固定阈值决策。

## 可复核产物

- 探针：`artifacts/bc100_rootcause/audit_objective_optimization.py`
- 数值证据：`artifacts/bc100_rootcause/objective_optimization_evidence.json`
- timing 后验 CPU 探针与结果：`artifacts/bc100_rootcause/audit_timing_prior.py`、`timing_prior_evidence.json`
- 训练快照：`artifacts/bc100_convergence_20261110/crrl_bc100_notebook/metrics.jsonl`
- 更新 W&B 快照：`artifacts/bc100_convergence_20261110/wandb/history.json`
- 合法 mask 基线：`artifacts/bc_diagnostic_20261110/supervision/bc100/mask_baselines.json`
- 按卡落点先验：`artifacts/bc_diagnostic_20261110/supervision/bc100/card_tile_prior.json`

复跑使用已有 `F:\yzc\Miniconda\envs\cs285\python.exe`，`CUDA_VISIBLE_DEVICES=-1`，TMP 指向 `G:\Code\cr_rl\artifacts\bc100_rootcause\tmp`。CPU 探针可验证数学和实现机制，不能代替真实 183M checkpoint 的 GPU 拟合与 FP16 压测。
