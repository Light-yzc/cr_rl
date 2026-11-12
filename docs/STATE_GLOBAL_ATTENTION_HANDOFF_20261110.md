# KDA＋整局全局注意力与0.4秒决策间隔交接

日期：2026-11-10（北京时间）。本文件汇总侧聊中的方案，供主 Agent 结合正在进行的本地诊断实施。

## 用户确认的方向

1. 时序核心考虑 KDA 与全局 Attention 混合。**Attention 层读取开局至当前的完整因果序列，不采用 sliding window。**
2. 通过减少全局 Attention 层数、采用 GQA、更多使用 KDA 控制成本；暂不优先引入 MLA。
3. 统一决策间隔以 **0.4秒＝8个50ms模拟步** 为默认待测配置。0.5秒是后续提速候选，不做双频决策／历史摘要方案。
4. 先本地验证，再给出正式训练方案。本文不是已实现、已测得更强或已获准替换生产 checkpoint 的证明。

本次侧聊仅新增本交接文档，没有改训练代码、数据、配置或 checkpoint，没有启动训练或云实例。主 Agent 正在执行的实验由主 Agent 自行检查和接续；不要因本文重复启动 GPU 任务。

## 当前结构与待解决的计算问题

当前 BC checkpoint 为122.6M级模型：实体 Transformer 4层×256维，时序 Transformer 16层×768维，context128；每帧 dynamic4 pooling 最终汇总为一个256维 frame 特征。当前观测来自引擎结构化状态，不是模型生成的观测。

- 历史特征由共享实体编码器从真实历史观测生成；训练时使用当前权重重算并反传。
- 在线推理启用增量帧缓存时，旧帧实体特征可以复用，但**整个时序窗口仍重新运行**。
- 当前时序使用窗口内 learned absolute position 和 causal mask。不能直接追加普通 KV cache 并声称与旧滑窗输出等价。
- 当前没有地图 CNN，但已有实体坐标／Fourier特征，以及按卡牌条件化的18×32落点 query 对当前实体的 cross-attention。

已核查入口：

- `.tmp_latest/src/cr_rl/models/policy.py`：`encode_history`、`encode_temporal`、`_placement_logits`、`_joint_log_probs`。
- `.tmp_latest/src/cr_rl/models/incremental_frames.py`：`IncrementalFrameEncoder`，文件头明确时序仍全量重算。
- `.tmp_latest/configs/backends.yaml`：`tick_ms: 50`、`decision_ticks: 5`。

`.tmp_latest` 是本地诊断／同步源码树，与根目录生产源码可能不同；实施前比较版本，不盲目复制覆盖。

## KDA的准确含义

KDA不是“GRU分支与Attention分支加一个二选一门”的同义词。它在每一层内部维护多头矩阵状态，用当前 query 读取更新后的 state，输出中间特征。

以每头 `S ∈ R^(d_k×d_v)`、列向量 q/k/v 为例，省略多头、缩放、归一化和短卷积：

```text
S_bar = diag(alpha_t) @ S_prev
S_t   = S_bar + beta_t * k_t @ (v_t - S_bar.T @ k_t).T
o_t   = S_t.T @ q_t
```

- alpha：按key维度遗忘旧记忆；beta：控制delta写入。
- delta项先计算当前key对应的记忆预测，再写入与value的差异。
- q是时序层的特征query，不是最终选牌query。
- 每层／环境／座位／模型各有独立state。不能把一个state跨层、跨座位共用。
- 使用短卷积时，还需保存对应卷积状态。
- 递推state可能携带窗口外信息，但它是有损压缩记忆，不保证保存或精确检索全部过去。

Kimi Linear原版采用3:1的KDA与**全局MLA**混合。本项目下面的GQA候选是适配方案，不是照搬原版，收益未经验证。

参考来源（2026-11-10已联网读取）：

- 官方说明：https://github.com/MoonshotAI/Kimi-Linear
- KDA参考递推：https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
- KDA层定义：https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py
- MLA官方实现：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

## 首个候选结构

```text
当前结构化观测
       ↓
保留／迁移现有实体 Transformer
       ├──────────────────────────→ 当前实体 tokens
       ↓                                  │
frame256 → 投影512                          │
       ↓                                  │
[KDA → KDA → KDA → 全局GQA] × 2            │
       ↓                                  │
最终 history context                       │
       ├→ WAIT／手牌／技能分布              │
       └→ 所选牌＋history context＋实体tokens
                           ↓
                     条件落点分布
```

起始超参仅为待测值：

| 项目 | 候选 |
|---|---|
| 时序宽度／模块数 | 512／8，含6个KDA和2个全局GQA |
| GQA | 8个query heads、2个KV heads、head_dim64 |
| KDA | 首先验证匹配宽度的8头、每头key/value维度64；实际扩展率与参数数以实现统计为准 |
| Attention范围 | 所有已发生的真实时间步，causal；新对局清空 |
| 位置表示 | 与整局递增位置兼容，候选相对时间或RoPE；不再按滑窗反复重编号 |
| 首轮保持 | 实体分支与条件落点结构；不同时新增CNN、相对实体偏置和整套输入改造 |

每个block带各自的归一化、残差和FFN。KDA与GQA是**交替堆叠**，不是额外并行GRU融合分支。所有全局层都保留完整历史KV；KDA层只保留自身递推状态。

当前策略实现会预计算4张手牌各自的条件落点图，再组成 `P(card|s) × P(tile|card,s)` 的联合分布采样，数学上等价于先选牌再选该牌落点。新结构仍需让落点头读取最终history context、当前实体与对应卡牌；不可从实体编码器直接跳到落点而漏掉这些条件。WAIT不执行落点。

这不是旧时序主干的无损替换。512维新核心不能直接加载768维旧时序参数；若保留旧策略头，可评估512→768适配层，但不能宣称初始策略等价。保存独立架构版本、配置和新checkpoint，明确optimizer如何初始化／迁移。

## 全局注意力的规模估算

当前默认时间线为180秒常规＋120秒加时。4Hz约1200帧／座位；0.4秒决策约750帧。结算及特殊规则可能延长，对缓存上限要留余量或允许增长，不能将1200／750当作严格通用上限。

以下统一按512维、8个Q头、2个KV头、64维head、FP16 KV缓存、8路座位计算。只计KV，不含独立critic／其他模型的额外缓存、权重、optimizer、state或训练激活。

| 全局GQA层数 | 1200帧KV，8路总计 | 750帧KV，8路总计 | 1200帧最后一步attention FLOPs／座位 | 1200帧整局累计attention FLOPs／座位 |
|---:|---:|---:|---:|---:|
| 2 | 9.375 MiB | 5.859 MiB | 4.915 MFLOPs | 2.952 GFLOPs |
| 4 | 18.750 MiB | 11.719 MiB | 9.830 MFLOPs | 5.903 GFLOPs |
| 8，全GQA对照 | 37.500 MiB | 23.438 MiB | 19.661 MFLOPs | 11.806 GFLOPs |

公式（乘加计2 FLOPs）：

```text
KV bytes = L_global × T × 2(K,V) × H_kv × d_head × bytes_per_scalar × streams
attention FLOPs，最后一步 ≈ 4 × L_global × T × d_model
attention FLOPs，整局     ≈ 4 × L_global × d_model × T(T+1)/2
```

上述FLOPs不含Q/K/V/O投影、FFN、KDA、实体编码、策略头、反向传播及内核调度，不能据此推算已达到的毫秒延迟或端到端提速倍数。

- KV cache下，每步只处理新帧；全局注意力单步随T线性增长，整局累计平方增长。
- GQA减少KV存储和K/V投影；保留8个Q头时，不会把attention匹配／读取计算直接降到四分之一。
- 对比参考：同为8层512维、8路、1200帧，普通MHA的FP16 KV约150 MiB，也并非很大。因此不因长文本MLA收益而预设本项目必须采用MLA。
- 0.25→0.4秒使全局token约变为62.5%，理想整局attention项约变为39.1%；总训练／采样耗时不会必然按此比例变化，模拟器依然运行50ms物理步。

初步判断：**整局全局Attention在这个序列规模上可作为可行候选；真实吞吐、训练内存和学习效果必须本地实测。**

## 为什么选择0.4秒，部署1秒不等于全局操作锁

当前数据表中Knight、Giant、Pekka等DeployTime为1000ms，Golem为3000ms。这限制的是单位部署过程，不能据此认定玩家期间不能发送其他牌。

当前模拟器还实现命令发出到执行的固定20 ticks延迟：`Standard1v1Battle.PLAY_DELAY_TICKS = 20`。`PlayerCommands`允许不同牌的多个合法命令同时等待，按已承诺圣水和同卡pending等规则限制，而不是全局锁1秒。此处描述本项目当前后端实现，不冒称已经验证所有线上客户端细节。

官方案例支持亚秒时机有战术意义：公主曾将部署1.0→1.2秒、首击0.5→0.3秒以改善高延迟玩家桥头拦截；Vines在2026年3月将部署0.4→0.9秒以增加反应空间。资料用于区分机制，不用于推导某个间隔已获胜率验证。

- https://supercell.com/en/games/clashroyale/blog/release-notes/march-balance-changes/
- https://supercell.com/en/games/clashroyale/blog/release-notes/march-balance-changes-2026/
- `.cache/replay-viewer/tables/16.635.3/characters.json`
- `.cache/replay-viewer/tables/16.635.3/battle_timelines.json`
- `.tmp_crforge/core/src/main/java/org/crforge/core/battle/unit/PlayerCommands.java`

| 固定间隔 | 近似均匀事件下等待下一次决策的均值 | 最大额外网格等待 | 5分钟token |
|---|---:|---:|---:|
| 0.25秒，当前 | 125ms | 小于250ms | 1200 |
| **0.40秒，默认候选** | **200ms** | **小于400ms** | **750** |
| 0.50秒，后续提速候选 | 250ms | 小于500ms | 600 |

这是决策网格等待，不是总命令生效时间或部署时间。0.4相对0.25减少37.5%token；0.5相对0.4再少20%。0.4是折中候选，尚未证实最优。

## 数据与训练器必须配套

1. **BC不可直接隔行删帧或仅改运行配置。** 现数据为0.25秒网格，0.4秒无法整步下采样。优先从原始replay命令与50ms模拟轨迹重建；保留原命令时间与状态／标签因果对齐。
2. 明确新网格上的动作映射规则。审计一格内多个出牌／技能、标签移动、执行状态变化、遗漏与非法率；不能静默丢命令或把未来状态用于当前标签。若采用格内延迟动作队列，它属于额外动作空间变更，不应暗中混入首轮。
3. 更新metadata、decision_ticks、时间戳／位置、数据版本与hash。原数据保持不变；报告监督动作保留率和时机量化误差。
4. 从随机单状态窗口改为同局同座位的连续chunk训练，复用每帧计算并监督多个时刻。划分仍按对局，不能将同局chunk分到训练和验证两侧。
5. 区分记忆跨度和梯度跨度。跨chunk保留state／KV可以看完整历史；detach会截断梯度，不能称为完整整局BPTT。需要明确选择、内存成本和训练语义。
6. 权重更新后state／KV可能过时；有限burn-in不能保证恢复所有远期记忆。定义重建／近似策略并单独验证，不能未经检查长期复用旧权重缓存。
7. PPO接入时记录或重建序列起点状态，保证更新前old_logp与采样一致；跨局／座位／对手模型切换不串state。若为省算跳过强制WAIT，仍须按真实时间推进记忆或明确缺帧语义。
8. 决策间隔变化时，同128 token覆盖32秒→51.2秒。比较需同时报告真实时间跨度、token数、独立监督动作数，不能只匹配update步数。
9. PPO的gamma、lambda、每步奖励和计时统计要按物理时间审计。若原折扣按0.25秒步定义，保持每秒衰减时可用 `gamma_new = gamma_old^(0.4/0.25)`，lambda同理仅在需要保持该衰减口径时调整；终局奖励不因步长改变而简单放大。该项不影响当前BC loss。

## 主Agent建议执行顺序

### 阶段一：隔离验证核心正确性与硬件支持

- 先检查主线程现有本地实验结果，避免架构试验覆盖输入／归一化诊断结论。
- 新建独立诊断模型，保留现有生产文件和checkpoint。先检查KDA／GQA所选实现能否在RTX4090及目标V100上运行；不能假设最新FLA、Triton或FlashAttention内核都支持V100。
- 使用现有环境，不重新下载Torch／建大环境。所有脚本、TMP、cache、数据和输出放G盘；本地限制RAM，GPU任务串行。此前可用解释器为 `F:/yzc/Miniconda/envs/cs285/python.exe`，运行前核实。
- FP16为目标，验证短／长序列的loss、梯度和state范围。KDA参考实现内部使用FP32计算，不能因接口接受FP16就宣称纯FP16递推已稳定；实际state／累积dtype及其内存成本要单列。
- 核对逐帧递推与chunk前向／梯度、因果性、padding、对局reset、不同chunk划分、batch内不同长度及缓存上限。

### 阶段二：架构比较，暂保持0.25秒以便归因

- A：现有模型作参照。
- B：相同512宽度／8层预算的全局GQA核心，作为没有KDA的对照。
- C：本文6KDA＋2全局GQA候选。B/C使用一致的输入、监督、时序长度、损失归一化及优化预算；报告参数量差异。
- 不因小集训练loss下降就发布。看新的未调参控制集／完整验证的timing、选牌、条件落点及原协议实战；主线程反复查看的64状态开发集不是新的最终留出证据。

### 阶段三：在选定架构上比较0.25与0.4秒

- 仅改变决策时间尺度及必要的数据／折扣换算，使用对应重建数据，避免与架构更新混在一个无法归因的结果中。
- 记录出牌／技能保留率、命令延迟、桥头拦截／法术时机、故障／非法率和固定换边战绩。
- 0.5秒仅在0.4验收后作为后续提速比较，不预设一定更好或无损。

### 性能报告至少包括

- 模拟器、观测处理、实体编码、时序核心、落点头、前后向、optimizer、数据等待、保存各阶段时间。
- 早／中／末局长度下，单路与实际并行数的延迟、吞吐，尤其最长序列。
- 峰值RAM、显存、各模型KV／递推state大小；不把KV缓存大小当成训练总显存。
- 以相同监督量和实战门禁比较达到效果的墙钟时间，不只比较token/s。

在这些检查完成前，本文仅提供可实施候选，不承诺更快收敛、战力提升或具体加速倍数。
