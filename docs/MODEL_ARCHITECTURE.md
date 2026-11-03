# 当前通用策略架构（v2 修订版：架构版本 2 + 独立 Critic 训练版本 3）

更新：新 PPO 训练默认使用[独立 Transformer critic](INDEPENDENT_CRITIC.md)，
并保留下述 BC actor 的动作路径。本文同时涵盖当前已实现代码与 **v2 架构升级方案**（Tile Head 残差/FP32/2D 坐标编码、实体离散特征 Embedding 化与单帧推理缓存）。
新的训练 checkpoint 版本为 3，部署 actor 仍使用版本 2 的策略权重布局并标注 `policy_only=true`。

这是读取结构化游戏状态的 Transformer Actor-Critic，不是文本语言模型。通用性指同一套参数以卡组为条件控制不同组合；参数规模或支持卡组输入本身不能证明泛化能力。

```mermaid
flowchart TD
    O[最近128帧结构化观测] --> E[每帧实体 Transformer: 理解实体交互]
    G[每帧全局状态 Header + 手牌/卡组上下文 c_t] --> QP[Static + Dynamic Query 生成器: Q_t = Q_static + MLP(c_t)]
    E --> P[条件多查询实体池化: Cross-Attention(Q_t, K_t, V_t)]
    QP --> P
    G --> P
    P --> T[因果时间 Transformer]
    D[自身8张卡 + 敌方已公开卡牌] --> C[共享卡牌嵌入与池化]
    C --> P
    T --> H[当前决策向量]
    C --> H
    H --> A[WAIT / 4个手牌槽位 / 2种技能]
    H --> V[独立 Critic / 共享价值头 V]
    F[当前4张手牌的卡牌身份] --> A
    F --> Q[每张手牌的576个落点查询 + 2D网格坐标]
    H --> Q
    E --> X[FP32 查询对当前帧实体做交叉注意力 + Query残差]
    Q --> X
    X --> L[条件落点概率]
    A --> J[合法动作掩码下的联合分布]
    L --> J
```

## 输入和时间（含 v2 表征与动态查询池化升级）

模拟器每侧原始观测 7,024 个整数，经适配器转换为全局状态与最多 200 个实体的 34 维特征。全局状态包括资源、塔状态、手牌、卡组等；实体含类型、位置、生命及其他状态。网络额外显式读取当前手牌身份、自身完整卡组和敌方已经公开的卡牌，不直接输入敌方未公开手牌。这里只描述网络使用的字段，模拟器的信息可见性仍需持续审计。

每帧先在投影后的实体之间做 self-attention，再池化成一个帧向量；与全局状态编码组合后，组成时间序列。时间 Transformer 使用因果注意力和有效帧掩码，不跨对局拼接。当前决策每 250ms 一次，128 帧覆盖约 32 秒。

**v2 输入、动态查询池化与时序表征升级要点（v2 全新重训，不背负旧 v1 权重约束）**：
1. **实体离散 ID 与位掩码表征修复**：在 v1 实现中，场上实体的 `row`（兵种/建筑/投射物表 ID，`0~1466`）、`kind`、`state`、`buff` 仅除以常数归一化后送入线性层，`flags`（`uint16` 位掩码）直接除以 `65535.0`。**v2 升级方案**为 `row`、`kind`、`state`、`buff` 引入独立的可学习 `nn.Embedding`，将 `flags` 拆解为 16 维 0/1 位特征，并为 Header 中的 `next_card`、`pending_card`、`ability_champion`、`enemy_play_card` 共享 `card_embedding`。
2. **Static + Dynamic Query 实体注意力池化（核心升级）**：
   - 将架构职责明确划分为两层：**Entity Transformer 自注意力**负责理解场上实体之间发生了什么交互；**Dynamic Query Pooling** 负责根据当前局势与手牌资源，动态决定重点从实体集合中提取哪些信息。
   - 取代均匀平均的 `Masked Mean`（会被骷髅海等人海单位稀释关键威胁）和提问方式固定的纯 `Static Query`，v2 采用 **Static + Dynamic Queries**：
     $$Q_t = Q_{\text{static}} + \operatorname{MLP}(c_t), \qquad H_t = \operatorname{Attention}(Q_t, K_t, V_t)$$
     其中 $c_t$ 为当前帧的全局上下文（由当前圣水、手牌、双方卡组、双方 6 座塔实时血量及比赛阶段构成）。由于 $c_t$ 仅依赖当帧全局状态而不依赖上一帧时序 Transformer 输出，128 帧依然保持 100% 并行计算。
   - 支持四档消融配置：`Masked Mean` $\to$ `1 Static Query` $\to$ `4 Static Queries` $\to$ `4 Static + Dynamic Queries`（v2 默认）。
3. **卡组条件前置与 Rollout 单帧实体缓存**：将己方 8 张卡与敌方已公开卡牌的池化向量提前融入每帧上下文，使时序层能够结合双方卡组结构分析历史帧；在采样/推理阶段（权重冻结），缓存历史帧经过 `EntityFrameEncoder` 后的帧向量，每步仅对最新 1 帧计算 200 实体自注意力与动态池化，将 Rollout 吞吐提升 5~10 倍；训练反向传播阶段仍完整重算 128 帧窗口。

## 动作和价值（含 Tile Head v2 修复）

策略分两级：先选 WAIT、四个手牌槽位之一或两种技能之一；如果出牌，再选择 18×32 网格中的一个落点。每张手牌、每个落点的查询会关注当前帧实体，输出条件落点分布。

**Tile Head v2 关键升级**（详见 [`TILE_HEAD_DIAGNOSIS_20261103.md`](TILE_HEAD_DIAGNOSIS_20261103.md)）：
- **Query 残差连接（`tile_query_residual=True`）**：修复 v1 中 `tile_cross_attention` 仅将注意力输出送入 `tile_actor` 而丢失 576 个网格自身身份导致的空间坍缩（`queries = queries + attn_out`）。
- **FP32 落点头计算（`tile_head_fp32=True`）**：在 BF16 混合精度训练下强制落点交叉注意力与打分头以 FP32 运行，消除初期微小空间差异被 BF16 舍入抹平的问题。
- **2D 网格空间坐标编码**：为 $18 \times 32 = 576$ 个 `tile_queries` 注入与场上实体坐标 $(x/18000, y/32000) \in [0, 1]^2$ 对齐的 2D 网格坐标编码，使每个落点查询天然具备几何位置先验，无需靠稀疏标签死记硬背 576 个独立向量。

联合概率为 `P(手牌槽位) × P(落点 | 槽位)`。展平成 WAIT 1 个 + 4×576 个出牌动作 + 2 个技能，共 2,307 个动作。无法支付、不能放置或技能不可用的动作被合法掩码排除。不是在所有帧强制出牌。

BC 使用人类动作监督策略，不训练 critic；PPO 默认采用与 Actor 同构的**独立 Transformer Critic** 预测未来折扣回报（包含零和塔血差、皇冠差与终局胜负的合成回报，见 [`BC_TO_PPO.md`](BC_TO_PPO.md)）。

## 两种大模型

| preset | 参数 | 实体编码器 | 时间 Transformer | 上下文 |
|---|---:|---|---|---:|
| production | 119,541,766 | 4 层，256 维，8 heads | 16 层，768 维，12 heads | 128 帧 |
| general_300m | 309,233,670 | 4 层，256 维，8 heads | 24 层，1024 维，16 heads | 128 帧 |

两者共享动作格式、卡组条件输入和训练方法。309M 主要增加时间主干容量，不能仅凭参数数量断言更强；先建立 119M 的 BC→PPO 基线，再做等样本/等时间对照。

自博弈里的 learner、本轮行为参考和历史对手，都是这个架构的不同权重副本。本轮行为参考与对手冻结，只更新 learner。当前对手混合为 35% 当前自身快照、50% 历史 PFSP、15% 历史均匀；第一轮没有历史时使用自身初始快照。尚无独立训练的剥削者联盟，也没有蒸馏学生模型。
