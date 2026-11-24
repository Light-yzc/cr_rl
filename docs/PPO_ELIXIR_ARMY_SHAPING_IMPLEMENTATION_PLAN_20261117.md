# PPO 圣水价值战场势函数：最终实施计划（Agent Handoff）

日期：2026-11-17。项目：G:/Code/cr_rl。状态：**设计方案，尚未实现/验证**。

## 0. 任务与边界

目的：在现有 PPO 终局输赢、塔血、皇冠奖励之外，增加**按实际出牌成本、剩余血量、单位类别加权**的战场单位价值势函数。敌方单位受损/死亡提供即时正向信号；我方单位受损/死亡提供对称负向信号。多单位卡、死亡分裂、产兵建筑和持续召唤必须**按一次出牌的总价值预算分摊**，不因召唤数量或频繁重生而重复计费。

严格要求：

- **仅 Potential-Based Reward Shaping（PBRS）**：`r_army = gamma * Phi_next - Phi_now`，真实终局 Phi_next=0；不可附加独立击杀奖励、独立阵亡扣分、消耗圣水罚分。
- 使用当前 PPO 实际 `gamma`，**不要硬编码 0.9995**。不更改 terminal win/loss、塔血/皇冠奖励、GAE、policy ratio、优势计算等已有语义。
- 现有 `--army-reward` 默认 0.0；旧 `army_potentials()` 是**按实体数量**计权，不按费用。新功能必须显式 opt-in；关闭时与旧 checkpoint 的 reward metadata、数值结果完全兼容。
- 不允许模型拿到额外环境私有信息作为 policy 输入；reward 侧使用经证实的模拟器实体归属信息即可。不得为无法确定来源的实体猜测完整卡牌成本。
- 不修改/中断当前 Colab GPU、CPU 分片筛选或其他长训工作；先本地 CPU 单元测试和可控小规模对照，再单独启动新实验。**此任务不要求启动 GPU。**

## 1. 已确认的项目现状

1. `src/cr_rl/training/rewards.py` 的 `RewardConfig` 含 `army_potential_weight`，默认 0；`army_potentials()` 以 `kind==1`、`side`、`hp/max_hp` 计算均匀实体价值（包含一般部署建筑，排除塔），血量函数为存活时 `0.3+0.7*hp/max_hp`，最后除以 10。
2. `shaped_rewards()` 已正确使用 `gamma*Phi_next-Phi_current`、真正 terminated 时消去后继势函数、truncation 保留后继势函数、fault 时屏蔽 shaping，返回 `hp/crown/army/overflow` 分量。
3. `src/cr_rl/training/state_selfplay.py` 在每次 `env.step` 前保存 `raw`，然后读取 `following`、`actions`、`terminated/truncated/faulted`，调用 `shaped_rewards()`；按对局 lane 和座位记录 `_reward_army`；同一实例可跨 PPO rollout chunk 存活。
4. `scripts/train_selfplay.py` 暴露 `--army-reward`、`--tower-shaping` 及 `--allow-reward-migration`；`state_selfplay.py` 对 checkpoint 的 `reward_config` 严格对比。
5. v4 entity schema 的 `schema.entity` 是动态获取的，相关字段包括 `id`、`row`、`side`、`kind`、`hp`、`max_hp`；**尚未证实 `id` 在整个对局的稳定性，或存在可靠的 parent-cast/parent-spawn 来源字段。** `client/schema_adapter.py` 有从单位名称反推卡牌的 `_spawn_to_card`，但它不是一次出牌与实体的一一因果证明。

## 2. 数学定义和数值原则

将每次成功出牌或独立付费召唤记为一个 `cast group`，记 `g`；同一玩家连续打出两次同名卡，必须有两个独立 group。

```text
effective_budget(g) = actual_cost(g) * category_factor(g)
Phi(seat, state) = beta * SUM_over_live_groups[
    side_sign(seat, g) * effective_budget(g)
    * SUM_over_live_members_i[share(g,i) * health_value(i)]
]
r_army(t) = gamma * Phi(t+1) - Phi(t)
```

- `side_sign`：从座位视角己方 +1，敌方 -1；同一对局两座位应尽量由同一个世界实体台账生成互为相反数的势函数。
- `actual_cost`：优先**本次成功出牌实际消耗**的圣水；若后端无法直接给出，才从与当前后端 revision 匹配的卡表 `ManaCost` 映射，遇 Mirror、英雄技能、进化、费用变化等不能确定的情况不要猜数值。
- `share(g,i) >= 0`，并且对于该次出牌所有可能的初始、分裂、未来召唤成员，**预算分配总额 <=1**；份额出牌时确定，不因当前剩余单位数减少而重新归一化。
- 初始 `health_value(i)` 采用现有兼容形式：`0`（死亡/不存在），否则 `0.3 + 0.7 * clip(hp/max_hp,0,1)`；另外允许独立消融纯 HP fraction，但**不得在第一版未经对照擅自改变既有基线血量函数**。
- 初始类别系数仅作为实验配置：普通部队 `1.0`、普通防御建筑 `0.4`、产兵建筑 `0.6`、由直接付费法术产生的单位 `1.0`；公主塔和国王塔 `0`，继续由现有塔血/皇冠负责。此处数字是可调的**实验初值**，不是已经测得最优。
- 第一批候选 `beta=0.002, 0.004, 0.008`（单位：每个有效圣水点的势函数幅度）；优先 `0.004` 做 smoke，是否保留由实际 shaping 分布/胜率决定，不能单凭示例数值直接上长训。
- 不因产兵建筑自然到期/超时额外发“击杀奖励”；其自然消失仍应导致正常的势函数变化，保证望远镜抵消成立。

### 2.1 各类卡如何分摊

| 类别 | 规则 | 反例/约束 |
|---|---|---|
| 单体普通部队 | `share=1`，整卡圣水预算给本体 | 同名卡两次出牌不可合并 |
| 一次召唤 N 个单位 | 每个 `share=1/N`；不同类型可通过静态配置给固定权重，再以原始全部成员总权重归一化 | **禁止**每个都计整张卡成本，禁止幸存 N 变小后重新加价 |
| 死亡分裂 | 母体和未来子体的份额提前预算，例如母体 0.7、全部子体合计 0.3，具体由卡表/回放抽查确定 | 分裂时不能重新给子体一整张卡成本；有不确定的子体应当拒绝精确映射 |
| 普通防御建筑 | 用建筑类别因子折减整卡预算；自然过期只做势函数消减 | 防御建筑和塔不可混淆 |
| 持续产兵建筑 | 固定 `effective_budget` 再分本体/未来产兵预算；例如本体 50%，全部预计产兵合计最多 50%；未产生的份额不重新分配 | 生成次数多不能使有效总预算超额；不得把产兵再按完整卡费单独计价 |
| 持续召唤法术 | 原法术实际支付成本按固定理论最大/后端明确可证明的总召唤额度分配；异常超额召唤不能追加预算 | 不可按当前已出现数量重新分配；法术直接塔伤仍由塔血奖励处理 |
| 进化、镜像、英雄能力 | 根据该次真实施放形态与实际消耗建立单独预算；技能有额外消耗时单独预算 | 不可用基础卡费用替代所有变体，也不得凭单位名猜来源 |
| 非出牌产生的实体/不可证明来源 | 明确记录为未知，`share=0` 且统计覆盖率，或直接阻断新模式启用（按预检门槛） | 不允许“没映射到就按一张完整卡算” |

### 2.2 大前提

PBRS 的策略排序不变性质依赖**一致的 gamma、可定义的状态势函数、终止状态为零、时间截断的正确 bootstrap**等前提。实际 PPO 有函数近似、GAE、有限样本、状态可见性和非完美执行，不能声称应用 PBRS 就一定不影响胜率。给出 test 来验证代数性质，并用独立对照验证性能。

## 3. 实施阶段（按顺序执行，禁止跳过）

### P0：后端能力核查，先交付可审计报告

查看 runtime schema `entity.id/row/side/kind/hp/max_hp`、动作回执、backend `infos`、
`crforge_env` 数据和已安装 native 卡表。确认：

1. entity id 是否跨帧稳定、是否会复用；same side 的相同单位同时出现时能否区分。
2. 是否有真正的 `spawn -> cast group` / `spawn -> parent entity` 因果字段；如果没有，正常出牌能否依据 **已确认成功的动作 + 本 tick 的新 entity id + 类型/位置/时间** 无歧义归因。
3. 卡表是否能可靠解析初始召唤数量、死亡分裂、持续产兵、技能召唤和各类费用。
4. 在当前 PPO 的 8 tick (0.4s) 间隔下，一个 tick 内多个生成/消失事件会不会被观测采样漏掉。**不能把仅仅“实体出现在下一帧”误当精确击杀事件。**

输出后端支持/不支持矩阵、未知类别清单及至少 3 个原生模拟器回放证据。**若无法可靠追踪母子关系，不允许对该特殊类别做“精确卡费分摊”的虚假承诺**；可以先只对可证明的普通单体/静态群体出牌开新模式，其他类零值并报告覆盖率，或者先完善后端来源事件。

### P1：卡牌价值配置与每局来源台账

增加版本化的卡费与 spawn 分配配置（例如 `configs/army_elixir_values.yaml` 和 loader），
来源优先使用已安装且版本固定的 native tables，特殊类需要显式 override、证明及测试。

实现 per-env/lane 的 `ArmyElixirLedger` 或等价状态机：

- `cast_id=(episode_id, seat, unique_cast_sequence)`；`entity_id -> (cast_id, share, lineage)`；按 backend 证实的 entity 生命周期更新。
- 出牌**实际确认成功**后建立预算；未执行、拒绝、法术无召唤、手牌过时不可凭 policy proposal 建预算。
- 新一帧的同一 group 总体价值不能超过有效预算；预算耗尽后新生成单位不给额外价值；避免在同一实体消失后把其份额再分配给存活成员。
- 对同时出牌、同类型多个来源、镜像、特殊进化，以及同帧“父死亡/子出生”设置无歧义映射规则；不能确定则不凭猜测赋值。
- 独立 episode/lane 复位；同一局跨 rollout chunk 时**保留 ledger**；真正结束或环境 reset 后清空。故障、truncation、完整 BPTT 与 checkpoint 恢复不能把上局来源带到新局。
- 只在配置加载/后端初始化时解析卡表，热路径按 id/数组处理；**不做逐步磁盘 I/O、大文件哈希、解压或重新读取 Parquet**。

### P2：接入现有 reward 路径，严格兼容旧模式

计划修改：`src/cr_rl/training/rewards.py`、`src/cr_rl/training/state_selfplay.py`、
`scripts/train_selfplay.py`；如确需后端支持，再单独说明并修改相应适配层。

- 增加 opt-in 的 `army_reward_mode=elixir_potential`（命名可与现有 CLI 风格保持一致），
  原先 `--army-reward` 的均匀单位势函数保持完全相同；默认关闭不变。
- 因新 ledger 有状态，计算顺序必须确保 **pre-step Phi 来自旧实体状态**，
  env.step 后处理动作执行与实体出生/消失，再计算 **post-step Phi**，最终调用
  与现有完全一致的 `gamma*post-pre`、true terminal 清零、truncation bootstrap、
  fault masking。不要将新类强行塞入 stateless `army_potentials(observations)` 而丢失 lineage。
- 旧 `RewardConfig.metadata()` 对旧模式的结果保持原始 v1 契约，避免让旧 checkpoint
  因无关字段变动无法 resume；新模式使用明确的新版本、成本表版本、类别系数、
  beta、生命周期配置与 provenance，在 checkpoint 存储，并严格检测 resume mismatch。
- 只在**新实验目录/新 checkpoint 分支**通过现有 `--allow-reward-migration` 显式迁移；
  原 PPO optimizer、Critic、新 reward 尺度的复用风险需记录，可做 Critic 重热身对照，
  不可默默覆盖正在进行的 U929/PPO 长训。
- `_reward_army` 和现有 `hp/crown/overflow` 日志保持可比较，并额外记录
  `army_phi_before/after`、`army_value_own/enemy`、`unknown_lineage_count`、
  `mapped_member_rate`、`budget_violation_count`、`natural_expiry_count` 等。

### P3：CPU 测试（必须全部通过）

在 `tests/test_reward_shaping.py` 基础上增加新测试文件，例如
`tests/test_army_elixir_shaping.py`，至少覆盖：

1. **预算守恒**：单体 4 费总额 4，3 费/5 单位每个 0.6；
   死 1 个仍按原 N=5，其余单位不涨价；同名卡二次出牌预算互不污染。
2. **分裂/产兵**：父死亡+子生成的同 tick 交易不产生超额预算；
   产兵 1 次和 10 次都不超出已声明的整卡额度；建筑自然到期不算独立击杀分。
3. **实际执行**：被拒绝/跳过的出牌绝不创建 cast group；
   同类型多来源的并行情况没有准确证据时 fail closed；来源 ID 重用不造成幽灵价值。
4. **双座位**：镜像状态下两个座位的 army potential/reward 互为相反数，
   未知 `hp/max_hp`、实体数上限、法术与塔不造成 NaN。
5. **时间与抵消**：真实 terminated 下一势函数严格 0；
   truncated 不强制清零，faulted shaping 为 0；不同 gamma 对应
   `sum_t gamma^t r_army(t) == -Phi(s0)`（真正终局）；
   部署后死亡的完整过程无永久重复罚款。
6. **旧兼容与恢复**：旧模式 reward 数值/metadata/checkpoint 不变；
   新模式 resume 需要明确迁移/一致参数；
   rollout chunk 边界来源连续，reset 后 id 无污染，vector lane 独立。
7. **性能**：原生模拟器上的少量真实对局 smoke，不依靠纯造假数组证明出生、分裂追踪；
   对比关闭模式的每环境步吞吐，不允许引入明显的 Python 全量查找/数据 I/O 开销。

新增测试同时覆盖 normal PPO 与 full-episode BPTT collector 路径；运行项目原有
`tests/test_reward_shaping.py`、相关 PPO resume/regression，提交命令与结果。

### P4：微型实验与诊断门槛

用独立 experiment/run dir、相同起点 U929 Actor/Critic、相同对手池/牌组分布、
相同随机种子与采样预算，至少比较：

- A：当前 PPO（不启用 army shaping）。
- B：原有均匀单位势函数（例如 `--army-reward 0.015`）。
- C：新圣水价值势函数（首测 beta=0.004，后做 0.002/0.008 敏感性）。

先做少量环境步、1~2 个 PPO learner update 的 smoke：有限数值、反向梯度、
GAE/advantage、PPO ratio、KL、Critic loss、checkpoint resume、
两座位及对手池、终局/截断奖励正确。再做同预算固定评估牌组对照，输出
终局胜率、塔血/皇冠、圣水交换、WAIT/部署比率、平均与 P95/P99 单步 army reward、
army shaping 与 terminal reward 比例、来源映射覆盖率和预算违规计数。

任何预算超额、未确认动作创造价值、不可解释的正负脉冲、terminal telescope
失败、GPU NaN 或改变旧 baseline 行为，均为 blocker；如“映射覆盖率低”也须报告，
不可通过调大 beta 掩盖。**不能只因总 reward 变高就宣称策略改进。**

### P5：交付物与停机条件

交付代码、单元/集成测试、版本化配置、preflight 报告、最小对照实验命令与结果，
以及简明迁移说明 `docs/PPO_ELIXIR_ARMY_SHAPING_RESULTS.md`。对已运行训练保持不改动。

必须在结果文档逐项列出：backend 来源是否可靠、各类卡覆盖率、未知卡和 fallback、
是否需要修改模拟器、测试数量、性能影响、实验指标、已知限制以及**是否建议开启长期训练**。
未通过上述验收之前，新模式默认关闭，不可自动接入云端正在运行的训练。

## 4. 交给 Agent 的执行要求

按 P0→P5 顺序工作；先审计，发现没有可靠的 parent/cast provenance 就说明真实缺口，
不能凭名称或死亡时刻伪造精确归属。保留旧 reward 实现作为可回滚基线，修改最小化，
优先选择低成本、可缓存、可向量化的热路径。每完成一阶段跑测试，并输出 diff、
验收结果和未解决问题；能做到的部分可先交付，但**不允许宣称未实现的追踪与验证已经完成**。
