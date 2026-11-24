> 2026-11-16 状态更新：此文包含历史接口/缺口描述；当前接入状态与本轮验收见 [MODEL_INTERFACE_INTEGRATION_20261116.md](MODEL_INTERFACE_INTEGRATION_20261116.md)。

# 实机模型输入缺口交接（2026-11-16）

本次只审计并交接，不修改模型或实机字段推断逻辑。

## 1. 模型及接口基准

- 本地核对的最新导出：`artifacts/ppo_v3_v100/actor_export.pt`，PPO update **929**。
- architecture-v3、8 ticks / 0.4s，包含已训练的 `wait_hand_gate`。
- 原始模型输入：`int32[7024]`，224 个 header 字段 + 200×34 个实体字段。
- 另外传入 `bool[2307]` 合法动作掩码；它会直接改变模型可选择的动作。
- 实际字段顺序以 checkpoint 的 `config.header_fields/entity_fields` 为准。
- `CrForgeObservationAdapter` 会把实体 `id`、`target` 两列归零，再送入模型；
  目标坐标 `target_x/target_y` 仍会输入模型，不能因此停止解码目标引用。
- 其余数值/类别经归一化或嵌入进入模型。存在于接口并不证明模型依赖程度：
  本交接是覆盖审计，不是特征消融实验。

关键源码：

- `src/cr_rl/client/nulls_reader.py`：内存读取和 `NullsEntity` / `NullsBattleSnapshot`。
- `src/cr_rl/client/schema_adapter.py`：原始快照 → 模型字段。
- `src/cr_rl/client/schema_v4.py`：字段下标和单位。
- `src/cr_rl/data/observations.py`：实际模型预处理。
- `src/cr_rl/models/policy.py`：实体编码器。
- `.tmp_crforge/env/src/main/java/org/crforge/env/Schema.java`、`LadderEnv.java`：
  训练侧字段语义。这里用于核对语义，不表示所有客户端偏移已经验证。

## 2. Motion vector 不是全部缺失

`E_DX=12`、`E_DY=13` 已由连续两次模型观测的实体坐标差得到：

```text
dx = current_seat_x - previous_seat_x
dy = current_seat_y - previous_seat_y
```

单位是棋盘游戏单位，不是像素、不是每秒速度，也不是客户端瞬时速度。
与训练 schema 一致：本次新出现/上次未出现的实体用 (0,0)。负数表示方向，
所以 dx=-1 或 dy=-1 本身不能当作缺失证据。坐标要先转为己方视角再相减。

交接验收：两个座位、静止、移动、同帧重复转换、实体消失后重现、ID 复用、
tick 回退、漏帧后的位移，以及 projectile/area 的位移分别测试。
现有实现同 tick 重复调用 adapt 会更新历史位置，可能把第二次位移清零；
控制循环目前每帧只调用一次。接手方如果增加调用位置，必须保证幂等或缓存。

## 3. 首批需要补的实体输入

下表数字均为实体行内下标。除注明外，当前没有已验证的实时来源。

| 优先级 | 字段 | 现状 | 接手方需要提供的真实语义 |
| --- | --- | --- | --- |
| P0 | `state` (9) | `NullsEntity.state` 有槽位，reader 未赋值；adapter 用移动/目标/出生时间猜测 | 当前状态及客户端枚举到训练枚举映射；不是直接照搬枚举数字 |
| P0 | `attack_ms` (18) | 可接受可选值，但 reader 未读取，通常 -1 | 已经过的攻击计时，毫秒；不是剩余前摇 |
| P0 | `load_ms` (19) | 恒 -1 | 当前攻击 wind-up **剩余**毫秒；不能用基础 HitSpeed 减一个猜测值 |
| P0 | `speed_percent` (20) | 恒 -1 | 当前 Buff 作用后的移动倍率；中性100，停止0；不是当前是否真的在走路 |
| P0 | `hit_speed_percent` (21) | 恒 -1 | 当前攻击节奏倍率；中性100，眩晕/冻结等停攻0 |
| P0 | `flags` (10) | 只构造 alive/air/building/king 等部分位 | 检查并补齐训练定义的可见状态位，尤其无法移动/不可选中/隐身等；逐位核对，不复制客户端位图 |
| P1 | `deploy_ms` (17) | 未读取；按首次见到时间和表中部署时长推算 | 真实剩余部署毫秒，避免中途接入、漏帧、召唤等误判 |
| P1 | `aim_x/aim_y` (29,30) | 恒 -1 | projectile/area 等对象的实际瞄准或落点；转换到己方坐标；不要用当前坐标代替 |
| P1 | `time_left_ms` (31) | 恒 -1 | 相关实体剩余寿命，毫秒；法术区域/临时建筑等分别核对语义 |
| P1 | `radius` (32) | 恒 -1 | 当前有效作用范围，游戏单位；随阶段变化时不能一直使用静态表值 |
| P2 | `ability_ticks` (33) | 恒 -1 | 训练 schema 定义的技能剩余 ticks；无技能/不适用时遵循对应 sentinel |

注意：projectile/area 的 state=-1 等可能本来就是训练定义中的“不适用”。
请按实体 kind 区分“合法 sentinel”和“本应可知但未读”，不能把所有 -1 清零。

速度倍率可以由完整、验证过的 Buff 状态计算，也可以从已验证的有效属性读取。
但不能简单把所有 Buff 的表中数字相乘：叠加、减速、冻结、覆盖规则都需要验证。
只有确认 Buff 列表完整且为空时，中性100才有依据；`buffs=None` 不等于空列表。

## 4. 出牌及资源状态缺口（header）

这些字段对“攒费、等牌、避免重复出牌”尤其值得优先核对。

| 优先级 | 字段/下标 | 现状及要求 |
| --- | --- | --- |
| P0 | `elixir_reserved` (3) | -1；读取已接受但尚未兑现命令预留的圣水。控制器 pending 锁不等于服务端预留 |
| P0 | `hand_refill_ms` (6) | -1；需要真实补牌剩余时间，避免把过渡手牌当可出手牌 |
| P0 | `pending_count` (53)、pending记录 (65..80) | -1；需要命令卡牌/坐标/剩余ticks，与已发送未确认状态区分 |
| P0 | `play_open_in_ticks` (61) | -1；需要出牌窗口何时开放，不要仅凭 running 字节判断 |
| P1 | `full_bar_ms` (59) | -1；需要训练语义下的满费持续时间。中途接入只能从观测起累计，不能伪称掌握此前累计 |
| P1 | `last_refusal` (62)、`last_run_refusal` (63)、`illegal` (64) | -1；客户端可能没有与模拟器完全同构的拒绝码，需定义/验证映射，不应杜撰“成功” |
| P1 | `hand_pending` (15..18) | 仅由本机发送锁设置；如支持读取游戏命令队列，应补充真实补牌/占用状态 |

重要已知问题：曾读到 running 标志为 false 但 tick/time 仍推进，不能把该字节
单独用作禁止出牌的依据；已有 `BattleClockGate` 修复，交接时不要恢复旧门控。

## 5. 形态、技能及敌方历史

| 优先级 | 字段组 | 现状及要求 |
| --- | --- | --- |
| P1 | `deck_slots` (81..88) | -1；每张牌普通/觉醒/英雄槽标记。普通牌组也应在验证后给真实0，而非未知 |
| P1 | `deck_evolution` (89..96)、`hand_form` (97..100) | -1；实际觉醒循环/当前出牌形态，不能只靠卡牌名字猜 |
| P2 | 两个ability记录 (101..112)、拒绝字段113 | -1，技能动作始终被mask；如要使用技能，需提供schema对应champion/ready/cooldown/command信息及按钮校准 |
| P1 | `enemy_plays` (116..187) | 目前由出牌计数变化+新实体推断；法术、弹道、多个同时出生、复活等可能不完整，当前类型分类也需核对，不能全记成troop |
| P2 | `enemy_seen_evolution` (200..207)、`enemy_seen_evolution_need` (208..215) | -1；只记录符合训练可见性规则的已见信息 |

`enemy_elixir` 虽已读取，训练侧是否使用估计值、实机是否不恰当地使用精确隐藏值，
需要单独对齐。不要把敌方完整隐藏卡组/内部意图作为“缺失模型输入”塞进去。

## 6. 已有但仍需校准的输入

- HP/maxHP/shield/target/Buff 已有组件读取，但只验证部分对象布局；读取失败不能用
  0 假装“无血/无Buff”。Buff 永久持续时间 -1 也不是缺失。
- card等级按稀有度换算，entity等级目前基本按 raw+1：需与屏幕等级和训练等级核对。
- section/type/overtime 目前从 raw phase 推导，尚需验证真实对局阶段映射；不能因为
  该值大于0就未经验证认定是加时。
- 无目标时 target_x/y 当前写 -1，而训练 Schema.java 注释定义是 (0,0)：这是**语义
  对齐问题**，不是去找一个不存在的目标。修复前核对实际训练版本。
- 开局、中途接入和塔丢失时，塔缓存/皇冠推断需要测试，避免把“暂未看到塔”当倒塔。
- 2307维合法mask还依赖准确费用、形态、部署区域、pending和技能状态。原始特征补齐后
  要同时检查mask，不能只验证模型前向不报错。

## 7. 交付接口及验收要求

请优先交付 `NullsEntity` 的 `state/attack_ms/load_ms/speed_percent/hit_speed_percent`
等可选字段和相应序列化，及 `NullsBattleSnapshot` 中资源/命令状态的可选结构。
当前 Entity 仅已有 state/deploy_ms/attack_ms 三个可选槽位，其余需要新增并保持旧日志
from_dict兼容。adapter再映射到已有schema列，不改7024布局，不改训练侧字段顺序。

每个新增字段需要：

1. 来源、版本、单位、类型及枚举映射；说明直接读、表数据计算还是观测估计。
2. 连续片段覆盖移动、攻击、狂暴、减速、冻结/眩晕、死亡、出牌/补牌状态变化。
3. 按对象kind的适用范围及未知/不适用sentinel；不可跨poll缓存动态属性。
4. 控制单次读取预算，验证稳定性、P50/P95和完整模型新鲜度；失败快照不出牌。
5. 独立对照证据而非只检查“值不为-1”；保留raw和最终映射值便于复核。
6. 旧日志兼容测试、双座位坐标测试、采样间隔/ID重用测试；不放宽400ms门控。

优先交付：**攻击计时/前摇 + 速度倍率 + 真实状态/flags + 服务端pending/预留圣水**。
这些是审计建议优先级，不代表已通过消融证明一定改善胜率。

覆盖统计命令（只读旧日志，无ADB动作）：

```powershell
& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/audit_nulls_input_coverage.py
```

输出 `artifacts/nulls_live/input_coverage_20261116.json`。统计针对已有录制；不要当成
最新reader/真实对局覆盖的重新验证。本次没有声称补齐上述缺失字段。
