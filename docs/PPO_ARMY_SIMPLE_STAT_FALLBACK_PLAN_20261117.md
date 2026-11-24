# PPO 战场价值奖励：两档血量/攻击力保底方案

日期：2026-11-17。**Agent 实施计划，尚未实现。**
项目：G:/Code/cr_rl。

## 目标

保留现有 elixir_potential 对已映射单位的真实圣水费/固定群体分摊。
其余来源不明的普通战斗单位，以 HP 和攻击力估计一个**很小的圣水等价价值**。
不用追溯所有分裂体、产兵、母子关系，不开发复杂 DPS 或特殊技能估值系统。
新方法只能在独立实验开启；原 PPO 长训仍用旧 reward。

## 两档简单估值

1. **已精确映射**：完全采用现有 ArmyElixirLedger 的付费成本分摊，绝不再进入 fallback。
2. **未知的活跃战斗角色**：从原生 CharacterEntity 读取生命值及 UnitData 基础 HP、攻击值；投射物角色可在预处理时参考 projectile.damage。不能读取的字段作保守处理，不猜卡费。
3. 低档：基础最大 HP < H 且基础单次攻击伤害 < D，则价值为 **0.10 圣水等价/只**。
4. 高档：基础最大 HP >= H **或** 基础单次攻击伤害 >= D，则价值为 **0.30 圣水等价/只**。
5. H 和 D 在实施时通过同版本 native 卡表抽样小骷髅、蝙蝠、哥布林、骑士、火枪手、石头人等至少 10~20 种单位**离线定标**并写进配置。不同等级不能直接对比实时原始 HP/伤害；优先使用 UnitData 基础属性，不在热路径额外做等级缩放。
6. 价值还乘剩余血量函数：活着时 0.3 + 0.7 × clip(hp/max_hp,0,1)，死亡时 0。
7. 未知角色**每阵营**合计价值封顶 **1.0 圣水等价**：V_fallback(side)=min(1.0,Σ未知单位价值×血量函数)。大量小骷髅不会刷出几费额外估值。该上限只是风险控制，不声称“每张卡预算守恒”。
8. 不认识的**塔、建筑、飞行投射物、范围效果、无可靠属性的单位**保留 0，不通过小杂毛分档。特别不要把未知的高费主单位认成小骷髅；未知但 HP/攻击很高的实体最多也只计 0.30。

总势函数：

    Phi(seat) = beta × [(V_exact_own + V_fallback_own) - (V_exact_enemy + V_fallback_enemy)]
    r_army(t) = gamma × Phi(t+1) - Phi(t)

初始 beta=0.004（仅实验建议），禁止单独添加击杀/阵亡奖励，禁止靠调大 beta 掩盖准确映射覆盖率不足。

## 执行顺序

**P0：确认原生数据，快速标定。**
已检查 crforge: WorldEntity.getHitPoints()、WorldEntity.getDamage()、WorldEntity.getData()；
UnitData 存在 hitpoints、damage、hitSpeedMs、projectile、building 等字段。
Agent 要核对实际 JPype 调用，并从 native 运行时抽样确认 HP/攻击值、攻速与投射物攻击的语义，选出 H/D。注意 getDamage 是单次攻击伤害，**不是 DPS**。第一版不需要完整 DPS 模型。

**P1：低成本实现。**
- 修改 src/cr_rl/training/army_elixir.py：在 NativeArmyElixir.poll() **现有的同一次世界实体扫描**里，同时计算 precise 和 fallback，不能重复遍历 native 实体；ID 已经在 ledger.entity_to_slot 的角色排除 fallback。
- 修改 configs/army_elixir_values.yaml：添加 versioned opt-in 的 unknown_policy=stat_two_bucket、H、D、low=0.10、high=0.30、cap=1.0；旧版 zero_and_report 不受影响。后端与数据版本继续固定。
- 如需调整 src/cr_rl/training/rewards.py、state_selfplay.py、CLI，保持 old uniform 和已存在 exact elixir_potential 原样。新模式 metadata 必须区分，以防旧 checkpoint 静默迁移。
- 优先自动补齐**经证明无特殊召唤机制**的普通直接出牌白名单，提高精确覆盖率。特殊产兵/分裂不能冒充精确卡费；允许其不明子体进入低额 fallback。
- 不改 policy 输入、动作采样、PPO ratio、GAE、Critic、原胜负/塔血/皇冠 reward，不添加热路径磁盘 I/O、全表读取、hash 等。

**P2：必须通过的测试。**
- 单体、群体精确计价优先；未知小单位=0.10，大体型/高攻击未知单位=0.30；未知塔/建筑/投射物/非法 HP=0。
- 两个阵营各自封顶 1.0；100 只小单位也不会无限增长；存活多 step 无固定奖励，死亡、自然过期、生成只有势函数差分。
- 双座位潜能反对称；terminal 抵消望远镜恒等式、truncation bootstrap、fault mask；episode reset、跨 rollout chunk、多 lane、checkpoint 参数检查。
- 用真实 native simulator 抽测静态群体、分裂产兵与普通卡，并测热路径吞吐。旧模式的 reward 数值和 metadata 完全不变。

**P3：独立 A/B/C 对照。**
相同起点 U929、牌组/对手/随机种子/训练预算：
A 原有 uniform；B 扩充准确白名单、未知取 0；C B + 新两档 fallback。
报告 exact_mapped_rate、fallback_low/high_count、still_unknown_rate、
fallback_cap_hit_count、P95/P99 army reward、KL/critic loss、胜率/塔血、吞吐。
**精确率和近似覆盖率必须分开统计**；不因为 reward 数值升高就断言更强。

## 最终交付

提交代码、配置、测试、native 抽样 H/D 标定、独立实验结果，
写入 docs/PPO_ARMY_SIMPLE_STAT_FALLBACK_RESULTS_20261117.md。
原长训不受影响；测试失败或发生重复计价时保持旧奖励。
