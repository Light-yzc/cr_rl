# P0 原生来源审计（2026-11-17）

已先阅读完整实施计划。审计固定后端 `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`、tables `16.635.3`、8 tick 决策。未修改后端或运行中的 Colab 作业。

## 实证

命令：`F:/yzc/Miniconda/envs/cs285/python.exe scripts/audit_army_elixir_native.py`，设置 PYTHONPATH=src;.cache/vendor/crforge/python、JAVA_HOME=F:/JDK。
结果：`artifacts/army_elixir/p0_native_audit.json`，7 个原生案例，各 150 个决策，非合成数组。

| 实际出牌 | 实际费用 | 初始单位数 | 原生实体 ID |
|---|---:|---:|---|
| Knight | 3 | 1 | 5000006 |
| Archer | 3 | 2 | 5000006..5000007 |
| MinionHorde | 5 | 6 | 5000006..5000011 |
| Cannon | 3 | 1 | 5000006 |
| Golem | 8 | 1 母体 | 5000006 |
| Tombstone | 3 | 1 本体 | 5000006 |
| Graveyard | 5 | 0 初始直接单位 | 另有后续召唤 |

这些案例未观察到 ID 重用，但不是无限生命周期证明。`EntityHolder.add()` 使用每 kind 递增计数并在 1,000,000 取模；台账应在 episode 边界清空，检测冲突。

## 来源与支持矩阵

- `SeatIds.java`：observation id_version=2，各座位独立可见顺序编号，局内稳定且不复用。不能把两席位观察 ID 当世界 ID。
- `Standard1v1Battle.Play`：`result().placed()` 是成功证据，`units()` 给本次直接生成的全部单位对象；创建时已经分配世界 ID。即使同一决策内单位死亡，仍有原始 units 列表，不能按幸存数重新分费。
- `Standard1v1Battle.place()`：只有 placement 成功才运行支付函数；MirrorItem/VariantItem/EvolutionItem 保存实际 cost，不能用圣水条差分代替。
- 原生 Character/WorldEntity 有真实 HP 和全局 ID；reward 可从 `VectorBattle._env.env(i).battle()` 读取，不污染 policy observations。普通 VectorBattle 和 IndependentBattles.lanes 都可接入。
- 分裂/产兵等 `CharacterSpawns.spawned()` 的默认 castId 为 UNMODELLED；不是所有孩子都继承 deployIndex/castId。WorldObserver 有出生/变形等事件，但 Python observation API 没有统一 parent/cast 事件流。
- 0.4 秒抽样可以漏掉中间出生死亡，不能据此计击杀。新奖励只比较边界势函数，不统计独立击杀。

首版实现范围：配置明确允许的基础直接部署单体、静态群体、非产兵建筑；用成功 Play 对象和初始 units 固定均分预算。动态召唤、死亡分裂、产兵、法术召唤、Mirror、进化和英雄形态先整体零计价；统计 unknown，不按完整卡费兜底。

后续要完整覆盖特殊类别，必须补充统一原生来源事件（成功 cast 实付费用、parent entity、spawn/transform 原因与预算槽），并从固定卡表证明未来召唤额度。现有观察数组或 entity castId 本身不足以证明这些关系。本轮不修改固定后端，也不以合成测试冒充特殊机制的原生验收。
