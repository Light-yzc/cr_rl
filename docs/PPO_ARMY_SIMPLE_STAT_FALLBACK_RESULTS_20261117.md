# 两档未知单位估值：实施与验证

2026-11-17。已实现为独立 opt-in 配置 `configs/army_elixir_stat_fallback.yaml`；不改变原均匀模式或原精确配置。

## 能解决什么

未知分裂体、墓碑产兵、墓园召唤不再因为缺少来源 ID 而全部零价值。原生 CharacterEntity 的基础 UnitData 按固定参考等级 11 离线标定/首次按单位行缓存，包含远程单位 projectile 单次伤害；不把 DPS 与单次伤害混淆。每帧读取真实剩余 HP，仅用于血量比例。已在 `artifacts/army_elixir/fallback_calibration.json` 抽取 20 种原生 UnitData。

阈值 H=350、D=150（参考等级11）。骷髅81/81、蝙蝠81/81、哥布林202/125、投矛哥布林133/81、亡灵230/107、弓箭手304/112 为低档0.10；骑士1766/202、火枪手721/217、小石头人1039/84 为高档0.30。阈值是保守实验初值，未证明最优。

已知来源在台账中的单位绝不进入 fallback。未知建筑、塔、投射物、区域效果、无可靠血量/伤害单位仍为零。每边 cap=1.0，双边分别封顶；折扣势函数差分、终局清零、截断保留不变。没有独立击杀分。

## 重要限制

- **这是有上限的近似覆盖，不是来源修复或每张卡预算守恒。** 特殊整张卡不进入精确路径，故不存在先把完整母体按整卡计费、再给孩子加值的做法。
- 上限饱和时损失一只小单位可能仍在1费上限，不一定立刻得到变化信号。
- 未知大单位最多0.30，会低估昂贵主单位；效果是否改善必须独立对照。不能因近似覆盖率升高声称来源映射已解决。
- 原生数据缓存按单位行，未知单位首次出现时走少量 Java 属性读取；不读文件、不 hash、不遍历表。后续只用缓存；实体扫描与精确路径共用。
- 全面特殊 paid ability/clone/transform 源事件依然未实现；它们最多按可读普通战斗角色近似值，而不是实际费用。

## 已执行测试

`PYTHONPATH=src;.cache/vendor/crforge/python JAVA_HOME=F:/JDK`，cs285 Python：

```
python -m pytest -p no:faulthandler -q tests/test_reward_shaping.py tests/test_army_elixir_shaping.py tests/test_army_stat_fallback.py
```

25 passed。包含真实原生 MinionHorde 固定6人且排除 fallback、Tombstone 出骷髅、Graveyard 召唤、Golem 死亡后出现 Golemite；检查双边cap、反对称、零重复计费。预算槽分裂/产兵测试只证明代数台账，不冒充已实现的精确特殊来源。

`scripts/benchmark_army_elixir_native.py` 的固定普通牌组，热身后 fallback 总扫描约0.11–0.12ms/环境步（完整 JVM 纯模拟吞吐的比例开销明显，绝对值小）；此牌组本身是精确来源，不能代替大量未知单位场景。GPU端独立 A/B/C 和长期效果尚未完成，**暂不推荐覆盖长期训练 reward**。

启用（新实验目录，保留固定U929）：

```
--army-reward-mode elixir_potential --army-elixir-beta 0.004 \
--army-elixir-config configs/army_elixir_stat_fallback.yaml --allow-reward-migration
```

reward metadata 包含完整配置与校准版本，旧精确模式/新 fallback 相互恢复必须显式迁移。原始 uniform v1 metadata不变。
