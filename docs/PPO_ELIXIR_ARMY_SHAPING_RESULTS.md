# PPO 圣水加权势函数验证

P0 原生来源证据见 [审计](PPO_ELIXIR_ARMY_SHAPING_P0_AUDIT.md)。精确版用成功出牌记录的实际费用和初始单位对象分账；群体预算固定，未确认出牌无价值。局内来源台账跨 rollout 保留，新局和进程恢复后随模拟器重置。旧均匀模式保留原有 v1 契约。

普通静态精确部署通过了 Knight、Archer、MinionHorde、Cannon 验证。石头人分裂、墓碑、墓园不能做精确归因；仅另一个显式实验配置提供两档近似估值，详见 [fallback 验证](PPO_ARMY_SIMPLE_STAT_FALLBACK_RESULTS_20261117.md)。Mirror、进化、英雄技能以及复杂变形没有精确验收。自然到期通过势函数变化计入，不发独立击杀分。

L4 对照在隔离目录执行：共同 U929 起点、相同模拟器种子和牌组，每组两轮、每轮 128 个环境步，保持原选牌熵 0.003 与落点熵 0.0005。第 931 轮结果：

| 奖励 | Actor optimizer steps | Approx KL | Critic loss | 每轮秒数 |
|---|---:|---:|---:|---:|
| 无单位势函数 | 4 | 0.0000281 | 0.00679 | 23.32 |
| 均匀单位势函数 0.015 | 4 | 0.0000288 | 0.00670 | 23.39 |
| 精确圣水 0.002 | 4 | 0.0000267 | 0.00684 | 23.77 |
| 精确圣水 0.004 | 4 | 0.0000241 | 0.00694 | 24.21 |
| 精确圣水 0.008 | 4 | 0.0000198 | 0.00720 | 24.36 |

0.004 的 checkpoint 恢复后完成 U932；完整对局 BPTT 也完成一次更新。上述 PPO 样本无预算超额；精确模式 0.004 的 U930/U931 单位帧映射率约 24%/45%，完整对局约 26%。L4 回归 60 项通过、6 项双 GPU 测试跳过；补充 fallback 的本机 reward 回归共 25 项通过。

固定对手评估每组仅四局：无单位势函数、均匀、精确 0.002、0.004 和 0.008 均为 1 胜 3 负。**没有证据表明新奖励提升胜率，不建议用于当前长训。** 上述已完成的 GPU 运行早于 P95/P99 字段接入，不能由 summary 反推该分布，也不能把费用支出等同于圣水交换。

复现使用独立实验目录、原始 U929 与 `scripts/run_army_elixir_experiments.py`。迁移使用 `--army-reward-mode elixir_potential --army-elixir-beta 0.004 --army-elixir-config configs/army_elixir_values.yaml --allow-reward-migration`；不要在原 checkpoint 所在目录运行。沿用 Actor/Critic/optimizer 只证明短程可继续，Critic 重热身和尚未覆盖的特殊来源需要独立验证。
