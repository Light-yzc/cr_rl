# BC时机过滤对PPO的影响核查（2026-11-05）

本次只读代码和已保存第44–47轮PPO轨迹，不增加训练或测评，不改变当前配置。

## 结论

BC邻近WAIT过滤没有进入PPO在线训练；没有发现PPO在动作之间插入假帧、扩展动作标签或
压缩真实时间轴。BC学到的时机偏差仍可通过初始化权重间接带入PPO，不能据此认定毫无影响。
当前4Hz决策、BC导入的时间窗口归类、原生出牌执行延迟是另外的时机因素，须分别讨论。

## 三种不同的处理

1. `replay_supervision.isolated_wait_rows()`仅从BC的训练WAIT索引中剔除同局同座位动作前
   20tick/后10tick的样本；不改正动作标签、不删除历史观测。该规则不按卡牌种类区分，
   因此法术附近“此刻应再等”的负监督也会减少。这是合理的潜在精确时机风险，尚未用对照试验证实其影响量。
2. `bc_batch.build_bc_batch()`的padding用于历史不足时的左侧零填充，同时time_mask=False；
   不在两次动作之间插帧，不把被过滤的WAIT从历史里抽掉。PPO的EpisodeTrace窗口也仅左侧补零。
3. BC importer使用`label_replays(..., decision_ticks=5)`。固定crforge
   `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`的`ReplayLabeller.java:55`明确将人类指令
   按givenTick归入所在decision window，观测取窗口起点。窗口内具体偏移没有成为动作输出。
   因此“过滤不移动标签”不代表BC保留了逐tick的完整人类出手精度；5tick窗口最多含0–4tick偏移，
   即0–0.20秒。这是原有标签量化，与前后1秒/0.5秒的WAIT过滤不同。

## PPO实际学习链路

- `selfplay_v2.collect()`逐步保存真实obs/action/next_obs/reward/终局，直接交给模拟器执行。
- `optimize()`先在完整时间轴计算Actor/Critic目标，再按choice-only选择Actor行。
  `action==WAIT`不会被自动排除；只要存在其它合法选择，WAIT也可成为Actor样本。
  强制WAIT仍保留在GAE、Critic及历史中，只有Actor直接优化被跳过。
- KL参考是每轮`copy.deepcopy(self.actor)`的当前行为副本；没有加载固定BC模型作为动作监督。
  固定BC对手属于环境对手，不向Learner提供交叉熵标签。PPO可以修正继承偏差，但不保证已经修好。
- 新fastpath仅在整批没有其它合法动作时跳过Actor前向，仍追加每一帧和保持Categorical RNG；
  可出法术的决策不会被该规则跳过。

## 已保存轨迹的CPU审计

| Update | Actor choice行 | 其中WAIT行 | 下一秒内会有动作的WAIT候选 | 连续观测核对数 | 错位数 |
|---|---:|---:|---:|---:|---:|
| 44 | 735 | 525 | 320 | 3068 | 0 |
| 45 | 670 | 489 | 296 | 3066 | 0 |
| 46 | 1196 | 993 | 458 | 3066 | 0 |
| 47 | 798 | 604 | 302 | 3066 | 0 |

合计12,288个环境transition，12,266次同局、非终局后的完整next_obs→下一obs逐元素核对，零错位。
全部非终局状态推进5tick，观测decision_ticks均为5；所有动作均在当时合法mask中。
1,376个动作前1秒内的WAIT仍属于实际Actor训练候选，证明没有套用BC邻近WAIT剔除。
这些是候选行，不等于KL早停前每行都实际执行了梯度更新；近邻动作包括卡牌和技能，并非法术专项成绩。
原始小型报告和CPU脚本位于`artifacts/ppo_timing_audit_20261105/`。

## 精细法术时机的实际限制

- 模拟器20Hz、PPO每5tick决策一次，即0.25秒。若合适出手点在两次决策之间，当前动作空间
  不支持提交额外的子步时间偏移，也不能在已有决策间临时响应。
- 轨迹观测的`play_delay_ticks`均为20，即1秒。固定模拟器的`LadderEnv`/`LiveBattle`将普通出牌
  指令通过同一PlayerCommands规则排队；请求时间和执行时间不同。之后卡牌还可能有自身生效/飞行过程。
  这不是BC padding附加的一秒延迟，也不能只用动作请求时间判断法术是否打准。
- 当前还有按局面加WAIT偏置的ElixirHoldingPrior。它影响出手倾向，与BC过滤独立；本次未改。

现有审计能排除这几轮的时间轴错位和PPO复用BC过滤，不能证明法术战术时机已经正确，
也不能把某次失误单独归因于BC。专项验证应区分请求tick、实际执行tick、目标位置与命中结果，
再决定是否开展BC法术差异化margin或更细决策频率的受控实验。本次不额外占用48场评估预算。
