# PPO 间歇 BC 的数据筛选

> 已加入最终数据集的 Validation 重分配：全量 52 分片结束后才将一部分
> v1 Validation 转 Train；固定评估牌组和历史分区不改变，非 Heldout
> 数据目标为 90% Train / 10% Validation。参见
> QUALITY_BC_VALIDATION_REBALANCE_20261117.md。下述示范准入标准不变。

本轮实现的是独立的只读元数据筛选入口，不修改正在运行的 PPO、采集器或原始数据。
它输出候选索引，不是可直接交给 trainer 的 dataset.manifest.json。

## 默认准入条件

- 只取原 split registry 的 train，按 canonical_game_signature 去重；未知划分拒绝。
- Ranked 来源；这不证明玩家为高手，当前未验证排名。
- 源事件全部执行、覆盖率 100%、零遗漏、无 simulator fault、pending 已结清。
- 整局至少 20 个源动作事件，排除只有极少动作的异常/无效示范。
- 只允许执行审计、原生执行记录、唯一技能来源解析和至多 4 tick（0.2 秒）的非负时间网格对齐。
- 禁止额外 time_delay、手牌替换、落点修正、拒绝执行、跳过技能；未知修复类别拒绝。
- 一次污染排除整局，不把修复后的后缀当作恢复干净；不只屏蔽污染帧。
- 检查执行条目与事件总数、执行成功标记及时间差；缺少审计证据拒绝。

4 tick 是旧 5 tick 数据的网格对齐容差，不是原局状态误差上限。允许它不代表已证明重建
战场与原人类客户端相同。源状态和重建状态的动力学差异无法由事件表单独验证。

## 使用

```powershell
$env:PYTHONPATH='src'
& 'F:/python/python.exe' scripts/select_quality_bc.py `
  --metadata artifacts/bc_local_data_20261110/data/human_bc_reconstructed.json `
  --split-registry artifacts/bc_local_data_20261110/data/split_registry.json `
  --out artifacts/quality_bc_20261117/selection_v2.json
```

输出不可覆盖，记录源 metadata/split SHA256、逐局准入结论和排除原因。
脚本仅读元数据，不解压 13.8 GB observation，不加载 Torch，不连接模拟器。

## 本地审计与后续接入

293 局本地样本最初通过结构/执行筛选的 train 候选有 5 局，其中一局在 303.5 秒
的记录中只有 2 个源事件，因此新增默认最少 20 事件门槛。其余 4 局共 135 个源事件。
不能把这几个候选作为足够大、已验证战术质量的专家池反复训练。

下一阶段需要：

1. 在云端全量源 metadata 上运行相同筛选，不为凑数量放宽门槛；保留原验证/留出划分。
2. 对候选验证压缩数组 hash、原生输入契约、mask 和 flags、因果观测、历史连续性；
   明确源/模型 backend 版本差异。失败即排除，不修成“合格”。
3. 复用 event_preserving_resample 的 0.4 秒 WAIT 合并和完整动作保留，对第二次映射的
   延迟另行核验；旧 0.25 秒的对齐延迟不能与新映射延迟混为一项。
4. 原始数组仅逐局/磁盘 mmap 读取，不整包载入 RAM。产出新的 immutable dataset manifest。
5. 抽查高费留费、防守、法术的前后上下文，核验示范策略质量；先和无辅助 BC 做小规模对照。
6. PPO 辅助 BC hook 尚未实现；应位于全部 PPO 更新之后、下一轮新 rollout 之前。
   BC 数据只做 Actor 监督，不冒充 on-policy PPO rollout，不给 Critic 直接套专家回报。

当前输出固定 training_ready=false、ppo_rollouts_allowed=false；只有上述验收通过后才可发布训练池。
