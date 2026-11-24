# 实战录制利用率与实时写盘修复

## 原因

原训练导入器把“没有下一条decision确认”当作整局执行不明，只允许开局played_count=0
时排除失败点击。真实日志中，很多最后一张牌仍在server pending队列，对局便结束；终局
原生played_count和四张手牌均未变化。它们可证明未执行，不能据此丢掉整局。

另一类拒绝来自固定0.8秒门槛：17tick的轻微抖动、首帧CUDA冷启动，以及日志同步写盘的
长停顿被同等拒绝。例如某些帧read/inference合计只有几十毫秒，记录完成却花了1秒以上。
这不代表能恢复未观测的物理帧；必须区分严格PPO与离线AWBC用途。

## 执行对账

- 进行中的超时：要求原决策、receipt、紧随其后的decision出牌计数一致、手牌一致，
  且server pending和reserved elixir为0；有原生snapshot时它也必须一致。
- 收官超时：要求后面没有新提交，终局已结束，原生四手牌和计数与发送前完全一致。
  对局结束后pending不会继续执行，因此可排除该动作标签。
- 保留原proposal/action/logp与全部观测，只令policy_loss_mask=False；不改成WAIT，
  不伪造confirmed。证据不足仍拒绝。
- 新采集器在终局直接记录rejected及对账证据；确认单次出牌必须恰好计数+1。

## AWBC 时间契约

`load_episode(run,purpose='ppo')`仍要求所有间隔<=16ticks；`as_ppo_segment`拒绝超限数据。
`purpose='awbc'`允许不规则时间（最长10秒工程上限），但继续检查完整决策ID、同一对局、
合法proposal、概率、logp、执行对账和真实终局结果。超过10秒仍隔离。

AWBC不构造PPO importance ratio。保留实际step_ticks计算MC折扣，不补造中间观测，
不把停顿填成若干WAIT。间断前后两个Actor标签屏蔽，真实历史仍从开局推进记忆。
轨迹中的spell veto和forced WAIT继续单独排除Actor标签。Value影子头按真实回报与
采集版本交叉拟合，PPO Critic不直接吸收这些离线回报。

playbuffer与独立AWBC入口已改用此契约；loader一次计算actor_keep，取消再扫一遍gzip。

## 后续采集性能

新增OrderedRecorder：一个有界有序后台队列保存诊断与事件，JSON/gzip/磁盘/stdout写入
不再阻塞推理与触控。队列满/后台失败会报错停止，不能悄悄丢帧；结束时flush后才发布
collection_complete。模型概率、输入、随机采样规则保持不变。

匹配前额外预热6/32/96/200实体形状和落点路径，然后重置记忆，避免空WAIT预热遗漏真实
实体attention首次分配。此操作不会在正在打的旧child中途热替换。

## 实测

一次复核53局完整且有结果的会话，AWBC53/53通过；21,442帧保留，103条policy标签因
执行/时序等原因屏蔽（另有原有choice/veto掩码）。这是AWBC合同可用率，不是胜率，
不含没有结束的采集尝试，也不保证未来数据永远100%。

新后台写盘版本已经完成两局：291帧与483帧，最大间隔分别12/13ticks，状态年龄P95
约118.7/132.7ms；两局均写入reward、通过严格PPO导出并返回大厅。样本仍少，不能据此
保证所有机器和后续长时间都保持该表现。

Colab CPU最终回归已通过61项（54.03秒），覆盖两种时间合同、终局/中途对账、异步
顺序/错误传播、旧录制、playbuffer、AWBC及loader单次actor_keep计算。
后续检查的57个已结束且有结果会话也通过AWBC导入；实时采集仍继续，数量会增长。

## 复核命令

```powershell
$env:PYTHONPATH='src'
& 'F:/python/python.exe' scripts/backfill_nulls_results.py `
  --sessions artifacts/nulls_live/preset_60_20261117 --validate --purpose awbc
```

输出result_awbc_readiness.json；严格PPO使用--purpose ppo，输出result_training_readiness.json。
