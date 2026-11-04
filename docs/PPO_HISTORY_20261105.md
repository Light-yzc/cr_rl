# 历史对手池接入与数值审计

## 已完成

- 三桶采样：当前策略30%、历史PPO 50%、固定BC 20%。历史桶使用按卡组EMA的PFSP，
  保留uniform floor；空历史桶回退BC，单独记录requested/actual类别。
- 历史保留与强度晋升分离：历史候选约20k有效样本一次，仍须合法完整对局检查，容量8。
- 历史版本使用`2^30 + generation`，当前版本必须低于`2^30`。保留轨迹v3非负编号约束，
  兼容旧池BC编号0，避免generation14历史对手被误判为当前behavior14。
- seed载入逐一核对SHA、完整参数形状、generation、引擎来源和有限数值；不消耗恢复的
  torch RNG。generation6/10/14真实导出manifest已登记，实际载入时仍重新校验文件。
- Linux202项测试通过；真实模拟器tiny跨两个update检查通过：1536有效learner样本，
  512冻结对手决策均未计入learner mask。测试seed为tiny合成版本号，不能当作真实PPO强度。
- 上一轮64局评估已独立复算，GUI小型合档已下载，更新本地viewer最新指针。

## L4当前状态

会话`ppo-history-l4`，endpoint `gpu-l4-s-ass1b0-d7x7ceaw5b59`，Drive已挂载。
固定引擎构建、最新202项测试与Ruff完成。未启用subagent。

首次阶段目录：`MyDrive/cr_rl/runs/122m_ppo_history_20261105/`。
从generation14完整恢复后，在训练前CUDA审计被门槛拦住，未进行新梯度更新。
原始失败报告`cuda_policy_parity.json`与日志保留，controller状态为failed；不覆盖成成功。

| 路径 | BF16最大selected log-prob差 |
|---|---:|
| full/prepared，batch4 | 0.00751 |
| full/prepared，batch8 | 0.01480 |
| 帧缓存，batch1/4/8 | 约0.076 |

FP32各路径最大差约9.35e-5，在原5e-4门槛内；BF16缓存超过原0.03门槛。
BF16缓存复算原始old log-prob最大差0.08439，也超过原0.05门槛。
不能放宽门槛或改写old log-prob来宣称通过。正在同一冻结64状态集合上隔离测试
帧编码器FP32、时序仍混合精度的候选，未修改生产模型或已有checkpoint。

## 后续验收

先定位并通过数值审计，再从同一generation14完整状态开启8轮历史池训练。
网络规模和有效batch保持原配置；不能用中间generation10的测试成绩改选起点。
计划generation18/22各对generation6与准确BC5096，合计64局换边，固定新seed7000000、
sampling seed20261106；包含原生回放审计。失败阶段已登记该协议但没有执行比赛。
每轮Drive发布完整checkpoint、历史池、rollout和日志，最终全部核验后释放L4。
若数值修复改变实际精度配置，必须记录、单独测试并保留失败证据。

## FP32帧编码修复与r2正式启动

同一64状态候选通过原有阈值：FP32最大log-prob差9.346e-5、BF16最大路径差
0.01549、最大TV0.001665；对原始存储old log-prob的BF16差0.049158，小于原0.05门槛。
原始概率未改写，阈值未修改。新增`frame_encoder_fp32`配置，默认false保留旧权重行为，
此阶段显式true；Actor/Critic帧编码及梯度走FP32，时序等部分保留原混合精度。
参数张量形状、层宽和网络参数数量不变。历史对手保持各自原有精度配置。

最新203项Linux测试及Ruff通过，包含精度配置保存、帧编码梯度与原参数严格载入。
启动r2时，实际配置版CUDA审计再次通过，与隔离候选数值相同。第15轮已开始真实采样，
控制器与trainer的/proc存活检查完成；首次失败阶段没有训练更新，r2仍以generation14
同一完整checkpoint为起点，不继承失败阶段的任何优化结果。

当前Drive：`MyDrive/cr_rl/runs/122m_ppo_history_20261105_r2/`。
W&B：[itnqtrmf](https://wandb.ai/weidemingdeguge-test/cr-rl/runs/itnqtrmf)。
真实6/10/14权重SHA与`history_bootstrap.json`一致；保存历史、强度晋升不混为一谈。
固定8轮及18/22的64局评估已登记；首轮备份见下节，最终结果仍待。不能将启动称作收敛。

## generation15发布与独立验收

已完成4,545有效learner决策，累计72,246；Actor21优化步、Critic288步，3场正常终局、
故障样本0。Actor实际总step186，Critic4464；预热仍为2轮/9局，没有重置。
Actor达到KL门槛后正常结束，Critic四个epoch已完成。更新前old log-prob复算最大差
0.006535；value loss均值0.001868，这些不代替胜率评估。

CPU独立SHA核验完整checkpoint（2,919,967,989字节）与pool通过，记录在
`generation15_cpu_readback.json`。checkpoint SHA为
`3a96714e2c681ed00f447ef9dada0a1e5595a01dcdff45bfd45df228f7ab84f2`。
checkpoint确认Actor/Critic帧精度、两个优化器、历史保留时钟及6/10/14对手均已保存。

只读审计器`audit_history_rollout.py`在不可变manifest SHA校验后审计完整轨迹：
1,599冻结对手决策（generation10:725，generation14:874）均未进入learner mask。
当前方有216条选牌指令，占有效决策4.752%；1,716帧有合法牌可出，其中87.413%选择WAIT。
指令数量不等同于实际成功出牌；日志稀疏采样点的play_fraction=0不代表全局不出牌。
证据：`history_rollout_000015_audit.json`。最新新增审计单测及Ruff也通过。

首轮含发布墙钟927.68秒：采样116.87、优化744.08、序列化10.82、备份53.21秒，
有效样本4.90/秒，峰值torch allocated10.29GiB。FP32精度修复增加优化成本；对手混合
和实际Actor步数也改变了，因此不能将跨阶段耗时变化全部归因于某一工程项。
后续仍需以当前精度做受控吞吐测试。第16轮连续推进，首轮末未结束游戏未被重置。
