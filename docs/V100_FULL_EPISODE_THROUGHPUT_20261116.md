# 双 V100 整局 PPO 吞吐优化

## 测量方法与收益

基线为真实生产 u709 的 36 环境、空间编码微批次 32、整局 BPTT、时序 block 重计算：约 140.45 learner frames/s。

`benchmark_full_episode_throughput.py` 采集一批完整对局，固定其中 16 条轨迹共 7015 帧，对每个配置执行相同 Actor/Critic 优化路径。测试学习率和 weight decay 设为 0，模型权重不变，保留相同 KL、参考策略、锚点、choice mask、GAE、全局梯度和 optimizer 分组；每个配置都是 Actor 4 步、Critic 3 步。临时基准数据只写 Intel 数据挂载，完成后删除。

| 空间编码帧批量 | 输入特征缓存 | 相同数据优化耗时 |
|---|---|---|
| 32 | 关闭 | 16.17 秒；末尾复测 15.40 秒 |
| 64 | 关闭 | 12.99 秒 |
| 128 | 关闭 | 13.08 秒 |
| 64 | 开启 | **12.34 秒** |
| 128 | 开启 | 12.62 秒 |

采用 64 + 缓存，比 32 两次测量的平均耗时低约 22%。没有扩大 PPO 的策略/价值有效 optimizer batch。128 在另一批初步测试触发过数值检查，重测虽通过，收益也不足以替代 64。

## 实现

1. 空间编码与落点头的 `frame_microbatch` 从 32 增至 64，减少小批量前向/重计算的 Python 与 CUDA 调用。Actor/Critic 序列微批次仍为 4/4，保持原 optimizer 分组和更新节奏。
2. `state_frame_cache` 在一次 update 中将规范化 FP32 header、entity、mask 和手牌 ID 缓存在训练 GPU 上，由参考策略、Actor 与 Critic 复用。缓存的是模型输入，编码器和时序层每次照常前向/反向；没有复用旧网络表示或截断梯度。每卡上限 768 MiB，显存不足可回退，退出作用域（包括异常）立即释放。基准缓存每卡约 35.6 MiB，准备约 0.04 秒。
3. 逐泳道裁剪 memory 时，一次读取整批 positions，替代每个泳道独立 `.max().item()` 的同步。
4. 整局采集将裁剪后的 KV 前缀预留到几何容量桶，避免冻结对手每步扩到实际长度两倍并同步读取最大长度。只在整局模式开启；连续跨 update 的兼容模式保留原容量上界校正，避免悲观 length_bound 累积导致过量分配。
5. 新增采样计时：模型推理、模拟器步进、其余整理分别记录。基线一次采样约 61 秒，其中模型约 56.3 秒，模拟器仅 2.5 秒；增加 CPU 模拟线程不是主要提速方向。
6. 修复安全退出时补存 checkpoint 后未执行常规保留策略的问题。仍保留最新两个恢复点和一个周期备份，同时保护现役/保留池引用的对手；此次清理旧 u696/u697 恢复点回收约 8.15 GiB，保留 u708/u710、u640 周期备份与独立 best 文件。

## 完整规模测试

48 环境、frame_microbatch=64、输入缓存开启，连续两轮均通过：

| 更新 | 有效帧数 | 总耗时 | 吞吐 | 初始 log_prob 最大误差 | GPU0/GPU1 峰值 allocated |
|---|---:|---:|---:|---:|---:|
| u711 | 31723 | 179.22 秒 | 177.01 帧/秒 | 0.00463 | 7.23/7.84 GiB |
| u712 | 35475 | 199.91 秒 | 177.46 帧/秒 | 0.00333 | 7.08/7.87 GiB |

相对原生产约提升 26%。不同规模采样不是逐局完全相同的严格 A/B，但两轮端到端吞吐一致；上面的固定轨迹基准用于单独验证执行路径收益。

完整测试仍保持整局反传、current 双边收集、PFSP 强对手池、固定 BC 锚点、主副修卡组抽样、随机等级、独立 Critic、全部 KL 与 AMP 保护。训练阶段观察到双卡约 94%–96% 利用率，其中一张 V100 达约 250W；收尾采样阶段的利用率依然随活跃对局数变化。

## 监控与复现

- `performance/frame_microbatch`
- `performance/rollout_model_seconds`、`rollout_env_seconds`、`rollout_bookkeeping_seconds`
- `performance/state_frame_cache_cuda_0_mib`、`state_frame_cache_cuda_1_mib`
- `performance/state_frame_cache_prepare_seconds`
- 原有吞吐、双卡峰值显存、initial log_prob、overflow 与 `train_traj/*` 面板。

`CACHE_STATE_FRAMES=0` 可关闭输入缓存；`FRAME_MICROBATCH=32` 可恢复原空间编码批量。整局模式和正常每 800 update 保存不变，额外 validation 保持关闭，目标仍为 u250000。

固定轨迹基准输出在 `artifacts/ppo_full_throughput_grid/`；完整 48 环境验证在 `artifacts/ppo_full_throughput_48_smoke/`。基准和冒烟测试均不发布训练 checkpoint 或联赛快照。

## 最终配置与回归

64 环境另做一次完整测试：41897 帧、254.73 秒，164.47 帧/秒，低于 48 环境；双卡峰值 7.40/7.89 GiB，进程峰值 RSS 7.51 GiB。Actor/Critic 均无 overflow、无 KL 拒绝，初始 log_prob 误差 0.00478。因此最终选择 **48 环境 + frame_microbatch 64 + GPU 输入缓存**。

数值校准记录：48 环境第一轮 Critic 有 2 次 AMP overflow 跳步，下一轮 0；Actor 三轮均为 0、无 KL 拒绝、所有指标有限。恢复点 Critic scaler 为 scale=32、growth_tracker=94/100，恢复后很快触发动态增长；GradScaler 保留了正常回退与跳步保护，不会把非有限梯度写入参数。不能把该次跳步描述成“所有更新均无 overflow”。128 编码批量重测通过，最终选择 64 以实测速度为依据。

另去掉整局模式下的重复 raw prefix trace：完整 learner segment 已保存训练所需的每一帧；GPU 上的 KDA/GQA memory 与完整反传均保留。只有跨 update 的兼容模式需要保留额外原始前缀。新增开局 episode_step 检查，防止把缺少前缀的中途对局当作完整局。该改动避免已结束泳道的数组视图长时间保留整批 raw 数据，降低 RAM 峰值。

已从生产 u710 模型与优化器状态恢复长训，PID 96033，物理 GPU 1/2；目标 u250000、checkpoint 间隔 800、validation 关闭。CPU 回归 149 项、GPU 专项回归 7 项通过。最终状态记录在 `artifacts/ppo_full_throughput_48_smoke/regression_final.log` 与 `verification_summary.json`。

恢复后的生产首轮 u711 已完成：205.24 帧/秒，初始 log_prob 误差 0.00897，Actor overflow=0、Critic AMP 跳步=1，无 KL 拒绝、无服务重启。该首轮使用了去除重复 raw prefix 的最终实现；205 帧/秒是单轮实训观察，稳定比较仍以两轮 177 帧/秒为依据。原始结果保存为 `production_first_update.json`。

## 按散热条件交换双卡分工

应用户要求，安全完成并保存 u729 后，将启动默认顺序从 `GPU_IDS=1,2` 改成 `GPU_IDS=2,1`，随后恢复长训。已按 CUDA UUID 核对：逻辑 cuda:0 对应物理 V100#2（PCI 06:00.0），负责主要 Actor 与参考缓存；逻辑 cuda:1 对应物理 V100#1（PCI 04:00.0），负责独立 Critic、冻结对手及 Critic 完成后的 PeerActor 协助。4090 不参与训练，NVLink 双卡协作继续保留。交换前采样观察卡 1 约 71–78°C、卡 2 约 49–57°C。没有修改功耗上限或 PPO 超参数。
