# 本地 CPU 并行与双卡采样验收（2026-11-08）

用户授权扩大新采样并优化 CPU 多线程，同时要求 traj 采集也使用两张 V100。
正式训练在完整 update152 后暂停，保持完整 Actor/Critic、Adam/scaler、训练场次目标1258。
运行数据继续位于 Intel SSD；没有使用 GALAX。

## 实现

- IndependentBattles 原先逐 lane 调用 step，忽略传入线程数。现使用持久线程池，
  各 lane 仅写自己的数组行，主线程等待全部完成后才推进下一决策；异常时也 join 全部任务。
- CLI 分别控制 env-threads、torch-threads、torch-interop-threads，避免把不同线程池混为一谈。
- dual-rollout 在第二张卡创建本轮冻结 Actor 副本。按 lane 固定分配 learner 流，
  与第一张卡 Actor/历史对手并行；副卡 Actor 和主 Critic 推理在同一工作线程依次运行，
  保持各模型增量帧缓存独立。记录原模型实际采样 logp；不改变合法 mask。
- 副卡 Actor 副本只在 collect 生命周期内存在，进入 PPO 后释放；Critic 双卡训练仍按原逻辑运行。
- 对手采样缓存容量随环境数增加，避免超过2个活跃历史模型时每帧驱逐/重新读盘。
  进入 PPO 前保留最近2个模型，释放其余权重/缓存，为激活和 prepared inputs 留出空间。
- 所有训练超参数、reward、Actor2遍/Critic1遍、有效batch64、KL/logp门限保持。

## CPU 模拟器原生对照

相同 seed、合法请求，1152步（128步预热、1024步计时）；计时仅包含 env.step，
状态/合法mask/reward/done 按逐步完整数组计算 SHA256，串行/并行完全相同。

| 环境数 | 串行 step 秒 | 并行 step 秒 | 工作线程 |
|---|---:|---:|---:|
| 2 | 0.4047 | 0.1841 | 2 |
| 4 | 0.3498 | 0.2284 | 4 |
| 8 | 0.5431 | 0.3113 | 4 |

包含 JIT/机器状态差异；这是模拟器推进的局部对照，不代表整轮提速倍数。
证据：artifacts/local_dual_v100_checks/cpu_lane_benchmark.json。

## 真实模型采样对照

所有配置从同一完整152恢复，真实122M Actor、Critic、历史对手、128帧、原奖励。
先采128步预热，再测每lane768步；不做 optimizer、不发布权重、不修改正式计数。
新增环境会增加并行 lane；不同 GPU RNG 分配会改变采样轨迹，因此不将不同局面下
choice 数差异认定为因果效果。所有保存动作合法，logp有限，无模拟故障。

| 环境 | env线程 | Torch线程 | 双卡Actor采样 | 秒 | 新learner/秒 | 新choice/秒 |
|---|---:|---:|---|---:|---:|---:|
| 2 | 1 | 4 | 否 | 17.219 | 123.81 | 9.58 |
| 2 | 2 | 1 | 否 | 16.990 | 125.49 | 9.71 |
| 4 | 4 | 1 | 否 | 20.480 | 187.50 | 15.43 |
| 4 | 4 | 1 | 是 | 20.445 | 195.25 | 15.02 |
| 4 | 4 | 4 | 是 | 19.841 | 201.20 | 15.47 |
| 8 | 4 | 1 | 是 | 40.589 | 189.21 | 14.86 |

选择4环境、4模拟器线程、Torch intra4/interop1、dual-rollout。
相对原2环境的本组探针，新learner吞吐约+62.5%、choice约+61.5%；非完整PPO整轮结果。
双卡Actor采样在4环境相同Torch线程的独立收益很小，主要提升来自批量扩展、缓存和并行。
8环境本组没有更快，不因GPU占用更高就选择它。

本轮46项相关测试通过，包含原生并发缓冲隔离、错误join、两GPU Actor权重/FP16
logp一致性、PPO批次、Critic梯度汇总、缓存释放、训练预算与保留、TensorBoard。
证据：rollout_benchmark_summary.json、rollout_benchmark/*/result.json。
首次隔离探针因奖励配置未完全沿用、另一次因对手相对路径未映射而终止；修正后才执行正式对照。
扩展环境发现cache2反复加载问题后停止该探针，改为cache>=env数；不计入完成结果。
吞吐探针对局独立登记，不当作新增训练场次或策略强度评估。

## 正式运行

默认4环境×3072步，较原2环境增加一倍每轮环境转移；不是增加旧样本训练遍数。
从152恢复的新边界记录在 Intel resume_boundaries.json；未完成旧片段不计为完整训练场次。
TensorBoard继续记录 loss/KL/有效新样本吞吐，并新增 rollout_parallel 统计。
首个完整153更新已保存：15390个新learner、1299个choice，采集96.8秒、优化207.3秒，
双卡Actor分别采8373/7017行；两卡训练峰值12.27/12.43GiB。原入池子进程因父进程保留
CUDA空闲缓存而OOM，未入池；已在后续代码中只于入池前释放空闲CUDA缓存。不能将这轮
345.5秒（含失败入池）当作全部检查通过的稳态吞吐。

用户随后报告主机RAM爆满，进程已退出；未完成154不计入训练。153完整checkpoint已在
独立systemd内存限制内读回通过：2332个浮点tensor有限、两份scaler128、Adam2682/22160，
SHA256 0f25f95a191d0af030ab9540ebec4d82902ae6e4ca2d6226eb5f0fa39ff1699f。
验收单独运行，MemoryHigh1GiB/MemoryMax2GiB，实测峰值1GiB、swap137.7MiB，5.44秒完成。

## 主机内存修复

- Adam的CPU标量计数器clone后再释放恢复对象，避免小scalar一直持有完整checkpoint映射。
- compact-host-memory模式下，snapshot和EpisodeTrace共享本轮拥有的不可变数组；
  非终局时复用上一条next_obs/next_masks，避免同一状态存多份。
- 每轮结束将未完成episode的视图拷贝为独立小数组，避免一条存活lane保留整批多lane存储。
  随后gc和malloc_trim，归还已释放CPU堆；默认EpisodeTrace.append仍复制以兼容可变调用者。
- 默认启动器使用临时systemd用户服务：MemoryHigh5GiB、MemoryMax6GiB、MemorySwapMax1GiB。
  此限制覆盖训练和入池子进程；不能因训练内存增长挤满整机。不会修改全局swap或其他应用设置。
- 禁止将全checkpoint独立验收与训练叠加；验收在停止训练时、独立2GiB上限内运行。


内存修复相关56项回归通过，另增加RAM指标解析测试通过；测试覆盖共享视图轮末脱离、
默认可变输入仍复制、mmap标量计数器独立、原生并发、GPU logp/权重一致性、PPO及保留。
恢复后实测RSS约2.0GiB，PSS file约0.42GiB（先前进程file约2.58GiB）；
服务初始约3.3GiB，整机可用约8.2GiB；限制已由systemd读取确认，不只是文档约定。
过程中MemoryHigh触发回收并使用有界swap，不能把早期swap=0说成整个训练阶段零swap。
实时证据：ram_after_fix.jsonl；TensorBoard新增ram/曲线，侧车不导入Torch。


RAM修复后完整update154验收：17064有效样本、1286 choice；采样120.375秒、
优化277.012秒、整轮418.261秒。内存服务峰值5.002GiB，使用有界swap，
max/oom/oom_kill均0；154 checkpoint约2.8GiB已发布，保留153/154两份，
本轮清理2932447748字节。训练已进入155。没有并行启动完整Torch读回验收。
TensorBoard HTTP验证通过，ram/指标已出现。证据：ram_update154_acceptance.json。
此验收说明RAM保护与完整训练可运行，不代表整轮速度已达最优或低费策略已修复。


第176轮停止及恢复：175已完整保存（1230382样本、1156场），176在Critic梯度
范数非有限时由clip_grad_norm_保护退出，未见本次RAM/显存OOM报错。修复AMP路径：
溢出时不提交Adam，降低scaler、清空梯度、记录跳过批次；正常FP32严格检查仍保留。
Actor/Critic提交计数排除溢出批次，TensorBoard增加amp_overflow/指标。
单元验证包含无权重/Adam污染、下一批恢复、两个优化器的样本计数；双V100回归通过。
已从175恢复，重新采集176；第176失败时未保存的更新丢弃，模拟器重置。
真实溢出批次恢复尚未观察到，不将单元测试替代整轮验收。证据：amp_overflow_recovery_176.json。
