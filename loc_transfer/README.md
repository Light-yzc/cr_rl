# 双 V100 本地迁移

本机物理 GPU 0 为 RTX 4090，GPU 1、2 为 16GB V100，拓扑为 NV5。
启动器固定只暴露 V100；进程内 Actor 使用 cuda:0，Critic 使用 cuda:1。
Actor/Critic 使用固定 rollout targets 并行优化，独立优化器、梯度裁剪、KL 门限不变。
采样时 Critic 与 Actor/历史对手推理并行；训练输入跨设备传输使用 CUDA P2P。
首个正式 3072 步 update 已完成，实测约 271 秒，详见验收记录。

## 迁移文件

见 [DOWNLOADS.md](DOWNLOADS.md)。update 142 完整 checkpoint 已从用户提供的 Drive 下载并归档到 HDD，并通过 SHA256。
Actor 导出、随机初始化 Critic 均不能替代原 Actor/Critic/Adam/RNG 状态。
迁移源文件保留，下载后的完整 checkpoint 会按 SHA256 校验；失败则停止。

## 环境

```bash
bash loc_transfer/bootstrap_v100.sh
```

使用独立 `.venv-v100`、Python 3.12、PyTorch 2.6.0 / CUDA 12.4、Java 21，
模拟器固定到 configs/backends.yaml 的 commit 和 data_version。
不要用默认 `uv run` 自动同步为新的 CUDA/PyTorch，后续版本可能不含 sm_70。

## 验收与运行

```bash
CUDA_VISIBLE_DEVICES=1,2 .venv-v100/bin/python -m pytest \
  tests/test_dual_v100.py tests/test_separate_critic.py tests/test_local_retention.py -q
UPDATES=2 bash loc_transfer/run_v100.sh
```

首次从 update 142 恢复，随后从本地最新完整 checkpoint 继续；默认最多 600 个 update，
达到既有目标 1258 场后停止。沿用迁移 command.json 的
4 环境 × 3072 步（原为2环境）、Actor 2 epoch、Critic 1 epoch、有效 batch 64、LR 和奖励设置。
Actor 微批次 8、Critic 16，有效 batch 64；密集输入自动缩小微批次，双卡 FP16 autocast，参数和 Adam 状态保持 FP32，
Actor/Critic 分别保存和恢复 GradScaler。非有限梯度会中止更新，避免发布损坏状态。
两 update 的真实恢复验收与首个正式 3072 步 update 均已通过。
详情见 [本地双卡验收记录](../docs/LOCAL_DUAL_V100_20261108.md)。

后台持续续训：

```bash
.venv-v100/bin/python loc_transfer/start_v100.py
```

```bash
# 调整微批次时保持 BATCH × ACCUM = 64；以实测速度和峰值显存决定。
BATCH=8 ACCUM=8 UPDATES=2 bash loc_transfer/run_v100.sh
# 可用 FP32 做数值/速度对照。
AMP_DTYPE=fp32 bash loc_transfer/run_v100.sh
```

跨硬件/精度和微批次变更会改变随机轨迹；迁移回执登记模拟器重置边界。
不得把硬件迁移声称为逐轨迹相同的续训。

## 磁盘保留

运行目录入口为 `artifacts/local_dual_v100`，本机链接到 Intel SSD
`/run/media/regen/本地磁盘/cr_rl/local_dual_v100`，与迁移源分开。
每轮原子替换 latest，使用硬链接保留最近 2 个完整恢复点，不额外累积周期点（启动时 Intel 分区空闲约 15GB）。
对应 pool 一并保留，最近 2 个 rollout 保留。旧 checkpoint/export/pool 及时回收。
只回收运行目录中不再被当前 pool、保留恢复点、活跃对手或评估/发布任务引用的对手文件；
迁移源和目录外权重不会被删除。回收字节数写入 progress.jsonl 的 storage_retention。
备份发布目录仍使用自身发布协议；本地启动器不删除云端归档。

## 双卡负载均衡与监控

默认启用 `--balanced-critic`：Actor 优化时两卡分别训练 Actor/Critic；
Actor 完成后，其 V100 加入 Critic 数据并行，每卡至多 16 样本。
副卡实体预算为 768（行数×最大实体数），密集输入降到 8 或更小，
为 Actor、Adam、冻结参考与对手缓存留出空间。
两卡梯度通过 NVLink 求和，只在主 Critic 上执行一次全局归一化、裁剪和 Adam；
下一步向副本广播最新参数，有效 batch 仍为 64。密集输入自动缩小微批次。
相同真实 128 帧输入的隔离测试中，密集输入的 128 样本耗时 3.156 → 2.369 秒，
吞吐提升约 33%，两步后最大参数差 4.36e-6；这项数字仅代表 Critic 阶段。

```bash
.venv-v100/bin/python loc_transfer/start_tensorboard.py
```

打开 http://127.0.0.1:6006 。提供 loss、KL、熵、回报、训练对局胜率、
新样本/秒、各阶段耗时、双卡利用率/显存/功率、磁盘空间及回收量。
`critic_step/critic_peer_rows` 显示每轮累计分给副卡的 Critic 样本；
`critic_parallel` 记录每轮分工和 NVLink 梯度/权重传输字节。
训练对局胜率受对手采样影响，不能直接当作固定基准评估胜率。
日志侧车不导入 Torch，不占 GPU；断点读取进度保存在 Intel 运行目录。


## 采集阶段 CPU 与双卡

本机基准选择4环境、4个模拟器工作线程、Torch intra4/interop1；8环境未更快。
`--dual-rollout` 让两张V100分担当前Actor采样，副卡同时负责Critic推理。
采样缓存最多4个对手，PPO前缩到2个，副卡采样Actor也在PPO前释放。
原有双卡Critic训练继续开启，reward、有效batch64和KL保护保持。
本组采样探针新choice吞吐9.58→15.47/秒；不将其当作整轮提高61%。
详情：[CPU与双卡采样验收](../docs/LOCAL_ROLLOUT_CPU_20261108.md)。

```bash
# 参数可覆盖；以真实choice/秒及整轮吞吐选择，不盲目增加线程。
ENVS=4 ENV_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-v100/bin/python loc_transfer/start_v100.py
```


## 主机 RAM 保护

后台启动器默认创建临时用户服务 `crrl-v100.service`，覆盖训练与入池子进程：
MemoryHigh=5GiB、MemoryMax=6GiB、MemorySwapMax=1GiB。不改系统全局swap或其他程序。
使用 `start_v100.py` 启动以保留这些限制；直接调用shell训练脚本会绕过服务的内存保护。

`--compact-host-memory` 共享本轮不可变轨迹数组，复用相邻状态，轮末将未完成episode
压成独立存储，并归还CPU堆；Adam CPU计数器不再持有完整checkpoint文件映射。
全checkpoint独立验收安排在训练停止期间，并单独限制在2GiB内存内，避免重叠峰值。
TensorBoard `ram/` 提供训练RSS、训练swap、服务当前/峰值/上限和整机可用内存。

```bash
systemctl --user show crrl-v100.service -p MemoryCurrent -p MemoryPeak -p MemoryMax
```
