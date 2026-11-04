# PPO 训练效率优化：2026-11-04

用户要求检查 CPU、RAM、GPU 利用率并提高真实训练速度。

## 实测与选择

原 L4 在采样时 GPU 利用率约 27%、显存 4.8 GB，进程 RSS 约 3.1 GiB，系统可用 RAM 约 47 GiB。采样与训练阶段不能混在一起比较；低 RAM 占用本身不是需要填满内存的理由。

在现有同一台 L4 上暂停训练，从已完成并备份的 PPO update 2 做 Critic 前向/反向对照。用真实 rollout 1 的相同 64 个固定样本，保留 128 帧历史和原模型；每个配置先预热一次，再取两次耗时中位数。该局部微基准未包含 Actor、采集和 Drive 写入，不能声称整轮训练也提高相同比例。

| 输入方式 | microbatch | 累积次数 | 64 样本耗时 | Critic 基准峰值显存 |
|---|---:|---:|---:|---:|
| 原按 batch 重解码 | 4 | 16 | 1.5075 s | 2.48 GiB |
| 单轮预解码 | 4 | 16 | 1.2099 s | 2.48 GiB |
| 单轮预解码 | **8** | **8** | **0.9882 s** | **3.71 GiB** |
| 单轮预解码 | 16 | 4 | 1.0555 s | 6.16 GiB |
| 单轮预解码 | 32 | 2 | 1.2200 s | 11.07 GiB |
| 单轮预解码 | 64 | 1 | 1.2425 s | 20.47 GiB |

采用 8×8，有效 optimizer batch 仍为 64。Critic 微基准吞吐增加约 52.5%，耗时减少约 34.4%。更大的 batch 并未更快，64 的显存也不足以给完整 Actor/Critic/behavior 训练留足余量。基准只创建 Critic，不能将这里的显存值当作完整训练峰值。

## 代码改动

- `PreparedHistoryBatches` 按每个对局/座位的必要连续帧范围预解码一次，历史窗口只保存帧索引，重叠帧复用。
- 缓存 float32 归一化观测、离散卡牌 ID、mask 等固定输入，不缓存任何参与训练的 embedding 或特征；训练主干梯度保持完整。
- 默认最多缓存 2 GiB 输入到 GPU，且至少预留 8 GiB；不满足条件则保留在 CPU。当前基准的 6144 个独立帧只需约 63.8 MiB，准备耗时 0.42 秒。
- 保持原有每批实体数量裁剪、episode 边界、左侧补零、合法动作 mask、时间 mask 和手牌槽位顺序。
- `selfplay_v2` 每轮只构建一次输入缓存并记录准备、优化耗时及缓存大小。空完整对局的预热轮次仍允许跳过 Critic 更新。
- `colab_start_ppo_trial.py --resume --batch-size 8` 校验 Drive 已发布 generation 的 checkpoint SHA 与 pool SHA，恢复优化器/RNG/预热进度，只执行原计划剩余轮次。原 run_config 存档，不重新计算总 4 小时预算。

保留 2 个环境×1536 步、不改模型、学习率、奖励、折扣、epoch、PPO/KL 目标。此次重启丢弃未提交的 update 3 部分轨迹及未结束对局；update 1/2 权重、优化器和备份保留，不重做 Critic 预热。此为原训练器既有的重启语义。

## 验证与证据

- 不同 episode、seat、实体裁剪和左补零的输入逐元素、dtype、shape 与旧路径完全一致。
- 实际小型 Critic 的输出与所有参数梯度在 CPU 对照中逐元素一致，确认不是缓存掉梯度。
- 真实 L4 历史输入的缓存/原实现逐元素一致。
- 本地完整测试 146 项通过，修改文件 Ruff 与 diff 检查通过。
- 实测 JSON：`artifacts/ppo_pilot_20261104/efficiency_benchmark.json`；复现脚本 `scripts/benchmark_ppo_efficiency.py` 针对这次暂停在 update 2 的运行，勿在真实训练并行时执行。

恢复后需实际检查 `inputs_prepared`、Critic/Actor 优化、KL、非有限数值、故障计数与下一轮备份。完整训练的最终吞吐须以这些运行指标为准。

## 恢复后的真实训练抽样

从 update 2 恢复后，update 3 已完成轨迹采集并进入 Critic 优化。固定输入缓存准备 0.864 秒、63.8 MiB，存储于 CUDA。最近 40 个同 epoch 优化步间隔中位数 1.017 秒；每步有效 batch 64。

8 次约一秒间隔抽样：GPU 利用率 86%–95%，显存 7094 MiB，进程 RSS 约 4.45 GiB，进程 CPU 约 100%（psutil 的单核百分比口径）。这是 Critic 训练阶段的真实观测，不能与先前采样阶段的 27% GPU 利用率直接作前后对照。训练阶段 GPU 已持续繁忙，不需要填满 RAM/显存来提高表面占用。

本机保存 `artifacts/ppo_pilot_20261104/efficiency_live_report.json`，远端相同文件位于对应运行目录。部署时远端 4 项新增输入/梯度测试亦通过；源代码和微基准已校验备份到 Drive 的 `efficiency_source/` 和 `efficiency_benchmark.json`。

## Actor 与首轮完整优化验收

update 3 已完整保存并发布 Drive generation 3：Actor 真正优化 20 步，Critic 296 步，故障样本 0；Actor 触发已有按有效 batch 计算的 KL 限制后停止该轮 Actor 更新，属于保护机制。全轮 518.76 秒，其中优化 331.20 秒、输入准备 0.884 秒；该轮包含此前未执行过的 Actor 阶段，不可与仅 Critic 预热轮次直接比较整轮速度。

首步 Actor reference KL 约 4.6e-10，新旧 log-prob 最大误差 0.000448；后续第12步 reference KL 0.00609，Actor 阶段抽样 GPU 92%、显存 7900 MiB。包含真实落点头反向传播后仍未溢出。证据：`efficiency_actor_verified.json`。

用户随后恢复长期目标，需完成训练后检验强度。为避免旧控制器六轮结束立即释放 GPU 而跳过评估，在 generation 3 备份校验后进行了第二次明确的进程重启，丢弃未提交的 update 4 部分轨迹，从 generation 3 恢复剩余三轮；已完成的20个 Actor 优化步保留。已接入完成后自动评估选项：预注册 held-out pair 16–31 共32局及训练三卡组6局，均对阵准确的修复版BC起点、无额外WAIT偏置；全部报告和回放备份后才释放GPU。期间评估失败保留现场。边界2→3、3→4的重置会显式写入审计，不声称跨重启对局连续。未改变正式BC晋级门槛。

增加评估配对审计测试后，本地全套151项测试通过。
