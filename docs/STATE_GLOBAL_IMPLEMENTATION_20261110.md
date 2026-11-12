# KDA＋全局GQA实现与L4验收（2026-11-10）

**最新状态：按用户最终确认，已启动新的0.4秒离线BC重采样阶段。** 合并WAIT、动作后移、保留全部原BC记录的非WAIT标签；293局转换通过，已观察新更新257～288并备份。当前方案和运行入口见[STATE_EVENT040_TRAINING_20261110.md](STATE_EVENT040_TRAINING_20261110.md)。下文是旧原生重建试验的历史记录；该旧试验停在update256，其部分轨迹数据不再续训。

## 已实现结构

首个训练候选为 **182795519参数**：保留4层256维实体Transformer及dynamic4池化、当前实体到条件落点的cross-attention；时序改为1280维、`[KDA,KDA,KDA,GQA]×2`，20个Q/KDA头、5个KV头、head_dim64，SwiGLU FFN。相对122621959参数旧BC增加约49.1%。

KDA使用逐key遗忘与delta矩阵写入，分块低三角求解和逐帧递推有数值/梯度对照；不是GRU替代。没有依赖FLA/Triton，当前实现未包含Kimi原版所有短卷积等细节，因此称本项目KDA/GQA候选，不冒称完整Kimi Linear复刻。

两层GQA保留开局以来全部真实帧KV，RoPE使用整局递增位置；不滑动截断。KV按需扩容。不同模型、座位、对局和权重版本的state显式隔离，更新权重后旧state拒绝复用。即便只有WAIT合法，仍推进记忆。

精度为FP16 autocast/projection与KV，FP32参数/Adam、KDA矩阵状态/累积、帧与落点安全路径、概率边界；没有宣称纯FP16递推。此前超长buff仍存在，因此不把已验证的FP32帧边界直接移除。

## L4端到端实测

证据：`artifacts/state_global_20261110/l4_bench1280.json`。单台NVIDIA L4，满200个实体压力夹具，每组8次预热后40次完整决策；包括CPU观测适配、H2D、模型/状态更新、采样及CPU动作返回，CUDA同步。前缀由重复真实观测构造，权重为随机性能夹具，不是战术评估。

| 并行座位 | 起始历史帧 | p99 ms | 最大 ms |
|---:|---:|---:|---:|
| 1 | 1 | 28.03 | 28.03 |
| 1 | 375 | 28.52 | 28.57 |
| 1 | 750 | 36.29 | 36.66 |
| 1 | 1200 | 28.36 | 28.60 |
| 8 | 1 | 34.04 | 34.19 |
| 8 | 375 | 31.93 | 32.31 |
| 8 | 750 | 30.73 | 30.77 |
| 8 | 1200 | 34.44 | 35.01 |

320次测量均低于400ms，也通过p99<300ms的余量门槛。最高CUDA allocated约982MiB。此为已加载、已预热模型的实测，不包含checkpoint启动、网络往返或模拟器物理计算；不是操作系统未来调度的硬实时保证。真实对局评估另外报告每次完整决策的延迟及400ms超限次数。

本地RTX4090先行完成768/1024/1280宽度各3次混合精度更新，初始1200步历史后继续32步训练，loss/梯度/state均有限、scaler保持32。该随机特征数值探针不证明真实BC收敛。

## 0.4秒数据重建

从原始`IL_Replay_replays.parquet`（SHA256 `8f99ddd7b6f460c3efaf6873b4727e746a9e9edd8c796d120084aaaa68b4e261`）重新模拟，**不是隔行删旧5tick帧**。完整源离线核对293局、25004命令，source SHA、row/tag、命令数和卡牌映射通过。

命令映射到不早于源时刻的8tick网格；同座位碰撞按FIFO等待，另保留native合法性/执行确认与修复审计。网格等待最高7ticks（350ms）不等于总执行偏差；圣水、pending和既有执行器确认等待可增加额外延迟。完整源规划有102次同座位碰撞，未提前命令。

每局输出压缩NPZ、外置修复ledger与命令保留率/延迟统计；旧0.25秒数据保持原样。当前模拟器补丁SHA与旧数据分别记录，重建是新后端轨迹，不冒称旧状态精确复刻。

同局两座位和所有chunk固定到同一分区。原registry逐identity核对293/293命中，源划分175训练/53验证/65留出；最终重建接受集各分区数量由实际验收决定，不将源数量当成已经重建成功数量。

## 连续BC与验证

- 每局座位从tick0连续读入、chunk32，多时刻监督；跨chunk保留完整forward memory并detach，属于TBPTT，不是整局BPTT。
- 同一局座位处理完才做一次optimizer step；下个座位从空state开始，避免缓存由已变更权重产生。
- 条件loss按整个有效episode的N/非WAIT/普通出牌计数归一化，微批贡献直接相加；timing目标prior0.20，抽样prior使用真实连续训练行比例。
- 只计算普通出牌监督行的落点头，WAIT/技能监督目标及全部参数梯度与全落点计算做了等价性测试。评估/推理仍按完整策略输出。
- 初始兼容实体/读出张量来自旧epoch5 BC，新时序与不匹配维度重新初始化；不迁移旧时序optimizer，不宣称初始策略等价。
- 单L4有界3epoch试验，默认每32局座位保存最新、每64局座位短验证；阶段离线验证固定16座位（开发监控，不等于全53局验证）。只保留最新训练checkpoint与一个上一阶段模型，源BC使用已有冻结文件。
- 每epoch对旧BC做8局固定换边；第2/3epoch额外各对上一epoch做8局，共40局，保留48场预算余量。使用既有留出卡组、固定种子，0故障/非法为有效性要求；成功出牌费用另做native replay audit。
- 两边都以0.4秒决策。旧BC原本在0.25秒训练，因此旧BC对战带有网格迁移差异；分开报告离线拟合与实战战力，不将自博弈1:1当作已收敛。

## 文件与运行入口

- 核心：`src/cr_rl/models/state_temporal.py`、`state_policy.py`。
- 连续数据与训练：`src/cr_rl/training/state_bc.py`、`scripts/train_state_bc.py`。
- 重建：`scripts/rebuild_bc_interval.py`、`src/cr_rl/data/interval_rebuild.py`。
- 性能/数值：`scripts/benchmark_state_policy.py`、`probe_state_numerics.py`。
- 实战：`scripts/evaluate_state_policy.py`。
- 流水线：`scripts/colab_state_pipeline.py`；临时打包工具`.tmp_latest/package_state_candidate.py`。
- 新增CPU测试共70项：temporal22、policy7、stateBC25、interval6、evaluation10。源树兼容问题修复后分别通过；远端还会再执行整套。

打包以当前`.tmp_latest`的BC/补丁模拟器代码为底，覆盖根目录新state模块及训练器，避免把根目录旧未patch setup混入。实际发布文件以run中的`source_manifest.json`和源码包为准；其他旧PPO修改不被整体覆盖。

运行时：`state_global_l4_20261110`，唯一L4。

Drive输出：`MyDrive/cr_rl/runs/state_global_183m_040s_20261110`。实时阶段见`status.json`；训练见`training/metrics.jsonl`；实战见`evaluation/epoch*_bc`、`epoch*_old`。数据提交manifest在shards之后，checkpoint临时写入后原子替换。恢复使用持久阶段记录、固定上一轮快照和Drive最新权重，避免重跑/覆盖错误轮次。

本地仅G盘新增文件，沿用F盘现有conda Torch；没有重新下载本地Torch。Colab只补缺失依赖，不重装Torch。

## 实际启动与首次监控

新L4已完成Drive交互挂载、补丁后端构建、远端70项测试（全部通过），并实际重建完整293局源命令，接受288局、297450行。原始replay ID 84/124/130/181/265没有通过重建门禁；接受集实际分区为173训练/50验证/65留出。源命令完整可用不等于每局都执行到源末尾；重新模拟提前终局会降低执行保留率，详见每局audit，未把它报告成100%。

训练实际173局座位×2=346个更新/epoch，每个座位从开局开始连续训练。训练监督175612行，其中11359非WAIT、10886普通出牌；自然行动作比例约6.468%，目标prior仍0.20。当前模型约803万兼容参数从旧BC迁移，其余新时序/不匹配读出重新初始化，因此初始loss不能与旧BC直接当作同起点比较。

首次固定16验证座位（9637行）结果：

| 指标 | update0 | update64 |
|---|---:|---:|
| 同口径监督目标 | 15.1306 | 7.0142 |
| 加权时机loss | 1.1306 | 0.4605 |
| 条件选牌NLL | 1.2855 | 1.2940 |
| Gaussian条件落点loss | 12.7145 | 5.2597 |
| 选牌准确率 | 28.34% | 41.28% |
| 确定性动作召回 | 97.28% | 0% |

总目标下降主要来自新读出的初始化适配；条件选牌NLL未改善，确定性输出转向WAIT。随机策略仍可能出牌，不能把recall0解释为从不出牌；同样不能把这一loss下降称作实战提升。后续按旧BC/上一epoch对战结果决定。

已观察到update83仍运行，scale32、无溢出跳步，RSS约2.1GiB。初期裁剪前梯度峰值约3114，后期仍偶有尖峰，有全局clip1保护；有限性通过不等于完全平稳。权重及Adam通过latest原子保存并备份。默认资源守护在全部任务结束后释放唯一L4；失败持续15分钟才释放，留修复窗口并避免长期空转。

当前是BC新架构试验，未接入正式PPO训练；PPO流式状态及old_logp重建必须另行验收，不能将旧窗口PPO直接指向这个模型。
