# 0.4秒、合并WAIT且保留动作的新BC阶段

## 当前执行方案

按用户最终确认，实际训练、推理与对战评估都使用8个模拟器tick，即0.4秒决策间隔。此前重新模拟原始人类命令导致提前终局、丢失后续动作的方案已停用；不恢复那份部分轨迹训练集。

新阶段从原先已经记录的完整BC数据做离线重采样：每座位的非WAIT动作按原顺序映射到不早于原时刻的网格；碰撞时继续向后排队，不覆盖动作。

```text
scheduled_tick = max(ceil(source_tick / 8) * 8, previous_action_tick + 8)
```

出牌、技能、原有flags与合法mask逐条保留。动作对应的原pre-action observation不变，仅decision_ticks字段改8；原tick/time_ms仍保留。另存source_rows/source_ticks供回查。WAIT只从当前逻辑时刻以前、下一条未发出动作以前的原始等待帧选取，避免把动作执行后的状态配给该动作标签。

这是`offline_delayed_bc_supervision`，属于BC离线增强；没有重新模拟延迟后的战场，不声称这些状态是新的原生0.4秒轨迹。数据明确禁止当作PPO rollout。实战验证仍通过模拟器以0.4秒运行。

## 已完成的全量转换验收

| 全293局原记录 | 转换前 | 转换后 |
|---|---:|---:|
| 数据行 | 492000 | 307914 |
| WAIT | 470943 | 286857 |
| 非WAIT | 21057 | 21057 |
| 普通出牌 | 20134 | 20134 |
| 技能 | 923 | 923 |
| WAIT占比 | 95.72% | 93.16% |

合并184086行WAIT，WAIT行数减少39.09%，总行数减少37.42%。原BC记录中的全部非WAIT动作身份、顺序、状态、mask、flags校验通过；动作没有提前，最大后移7ticks=350ms。本份BC数据没有额外FIFO碰撞排队。原来的无效/flagged标签仍按原训练规则过滤，结构保留不等于强行监督非法标签。

这里的21057条是完整BC记录中的非WAIT标签，不能改称原始人类parquet的25004条命令全部被重建。两者来源与口径不同。

单大NPZ按ZIP成员流式解压到磁盘memmap，逐局输出shards；转换后约15GB临时数组已删除。转换清单与逐局hash均发布到Drive，manifest最后提交。

## 模型、初始化与L4验收

- 182795519参数，时序宽1280，`[KDA,KDA,KDA,GQA]×2`；6层矩阵递推、2层整局GQA。
- 从已保存的architecture-v3 update256完整加载全部模型张量，包括时序层；新数据阶段重置Adam、scaler与跨局memory。优化计数接续256，首个新更新为257。
- FP16 autocast与KV；已有数值安全边界保持FP32，包括参数/Adam、KDA矩阵累积、帧编码与概率计算。不能称纯FP16全路径。
- 本轮L4用已训练parent256权重，满200实体压力夹具，1/8并行座位，历史长度1/375/750/1200；每组30次测量，共240次。
- 最高端到端warm延迟34.395ms，最高p99为33.779ms，全部低于400ms门槛。包含观测适配、H2D、模型/状态更新、采样和动作返回；不包含启动加载、网络或模拟器物理计算。
- 历史前缀使用重复观测构造，性能验收不等于战术效果证明。真实对战另记录延迟。

## 训练与评估安排

固定原对局identity划分175训练/53验证/65留出；每个对局的两座位各从开局连续读入。chunk32、frame microbatch8，跨chunk保留forward memory并detach；每座位完成后一次optimizer step。每epoch共350个座位更新。

学习率3e-5、timing目标动作prior0.20，实际sample prior从新训练集计算；条件选牌与落点分别归一化。每32座位保存并校验上传最新checkpoint，每64座位在固定16验证座位监控。只保留最新训练checkpoint和一个上一阶段权重文件，不累计每步checkpoint。

共3epoch；每epoch8局对旧BC、8局对自己的旧快照，合计48局。epoch1的旧快照为parent256，后续为上一epoch。每组4个固定卡组配对并换边，两边均以0.4秒决策。旧BC原在0.25秒训练，结果须注明其决策网格迁移差异。

实际监控已完成新更新257～353：97/97更新成功，loss与梯度有限，FP16 scale32，无溢出跳步，最高训练RSS约2272MiB。已通过CPU mmap回读确认本地与Drive完整checkpoint均为update352、next_episode96，2187268043字节，数据指纹与新manifest一致；203个模型张量与199个Adam state项存在。首次更新裁剪前梯度约534.5，后续仍有波动，clip1生效，不能仅凭“有限”就称全面健康。

固定16验证座位共8731有效行，已完成起点与第64次新更新验证：

| 指标 | update256起点 | update320 |
|---|---:|---:|
| 同口径总loss | 6.8203 | 6.7876 |
| timing loss | 0.4518 | 0.4611 |
| 条件选牌NLL | 1.2445 | 1.2330 |
| 落点loss | 5.1240 | 5.0936 |
| 条件选牌准确率 | 42.56% | 40.15% |
| 确定性动作召回 | 0% | 5.72% |

变化很小且选牌准确率下降，仍有明显WAIT偏置，不能用高总体准确率或loss下降代替实战胜率，也不能声称已经收敛。首次对战在本epoch结束（约update606）后。

## 路径与恢复

- 唯一L4会话：`state_global025_l4_20261110`。名字里的025是创建时旧名，实际运行参数`--decision-ticks 8`。
- 云端工作目录：`/content/state_event040`。
- Drive：`MyDrive/cr_rl/runs/state_global_event040_20261110`。
- 入口：`scripts/colab_event040_pipeline.py`；实时`status.json`、`training/metrics.jsonl`，对战`evaluation/epoch*_bc`与`epoch*_old`。
- 新阶段中断后必须对新数据manifest恢复`training/latest.pt`；不得把parent256冒充新阶段精确resume。
- 源码以本次上传的`source.tar.gz`/`source_manifest.json`为准。后续恢复修复单独备份，不改已提交数据fingerprint，也不把未来代码冒称当前进程已热更新。
- 已额外部署并启动`colab_event040_resource_guard.py`，确认活trainer时只等待。完成时、或失败持续15分钟后，必须先等pipeline锁空闲、无活任务，确认Drive仍挂载并流式校验/补传latest，再释放L4。Drive失效或备份失败会阻止释放；实际释放API尚未触发。
- 恢复revision2（`source_recovery_revision2.tar.gz`/对应manifest）补了未提交重采样输出回收、评估checkpoint轮次和权重hash绑定、禁止缺失上一epoch对手时替换为parent。未来重启入口为`/content/state_event040/recovery_revision2/scripts/colab_event040_pipeline.py`；运行中orchestrator仍是最初版本，未中断trainer。当前数据未重新生成。
- 发布源码解包后CPU测试68项通过：resampler6、recovery11、stateBC40、statePolicy11。
- 本地证据：`artifacts/state_global_20261110/event040_monitor_snapshot.json`。本地实验、下载和缓存均在G盘，未安装新Torch。
