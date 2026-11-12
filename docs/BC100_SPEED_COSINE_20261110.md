# BC 100 epoch、L4速度验收与余弦调度

本次沿用0.4秒离线BC数据、183M显式状态模型、175训练局/53验证局/65留出局。恢复源为已完成第一轮的update606，Adam/scaler/RNG及数据划分保持；总目标改为100个epoch，含已完成的一轮。每轮350座位更新，无溢出时终点为全局step35256。

## L4真实训练对照

同一个update606完整checkpoint、同两条完整训练座位轨迹，每组恢复同模型/Adam，lr=0保护权重。先做无计时事件的完整训练吞吐测量，另一次CUDA events分阶段分析；不把插桩计时当作实际吞吐。各组FP16 scale32、无溢出，参数分块SHA前后一致。

| 配置 | 两座位耗时 | 历史帧/秒 | 相对基线吞吐 |
|---|---:|---:|---:|
| frame8基线 | 5.090s | 177.2 | 1.00× |
| 延迟指标回读 | 5.045s | 178.8 | 1.01× |
| frame16 | 3.937s | 229.1 | 1.29× |
| frame32 | 3.364s | 268.1 | 1.51× |
| frame16＋TF32 high | 3.942s | 228.8 | 1.29× |
| frame16＋fused AdamW | 3.893s | 231.7 | 1.31× |
| frame32＋TF32 high＋fused | 3.300s | 273.3 | **1.54×** |

主要收益来自帧编码微批量增大，减少实体编码调用；时序chunk32、TBPTT、每座位一次optimizer step保持。两种frame大小的FP16矩阵形状变化也会产生小数值差，不称逐位等价；最快组固定探针最大动作概率绝对差0.000201，小于本次0.001准入线。

插桩首座位：帧编码0.607→0.165s（含在forward内，不能再次相加）；forward/loss1.132→0.678s，backward1.225→0.818s。数据准备/H2D约0.029s，指标约0.032s，当前主要耗时仍为模型前后向。profile阶段CUDA区间可能包含设备等待，仅作定位。

这是两条真实轨迹的有界测量，**不包含Drive发布、验证或整个epoch的冷缓存**，不将1.54倍称作完整训练已测倍数。初始293个Drive shard串行缓存约198秒、恢复点选择及复制约80秒；之后训练只读/content。

另用实际完整模型做32帧×200人工重复实体的FP16反向及fused Adam压力检查：全部输出、loss、unscale梯度和state有限，参数SHA不变，峰值allocated4.06GB/reserved4.35GB。KDA矩阵状态FP32、GQA缓存FP16。该探针从空state开始，不能代表整局最坏显存或战术正确性。

证据：`artifacts/state_global_20261110/bc100_l4_profile.json`；云端`/content/bc100/dense_probe.json`。只保留一台L4会话`bc100_speed_l4_20261110`。

## Warmup与余弦

- 从原constant恢复点明确交接一次，保留模型/Adam/RNG。`--start-lr-schedule`会记录父checkpoint SHA与anchor606。
- 最初LR3e-6；350次成功更新预热到峰值3e-5，余下成功更新余弦衰减到3e-6。
- 第一轮已完成，默认调度余量为99×350=34650次成功更新。FP16溢出跳步不推进调度；若有跳步，epoch结束可能略早于LR到达地板。
- 配置和调度位置写入checkpoint。正常resume自动继承，拒绝重新warmup或静默改变100轮终点。
- 训练行记录实际`learning_rate`、下一步LR、schedule step；W&B将这些字段转为标量曲线。

## 运行与恢复

入口为`scripts/run_colab_bc100.py`，先做4个真实更新的交接验收并保存，再从该完整点恢复剩余100轮预算。这4步计入总预算。数据与源码只从当前必要文件缓存，训练读取本地磁盘；每128个座位及epoch结束保存latest，验证每350座位/epoch结束，固定16验证座位。

Drive目录：`MyDrive/cr_rl/runs/state_global_event040_20261110/bc100_cosine`。源码包、文件SHA清单、notebook、性能报告、状态及最新训练checkpoint在此同步；不复制原始重建资料、历史权重或评测模型。

Notebook为`notebooks/colab_bc040_wandb.ipynb`，默认100轮/350 warmup/3e-6地板及相同速度参数。它自动选择原阶段、旧notebook、本阶段备份和本地缓存中最新有效完整点，避免回退，并保留调度状态。

W&B优先使用运行时`WANDB_API_KEY`；网页notebook还可读Colab Secret。CLI新L4目前未获得该环境变量、Secret访问超时，因此该运行时在没有凭据时明确记录offline模式和JSONL，不能冒称已在线上传；notebook在用户的已有环境变量下使用online。在线与offline游标分开，后续可从JSONL补传。

100轮是用户设定的训练预算，不是收敛承诺。轨迹验证与固定对手胜率分别判断，保留原留出对局，不因预算变长重新分区。

## 实际开训与交接回读

北京时间21:21左右，4步交接已保存到本地与Drive的update610，两端均为2187271307字节；CPU mmap回读确认anchor606、成功调度位置4、100轮horizon34650、next LR3.308571e-6，Adam paramgroup LR与调度一致、fused状态已保存。

随后正式进程11367从610接续，不再带`--start-lr-schedule`。最新实测到update658，交接以来52次更新全部成功、scale32、无溢出；warmup位置52，下一步LR7.011429e-6，训练RSS约2.21GB。当前最新持久点仍610，按128座位/epoch结束继续保存。

源码、manifest和notebook在Drive逐文件SHA回读一致：源码包SHA `a916f58d29c9baa61c29d5b07d6964347bdb3cbbb93cb9d240b566601910a792`；notebook SHA `b371acde1e8e32150687c9595d9cded7a89ee49e5fa3b54fa42492e5f32db4fa`。部署源码归档保持版本不变，此段实时回执为后续独立监控记录。

相关测试71项，加协调器5项，均通过；最终发布包的协调器与notebook另7项抽验通过。资源守护10892已启用：活任务/锁未释放时不关闭；完成或失败持续15分钟后，先验真并补传最新checkpoint再释放。

本地回执：`artifacts/state_global_20261110/bc100_release_verification.json`与`bc100_l4_dense_probe.json`。W&B当前CLI实例明确offline，未生成在线run链接；源码/notebook会在用户环境变量可用时使用online。
