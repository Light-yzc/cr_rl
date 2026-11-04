# PPO generation 6 后续推进（2026-11-04）

> **22:45最新：** 固定追加8轮已完成，Drive generation14已发布，累计67,701有效样本。
> 64局预登记换边评估正在运行，尚未产生完整强度结论。新增工程候选197项测试、真实
> L4微基准、tiny两轮训练与CPU第三轮恢复通过，详见[工程实测](RL_ENGINEERING_SPEED_20261104.md)。
> 第14轮原始820.04秒包含约206秒的独立工程基准停放，不能当作训练退速证据。
> 停放后恢复同一进程和未完对局，没有重启或丢弃learner样本。

本轮从 GitHub `04b3709` fast-forward 接手，已阅读 `plan.md`、`todo.md`、
128 局结果与数值诊断。通过仍在线的 CPU 会话重新读取 Drive，而非沿用旧 VM 的 PID。

## 起点核查

- 128 局完成标记与 summary 的 SHA 一致：
  `e5a36ce149f68a0262afbbc968bcfee584705402e17145b1575d7c455eec9109`。
- 已完成结果仍为 76 胜 52 负；本轮不重复这批评估。
- 完整 generation 6 checkpoint：2,919,967,669 bytes，SHA
  `6f15c4f7fd65e8e291b10316f38b071b0c8a118d97d0251fb726edb5aac973cb`。
- 保持 122.62M 模型、三套已验证普通形态卡组、独立 Critic 和原超参。

## 本轮执行

仅新增一台 L4：`ppo-continue-l4`；原 `drive-reader-cpu` 保留。
新 L4 已经用户单独授权、挂载 Drive，安装固定版本引擎与 Java 21。
Linux 最新完整测试 **189 passed**，Ruff 全仓检查通过。

新增 `validate_cuda_policy_parity.py`：用不可变 Actor5 和 rollout6 SHA 约束源文件，
从两个 lane、双方 learner seat、WAIT/非 WAIT 和不同实体密度中取真实历史。
开头缺失历史的跨 update 片段只在可还原完整窗口后使用，不伪造 padding。
对 FP32/BF16、microbatch 1/4/8、完整历史/输入缓存/帧缓存做概率对照。
原始 old log-prob 不改写。

在运行前固定数值保护范围：FP32 选中动作 log-prob 最大差 0.0005、TV 0.0001；
BF16 对应 0.03、0.01；BF16 复算相对原记录最大 log-prob 差 0.05。
这些是有限数值误差的工程检查，不是位级一致性声明或强度验收。
检查失败时不自动训练，也不自动放宽范围。

新增 `continue_ppo_training.py`，完整恢复 generation6 的模型、双优化器、RNG、
预热计数和对手池。新输出目录避免覆盖旧 pilot。
数值检查通过后，固定追加 8 个 update，目标 generation14，墙钟预算 3 小时。
每轮保留现有 Drive 完整 checkpoint、Actor 导出、轨迹和日志发布机制。
进程重启导致旧未完对局重置，后续连续 update 保持同一模拟器。
reference 仍是每轮冻结 behavior，并非永久 BC 锚点。

## 实时状态与证据

2026-11-04 21:22（Asia/Shanghai）：generation7 已完整训练、发布到 Drive 并独立 SHA
核验；generation8 正在推进。第7轮新增3,072有效样本，累计32,204；Actor更新11步，
Critic更新192步，5场完整终局，faulted_samples=0。平均KL 0.00702，reference KL
0.01176，value loss 0.00301；KL保护正常触发。该轮453.56秒，峰值显存7.11GiB。

第7轮完整 checkpoint：2,919,967,669 bytes；SHA
`b0370810c9b44ddee20d19c6f856f53de1beeb0958e27c505ab9977ff7f064dc`。
独立读回证据：`generation7_readback.json`。训练/备份路径正常不代表新强度已经提升。

数值审计共 64 个样本，lane/seat 四种组合齐全，历史长度 28–128 帧、实体数 5–42。
FP32 各路径选中动作 log-prob 最大差 0.00006485、最大 TV 0.000002262；
BF16 对应 0.0227356、0.00408894。BF16 复算相对原记录最大差 0.0157757。
结果支持被测范围内误差有界，不声称严格位级一致或所有状态已覆盖。

后续评估控制器已实际运行（本 VM PID 7294），等待训练完成，不与训练争用 GPU。
已预登记 generation10、14 分别对 generation6、准确 BC 起点，共 64 局：
pair16–23，物理 seed base6000000，sampling seed20261105，固定两块四配对。
逐块验证模型/代码/卡组/seed/seat，逐局原生重放，发布 GUI 回放与聚类配对统计。
这是一轮固定学习曲线诊断，不按中途胜率追加测试或选择最终 checkpoint。

W&B 真实事件镜像已连接：[本轮运行](https://wandb.ai/weidemingdeguge-test/cr-rl/runs/bbkxtmn1)。
使用用户此前授权的 key，通过远端进程环境传递；密钥不写入源码或公开报告。
日志镜像独立于训练进程，记录真实 Critic/Actor 更新、价值目标、终局和 KL 指标。
训练原始 JSONL 仍按每轮完整备份到 Drive。
W&B服务端已实际读回 Critic optimizer1948、value loss0.00275、Actor KL0.01024；
证据 `wandb_server_readback.json`，不是仅凭创建run或本地打印判定上传成功。

完成释放保护进程已运行（本VM PID10044）：必须确认generation14完整权重、
对手池、轨迹SHA、全部64局汇总和每份原生回放备份一致，才释放本台L4。
失败保留现场，不影响CPU读盘会话；不会仅因旧status写completed就删除运行时。

- 云盘目录：`MyDrive/cr_rl/runs/122m_ppo_continue_20261104/`。
- `continuation_protocol.json`：本轮固定预算、来源和数值范围。
- `continuation_status.json`：控制器与实际子进程状态。
- `cuda_policy_parity.json` / `.log`：实际数值对照结果。
- `evaluation_protocol.json`、`evaluation_status.json`：预登记的64局学习曲线。
- `wandb_server_readback.json`、`generation7_readback.json`：服务端与新权重实际核验。
- `release_verified.json`：最终核验标记已存在，generation14、64局评估与工程grid完成后释放L4。
- 训练本地目录：`/content/cr-rl-lab/artifacts/122m_ppo_continue_20261104/state/`。
- 连接观察器仅刷新同一 endpoint/kernel，不创建实例、不执行假计算。

训练后的固定权重强度评估与遗忘检查仍是后续必做项；追加训练结束不等于最终模型达标。

## 21:30 后续核查与指标补齐

generation8 已发布，累计37,322有效learner样本；本轮新增5,118，Actor优化5步，
Critic优化320步，4场完整终局，faulted_samples=0。平均KL0.00890，value loss0.00764；
原始log-prob最大复算差0.00617，KL保护触发。generation9正在采样，实际trainer PID6392
仍存活，未因一次SSH断连而重启。第8轮起点[648,454]等于第7轮终点，跨轮对局连续。

检查发现v2没有发出`targets`事件，导致W&B explained variance图为空。
只升级并重启了独立指标镜像（新PID11577），**训练进程没有重启**。
镜像现同时上传每轮summary、奖励分量、吞吐和峰值显存，并从已发布轨迹重建价值诊断：
使用原`old_values`、`bootstrap_values`、终局/截断边界及原gamma/lambda计算完全相同的GAE
target，取有效learner行统计。它度量更新前冻结Critic，而不是重新训练后的拟合质量。
零方差target不伪造explained variance。相关边界/源文件不变性测试通过。

第8轮GAE目标标准差0.17662、advantage标准差0.16443，更新前explained variance
**0.13328**，已从W&B服务端独立读回；这仅表明价值估计具有有限解释力，尚不支持
Critic已经成熟的结论。证据为云盘`value_diagnostics_000008.json`和服务端读回报告。
指标镜像故障不会中止或改写训练。

## 固定追加预算完成与独立复算

已完成generation7–14，累计67,701有效learner样本，新增各轮故障样本0。
模型仍为122,621,959参数、三卡组通用策略，没有中途替换架构或工程候选。
完整模型、两个优化器、RNG、预热状态与原始rollout已存Drive。

固定64局评估结果（每个比较16局、8个换边配对）：

| 比较 | 胜–负 | 配对95%区间 |
|---|---:|---:|
| generation10 对 generation6 | 13–3 | 62.5%–93.75% |
| generation10 对 BC5096 | 9–7 | 37.5%–75% |
| generation14 对 generation6 | 7–9 | 31.25%–50% |
| generation14 对 BC5096 | 10–6 | 50%–81.25% |

CPU独立读取并复算64局，核对权重、固定协议、换边、原生回放SHA与结局一致。
汇总SHA `7dbe5a67981040b818a1520a1553edd69408915ad8dfabc57a068e951126a45d`。
证据：`independent_cpu_readback.json`，GUI合档`learning_curve_all_matches.jsonl`。
最终轮相对generation6未显示改善；不能挑选中间表现较好的generation10当作预登记优胜者。
generation14的满费合法等待均值对6为0秒、对BC为0.75秒，行为改进不等于强度持续提升。

下一阶段按用户要求优先接入真实历史PPO对手：30%当前、50%历史、20%固定BC。
从generation14完整状态继续，generation6/10/14均冻结为对手，约20k样本保存历史候选，
容量8；保留历史与最强模型晋升分别处理。历史版本使用高位保留编号区间，避免与当前
update编号碰撞，历史座位不计入learner_mask。新代码Linux202项测试通过，真实模拟器
tiny检查正在进行；新L4训练尚未启动。架构保持不变。
