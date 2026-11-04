# RL 工程提速：实施与实测（2026-11-04）

目标是在相同 learner 样本和有效优化 batch 下减少 CPU、重复前向与历史备份开销。
不改变奖励、旧概率、GAE、模型宽度、合法动作或历史窗口含义。
当前八轮有界训练仍使用原版；工程候选在独立 `/content/cr-rl-speed-review` 验证。

11月5日续接：上述八轮已结束。新历史池阶段默认启用增量推理/备份，并显式使用FP32
帧编码器解决新冻结状态上的BF16缓存数值超限；此精度改变可能影响优化耗时，不能
直接把旧BF16微基准加速比例移用于新阶段。参考缓存与延迟指标仍为可选项。

## 实测与限制

- 从真实运行的 trainer PID6392 采了两段各25秒、40Hz的py-spy栈。
  首段跨越采样/优化边界，已按实际栈重新分组，不能把整段称为纯采样。
  共有634个采样栈、906个优化栈，另有准备/空栈样本。
  采样栈命中帧缓存查表/bytes生成、窗口组装，支持先优化CPU路径。
  优化栈经常停在标量CPU拷贝，但这也包含等待GPU完成，不能直接推断可移除比例。
- 真实 Actor10 权重、真实 rollout10 状态、L4、128帧、4个座位流，三次交替顺序：
  原版32个测量步中位数1.472485秒；增量接口0.847327秒，**1.7378×**。
  被测状态选中动作log-prob差与TV均为0；不外推全状态位级一致。
  这是同步推理微基准，**没有包含模拟器、优化器或Drive备份，不是整轮加速比例**。
  测量时用pidfd暂停并恢复同一trainer，未重置模拟器、权重或RNG。
- 原版训练update12、14含工程基准停放时间，不拿这些原始墙钟做前后提速对比。
- 最新隔离回归197项通过；联合CUDA、真实Drive历史I/O、tiny完整训练和恢复均已通过。

## 已实现的候选

1. `IncrementalFrameHistory`与`IncrementalFrameEncoder`：
   以episode/seat/帧序号和环形槽位替代整帧bytes键；不逐步复制完整原始历史。
   帧特征写入GPU环形缓冲，时序Transformer仍重算完整因果窗口。
   每个模型独立按参数版本和精度使缓存失效；保留原始帧供更新后重新编码。
   bootstrap preview不改变历史，跨局、128帧滚动、模型更新和过期请求有测试。
2. `publish_incremental`：
   第一次扫描已有轨迹，以后只处理新generation的轨迹与变化的活跃对手。
   receipt绑定备份目录及源元数据；旧数据每16轮重新完整校验，恢复仍完整校验。
   新checkpoint同步序列化与发布，所有新增文件校验完才推进latest，不后台读活模型。
   外部破坏旧云盘文件会在周期审计/恢复时发现；不能声称每轮重新读过所有历史字节。
3. `ReferenceDistributionCache`：
   一轮内按需缓存完整冻结reference输出，不缓存可训练特征。
   miss仍用原完整microbatch，避免为早停后根本不用的样本做整轮预计算。
   复用时需检查BF16微批次形状差异；默认未开启，当前KL常在第一epoch早停，未必有收益。
4. `defer_scalar_metrics`：
   Critic loss保留在设备上，到有效batch/最终汇总再读取；Actor指标合并读取。
   非有限loss检查和有效batch KL拒绝仍保留，不取消数值保护。
   CPU回归验证参数更新、KL判断与汇总一致。
5. 每轮独立`timings/update_XXXXXX.json`记录采样、优化、候选接纳、序列化、日志、
   备份、真实总墙钟与有效样本/秒，避免原summary在备份前截时造成误解。

候选CLI开关：`--incremental-inference --incremental-backup --cache-reference --defer-scalar-metrics`。
验证后，后续v2启动默认启用增量推理与增量备份，可用对应`--no-...`参数回退。
reference缓存、延迟标量仍显式开启；当前有界训练与固定评估未热替换。

## 证据位置与当前工作

Drive根目录：`MyDrive/cr_rl/runs/122m_ppo_continue_20261104/`。

- `engineering_benchmark/inference_streams4.json`：首个实测微基准。
- `engineering_benchmark_v2/`：联合隔离验证，含source、tests、ruff和各项结果。
- 联合控制器PID23360已完成；测过向量化接口、reference/scalar组合、历史payload备份，
  tiny两轮实际PPO贯通全部开关，之后另在CPU恢复到第三轮。
  备份微基准明确用极小公用权重占位，仅测历史数据增长开销，不假冒2.9GB模型发布。
- `artifacts/continue_20261104/profile_classified.json`为本机小型栈归类结果；原始栈在
  当前VM的`artifacts/122m_ppo_continue_20261104/profiling/`。

六卡组普通形态的接口检查已在CPU完成144局、36有序配对各4次，0故障/截断/非法，
12局逐位重放。新配置`configs/decks_expansion6.yaml`与留出卡组卡牌集合不重叠。
Drive证据：`runs/expansion6_acceptance_20261104_r4/acceptance.json`。
前三次尝试均在产生对局前遇到CPU依赖/导入错误，已保留失败记录；不记为模拟器机制失败。
这是未来扩大训练的接口自洽证据，当前PPO仍三卡组，也没有官方客户端保真结论。

## 联合基准最终结果（22:45核对）

| 项目 | 原版 | 候选 | 解释 |
|---|---:|---:|---|
| 4流、128帧、32步推理中位数 | 1.45274s | 0.81646s | 1.7793×，这批输出差0；不含模拟器/训练/备份 |
| Actor两epoch、64样本/epoch、batch8 | 3.54732s | 3.09799s | reference+延迟指标，时间约减12.7%；零LR吞吐测试 |
| 同预算batch16、cache+defer | — | 3.15517s | 此次未优于batch8，不加大生产batch |
| 历史payload发布中位数 | 0.51155s | 0.11614s | 约4.40×，排除了真实大权重序列化/上传 |

备份generation12旧路径读云盘64,595,697字节，新路径5,378,995字节。
该微基准仅用32字节checkpoint占位与真实rollout payload；不把4.40×外推至2.9GB完整发布。
reference仅在重复遍历才节省前向；这次8次forward、64行cache hit，当前生产常在首epoch
KL早停，因此仍保留显式开关。延迟标量单独对Actor提升很小，没有“移除一半耗时”的证据。

额外真实CUDA reference验证：64状态、batch8、两次不同混洗，最大选中动作log-prob差
0.00002956，最大TV0.000002178，通过固定保护范围。CPU回归包含reference缓存后的
参数更新和KL判断一致；BF16不泛称全状态位级相同。

联合tiny运行：2轮、4终局、Actor32步、Critic56步、0故障，完整备份发布到generation2。
在CPU完整恢复到generation3后又分别更新2步，预热状态未重置，默认增量开关生效，
新云盘checkpoint SHA核验通过。CPU恢复不声称与CUDA后续轨迹数值等价。
tiny真实整轮墙钟38.16s和62.42s，已记录所有分段；这不是122M前后加速比。

联合基准source归档SHA：`ed84ab2809577100dc7f8d95444aefe9278b3d945fbb25845ab67cbb025ce385`。
后续默认开关改动与reference额外验证源码另存最终工程归档，不冒称是该原始基准源码。

后续grid已完成。Actor batch4/8/16分别3.5172/3.1039/3.1322秒；Critic分别
1.4066/1.2785/1.2857秒，本轮保留两者microbatch8。独立Actor/Critic微批量参数已接入，
各自累积次数保持有效batch不变。延迟读取未使Critic稳定加速，不默认启用。

实际collector环境2/4/8在固定512环境transition预算下测得38.00/70.46/117.25有效样本/秒，
但每环境步数为256/128/64，平均历史长度96.25/64.5/32.5，场上密度也不同，且没有终局。
因此这些结果不能证明完整128帧稳态3倍吞吐，不据此直接切换env8。
仍待完整历史控制的环境grid和122M真实整轮前后对照。

最新Linux全套202项测试通过。reference缓存复用不同batch顺序时允许明确的数值容差；
单独延迟读取仍要求参数更新及指标逐位一致，KL停止判定保持一致。
