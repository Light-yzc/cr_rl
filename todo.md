# Agent 交接 TODO

> **11月5日最新：** `ppo-history-l4`已挂载Drive、203项测试通过；历史分桶及tiny
> 两update检查通过（1536 learner / 512 frozen decisions）。首次CUDA失败证据保留。
> 帧编码FP32在同一64状态通过原门槛（BF16路径差.01549、原old logp差.04916）；
> r2已完整恢复generation14并载入真实6/10/14历史模型，第15轮已发布，第16轮推进。
> Drive `122m_ppo_history_20261105_r2`；控制器13220、trainer13779、评估13225、
> release13233、指标镜像13456；kernel9111。实时进程可能变化，须查/proc后使用。
> 第15轮CPU独立核验checkpoint/池SHA通过，累计72,246 learner样本，Actor21/Critic288步、
> 0故障；完整轨迹1,599 frozen decisions均排除在PPO外。固定到22，18/22共64局已登记，
> 最终强度尚未验证；当前仍3卡组训练，六卡组仅机制验收。
> W&B新run `itnqtrmf`。详见[当前交接](docs/PPO_HISTORY_20261105.md)。

> **最新接续（历史池优先）：** generation14及64局固定评估全部结束，CPU独立复算通过；
> Drive `122m_ppo_continue_20261104/independent_cpu_readback.json` 为独立证据，
> `learning_curve_all_matches.jsonl` 为64局GUI合档。generation14对6为7–9，对BC为10–6。
> L4已释放，无新训练正在运行。历史分桶30/50/20、20k样本历史保留、保留编号区间隔离与
> seed校验已实现，Linux测试与tiny真实模拟器检查推进中。下一步先验证再开新L4，
> 从generation14完整恢复，冻结generation6/10/14为对手；不得按测试结果改选10为最优起点。
> 工程grid已完成：Actor/Critic均保留microbatch8；env2/4/8测试历史长度不同，
> 不能据此声称稳态3倍加速。122M整轮受控对照仍待完成；旧状态见下方历史记录。

> **22:45最新：** 固定8轮追加训练已结束，generation14发布、累计67,701有效样本，
> 64局固定评估正运行，控制器7294和实际kernel监督均在线。工程候选197项测试与
> tiny训练/恢复通过，增量推理约1.78×、重复epoch Actor约省13%、历史I/O微基准约4.4×；
> 均不是122M整轮提速比。后续v2默认增量推理/备份，当前固定评估不变。
> 2/4/8实际环境与Critic独立batch/整轮受控对照还待做，现维持env2、batch8×8。

> **工程提速补充（22:15）：** 参阅[工程实测](docs/RL_ENGINEERING_SPEED_20261104.md)。
> 隔离候选197项测试通过；真实L4增量推理微基准约1.74×、被测输出差0，非整轮加速。
> 增量备份、按需reference缓存、延迟标量读取与整轮分段计时已实现，默认关闭。
> 联合基准PID23360等待真实采样阶段；原trainer6392仍为原版连续训练，不重复启动。
> CPU六卡组接口144局和12局逐位重放通过；当前PPO训练仍三卡组。

> **Windows 接手推进（2026-11-04 21:05）：** 已拉取 `04b3709`，通过原 CPU
> 独立核对 generation6 和 128 局完成 SHA；只新开一台 `ppo-continue-l4`，
> 用户已完成新 Drive 授权，186 项 Linux 测试与全仓 Ruff 通过。
> 21:22 更新：完整状态恢复和64样本真实 CUDA 概率对照已通过，generation7
> 已训练并独立核验Drive checkpoint，累计32,204有效样本；generation8推进中。
> 最新188项测试与Ruff通过，W&B真实指标服务端读回。64局冻结评估等待训练完成，
> 全部备份核验后的L4释放保护已运行。整个追加训练与强度验收仍未完成。
> 21:30：generation8已发布（37,322有效样本），generation9采样中。仅升级指标镜像，
> 未重启trainer；补齐真实GAE target诊断，第8轮更新前explained variance=.13328，
> W&B服务端读回；最新189项测试通过。
> 以 [本轮记录](docs/PPO_CONTINUATION_20261104.md)
> 和新目录 `122m_ppo_continue_20261104/continuation_status.json` 为准。
> 下方“本轮结束后不再自动开GPU”是旧设备交接时约束；本轮按用户继续推进指令执行。

更新时间：2026-11-04 20:40，Asia/Shanghai（补测完成后交接）。工作目录 `/Users/Regenin/Code/rl_training`。

**接手结论：128局复测已经补齐并独立复算：76胜52负（59.375%），按卡组配对聚类95%区间53.125%–65.625%，符号检验p=.01953125，通过这次固定复测的提升判据。模型仍为六轮PPO generation6，支持相对BC起点改善，不代表全卡组/官方客户端达标。下一步是CUDA数值对照和持续学习曲线，不要重跑这128局。**

## 1. 当前真实状态

- 当前HEAD：`f2079e9`（update todos，前序基线`16f1604`）。此前大部分改动已纳入该提交；当前仍有`plan.md`、`todo.md`、恢复控制器修改和新增恢复测试未提交，以实时`git status --short`为准；务必保留，不能reset/clean。`artifacts/`、`.cache/`、权重与私有日志被Git忽略，换机器只复制Git代码会丢失本地证据。
- 实时 `colab sessions` 仅有 **`drive-reader-cpu`**，标准CPU，endpoint `m-s-euw4b6-34lot7qz7dd2s`；Drive已挂载 `/content/drive`。无活动GPU；补测GPU已完成、备份并按配置释放。
- `ppo-seed-eval-l4` 已不在线。Drive旧status仍写running，这不是活进程证明；服务器assignment消失已确认。最后一批缺失，退出原因未确定，不得写成正常完成/已自动释放。
- 本地CPU连接观察器日志最后是 `watch_failed: TransportError`。观察器已退出不等于CPU VM消失；本轮服务器确认CPU仍在线。需恢复观察器时只附着现有session/kernel，不能自动新建。
- 用户要保留便宜CPU作长期读盘入口；Colab不能保证永久在线。用户希望单台GPU，CPU读盘会话可并存。补测GPU `ppo-eval-finish-l4` 已完成且释放。用户准备换设备，本轮结束后不再自动开GPU。
- 用户多次强调先把模型弄好，官方安卓客户端实测延后；不要把客户端安装/登录当成当前前置步骤。

## 2. 已完成与结论

### BC

- 结构化状态，无视觉。修复版 **122,621,959参数**，实体self-attention→时间causal self-attention→选牌/技能/WAIT及落点cross-attention。沿用121M实体/时间主干，升级局部读出；不是300M。
- 模拟器：crforge commit `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`，data `16.635.3`，50ms tick、5 ticks/decision（4Hz），原始seat状态7024维、联合动作2307。
- 人类BC数据是回放近似重建，不是官方内存轨迹。冻结979分片，Stage2数据SHA `e7edb3151eb064450335fd807533db8025be16f9d73f7ee8f7f82d43b82ca732`。
- 修复/原版对照均到step5096。相同step5088快验：slot33.64%→38.18%，落点L1 11.8→11.11，within2 20.00%→18.18%。只有局部收益，没有明显空间能力突破。
- 用户授权停止剩余BC扩大评估、先试PPO。**没有通过正式BC promotion gate，不得篡改门槛或补造晋级报告。**

### 六轮探索性 PPO（已完成）

- Drive：`MyDrive/cr_rl/runs/121m_ppo_pilot_20261104/`。
- 29,132有效learner样本，Actor54个优化步、Critic1756步、26场正常终局、faulted_samples=0。
- 前2轮独立Critic完整对局预热，后4轮Actor实际更新；同版本双方数据复用，历史对手只使用learner侧。
- 三卡组：`configs/decks_initial3.yaml`：巨人电车、野猪循环、墓园控制，普通形态。一个通用模型根据卡组输入使用三套牌。
- 2环境×1536步，Actor LR2e-5/2epochs，Critic1e-4/4epochs；microbatch8、累积8，有效batch64；gamma.9995、lambda.98，tower/crown shaping、overflow.001、referenceKL.01、targetKL.02。
- reference为每轮冻结behavior，**不是永久BC锚定**；当前池只有BC anchor，尚未达到250k learner样本候选门槛，不能声称长期历史池效果已验收。
- 在generation2、3后进行了明确的进程重启：模型/optimizer/RNG恢复，未完成游戏重置。不能声称模拟器精确恢复或这些边界连续。
- generation6已发布；恢复必须带完整checkpoint、pool和对手快照。`latest.json`、`checkpoints/update_000006.manifest.json`为恢复来源，Actor导出不能用于恢复optimizer。

### 原38局完整评估（已完成且核验）

- 冻结PPO6对准确BC初始化，无额外WAIT bias。
- 留出32局 **19胜13负（59.375%）**，配对bootstrap95%区间43.75%–75%，双杀/各胜一局/双败=5/9/2，sweep符号检验单侧p=.2265625；预注册提升判据未通过。
- 训练三卡组6局5胜1负，样本不足。
- 留出场均成功出牌18.50→24.06，满费等待22.50→5.20秒；仍有18.25秒连续满费等待。38局原生回放parity全部通过。
- 结论：行为改善、胜率点估计向好，**尚不能确认通用强度显著提升**。这些留出卡组曾用于历史评估，不是从未查看的最终保留测试。

### 效率与数值

- `PreparedHistoryBatches`每轮只归一化/解码固定观测一次，复用帧索引；不缓存训练embedding，主干梯度完整。
- L4同样本Critic微基准64样本：原4×16为1.507s，缓存8×8为.988s，吞吐约+52.5%；不是整轮加速比例。真实训练GPU86–95%，Actor92%，显存约7.9GB。
- 新旧初始log-prob最大差约.0155，CUDA BF16全覆盖一致性未验证。
- 使用真正Actor5与rollout6两个非WAIT完整128帧样本，CPU FP32缓存/完整概率最大差约7e-8，训练/推理模式也约8e-8；两例相同lane/seat，覆盖有限，不能代替CUDA BF16检查。
- 模拟器三卡组100完整随机合法局：0故障/截断/非法，3局逐位重放；61项上游专项机制测试通过。未证明官方客户端保真、完整Sparky电击重置/碰撞/目标切换对齐。
- 最近全量本地测试：**184 passed**；补测L4同样184项通过。原首次复测源码包对应177项。

## 3. 128局复测已完成（保留原恢复记录供审计）

### 已冻结协议

`artifacts/ppo_pilot_20261104/followup_protocol.json`；Drive原PPO目录同名文件。

- policy：固定Actor6；opponent：固定修复BC5096。
- 卡组 `configs/decks_eval.yaml`，pair16–31，共16配对；每组物理卡组/seed相同，只交换策略座位。
- 四组simulator seed base：2000000/3000000/4000000/5000000。
- 对应policy sampling seed：20261201/20261202/20261203/20261204（入口内部加pair-start）。
- pairing_seed20261103，每chunk4配对=8局，共16chunk=128局。
- 重复seed必须按deck pair聚类统计，不能把四个seed当四套独立卡组；完整固定批次结束再结论，不根据中途胜率追加到显著。

### 中断时已有120局（历史，已补齐）

本轮18:06通过CPU重新读取：120局累计71胜49负、无无效局；只是未完成批次的描述性统计，不能替代128局聚类区间和正式提升结论。

Drive：`MyDrive/cr_rl/runs/121m_ppo_seed_replication_20261104/`。

- `seed_2000000/pairs_016,020,024,028`：四块完整。
- `seed_3000000/pairs_016,020,024,028`：四块完整。
- `seed_4000000/pairs_016,020,024,028`：四块完整。
- `seed_5000000/pairs_016,020,024`：三块完整。
- **只缺 `seed_5000000/pairs_028` 的8局**。截至最近读取，无`summary.json`、无`completed.json`，每个已存chunk都有execution_audit，120局native parity通过。
- 保留原120局，补局必须使用原来评估源码和协议，不改变batch大小、seed、权重或WAIT bias。评估采样不是batch-size invariant。

### 原恢复执行步骤（已完成；不要重复启动）

- [x] 先 `colab sessions`、通过CPU挂载重新列Drive，确认缺失仍仅最后8局。不要用旧running status证明活进程。
- [x] 写恢复入口：验证每个已存chunk的权重/代码/deck SHA、seed/seat/pair与协议、evaluation SHA对execution audit、完整原生回放数；只跳过完整合格chunk，不删除/覆盖已有结果。
- [x] `scripts/colab_seed_replication.py --resume` 已实现：严格核对注册协议，已有chunk通过seed/seat/权重/代码SHA、evaluation-to-audit SHA、每个原生回放SHA校验后跳过；部分残留回放拒绝覆盖。新权重不变，评估/聚合源码SHA与原运行相同。184项本地测试通过。
- [ ] 评估原源码归档在原PPO目录 `replication_source.tar.gz`，268120bytes，SHA `7dd9a553f28cf1a308bb2b3910764c00c078b205ccbc61e6a7a390cc530562b1`；receipt同目录。先核对归档，再恢复到新GPU。
- [ ] 更省授权的方案：用已挂载CPU准备最后8局所需的权重/源码/表格小包，传到唯一新L4；GPU结束后结果传回CPU再写Drive。仅当传输已实际验证且保持权限边界时采用。否则给用户真实的新VM挂载OAuth链接。不能复用别的VM凭据服务地址或把未挂载当成已授权。
- [ ] 若新GPU需Drive授权，`colab drivemount --session NAME`务必`tty:true`，等待用户完成后才向PTY发换行。非PTY会提前提交400；确认已挂载再启动。
- [x] 补局后使用 `summarize_repeated_evaluations(protocol,names,evaluations)` 聚合全部16块，独立复算并校验SHA，再发布summary/completed。原128局若仍未齐，不能生成正式提升结论。
- [x] 汇总胜负、deck-pair聚类区间、符号检验、各卡组、成功出牌与满费等待；满足预注册规则才写“本固定复测显示提升”，不外推高手或官方客户端。
- [x] 更新本todo、plan、README和结果文档，备份完整回放/代码/报告后释放唯一GPU，CPU保留。

### 完成证据与迁移包

- 128局=76胜52负0平，最后8局5胜3负；16块均有完整原生审计及回放，CPU独立读回，本地聚合器逐字段复算一致。
- 完成summary SHA：`e5a36ce149f68a0262afbbc968bcfee584705402e17145b1575d7c455eec9109`；Drive原复测目录有`summary.json`、`completed.json`。
- 本地证据 `artifacts/ppo_seed_replication_20261104/final_readback.json`；解释见 `docs/PPO_REPLICATION_RESULTS_20261104.md`。
- 新设备包：`MyDrive/cr_rl/handoffs/20261104/cr_rl_handoff_20261104.tar.gz`，相邻`archive_receipt.json`记录SHA。先核对SHA、解压、读`RESTORE.md`，克隆包内`repository.bundle`后运行overlay脚本恢复未提交文件。包内manifest逐文件校验，已实测clone+overlay可还原。
- 包不含模型、BC原始数据、venv和凭据；模型/优化器仍在原PPO Drive目录。新设备需正常登录Colab，不能复制OAuth文件。
- 旧“只缺8局”的描述均为中断历史，不是当前待办。

## 4. 后续计划（仍未完成，按结果推进）

- [ ] 客户端接入方案已调研并记录于 [状态读取与出牌桥接](docs/CLIENT_STATE_ACTION_BRIDGE_20261104.md)：优先验证 `cr-memory-reader` 只读状态，再接 ADB 触控及内存执行确认；当前主要证据来自 Nulls，官服兼容性和观测字段完整性均待验收。此为方案记录，不替代当前 PPO 训练优先级，也不表示已实现客户端闭环。
- [ ] CUDA真实权重检查多lane/seat、不同实体密度/历史长度的FP32与BF16、缓存/完整重算、microbatch变化对old/new log-prob误差的影响。不要伪造old概率或用更新后权重重新定义behavior。
- [x] 从generation6完整恢复追加到generation14并完成固定64局学习曲线；最终轮对6为7–9，不能认定持续提升。
- [ ] 历史池新阶段从generation14固定追加到22，完成预登记64局换边与原生回放审计。
- [ ] 当前弱卡组及长满费等待分解见 `deck_diagnostics.json`。不得简单提高惩罚或对所有局面加WAIT偏置就声称修复；最长等待也出现在赢局。
- [ ] 历史池已接入6/10/14，约20k样本保留候选；仍待首个自动新快照接纳及长期遗忘评估。
- [ ] 历史池阶段后再做学习率对照：resume加载optimizer会恢复checkpoint的param_groups，
  不能只修改CLI就宣称学习率已变；需明确迁移并读取实际optimizer学习率验证。
- [ ] 扩大至4–6套通过机制验收的普通形态卡组，版本化记录覆盖；不要把广泛BC卡组说成PPO实际覆盖。
- [ ] 300M等样本/等墙钟学习对照、V100实时P95、完整保真/客户端接入、通用后电车专精及蒸馏均未完成。用户把客户端延后；无需此时催安装登录。
- [ ] `plan.md`累积了大量互相覆盖的历史状态，接手以本todo、当前Drive和最新结果为准；不能把全文所有历史方案同时视为已实现。

## 5. 文件与工具入口

| 用途 | 入口 |
|---|---|
| 主PPO | `scripts/train_selfplay.py` → `src/cr_rl/training/selfplay_v2.py` |
| 六轮监督/恢复 | `scripts/colab_start_ppo_trial.py`（硬编码6轮，用于本次pilot；不要直接当下一阶段任意轮数启动器） |
| 冻结评估 | `scripts/evaluate_policy.py`，支持独立simulator-seed-base/pairing-seed |
| 原pilot完整评估 | `scripts/evaluate_ppo_pilot.py`，历史结果保留旧代码SHA |
| 128局控制器/聚合 | `scripts/colab_seed_replication.py`、`src/cr_rl/evaluation/repeated.py` |
| 原生执行审计 | `scripts/audit_policy_replays.py`、`scripts/java/ReplayExecutionAudit.java` |
| 每卡组诊断 | `scripts/analyze_policy_games.py` |
| 数据输入缓存 | `src/cr_rl/training/prepared_history.py` |
| 连接观察 | `scripts/watch_colab_session.py`、`src/cr_rl/training/colab_connection.py` |
| 历史完整结果 | `docs/PPO_RESULTS_20261104.md`、`docs/PPO_DIAGNOSIS_20261104.md` |
| 性能/验收矩阵 | `docs/PPO_EFFICIENCY_20261104.md`、`docs/PLAN_STATUS_20261104.md` |

重要本地证据：

- `artifacts/ppo_pilot_20261104/final_drive_review.json`：38局最终报告与generation6清单。
- `artifacts/ppo_pilot_20261104/game_diagnostics.json`、`deck_diagnostics.json`：原生执行细节。
- `artifacts/ppo_pilot_20261104/cpu_policy_parity.json`：两个真实样本CPU对照。
- `artifacts/ppo_seed_replication_20261104/drive_review_incomplete.json`：120局已核验读取快照。
- `artifacts/ppo_seed_replication_20261104/connection.jsonl`：assignment消失与传输异常记录，不能从中断直接推断唯一原因。
- `artifacts/drive_reader_cpu_20261104/worker.json`及connection日志：CPU旧watcher状态；PID可能已失效，启动前查真实进程。

## 6. 关键哈希

- Actor6：`2bf44455826ec00d7d58962ce3e07c93a93e0ca02bd053951ff5585b814bce54`。
- BC修复5096：`ca7c91d1b54bb0cc326bfc04aff3652144de1bf8113deae20b78119c0326d1aa`，1494262013bytes。
- 原38局summary：`caec72d353916c5411a3a6c5b1a7d0a60cf42615f01f312e93dc633697523396`。
- 128局评估代码：`eea0f15c851128fce9cbf4bb7b57922ca70b20a06bcaf31f6a2a720ebb9d022d`。
- 128局聚合代码：`ade8ca41935fde60bcad8ee5135e9f4f42c943778dcb853873069aad7bf42e28`。

这些是定位与校验线索；真正恢复前仍以Drive不可变manifest与文件SHA逐一复核。

## 7. 已知操作陷阱

- **先查 `colab sessions` 再SSH**。`colab ssh --session NAME`在名称不存在时可能自动创建CPU，不得用它探测已消失GPU。
- 同一会话一次只开一条SSH桥，避免429。一次传输超时不是VM死亡；查服务器assignment再决定。
- 活动kernel有长任务时不要再排`colab exec`；用SSH检查或已挂载CPU读取Drive，避免命令排队误判卡死。
- 本地CLI Python：`/Users/Regenin/.local/share/uv/tools/google-colab-cli/bin/python`。jupyter-kernel-client已升级1.0.2。
- 连接观察器只续同一endpoint/kernel，不运行假计算、不创建实例；SDK `google.auth.exceptions.TransportError`现已纳入可重试异常，默认生产failure-limit20。主机需在线，不能保证平台不回收。
- 不输出OAuth/代理token、凭据文件或带token的完整错误URL；授权链接只交给用户本人。
- 原始大型BC数据继续云端保存；本机磁盘约17GiB空闲，不要整库下载。先前尝试发布引擎表格缓存因SSH超时未成功，不能声称该缓存已存在/可恢复。

## 8. 基础检查

```bash
cd /Users/Regenin/Code/rl_training
.venv/bin/python -m pytest -q
.venv/bin/python -m cr_rl.cli doctor
colab sessions
```

本次交接只写状态与TODO，不改训练算法、不丢弃权重、不宣称完成整个长期目标。第3节已完成；接手先验证迁移包和Drive完成标记，再按第4节推进。
