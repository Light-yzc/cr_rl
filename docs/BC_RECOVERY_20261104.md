# 2026-11-04 L4 断点恢复与对照评估

## 最新状态：运行时再次消失，尚未完成对照

### 内核监督遗漏及修正

复查实际启动代码后确认：上述两次BC trial只通过`start_new_session=True`运行后台训练/备份/释放守卫；没有把仓库已有的前台kernel monitor接入本轮入口。后台进程存活与Drive心跳不等于Colab内核持续执行。官方CLI说明已取消旧keep-alive daemon，以kernel activity维护liveness；缺少前台监督是本轮明确的启动遗漏，也是空闲回收的可能因素，但没有平台终止记录，不能认定为唯一或已证实原因。

已新增`scripts/colab_resume_bc_repair.py`：必须在已准备且已挂载Drive的现有Colab内核cell内执行，启动/接管真实controller后持续检查其进程、状态和备份阶段，完成/失败/超时退出；不模拟点击、不制造虚假GPU负载、不自动创建新VM。守卫在controller首次发布status后才启动，避免此前启动竞态。`kernel_supervision.json`记录kernel/controller身份和监督预算。相关故障/完成监督测试已通过，尚未在新运行时实测长时间存活，不能宣称今后不会中断。

```sh
colab exec --session SESSION --file scripts/colab_resume_bc_repair.py --timeout 15000
```

这个命令应作为实际kernel执行运行，不能只通过SSH在后台启动该脚本后把它当成kernel活动。仍保留断点备份与恢复；Colab寿命、额度和平台回收不受此入口保证。

本次实时查询Colab服务器无原L4；Drive status最后更新时间北京时间11:47:22，仍为training/baseline，最后日志step4592。release_guard仍为armed，没有正常释放或任务完成记录，不能确定平台中断原因。修复组已完成并备份step5096（SHA记录`ca7c91d1b54bb0cc326bfc04aff3652144de1bf8113deae20b78119c0326d1aa`）；基线最新备份step4480（SHA记录`47eeebffd9dc1c68cd352c6169037fbf7ae3c3cfbf92c52f4ecfbd413e4118d3`）。本次只读取Drive记录/文件元数据，未重新下载权重重算SHA。

没有comparison.json、evaluation目录或完整验证/对战结果，PPO未开始。恢复基线需先校验4480，再新增616步到5096；不能根据较晚的4592日志直接跳过未备份更新。修复组step5088快速验证选牌38.18%、落点within2为18.18%、L1为11.11，尚无明显提升证据。

状态检查中SSH CLI在原会话缺失时自动创建过临时CPU会话，已立即停止并再次确认服务器无活动实例；没有启动新GPU。以后必须等sessions确认目标实例仍存在后再调用SSH，不能并行执行这一存在性检查与SSH。

以下为此前成功恢复及执行协议，不能当作当前仍在运行的证明。

本页覆盖此前 `bc-repair-l4` 仍在运行的状态描述。旧运行在修复组日志step4944处中断，持久化checkpoint为step4864；基线组未开始，无法根据旧日志得出修复有效的结论。

## 已核验

- 只有一个新L4：`bc-verify-l4`。Drive已实际挂载；认证传播成功后曾出现挂载错误/控制连接断开，重试及内核重连后确认MyDrive目录可读。
- 锁定的模拟器commit及16.635.3数据表已编译。训练开始前远端127项测试通过（13.30秒），本地127项通过；新增只评估模式以真实小数据验证零更新输出。
- 冻结的979个数据分片均已恢复，manifest与原协议一致。
- 原parent4096及修复组4864分别校验备份SHA；4864同时CPU载入并核对步数、数据SHA、readout_version。修复组SHA：`aa59613954d1a4ffba06c3fe07ed237e68c180f5cd23c2a965c082f9e44fdd03`，大小1,494,224,317字节。
- 本次新VM从Drive恢复验证缓存成功：压缩50,448,245字节，还原12,019,731,564字节，校验下载3.46秒、解压写盘49.25秒、总53.95秒。这是新VM恢复的实测；不代表所有云盘或整套数据的启动耗时。
- 已恢复原epoch的next_sample=73728，保留optimizer/RNG/sampler状态。丢失的4864之后日志已归档到Drive的interrupted目录，当前规范日志删除这些未保存更新后再续写，避免画出重复/倒退步数。
- 已实际推进到step4880（恢复后16次更新）：batch loss6.38036，落点loss4.95471，梯度均为有限数；L4当时利用率45%、显存9988MiB。释放守卫状态armed。它只证明续训在执行，尚不是修复组完成或模型提升的证据。

## 后台执行内容

同一原运行目录 `artifacts/121m_bc_repair_20261104`，同一Drive运行目录；保留上次源码快照。入口：

```sh
python scripts/run_bc_repair_trial.py --resume-trial --evaluate-results --evaluation-pairs 8
```

1. 修复组从4864续训232步，目标5096。
2. 原版基线从同一4096及optimizer初始化1000步，目标5096。
3. 父4096、修复5096、基线5096在相同固定较大验证集合上评估（balanced 1024行、natural 4096行），无optimizer更新。落点与选牌为条件指标，时机独立解读。
4. 两个候选分别对父4096做8组换边、各16局模拟器对战，分批4组以限制显存；相同卡组池、种子、sampling protocol，WAIT bias均为0。它是小规模实战筛查，不能证明官方客户端强度，也不能将组合处理效果归因到某一个改动。
5. 写入 `comparison.json` 和 `evaluation/summary.json`；日志、对战原始记录、权重与结果均同步Drive。不会自动晋级或启动PPO。

控制器与释放守卫均为独立后台进程，不依赖本地终端或聊天保持连接。守卫在开始时因status文件尚未生成留下过一次等待错误；status生成后已重新启动并核验armed状态。只有两组完成且请求的评估完成/备份校验通过才按成功流程释放；若失败，先保存现有权重与日志。Colab平台仍可能在此之前回收VM。

实际最新训练步数必须读取Drive `status.json`、组内日志及 `bc/latest.json`；本页不是最终完成或强度提升声明。
