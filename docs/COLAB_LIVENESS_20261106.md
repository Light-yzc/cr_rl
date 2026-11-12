# Colab连接维护与被回收后恢复（2026-11-06）

已启用实际连接维护和临时防休眠。不能保证Colab无限运行，也不使用模拟点击、虚假GPU任务或旧keepalive ping。

## 官方规则与区别

本次重新读取[官方FAQ](https://research.google.com/colaboratory/faq.html)与[当前CLI会话文档](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)。原文保存于 artifacts/long1000_20261106/liveness/faq.txt 与 cli.txt。

- 空闲回收、运行时最大寿命、连接凭据到期是三件事。刷新连接只能处理后者，不能突破平台配额或寿命。
- FAQ说明限制动态变化，一般最长约12小时；Pro+在算力单位充足等条件下可达24小时。未核实本账号套餐，不据此承诺24小时。
- CLI于2026-09-25移除旧keepalive daemon，当前后端依赖实际kernel活动；不沿用旧保活命令。
- 本次查询账户剩余175.13计算单位、当前消耗约2.02单位/小时，仅为查询时快照。

## 当前已启动

| 项目 | 实例/进程 | 行为 |
|---|---|---|
| L4连接观察 | ppo_long_l4_20261106 / 本机26288 | 每30秒观察、300秒刷新凭据、最长2700秒轮换连接 |
| CPU连接观察 | cr_cpu_review_20261106 / 本机25548 | 同上，严格核对endpoint/kernel身份 |
| L4实际任务监督 | kernel3538 / exec96945 | 监督训练controller17752、trainer17780与实际进度 |
| CPU实际验真监督 | kernel5570 / exec43902 | 监督独立verifier12907及最终验收 |
| 本机临时防休眠 | 进程6284 | watcher存活期间请求系统保持唤醒，最长24小时 |

连接守护仅连接、检查与续凭据，先建立替代连接再关闭旧连接；不执行虚假任务、不新建VM/kernel、不重启训练。日志不记录token。两台kernel当前均busy，来自实际任务监督。尚未实测本轮首次连接轮换或跨凭据到期，不据此宣称永久有效。

本机防休眠使用线程SetThreadExecutionState，退出会撤销；不改永久电源计划、不阻止手动睡眠/关机。本机必须在线，关机后本地连接观察停止；云端任务是否继续仍由Colab决定。

## 回收与异常恢复

每轮发布不可变完整checkpoint，再原子更新latest；独立CPU核对SHA、模型/两Adam有限值、优化器连续与合法on-policy动作。断线先检查原endpoint/任务，不能直接创建另一份训练。

若VM已回收，从最后完整已发布轮恢复并登记模拟器边界；未完对局不算完成。需要新运行时Drive授权时由用户交互完成，不自动代授权。评估ledger防止重打已完成场次。

预算为新增1000训练局、48评估局。最终完整状态和CPU验真通过后，监督器按既定grace释放运行时；不是无限占用。

## 当前验收

恢复后的88/89两轮独立检查通过，完整89号41/1000，90继续；评估4/48、对起点84号2–2。GPU/CPU监督心跳新鲜，watcher日志首次检查通过，本机防休眠状态active、两watcher存活。

本机证据目录：artifacts/long1000_20261106/liveness/；两轮检查摘要：artifacts/long1000_20261106/monitor_completion_summary.txt。
