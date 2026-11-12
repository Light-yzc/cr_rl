# Actor学习率与监控修复执行记录（2026-11-05）

用户在[异常审计](PPO_LEARNING_AUDIT_20261105.md)后授权实施。当前先落地监控语义修正与
有界Actor LR对照；Critic/奖励/门控分开处理，不把多项策略改动同时加入。
训练预算仍为300场，策略评估总预算48场，不开subagent，不公开发布私有运行细节。

## 监控修复

- `rollout_value_diagnostics()`重建独立目标：Actor优势保留原lambda，Critic return/EV使用
  checkpoint metadata中的critic_gae_lambda。显式记录两个lambda、Actor return/EV、
  Critic target MSE及bias；旧无独立lambda轨迹继续按原语义计算。
- `optimize_separate()`分别记录已提交与被KL拒绝的样本、指标，汇总只使用已提交Actor batch。
  记录名义两遍样本呈现数、实际提交/拒绝数、唯一状态覆盖、Actor/Critic梯度裁剪次数。
- Actor与Critic日志中的loss都按完整有效batch、按样本数加权汇总，不把最后一个microbatch
  当完整batch，也不让不足一个microbatch的尾部获得与满batch相同权重。
- selfplay summary保存`sample_utilization`和`rejected_policy_metrics`。后续wrapper逐轮保存
  原始轨迹重算的实际Actor/Critic价值目标诊断；这是采样时pre-fit诊断，不冒称训练后泛化。

Linux267项回归及Ruff通过。小模型分别覆盖deferred true/false、KL接受/拒绝，
新旧实现Actor/Critic参数及完整两个Adam状态逐位相同，step数、早停与初始logp检查一致。
这些修复只改变统计和日志，不改变生产损失、梯度、GAE或KL决策。

## Actor LR开发对照

固定源：真实12卡组50号完整checkpoint；冻结rollout51，补上rollout50的历史prefix。
真实模型128帧历史，精度保留FP32帧编码及其它BF16路径，batch8×8=64，两遍Actor。
Actor优势仍用原采样V_old、gamma=.9995/lambda=.98，固定旧动作/logp、reference和归一化。
不训练Critic、不生成新对局，不把开发分支更新写回生产checkpoint。

候选LR为`2e-5 / 1e-5 / 5e-6`；每项用相同的两个shuffle seed 53/307，第二组反转候选顺序。
每个候选从同一个Actor和**独立深拷贝的完整Adam状态**开始，逐项断言源Adam step计数未变。
用固定128个choice状态检查优化后的Card/Tile KL；既有更新内KL门限保持.015/.06，
旧logp验收门限保持.05。

预先声明选择条件：两个shuffle均数值有限、固定集合Card KL<=.015且Tile KL<=.06；
实际提交样本数均不少于原LR，合计至少提高20%；符合条件时选择最高的候选LR。
此选择只检验数据利用和策略漂移，不使用训练loss选模型，也不能证明胜率提高。
不存在合格候选时不自动放宽门限，不把失败结果当成功接入。

## 公平性修正与失败证据

第一次开发试验启动后发现`optimizer.load_state_dict()`可能让CPU Adam step tensor在候选之间
共享；整轮标为无效，不用于选择。该隔离进程退出时有PyTorch CUDA清理异常，生产trainer
由pidfd监督finally恢复，未重启模拟器；停放182.74秒单独记录。
日志保留在`actor_lr_repair_0001/invalid_shared_adam_attempt1/`。
修正为完整独立Adam深拷贝并断言源step计数后重跑，生产权重和原始checkpoint没有被候选覆盖。

## 接入边界与恢复原则

GPU开发试验仅在原trainer的采样段停放，最多900秒，正常/异常均自动恢复同一进程。
这种暂停不改变模拟器或RNG，但耗时需从工程对照的整轮墙钟中扣除。

候选合格后，在完整checkpoint与当次应有短测均发布、新一轮刚开始的边界交接。
CPU逐项核对Actor、Critic、warmup、counter、torch/NumPy RNG及两个Adam；唯一允许差异是
Actor optimizer的显式LR覆盖，矩与step保持。Critic1e-4/2遍、KL.015/.06、max_grad_norm=.5、
有效batch64、12套卡组、奖励、历史、对手池与300/48预算保持。

该交接需要一次已声明的模拟器重置，未完成局和未发布初段不计完整局；最终CPU回放验真器
增加有完整恢复回执认证的边界，不放宽其它连续性检查。首个新checkpoint另独立验真。
仍保留原短测与卡组回调、release guard及progress tracker，不重复启动trainer。

本轮不启用每次早停减半或本轮事务式回退调度器。先看固定较小LR的实际利用率、
固定对手短测和墙钟，再决定是否需要带上下限的跨轮自适应机制。
更小LR可能允许更多Actor更新，实际耗时也可能增加，不能承诺数据利用提高等同免费提速。

## 正式对照中间结果（尚非最终选择）

shuffle53：原2e-5提交8步/512次样本呈现，约21.1%的名义两遍；固定集合Card KL .013059。
1e-5提交38步/2432次，完整两遍，Card KL .011934，Tile KL .000734。
两项源Adam step不变检查均通过。1e-5花124.70秒，原LR31.64秒，更多有效优化有实际计算成本。
这只是同一冻结数据、第一组shuffle的开发证据，不据此宣称胜率提高或提前接入。
shuffle53的5e-6也提交38步/2432次，Card KL .002427、Tile KL .000051；
shuffle307的5e-6同样38步/2432次，Card KL .001231、Tile KL .000381。

## 最终选择与接入

第二组shuffle307：1e-5完整38步/2432次，Card KL .000509、Tile KL .000228；
2e-5也完整38步/2432次，Card KL .002616、Tile KL .001435。因此不能说原LR在所有shuffle都失败。

两组累计提交：原LR2944次，新1e-5为4864次，提升65.22%，覆盖率从名义两遍60.53%到100%。
这仅针对一个冻结rollout、两个shuffle；不等于训练长期覆盖率或胜率已经提升。
1e-5与5e-6都通过预声明条件，按“选最高合格LR”选1e-5。
源old-logp最大差.005490，所有候选有限且源Adam step不变检查通过。

正式公平对照停放原trainer698.54秒后自动恢复（无模拟器重置）；
上一轮无效开发停放182.74秒已另外保留，不能把这些工程停放当生产性能数据。

23:27 CST从完整56号接入：总训练174/300场，源SHA
`ac521e8a838e1759bdba6f24e81d62d3778abd6c59b0fdc48f1beece2a7ff828`。
完整恢复Actor779步、Critic10158步、累计Learner221820、扩展阶段144局加原30局。
CPU核对模型、RNG、warmup、counter与Adam矩/step完全保持，唯一显式差异是Actor LR2e-5→1e-5。

新增56→57恢复边界，源未完成局步数[7,25]及未发布57初段丢弃、不计完整局。
控制器110678、trainer见该阶段processes.json、最终CPU110677、首轮CPU110702；
guard49777/tracker49954保持，kernel676和Drive挂载保持。
入口`/content/crrl_actor_lr_20261105/runtime_train.py`，初始恢复验收绑定56号；
旧speed入口绑定43号，不能再裸启动。若中断需重新准备最新完整恢复回执和重置边界，不能直接重跑。

原48场短测、12卡组及300场counter停止条件不变，稳定注册仍6，无自动晋升。
当前是固定1e-5阶段，没有启用自适应调度器或同轮回退重试；后者仍需单独验证。

## 首轮生产结果

57号已完整发布：总178/300场，累计Learner225805，12套卡组，第58轮连续采样继续。
655个choice状态全部访问两遍，提交1310/1310次样本呈现、唯一覆盖655、拒绝0；
Actor22步未KL早停，Critic126步/7970次样本呈现，保持两遍。
Actor/Critic LR分别1e-5/1e-4，old-logp最大差.005713<.05，0故障。
Actor已提交指标Card KL均值.001269、Tile KL .000423，汇总与拒绝指标现在明确分开。

57号整轮含发布542.67秒（9.04分钟）：采样90.07、优化402.29、本地保存13.69、
日志1.62、Drive35.00秒。更多有效Actor优化有计算成本，不能把LR修复包装成提速。
这只是新阶段首轮观测，不保证每轮覆盖100%，也不证明胜率提高。

核对时发现旧kernel监督在早前控制器交接后仍绑定旧PID、心跳停在旧时间戳；
已在原kernel676恢复认证当前controller110678的实际训练监督，CLI exec本地session54502。
监督只检查现有进程和300/48验真结果，不启动额外trainer；异常退出仍需先检查现有任务。
当前trainer110701、最终CPU110677、首轮CPU110702，guard49777/tracker49954保持。

23:37 CST首轮独立CPU回读完成：124份manifest文件、完整模型/两个Adam有限值、合法on-policy动作、
choice-only筛选、步数连续和LR覆盖均通过。首轮CPU进程110702已正常完成。
诊断57号分别输出Actor目标EV .20465、实际Critic目标EV−.01768；它们来自采样时旧值，
不是本轮训练后的预测能力证明。Critic校准问题仍待后续独立处理。

本地执行脚本与小型证据：`artifacts/actor_lr_repair_20261105/`；
私有Drive：`runs/122m_choice_train300_20261105/actor_lr_repair_0001/`。

## 后续生产观测（2026-11-06 00:45 CST）

已发布65号、212/300场，66号继续。57–65号提交10120/11002次计划样本呈现（91.98%），
8/9轮完整两遍，62号发生KL早停，0故障。首三轮100%只描述当时，不能保证此后每轮跑满。
同期>=5费请求15/1612（0.93%）仍少，未据此调整攒费先验。

降LR后的首两次预定短测：189场/60号对stable6为1–3，212场/65号对generation10为1–3。
二者每批4场，尚无胜率改善证据；同配对换边包含一组0–2。评估已用32/48，不追加。
成功出牌均费分别2.7467/2.6535，>=5费占比分别6.67%/4.95%，满费合法WAIT均值均0秒。
完整只读快照见`artifacts/holding_prior_review_20261106/snapshot.json`，解释边界见学习审计末节。
