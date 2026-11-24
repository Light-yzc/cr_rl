# 实机日志能否用于强化学习：检查结果

## 2026-11-17 最新只读复查

`audit_nulls_rl_readiness.py`现直接调用当前`load_episode`验证记录契约，接受独立的
`terminal.json`，区分记录可用于AWBC与已验证可直接用于on-policy PPO。坏manifest、
缺事件文件和非对象JSON诊断帧会记录为拒收原因，不中断其他会话的审计。
本机5局带结果记录中，1局通过、3局因超过0.8秒的观测缺口拒收、1局因执行不明确拒收；
通过的1局有26个可用于Actor的选择标签。明确stale跳过动作仅保留上下文，不作Actor标签；
无解释的未提交出牌及重复decision ID拒收。新报告为
`artifacts/ppo_aux_review_20261117/live_recording_readiness.json`，修改和测试记录见
`PPO_AUX_REVIEW_AND_FIXES_20261117.md`。下文保留此前时间点的审计过程，不应把历史
“没有导入器/终局采集”等描述当成当前状态。

## 2026-11-17 采集修复

重新启动 `scripts/start_nulls_policy.ps1`，等 `ready` 后从开局进入对战；全程不要用手替模型操作。读到同一局终止快照时生成 `terminal.json`，并提示输入 `win` / `loss` / `draw`；回车保留未知。未能读到终止快照时标为中断，不按离开战场自动判输赢。结果由操作者按结算界面确认，尚未验证自动胜负识别。可以在之后用 `scripts/set_nulls_result.py <run> --outcome loss --evidence "Saw defeat on results screen"` 补签同一局已保存的终局。

兼容的轨迹自动写出 `training/rl_episode.npz` 和 `training/rl_summary.json`。数据保存原始提议和对应 logp、控制器选择、确认动作、实际 tick 间隔和 +1/-1/0 的终局奖励。pending 时只允许 WAIT，时序记忆继续推进。拒绝遗失开局、间隔超过16 ticks、未确认出牌、无结果或旧版贪心记录；原始诊断日志仍保留。独立 gzip 成员减少意外退出导致的尾部损坏。

`scripts/start_nulls_policy.ps1 -Matches 5` 连续准备五局，每局独立目录；每次结算标注后等待下一次 `ready`，再进入私服对局。加 `-AutoBattle` 可识别当前校准的简体中文 900×1600 主界面「对战」和结算「确定」按钮，自动点对战并在局后回大厅；无法辨认按钮时停止。加 `-NoResultPrompt` 可连续采集，结算画面保存在 `result_screen.png`，之后再根据截图标注胜负。遇到运输断开、缺终局或其他中断会停止整批。Windows 回放校验与实战采集共用锁，应在采集结束后运行回放。

按卡组轮换采集（Windows，运行前在 MuMu 打开 Nulls 主界面或卡组页）：

```powershell
$env:PYTHONPATH = 'src'
& 'F:/python/python.exe' scripts/collect_nulls_presets.py --plan 2:10,1:5 --out artifacts/nulls_live/preset_batch
```

`--plan` 按列出顺序执行。达到一套的完整记录配额后才切换。要求冻结 checkpoint，逐局核对从对局读取的卡牌 ID 和形态，保留结果截图；同一个输出目录会从 `batch.json` 恢复，已计数会话不会重复算。若在途局中断，下次用原命令重跑时保留日志、标记跳过且不计入配额；重跑前先让客户端回大厅或卡组页。权重不符或已结束但终局证据不全仍会停止。配额统计的是录制完成，不代表已标注或通过 PPO 审核；训练前仍需按截图标注并运行轨迹校验。`--verify-switches` 只切卡组和返回大厅，不开局。

下一步用 `scripts/check_nulls_ppo_batch.py <run> [<run> ...] --checkpoint <matching-full-ppo.pt>` 在 CUDA 上重放，核对冻结 Actor 指纹和 old_logp。当前已接到训练段格式和变步长 GAE，但此命令只做训练前校验，不执行优化；尚未用新版实机对局验收。以下是修复前的历史审计，旧录制仍不符合 PPO 合同。

### 本机两局与小型更新验收

`play_20261117_002235_692` 与 `play_20261117_002756_739` 都自动从主界面点对战、结算并回大厅，截图核实分别为 1:2 和 0:3；不推断对手是真人还是机器人。第一局 608 帧，最大 36 tick 间隔，3 次提交超时。第二局 146 帧，最大 13 tick 间隔，开局前 4 次提交超时。当时连续观测到已出牌计数没有增长，但不能据此确认实际拒绝原因。两局均按规则拒绝进入训练。

采集结束后，用本机 RTX 4090 对第二局做纯概率回放：146 帧原始提议的 old_logp 最大误差 4.77e-7。另用一条合成败局、可变 tick 间隔及生产优化器做 CPU 小规模更新：Actor、Critic 各提交 1 步，终局回报为 -1。测试揭示并修正训练段字段名应为 `old_log_probs`。没有用这两局实战更新权重。下一步需定位开局点牌为什么无回执，并采集至少一局完整合规样本。

随后调整开局超时处理：普通出牌超时，且提交前、超时回执和回执后下一次观测的出牌计数均为 0、手牌没变，才排除该帧的 Actor loss；保留其时序和价值上下文。不能证实未执行的情况继续拒收。按此规则，第二局重新导出通过，排除 4 帧；第一局的观测缺口仍未通过。

结论：目前能用于输入/策略回放、失败分析和有限的行为学习材料，不能直接作为在线PPO训练批次。读取不到敌方隐藏手牌或真实圣水不是阻碍；缺的是终局真值、行为策略语义和可校验的转换契约。

检查代码：`run_nulls_policy.py`、`client/diagnostics.py`、`client/action_selection.py`、`client/live_control.py`、`export_nulls_training.py`、`nulls_reader.py`、现有PPO训练转换处理。新增只读检查工具 `scripts/audit_nulls_rl_readiness.py`，输出 `artifacts/nulls_live/rl_readiness_audit_20261116.json`。

## 实测记录

全目录19份session manifest：13份标记collection_complete，0份有已验证且非空的outcome，0份记录selection_mode=policy_sample。当前代码虽已加入policy_sample，但尚无本次检查可见的新版采集会话。

| 会话 | 帧数 | 已确认出牌 | 终局结果 | 记录问题 |
|---|---:|---:|---|---|
| play_20261116_221405 | 421 | 36 | 未知 | 已结束但未记录胜负；旧贪心门控；17次法术拦截 |
| play_20261116_221114 | 139个可解码帧 | 12 | 未知 | GZIP没有结束标记、无完整stop记录；有41tick间隔 |
| cards_only_20261116_final | 0 | 0 | 未知 | time_limit，未采到训练帧 |
| play_20261115_210255 | 345 | 17（另1次超时） | 未知 | 无manifest、无逐帧decision_id和单调时间、无显式logp，错误退出 |

最新完整会话421帧都具有int[7024]观测、2307维概率、合法动作ID、decision_id、match_id和读入/推理单调时间。全部421个decision事件按ID/tick对齐，概率和最大误差2.41e-7，存储logp与对应原始模型概率对数最大差3.38e-7，无解析错误。第一帧tick1，最后tick3383；420个间隔中380个为8tick，其余6—11tick。证明存储基本完整，但不证明实际行为策略的概率正确。

## 阻碍直接PPO的项目

1. **没有已验证的胜/负/平和真实终止分类。** manifest始终初始化outcome=null/outcome_verified=false，结束时只更新stop_reason等，不填结果。require_running_clock=True使reader在clock_ended时直接抛NotInBattleError，最终快照没有写入。非战斗、返回大厅、读取失效和真实终局不能仅凭统一battle_end事件等价处理；collection_complete=true也不是结果证据。
2. **记录概率不等于旧控制器的行为概率。** 旧贪心、阈值、近满费阈值、落点选择和法术拦截改写了分布。旧数据中p(action)是原Actor概率，不能拿来冒充控制器实际选择该动作的概率。新policy_sample已记录proposed_action，但仍有门控，训练必须选定动作语义：训练原始提议时把门控/执行作为环境的一部分，并保留对应提议logp；训练门控后动作时必须计算其实际分布。不能混用两种语义。
3. **执行过程尚未形成RL转换契约。** 存在pending期间推理、提议未发送、过时跳过、传输失败、确认超时等。确认目前由played_count增长且槽位轮换判断，不是完整服务端落点确认。未来需要明确“策略提议→发送/不发送→回执→下一观测”的关系，而不是把confirmed标签当作所有训练条件。被拦/未执行动作可以作为环境反馈保留，但不能假装已经成功出牌。
4. **未生成带边界和时间语义的训练序列。** obs[t+1]多数能从下帧重建，但最后终局状态缺失。reward、terminated/truncated、bootstrap、时间间隔处理与回合边界没有导出为可验证合同。当前导出器刻意输出reward=null、return=null、ppo_eligible=false，现有模拟器PPO入口也没有这些客户端数据的适配器。
5. **旧采样策略与新训练权重不能任意混用。** 新采集需冻结行为版本，关联Actor及训练用Critic/完整checkpoint身份，再做有限次PPO更新后重新采集。compute_value=False不是必须现场改为True的阻碍：可在固定Critic上重算价值，但需要对应权重和完整观测/记忆重放语义。当前manifest只明确关联Actor hash，未形成这种完整训练快照关联。

## 本次修正的导出标签问题

旧导出器把action=WAIT且reason=policy_wait都作为BC标签，最新会话中有17帧其实是spell_guard_veto后的WAIT，不能当作模型主动等待示范。本次已根据事件或诊断帧中的拦截信息排除这些标签，同时保留全部421帧作为记忆上下文。

修正后该会话：36个确认出牌标签、110个WAIT标签、275个无标签上下文。它们仍是模型/控制器生成的行为，不是专家标签，也未变为PPO合格样本。新增回归覆盖事件拦截、帧内拦截和正常WAIT；连同相关诊断/采样测试9项通过。未重写原始采集日志。

## 下一步最小补齐顺序

1. 结果采集：终局前后观测保留、可靠结果证据与来源；读不到结果时允许明确标记“未知/截断”，不伪造负奖励。必要时先通过人工确认结果打通试验，但结果声明必须有来源，不能把断线当输。
2. RL采集模式：固定行为权重及初始记忆状态，统一原始提议动作语义和概率；完整记录控制器干预与下一观测。不能简单关闭安全执行门禁来追求“纯采样”。
3. 转换导出：连续序列、实际tick间隔、奖励、真实终止/截断及价值bootstrap规则。首轮可使用真实终局+1/-1/0，不必立刻移植全部模拟器shaping，缺失终局不强行给0。
4. 冻结版本采集一小批新完整对局，离线重放检查old_logp、边界、延迟和动作映射；再进行有限更新。旧失败片段可用于场景设计/离线研究，无法可靠补出的隐藏行为信息不追造。

本检查没有读取或操作模拟器，没有开启训练。新导出修正与此文档产生在上传ZIP完成之后，不在先前7.36GB包内；后续训练/采集代码发布时需同步这些小文件。
