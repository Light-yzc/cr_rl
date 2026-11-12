# 模型与策略优化轨迹

> **当前任务（11月5日15:00 CST）：** 用户将预算改为只扩充训练，本轮累计至少300场有效
> 完整训练对局，最终评估固定48局（每个对手16局），不自动追加。原自动优化goal保持paused，当前是新授权的有限扩容任务。
> 原18号训练已结束，作为中间状态；旧18号评估/条件384局取消，原协议完整保留并记为superseded。
> 局数按唯一episode ID计、两座位一局、跨update不重复、无开局/故障/截断不计，counter随checkpoint恢复。
> 261项Linux回归/Ruff、真实小模型2→4局恢复验证通过；新controller34811/release36910、
> CPU337263恢复已通过并退出，337269负责最终验真；前台exec79236，仍用原L4/kernel。
> 已独立核验基线30局+扩容4局=34/300，19号checkpoint目标270/计数/回放ID一致；20轮继续。
> 19轮Actor11/Critic284步、0故障，原初概率最大差.007141；完整300局及最终48局评估仍待完成。
> 只读连接观察20656/30小时。通用提交ca56c4f仅本地；稳定注册仍6。[具体计划](docs/TRAINING_GAMES_300_20261105.md)。

> **历史状态（11月5日13:47 CST）：** 用户两份新方案已实现为默认关闭的组合候选：choice-only
> Actor/仅choice归一化、完整Critic、分解KL(.015/.06)、模型内攒费先验3、Critic lambda=.995。
> 252项Linux回归/Ruff、真实模拟器小模型8轮及真实122M generation10完整恢复通过。
> 新L4 ppo-choice-l4已挂载Drive，kernel8027（f17e1064-3153-49ad-89eb-1623ae72f7a8）；
> controller8712/guard8963，CPU311263，本地连接观察26328（choice_connection_r2.jsonl）。新行为512步CUDA审计通过原阈值；
> 正式trainer10256已发布首轮11：3072 learner/1556 choice，Actor24/Critic192步、4终局0故障；
> old-logp最大差.006624。首轮29文件/完整模型及两个Adam/动作与行选择CPU核验通过；
> 11–15五轮已发布：新增17322 learner、Actor88/Critic1088步、18终局0故障，第16轮采样中。
> >=5费覆盖18.4%→14.3%→5.4%→6.1%，14轮120s后仅.90%；完整8局MC价值预测仍弱。
> 分层只读审计已保存，PBRS恒等误差4.77e-8；不能据小样本宣称容量不足或已改善强度。
> 每参考128局/总384局新seed确认已事前登记，6项私有检查通过，CPU322165等待；
> 同一kernel条件入口已排队，必须192局+CPU提名才开局，未提名跳过，不自动修改稳定6。
> 固定从10续8轮到新18，再192局对parent10/stable6/BC各64；稳定注册仍6，无本轮强度结论。
> 1dddc65已获明确授权并push指定公开分支；新通用提交894857e仅本地codex/choice-ppo-recovery。
> 私有报告/计划/traj始终未公开，新执行计划docs/CHOICE_PPO_RECOVERY_20261105.md。
>
> **历史状态（11月5日10:33 CST）：** Critic-only两轮保存完成、累计新增7,949样本，Actor136/
> Critic500步、6终局、0故障；最终24号55份备份/完整有限数值/连续性独立CPU核验通过。
> Actor对照门控仍失败，0局强度评估，不晋升，不宣称Critic改动有效；稳定模型仍6。
> 零更新重复性诊断实际48历史（计划64），普通梯度差约5.6e-9，确定性三次精确一致；
> 这不能解释103步Adam后的.0008416参数差，原门槛和失败保留，下一步先复现完整更新序列。
> L4已验真释放，当前只有drive-reader-cpu；全部完整模型/数据在Drive，本地仅小报告和GUI。
> 1dddc65仍仅本地codex分支，公开推送确认未回复，GitHub未更新。本目标仍未达成新的稳定提升。
>
> **历史状态（11月5日10:25 CST）：** Critic-only首轮Actor公平性检查失败，原阈值保留：
> 输入12数组完全一致、RNG/配置一致、Actor同为103步，但参数最大差.0008416 > 1e-5。
> 已写passed=false回执，当前试验不进入96局评估/确认，不晋升，不称单因素效果。
> 23已完整发布、3943样本/Actor103/Critic248/2终局/0故障；24继续完成原预算后保存。
> 下一步仅做真实122M固定64历史的零更新CUDA梯度重复性诊断，并独立CPU核验最终训练备份；
> 两项结束后有条件释放该L4。尚未确认差异原因，稳定模型继续6。
> 公开推送仍未获用户明确确认，1dddc65仅本地；下方“运行健康/等待公平性”均为历史。
>
> **历史状态（11月5日09:58 CST）：** lambda=.995固定96局已全部完成且CPU验真55份备份/
> 完整模型优化器有限数值/96份原生回放/比分与门槛；对control15–17、对stable6为8–8，未提名。
> 已显式跳过条件确认，稳定版本继续6；最新GUI切换为这96局显示副本。
> 新独立Critic回报试验已启动：Actor GAE=.98，Critic TD(lambda)=1；其余同源22/六卡组/两轮预算不变。
> 237项回归、Ruff、真实122M恢复及所有Adam状态逐项核验通过；CUDA原阈值通过，23轮Critic优化中。
> 运行122m_critic_lambda1_online_20261105：controller161017、wrapper161021、trainer161667、guard161026，
> CPU271290；确认CPU272862（仅合格后96局，否则skip），同一kernel9111。尚无新试验强度结果。
> 通用提交1dddc65仅本地codex分支，公开推送自动审批拒绝，已发明确仓库/提交确认，尚未获回复。
>
> **历史状态（11月5日09:28 CST）：** lambda=.995两轮训练完成并发布24，新增7,742样本，
> Actor44/Critic488步、6终局、0故障；跨轮episode steps连续，累计104,875样本。
> 固定96局开发进行中，对原lambda=.98六卡组control已15–17，不满足原55%提名门槛。
> 对parent22为9–7、对BC为9–7（各16局）；control对BC和candidate对6仍在完成中。
> 必须完成全部96局与CPU全备份/回放验真，才结案。稳定版本继续6。
> 条件新seed96局确认脚本2项测试通过，同一kernel排队；未提名时跳过，不回头重复10号验证。
>
> **历史状态（11月5日08:51 CST）：** 历史10号固定96局和CPU独立核验完成：对6为18–14，
> 对22为22–10、对BC为17–15，三项均未过独立晋级门槛，稳定版本继续6。
> 已开始预登记lambda=.995单因素试验（与六卡组lambda=.98旧control对照）：同一完整22恢复，
> 23轮已发布：3943样本、Actor20/Critic248步、2终局、0故障；CPU已确认与旧control首轮
> 全部12组数组逐元素一致。24轮3799样本已采完、优化中；强度仍待固定96局评估。
> controller139918、wrapper139922、guard139927、CPU251868，同一kernel9111。
> 通用代码1ce1c5d已push，228回归/Ruff/真实122M两步CPU恢复与lambda保存验证通过。
>
> 上一项六卡组两轮与固定96局全部完成，CPU独立核验55份备份、
> 完整有限数值模型/优化器及96份原生回放/汇总通过；对原对照12–20、对稳定6号8–8，未提名。
> 最新GUI已切换为六卡组96局。三新增卡组有PPO样本，实际learner覆盖为5/6，不宣称六套全部训练。
> 历史10号独立复测已完成，latest GUI更新为其96局；仍无新晋升证据。
>
> 上一项batch256固定96局开发与CPU全部55份备份核验完成，
> 对batch64 control为12–20、对stable6为7–9，未提名确认，稳定模型仍为generation6。
> 三→六卡组课程对照两轮已完成并发布update24，新增7752有效样本、Actor216/Critic488步，
> 6正常终局、0故障，跨轮对局连续。CPU已复核23号完整SHA与六卡组迁移；最终全量读回待完成。
> 六卡组完整结果与唯一后续分支均已验真执行；当前仍未证明新模型变强。
> 最新通用代码7785156（b073e0a默认关闭WAIT推理开关及格式修复）已push，
> 227项回归/真实tiny两轮对照通过；真实详细报告仅保存本地ignored artifacts/私有Drive。

记录起点：2026-11-05，Asia/Shanghai。目标是依据可复核的对战结果改进模型和训练策略，
保留经过验证的版本，控制遗忘与训练回退。每轮训练权重可能波动；对外使用的版本应通过
固定评估后晋升。训练损失、训练内胜率和更大的训练步数均不单独作为变强证据。

## 当前实验与验收方式

- 当前Actor122,326,278参数（122.33M），独立历史Transformer Critic120,940,545参数
  （120.94M）；BC共享模型为122,621,959参数，PPO导出Actor去除了原共享value头。
  此轮精度修复没有改变各网络参数形状。原PPO覆盖三套卡组；现独立候选配置六套，
  需以新rollout的有效learner行审计实际覆盖。六套均已完成144局机制/接口验收。
- 历史池阶段从完整generation14恢复，固定追加到22。generation18与22分别对
  generation6、准确BC5096，8个卡组配对双向换边，共64局，已在观察结果前登记。
- 先完整报告固定批次，再决定下一项改动。中间checkpoint保留为候选；稳定版本的晋升
  需要对固定参考和历史策略的独立结果，另检查卡组退化、合法性及满费等待。
- 当前评估卡组曾参与早期评估；新seed提供复测证据，不能称作从未查看的最终测试集。
- 改动优先顺序：使历史对手池实际工作 → 检查学习/遗忘曲线 → 学习率受控对照 →
  满足机制和强度条件后扩展卡组。网络规模暂保持原配置。

## 已有结果与改动依据

| 阶段 | 可复核结果 | 判断与决定 |
|---|---|---|
| generation6 对准确BC | 128局新seed复测76胜52负；按卡组配对聚类95%区间53.125%–65.625% | 支持在该固定队列上优于BC；保留为比较参考，不外推全卡组能力 |
| generation10/14 学习曲线 | 各16局对6：10为13–3，14为7–9；对BC分别9–7、10–6 | 持续训练没有表现出单调改进，不能按中间测试成绩宣称10是预先选定优胜者 |
| 历史池核查 | 原阈值250k样本；累计67,701时池仅BC，6/10/14没有参与 | 优先接入历史策略；先于更换网络结构或同时调整多项超参 |

## 变更1：显式历史对手池

实施：每局开局30%当前策略、50%冻结历史PPO、20%固定BC；初始化历史包含6、10、14。
历史内部用按卡组EMA的PFSP，保留10%均匀成分。池容量8，约20k有效样本保留一次历史候选，
经过合法完整对局检查；历史保留与最强模型晋升分别处理。

正确性：历史行为版本使用`2^30 + generation`，当前版本低于`2^30`，兼容轨迹v3非负编号。
历史座位不进入当前策略的PPO样本。载入种子校验SHA、模型形状、引擎来源、generation及
有限数值；校验不消耗恢复的torch RNG。

结果：tiny真实模拟器跨两轮检查1536有效learner样本、512冻结对手决策通过。
真实generation15轨迹独立审计4545有效learner样本、1599冻结对手决策，混入数量为0。
generation16末累计开局抽样为当前2、历史5、BC2；少量开局的实际比例会波动。

决定：保持此阶段固定配置，观察generation18/22对战结果和首个新历史候选接纳。
目前尚无历史池已改善强度或遗忘的最终结论。

## 变更2：帧编码精度修复

发现：generation13行为权重与rollout14的64个真实状态CUDA审计中，BF16帧缓存最大
动作log-prob差约0.076，超过预登记0.03；复算原始old log-prob最大差0.08439，超过0.05。
首次阶段被门槛阻止，没有梯度更新，失败报告完整保留。

实施：增加保存于模型配置的`frame_encoder_fp32`，此阶段Actor/Critic帧编码使用FP32，
其余原有混合精度路径继续使用。参数形状、层数和参数规模保持原配置，旧checkpoint默认
行为保留；历史对手沿用各自原精度配置。

结果：同一64状态和同一阈值重新审计通过：BF16路径最大差0.01549、TV0.001665，
原始old log-prob最大差0.049158。原始概率和阈值未改。最新203项Linux测试、梯度检查通过。

代价：新阶段generation15端到端927.68秒，有效吞吐4.90样本/秒；其中采样116.87、
优化744.08、序列化10.82、备份53.21秒。精度、对手组合和实际Actor优化次数均已变化，
不能将旧微基准速度直接移用于此阶段。后续优化需在当前精度下做受控测试。

## 本轮检查点记录

| generation | 有效learner样本 | Actor步数 | Critic步数 | 故障样本 | 状态 |
|---|---:|---:|---:|---:|---|
| 15 | 4545（累计72246） | 21 | 288 | 0 | Drive完整checkpoint及池SHA经CPU独立核验；完整轨迹审计通过 |
| 16 | 3072（累计75318） | 14 | 192 | 0 | 已发布Drive；跨轮对局连续，额外全权重CPU核验尚未执行 |
| 17 | 3072（累计78390） | 20 | 192 | 0 | 已发布，固定对战待执行 |
| 18 | 3072（累计81462） | 28 | 192 | 0 | 已发布，固定对战待执行 |
| 19 | 3714（累计85176） | 28 | 236 | 0 | 已发布 |
| 20 | 3311（累计88487） | 5 | 208 | 0 | 首个约20k历史候选通过完整对局检查并入池 |
| 21 | 4598（累计93085） | 49 | 288 | 0 | 已发布 |
| 22 | 4048（累计97133） | 12 | 256 | 0 | 最终全部51份清单文件经独立CPU读回SHA核验，64局固定评估进行中 |

15末对局步数`[63,811]`与16开局相同；16末为`[125,932]`。这两次update边界未重启模拟器。
15预热仍为2轮/9局，Actor/Critic优化器实际总步数186/4464，未重置。

15整轮选牌指令216次，占有效决策4.752%；1716帧有牌可出，其中87.413%选择WAIT。
指令不等同于成功出牌。更新前GAE价值解释率0.3954，不代表拟合后价值验证或模型强度。

后续里程碑：generation20已发布，并触发首个约20k新样本历史候选接纳；池现为
BC anchor与6/10/14/20。6号历史已有4场统计，10号2场、14号6场；这些是训练池统计。
第21轮正在更新，最近采样累计93,085。历史保留不表示20已获得最强模型晋升。

固定阶段现已结束：14→22新增29,432有效learner样本、Actor177步、Critic1852步，
27场正常终局、故障样本0；各update边界游戏连续，预热保持2轮/9局。
最终29次开局抽样为当前6、历史17、BC6；比例是配置概率，小批次会波动。
最终完整checkpoint SHA为
`db7d196b2981687516a305c819e1f56c3f528fbb4d40bdb859bcfa11270b2fe6`，
2,919,967,989字节；CPU独立校验全部51份manifest文件和pool通过，
见`generation22_cpu_readback.json`。历史池为BC+6/10/14/20。

实际发现过Drive跨挂载同步延迟：CPU先看到清单、随后才看到大checkpoint。
未因此重启训练或覆盖文件；重新读取同一不可变文件后校验通过。释放/交接保护增加
独立CPU回执校验，要求generation、manifest SHA、checkpoint SHA和全部文件数吻合。
该改动完整211项Linux测试通过。当前保护进程41470，原训练/评估进程未重启。

## 完整行为审计：时机波动与低费循环

全部15–22不可变rollout经SHA校验后独立审计：29,432有效learner样本，19,720历史对手
决策，历史动作进入learner mask为0。满10费、有合法牌可出仍WAIT累计298.5座位秒，
其中18轮103秒、19轮137.5秒；该总量包含当前策略自博弈双方，不能直接除以终局数当场均。
分卡组后，18轮墓园贡献73.25秒，19轮墓园135秒，明显集中而非全卡组一致。

22轮满费等待降到0，但按实际卡组分解发现：

| 卡组 | 有效决策 | 出牌指令 | 平均指令费用 | 核心牌指令 |
|---|---:|---:|---:|---|
| 野猪循环 | 2061 | 139 | 2.029 | 野猪/火球/火枪手各1；其余136次为5张低费牌 |
| 墓园控制 | 1462 | 74 | 2.824 | 墓园0、毒药1；仅4帧有5费牌可合法打出 |
| 巨人电车 | 525 | 18 | 3.056 | 巨人1、电车0 |

这支持“出牌节奏/攒费能力值得检查”，不能仅凭零满费等待认定策略健康；卡牌指令也不是
成功部署。卡组样本量和局面不同，核心牌少不自动代表策略弱，仍以固定换边胜负为主要证据。
当前不直接提高溢出惩罚或注入强制核心牌动作。新增按卡组的原生执行诊断会核对真实
成功部署、费用与等待，避免只看全局均值；后续LR对照也用同样方式审计。
证据：`history_all_rollouts_audit.json`、`training_deck_behavior_audit.json`。

## 变更3：完整回报价值诊断与显式学习率迁移

新增只读诊断，用原始`old_values`预测同策略版本、完整正常终局的Monte Carlo塑形回报。
排除缺前缀、截断/故障、非连续步数和跨行为版本；不改写原始价值或训练目标。

| rollout | 合格完整对局 | 合格座位 | MC解释率 | MC MSE |
|---|---:|---:|---:|---:|
| 15 | 3 | 5 | 0.08679 | 0.39951 |
| 16 | 2 | 2 | -0.38312 | 0.29925 |
| 17 | 2 | 2 | -0.04918 | 0.65779 |

判断：完整终局的估值证据弱于自举GAE指标，每轮仅2–3场且排除了跨update长局，存在明显
抽样限制。尚不能据此判定Critic退化或应当扩容；也不能仅凭很低的TD拟合损失判定估值成熟。
后续保留这一诊断，必要时增加冻结策略的完整对局验证，再决定价值目标或拟合预算。

为后续Actor LR对照增加显式`--resume-actor-learning-rate`，在恢复Adam后修改实际
param_groups，保留step与moments；Critic LR可独立显式迁移。默认恢复语义保留原LR。
完整tiny续训验证：Actor实际1e-5、Critic仍1e-4，分别更新1步并成功保存/备份。
候选代码210项Linux测试及Ruff通过；当前14→22固定阶段仍为原2e-5，没有中途替换。

## 后续条件试验的预登记

- 触发条件：22轮的64局开发评估若对6或BC缺乏原协议的改善证据，则开展Actor LR对照。
  若两者已有改善证据，跳过学习率变更，优先做独立确认。
- 两分支都从同一generation22完整状态开始，各增加2轮：control2e-5、candidate1e-5，
  Critic/奖励/卡组/有效batch保持相同。固定环境决策预算，报告真实learner样本数和耗时，
  不宣称分支采样轨迹或有效learner数量完全相同。
- 预登记96局：candidate对control32局，两者分别对起点22各16局、对BC各16局；
  simulator seed base8000000、sampling seed20261107，物理卡组双向换边，逐局原生审计。
- 开发集偏好只能用于选择后续确认候选，不自动写最强模型指针，不将多项小样本测试解释为
  最终统计保证。失败、无差异与回退同样保留。
- 原64局后设置15分钟有界交接窗口；只向真实存活的后续controller和release guard交接，
  没有接续任务则仍在完整验收备份后释放GPU。

实际登记：`122m_actor_lr_trial_20261105`，controller37027、release guard37031，
当前处于`waiting_source_evaluation`；原训练/评估保持原配置。这只是候选试验已准备，
尚未得到LR对照结果。两阶段各有有限墙钟预算，不做无限空转。

稳定参考登记：Drive `runs/stable_policy.json` 当前指向generation6，实际Actor SHA与
128局复测completed/summary SHA重新核对一致。此记录代表已验证参考，不表示证明全卡组
最优。后续候选不会仅凭最新step或开发集点估计自动覆盖这个记录。

## 历史池固定64局结果与下一项决定

完整64局已结束，CPU独立复算所有汇总、策略/座位/seed身份以及64份原生回放SHA通过。
汇总SHA：`2e337a4267dcdb4435bc92902c206be4b1353dd4e1f585e36c04fc1652b9d2b5`。
每个比较均为16局、8配对，不能把两对手结果合成更大样本宣称统计显著。

| 比较 | 胜–负 | 得分 | 配对95%区间 | 场均合法满费等待 |
|---|---:|---:|---:|---:|
| generation18 对6 | 4–12 | 25.0% | 6.25%–43.75% | 9.156s |
| generation18 对BC | 4–12 | 25.0% | 6.25%–43.75% | 21.625s |
| generation22 对6 | 9–7 | 56.25% | 37.5%–75% | 0s |
| generation22 对BC | 10–6 | 62.5% | 50%–81.25% | 0s |

结果：18号出现明显退步，22号在该队列恢复，但所有比较均未通过原预登记改善判据。
因此历史池接入不等于已解决稳定性问题；不替换稳定参考6，也不挑18/22之外的中间权重
回填这个固定比较。22对BC的真实成功出牌平均支付2.404费，BC为3.087费；对6时
分别2.473/2.637费。低费使用倾向在闭环执行中也可见，但这不是独立的输赢因果证明。

决定：触发已登记Actor LR受控对照。两分支仍从同一22完整状态起步，各2轮，其他配置
保持原设定。control2e-5已开始第23轮，控制器46545、trainer47220；source21/rollout22
在实际配置下的CUDA审计通过，BF16复算原old log-prob最大差0.019001。candidate1e-5
按顺序等待，避免同GPU争用。真实kernel监督与独立释放保护已完成交接。

证据：`history_curve_independent_cpu_readback.json`（含按卡组的原生费用/部署/等待统计），
`learning_curve_all_matches.jsonl`（64局GUI合档）。详细审计向W&B追加的动作被自动审批
拒绝，理由为具体私有审计内容外传授权不足；报告保留在云盘，原有常规训练日志继续工作。

## 独立确认规则预登记（开发结果出现前）

仅当开发试验有明确偏好时，提名该分支；无明确偏好则不挑选候选。
候选对稳定参考6、训练起点22和准确BC分别32局，共96局；若此前跳过LR试验，
则22对14代替自身对局。配对起点24/28/32/36，与此次开发配对0–23分开；
simulator seed base9000000、sampling seed20261108。卡组身份和部分配对曾在更早研究
出现，不能称全新最终保留集。

三个比较都必须满足配对bootstrap95%区间下界严格大于0.5，并且换边sweep符号检验
单侧p≤0.05/3，才有资格进入独立晋升审查。这个保守规则不保证真实胜率单调提高；
配对间共用卡组也限制统计假设。代码不会自动修改稳定参考，必须再经独立CPU读回、
固定报告复算与96份原生回放SHA校验。缺失、不合法和未完成对局均不当成成功验收。

确认程序及实际actor绑定/回放篡改/门槛边界测试合计214项Linux回归通过。
已登记`122m_policy_confirmation_20261105`，controller50146、release guard50154，
当前等待开发试验；独立CPU读回worker169598待命。当前两LR分支的训练参数均未因此改变。

## LR对照执行检查（结果尚未出现）

control2e-5已完整完成23/24两轮并发布：7,766有效learner样本、Actor43步、Critic488步，
7场正常终局、故障0。23轮3,943样本/Actor29步；24轮3,823样本/Actor14步。
candidate1e-5已从同一22完整检查点恢复，实际配置CUDA审计通过，复算原old log-prob
最大差0.019001；第一轮23采样也为3,943样本，当前正在Critic拟合，尚无候选对战结果。

CPU独立worker171284待两组23轮落盘后比较全部冻结输入、动作、旧概率、奖励与bootstrap；
只比较首轮，后续轨迹允许因策略更新自然分叉。另由worker174578独立核验两分支24轮
全部文件SHA、优化器实际LR、预热保留和参数有限性。两项均只写私有Drive报告，不修改训练。

control24已完成独立CPU验收：55份文件SHA与pool匹配，Actor/Critic张量均有限，
Actor实际LR2e-5、Critic1e-4，预热保持2轮/9局，23→24对局连续。
完整checkpoint SHA：`6feb23ecd97d9882597b5a5df415a109b2d0824521ad2231186dad82b2d6be93`。
这证明控制分支可恢复且实际配置正确，不代表其强度胜过22或稳定参考。
证据：`control/independent_backup_readback.json`。

## LR 两分支完成与 KL 批次核查（11月5日 03:40 CST）

两分支均已完成 generation23/24，并经独立CPU核验各55份文件、完整模型/优化器、
有限张量、预热保留和跨轮对局连续性。control2e-5：7,766新样本、Actor43步、
Critic488步、7场终局；candidate1e-5：7,733新样本、Actor57步、Critic488步、
6场终局；故障均为0。candidate完整checkpoint SHA：
`8cb9e212a06cd25e606b7c83e16604225e46991738e48595866433fb4571882e`。
当前执行预登记96局开发评估，不据少量中间比分挑选赢家。

首轮23的12组数组逐元素完全相同，包括状态、动作、合法mask、奖励、V_old、
old_log_probs、bootstrap和策略/episode身份；两分支实际Actor LR分别2e-5/1e-5。
因此首轮数据控制成立；24轮允许策略更新后自然分叉。

首轮两者均应用29个Actor优化步，于第30个完整64样本批次停止。control的
reverse reference KL为0.0229108874，candidate为0.0244712933，阈值保持0.02。
该KL是当前分布对本轮冻结behavior的KL(new||behavior)，不是对永久BC的KL。
独立CPU从generation22的NumPy RNG重放所有重置、历史池更新及Critic/Actor置换，
与23轮最终RNG完全一致，精确重建停止批次。3943个有效决策中只有452帧有非WAIT
动作可选；第30批64帧仅5帧有选择、实际出牌1帧。已应用29批的有选择帧数为1–13，
没有整批完全无选择。该结果提示有效策略决策稀少，尚不证明KL实现错误或阈值应放宽。

同数据Critic前3步日志一致、第4步出现小数值分叉，所有dropout均为0；尚未定位
具体CUDA算子，不将其当作已证实根因。GAE、advantage、V_old和value targets在
Critic拟合前已冻结且Actor/Critic参数独立，所以这次Critic拟合分叉不会改变首轮
Actor目标；后续rollout价值预测可能不同。待固定评估完成后，可在原停止批次做
不更新权重的train/eval与精度对照，先区分数值基线与实际策略漂移，再决定下一项改动。

证据保留于私有Drive本轮目录：`first_rollout_comparison.json`、
`independent_branch_backups.json`、`kl_stop_batch_reconstruction.json`。
详细记录继续仅保存在本地和私有Colab/Drive，不推送到W&B或GitHub。

## 恢复可靠性

修复：恢复命令移除首次启动专用的`--history-seed-manifest`，避免向已有完整池重复导入。
保留优化器、精度配置、历史保留时钟和预热状态。命令回归检查和真实generation15的
CPU完整恢复/前向检查均已通过：恢复后仍有6/10/14，186/4464优化器步数及2轮/9局预热
保留，历史对手无梯度；只采集2个环境步，优化器未更新，checkpoint SHA未变。
自进程峰值RSS3.62GiB，未影响GPU正式训练。证据为
`real_history15_cpu_resume_validation.json`及日志末尾`SOURCE_CHECKPOINT_UNCHANGED`。

## 下一项优化的准入条件

1. 完成本轮固定64局，并审计每个历史对手、卡组和换边结果；先区分牌组差异、时机问题、
   落点问题和策略遗忘，再选具体改动。
2. 若Actor频繁KL早停而固定评估没有改善，比较较小Actor学习率，保持初始checkpoint、
   采样预算、有效batch和对手协议可比。读取实际optimizer.param_groups确认LR已迁移；
   仅修改CLI不能覆盖resume恢复的旧学习率。
   已有依据：15/16首epoch实际应用Actor样本分别为21×64=1344、14×64=896，
   占对应新数据29.57%/29.17%；Critic使用了全部有效样本。拟比较2e-5与1e-5，
   用同一冻结rollout的局部优化实验检查KL及覆盖，再用闭环训练验证强度。
   后续完整统计15–21：全部在首epoch KL停止，Actor有效epoch当量为
   0.296/0.292/0.417/0.583/0.482/0.097/0.682，约占计划两epoch预算的
   14.8%/14.6%/20.8%/29.2%/24.1%/4.8%/34.1%。这表示Actor应用的优化覆盖不足，
   不是Critic丢弃了样本。最大初始log-prob检查误差在20轮约0.03857，持续保留诊断。
3. Critic拟合/推理先做当前精度下的batch与吞吐对照。减少拟合epoch属于训练方案变更，
   需用独立价值诊断及固定对战验证，不能仅凭耗时晋升。
4. 只有验证集结果支持保留改动，才进入后续阶段；未通过、失败及无明显差异的实验同样
   记入本文件。强度验收与文件保存、历史池入池分别记录。

## 当前证据位置

- Drive：`MyDrive/cr_rl/runs/122m_ppo_history_20261105_r2/`。
- `continuation_protocol.json`、`evaluation_protocol.json`：冻结预算与评估。
- `cuda_policy_parity.json`、`generation15_cpu_readback.json`、
  `history_rollout_000015_audit.json`：数值与恢复/轨迹独立证据。
- [W&B](https://wandb.ai/weidemingdeguge-test/cr-rl/runs/itnqtrmf)。
- 首轮完整checkpoint SHA：`3a96714e2c681ed00f447ef9dada0a1e5595a01dcdff45bfd45df228f7ab84f2`。
- 详细运行交接：[历史池阶段记录](docs/PPO_HISTORY_20261105.md)，[TODO](todo.md)。

## 数值核查与后续监督接续（11月5日 03:53 CST）

独立CPU已完成原停止批次、相同source22权重的FP32 train/eval前向对照：
reverse KL均值4.961e-9、被采样动作log-prob最大差1.431e-6，未发现明显模式偏差。
与原BF16采样old_log_probs最大差约0.003639，旧概率没有改写。此项仅说明CPU FP32
路径；CUDA/BF16仍待同批次核查，不能用CPU结果替代。

固定96局开发评估持续推进，03:52时已进入candidate_vs_source/pair20。
独立CPU worker183821已登记，完成后严格验证两分支备份、源22/BC权重、96份原生
回放SHA、固定配对及汇总/选择规则，生成`development_cpu_readback.json`并合档
`development_all_matches.jsonl`，不依据中间比分选优。原独立确认及CPU核验保持。

同一kernel已在现有确认监督之后排队一个只读CUDA审计（本地观察session6536）。
仅当固定开发评估结束且确认完成/因无候选跳过时才运行，核查source22、control23、
candidate23的FP32/BF16和train/eval差异，并将KL分解为选牌/时机与条件落点两部分；
不做backward或optimizer更新，不修改训练配置/原old概率/稳定参考，最长600秒。
结果为`exact_kl_batch_cuda_audit.json`，状态`exact_kl_cuda_status.json`。

额外登记同台L4闲置保护66759，仅处理确认因无候选跳过的分支：等96局独立CPU回执、
CUDA只读诊断均完成，再留900秒接续窗口，完整验真后释放。明确候选仍由原确认guard
负责。后续真实实验可通过本轮`next_experiment_handoff.json`交接，必须包含新的
`backup`、存活的`controller_pid`及`release_guard_pid`，沿用实际cmdline核验，不能靠
写一个空文件保留GPU。失败状态保留供检查。恢复时先查assignment/真实进程，不盲信PID。

稳定参考仍为generation6；尚无新模型通过稳定改善门槛。详细轨迹仅在本地及私有云盘。

## Actor 独立有效 batch 候选（11月5日 04:02 CST）

基于停止批次的452/3943非WAIT统计，新增了只改变Actor梯度累积的配置：
`--actor-effective-batch-size`。Critic仍使用原`batch-size × grad-accumulation`，
行为模型、GAE/V_old、reference分布和KL阈值不变；Actor的KL仍在其完整累积批次
上计算。未修改当前LR开发评估，也未将候选配置宣称为训练结果。

实现细节：Actor有效batch可独立于Critic，末尾不足一批按实际样本归一化；CLI检查
Actor/Critic各自至少64且各自microbatch整除对应有效batch；summary额外记录
`actor_effective`和`critic_effective`，便于比较吞吐和KL覆盖。旧命令不提供该选项时
行为完全保持原来的有效batch64。

隔离Linux环境实测217项测试全部通过（12.41s，代码及测试SHA已写入私有
`actor_batch_candidate_cpu_validation.json`），并验证无效Actor batch会在启动模拟器
前拒绝。该候选尚未运行GPU；只有当前固定评估结果证明需要时，才从完整checkpoint
建立新的等采样预算对照，保留稳定generation6和历史结果作回退基线。

## Actor LR 开发评估结果（11月5日 04:12 CST）

固定96局和12个评估分片已完成，原生回放审计全部通过。candidate=1e-5与
control=2e-5均从同一generation22恢复，训练预算和对手协议相同：

| 比较 | 胜–负 | 得分 | 配对bootstrap95% | sweep符号检验p |
|---|---:|---:|---:|---:|
| candidate 对 control | 20–12 | 0.625 | [0.53125, 0.75] | 0.0625 |
| candidate 对 generation22 | 9–7 | 0.5625 | [0.5, 0.6875] | 0.5 |
| control 对 generation22 | 10–6 | 0.625 | [0.5, 0.8125] | 0.25 |
| candidate 对 BC | 7–9 | 0.4375 | [0.3125, 0.5] | 1.0 |
| control 对 BC | 10–6 | 0.625 | [0.5, 0.8125] | 0.25 |

预登记选择为`inconclusive`：candidate虽然直接比较胜control，但对BC反而较差，
且没有满足独立确认门槛。因此不提名candidate、不修改stable_policy.json、也不把
control提升为稳定模型；generation6继续作为稳定参考。这个结果否定“单纯减半Actor
学习率即可稳定变强”的假设，后续应优先处理有效动作样本覆盖/批次，而不是继续盲调LR。

固定评估的12个分片、96局原生回放和汇总SHA保存在私有Drive
`122m_actor_lr_trial_20261105`；独立CPU仍在逐文件读回，完成后再运行已排队的零更新
CUDA/BF16批次诊断。确认流程已按规则安全跳过无候选分支，稳定注册保持未变。

## LR 独立验真、CUDA 结果与冻结 batch 对照（11月5日 04:30 CST）

独立CPU已重新校验两分支完整备份、源22与BC权重、96份原生回放SHA、固定配对、
汇总和预登记选择，结论仍为`inconclusive`。summary SHA：
`8cf2f1934b2f0c786dbaf7db280de436fc5e9409f750a717165b9e5297b747be`。
回读工具最初要求每分片`all_matches.jsonl`而失败：真实评估输出仅有原生match文件。
已改为调用已有`prepare_replay_archive.py`生成原生GUI封装，原始match/SHA未改；96局
完整GUI合档SHA为`c8b557e34a2a37f9d7acea2a0acb8da65b4bd0effbb3fbe3dbb500dc4a784e02`。
本地viewer入口已指向该96局归档，详细报告只保存在私有Drive及ignored artifacts。

CUDA诊断在固定评估结束、GPU空闲后完成，没有backward/optimizer更新。复现原停止
批次：control23 BF16 reverseKL=0.02291095，其中选牌/时机0.00278078、条件落点
0.02013017；candidate23为0.02447135，其中选牌/时机0.00134636、条件落点
0.02312499。因此落点分布改变占主要部分，不把仅较小的sampled-action KL当作策略
整体稳定的证明。同权重source22的CUDA train/eval基线：FP32 KL0.00283441、BF16
0.00221157（CPU FP32此前约5e-9）。这一有限数值偏差需要保留，不足以解释全部约
0.023–0.024的已学习漂移。原KL阈值仍0.02。诊断SHA：
`43c9f1515fcb265c336164fe83bff6c66142a5aa222086e2fb7a024febbd4d46`。

下一步已登记为机制对照`122m_actor_batch_replay_20261105`：同一个source22完整
checkpoint及optimizer、同一份control rollout23（3943有效learner/452有选择帧）、
同一置换seed20261109，Actor LR2e-5/microbatch8/epochs2/原KL0.02，分别比较
有效batch64和256。GAE从冻结的原奖励、V_old、bootstrap重建并锁定；Critic更新为0。
这是冻结轨迹优化实验，不是新增在线generation，不据优化覆盖或loss宣称实战更强。
两组导出权重与完整轨迹仍只在云端。当前controller76600、guard76601，原闲置guard
66759已验真并正式交接；真实kernel9111已监督该任务（观察session80820）。
代码commit4a1b715，217项Linux测试通过；当前优化64组已进入原KL停止后的全轨迹读回。
最终需读取完整覆盖、全rollout KL及裁剪率后，再决定是否值得运行新的闭环训练。

## 冻结 batch 结果与在线验证启动（11月5日 04:49 CST）

冻结机制对照完成，CPU194482独立验证报告与两份导出Actor SHA、所有张量有限性，
Critic更新0、固定targets未改。summary SHA：
`2cc1c56a9844753ce260f27c53a1dc594edfc5248e09a0ebb9906e98b23b4aa3`。

| Actor有效batch | 优化步 | 实际应用行数/epoch当量 | 全轨迹reverse KL | 全轨迹clip fraction | Actor+读回耗时 |
|---|---:|---:|---:|---:|---:|
| 64 | 17，KL早停 | 1088 / 0.27593 | 0.015069 | 0.08395 | 102.04s |
| 256 | 32，两epoch完成 | 7886 / 2.00000 | 0.007071 | 0.05808 | 433.70s |

计数修正：原机制脚本以`steps×effective_batch`记录上界，因此256分支结果中的
`applied_rows_upper_bound=8192`、由它得到的2.0776不是实际覆盖。末批每epoch只有103行，
CPU回执按每epoch3943重算为7886/2.0，本表采用真实计数，不改写原始报告/SHA。
这项结果支持更大批次减少局部KL波动，但不证明胜率；总优化耗时也增加了，不能称整轮加速。

下一项已预登记并启动`122m_actor_batch_online_20261105`：仍从同一完整generation22
恢复，两轮在线续训，Actor batch256/microbatch8、Critic batch64/4epochs、Actor LR2e-5、
Critic1e-4、奖励/GAE/三卡组/对手池/采样预算均保持原配置。先通过同一原阈值CUDA审计
（原old logp BF16最大差0.019001），再进入训练；当前trainer81060已实际采样且未重启。

训练后固定96局新seed开发评估：candidate对原batch64 control24为32局（pair40/44/48/52），
candidate对parent22、BC、stable6，以及control24对BC各16局（pair56/60）；simulator seed
base10000000、sampling20261110、pairing20261103。选择规则在出结果前已写入protocol：
direct≥.55、parent/stable≥.5、BC≥max(.5,control_BC-.0625)才提名独立确认，不自动晋升。
这些是新的开发seed/配对起点，卡组身份有重叠，不称从未见过的最终测试集。

controller80404、wrapper80409、trainer81060、release guard80405；CPU独立worker197636
将验证首轮与旧control23全部12组数组逐元素一致，并核验最终完整状态/优化器/LR/预热、
96份原生回放及汇总。原机制guard76601已核验实际进程并交接。kernel9111实际监督训练，
当前本地观察session47839；失去观察不能据此重启trainer。代码新增正式续训batch入口，
217项Linux回归通过，通用提交d607675已push；详细轨迹未push。

本轮W&B镜像最初继承不含凭据的guard环境而退出，真实训练指标仍持续保存本地Colab/Drive。
尝试仅中断监控cell并利用原kernel既有凭据恢复，但原kernel也未持有该变量，因此目前
W&B镜像未恢复；trainer始终独立连续运行，不把W&B空白当作训练未推进。后续可恢复
常规指标镜像并从progress.jsonl补传，不需要重训，不向W&B上传详细私有审计。

## 预登记校验修订与在线首轮证据（11月5日05:10 CST）

配对索引问题：10套留出卡组只有45种唯一配对，原计划中的开发pair40/44/48/52及56/60、
确认64/68/72/76超出接口范围。合成96局确认测试在开始真实评估前直接报错，不能继续
执行该名单。已停止仅外层controller80404与等待器83676，实际wrapper80409和trainer81060
保持原进程、相同训练配置连续运行，所有checkpoint和优化保留；没有重训或看到开发比分。

修订后名单：开发direct配对0/4/8/12，其余比较16/20；条件确认24/28/32/36。simulator
seed10000000/11000000、sampling20261110/20261111、各96局、既定选择及确认规则均未改。
索引和卡组身份曾用于更早研究，这里仅是本次开发/确认之间不相交及使用新seed，不能
宣称全新从未见过的最终测试集。原错误协议保存为`protocol_invalid_pair_indices.json`；
新协议记录`supersedes_protocol_sha256`和修订理由，并提前遍历所有配对调用真实接口验证。

当前online controller85126以`--adopt-training`接管80409，confirmation controller85134、
guard85138已接管释放保护；CPU197636核查在线结果，CPU202103核查条件确认/无提名跳过。
同一kernel9111监督在线与后续确认（观察session50236/61224）。确认只在源96局完成、
CPU验真且既定开发门槛提名后触发；对stable6/parent22/BC各32局，三项都需配对bootstrap
95%下界>0.5且sweep单侧p≤0.05/3才具备独立晋升审查资格，不自动改稳定注册。
合成96局完整验收、改Actor与改回放拒绝测试加既有回归，实际218项Linux测试全部通过。

另一个机制限制：独立CPU核查发现冻结64分支Adam第一/末步总数343/359，随后256分支
第一步为360，源22真实时钟为342。`optimizer.load_state_dict`共享CPU step tensor，
前分支使内存中的源optimizer时钟增加17，导致后分支不是完全相同优化器起点。
已用极小CPU例子复现共享，并确认原完整checkpoint SHA仍为
`db7d196b2981687516a305c819e1f56c3f528fbb4d40bdb859bcfa11270b2fe6`，文件没有改写。
因此撤回冻结结果作为纯batch因果证据的解释，保留原数字及SHA供审计；文件验真通过不等于
实验公平性通过。新`mechanism_validity_audit.json`明确`mechanism_fairness_passed=false`。
未来冻结脚本改为每分支deep-copy完整optimizer并检查起点342/模板不变，当前不占用GPU重跑。

独立恢复的在线试验没有读取冻结导出权重，原22 checkpoint、优化器与RNG在新进程恢复，
实际起点342。CPU已确认首轮状态/合法mask/动作/奖励/V_old/old_log_probs/策略身份/
learner&valid mask/seed/episode/bootstrap共12组数组与原control23逐元素完全相同。
真实在线第23轮：3943有效样本、Actor32步完整两epoch、Critic248步、无KL早停、0故障，
初始logp最大差0.005835，单轮1116.64s。当前第24轮推进；这些只证明优化覆盖恢复及
受控输入成立，是否变强仍等完整96局开发与有条件确认。

## Critic 完整回报读回（11月5日05:12 CST）

对已发布rollout23做只读MC诊断，只使用完整、无故障且behavior version恒定的轨迹，
将采样时保存的V_old与实际折扣塑形回报比较：2局/3个learner座位、2479个相关帧，
explained variance=0.06485、MSE=0.44509；回报标准差0.67585、预测值标准差0.06385。
这是source22 Critic在首次新采样上的预测，不能把2479帧当成2479独立对局，也不能
凭两局认定Critic容量不足。小的GAE拟合loss仍不足以证明长程价值准确；若闭环结果仍弱，
应先检查训练目标/完整回报泛化与数据覆盖，再考虑扩大网络或增加拟合epoch。
证据：本轮`critic_preupdate23_complete_returns.json`，只在私有Drive保存。

## 连接观察续接（11月5日05:15 CST）

原本地watcher25404在固定18000秒预算结束后正常退出，日志为`watch_budget_elapsed`，
不是训练失败。已确认L4仍有真实训练和kernel监督，并启动新有界watcher31836，预算
180分钟，继续观察同一assignment和kernel UUID `3f55b42f-3e02-4081-8225-07d08366e095`。
实际已读到`connection_attached`与`connection_healthy/kernel_state=busy`，没有分配新GPU、
创建新kernel或重启trainer。stdout/stderr为ignored artifacts内`history_watcher_r2.*.log`，
完整连接时间线继续追加`history_connection.jsonl`。该监督不保证绕过平台资源上限。

## batch256 在线训练完成、进入固定评估（11月5日05:24 CST）

第23轮3943有效样本、Actor32/Critic248步、2终局、1116.64s；第24轮3936有效样本、
Actor32/Critic248步、4终局、878.33s。两轮均无KL早停和故障，初始logp最大差分别
0.005835/0.002640；Critic仍原4epochs/有效batch64，Actor有效batch256/microbatch8。
实际应用Actor行数为2×(3943+3936)=15758，不能用64×256=16384高估末批覆盖。
本轮与旧control的采集预算均为6144环境决策；learner座位复用和对局演变使有效样本
总数7879与control7766略不同，不能称两轮全部轨迹完全相同，也不能把增加拟合耗时称提速。

Drive最新manifest为24，wrapper80409已完成完整checkpoint SHA核验后正常结束；
完整权重2919968181字节，SHA
`0d42ddbb491f8376d81190fd573b39d7d389e0b60f02baea4dba27122d40c2a6`。
清单55份文件的独立CPU复核安排在96局评估完成后执行，当前不提前声称已全部读回验真。
外层controller85126已经实际启动首个换边评估进程90042（candidate_vs_control/pair0）。

完整回报诊断补齐：rollout24仅看完整恒定behavior的2局/3座位，EV=-0.02580、MSE0.62415；
两轮连续合并并排除2个混合behavior座位后为4局/6座位/4944相关帧，EV=0.02754、MSE0.53437。
这些小样本值不能判断网络容量，也不作为晋升依据；记录为后续Critic目标与长程泛化检查。
证据：`critic_preupdate24_complete_returns.json`、`critic_complete_returns_23_24.json`。

后续保持预登记的完整96局开发比较；只有既定门槛提名后才进入另一组96局独立确认。
不看中间比分改变checkpoint、配对、偏置、学习率或训练时长，稳定generation6保持不变。

## 默认关闭的 WAIT 推理候选与验收安排（11月5日06:00 CST）

固定开发评估持续运行，已观察到64/96局通过原生回放审计；未读取中间比分改方案。
本轮另准备工程候选`FrozenInference(..., forced_wait_fastpath=True)`：仅当整批状态只有
WAIT合法时跳过策略前向，仍通过原完整形状Categorical采样以保留Torch RNG消耗。
所有原始帧继续进入环形历史，下次真实选择时补齐窗口内未编码帧；Critic始终计算，
不改变Actor/Critic训练、奖励、有效batch、对手分布或当前评估中的任何源码。
默认关闭，当前所有固定训练/评估仍使用原路径；非法动作logit占位值不要求逐位一致，
测试比较的是合法概率、被采样动作、log_prob、随机数状态和后续窗口语义。

隔离CPU完整回归及私有确认fixture合计223项通过（当前公开测试222项，加私有适配测试1项），
覆盖缓存已填充后改变帧编码器权重、等待间隔后的恢复、超过窗口的缺口/环形回绕、
episode重置、preview不修改历史、stale/合法mask/current-frame校验，以及共享BC策略与
独立Critic不混淆。通用代码及测试提交6454387/0e139b2已push，详细真实数据/报告未push。
验收回执为`forced_wait_candidate_cpu_validation.json`，明确CUDA/速度尚未实测。

只读CUDA基准已在独立`/content/cr-rl-forced-wait-review`准备，未修改生产根目录：
`/content/benchmark_forced_wait_real.py`，源码SHA
`1ca1216da9532a01f291eade85b4a31e6871f8e816c7e5d3805ecda619ec673e`；
inference模块SHA为`224ac6b40b6a16b1cdce41febd66491eaefd3a90506207290af327dfa3021eec`。
先验证脚本在当前评估未完成时确实拒绝运行，避免GPU争用。实际候选基准仅在开发评估、
有条件确认及其CPU回执都结束后执行，且回执距现在≤120秒才启动，以留足现有900秒
释放宽限；总进程上限720秒。队列观察session95364，仍监督同一kernel；观察超时不代表
脚本失败，不能据此重复启动。

预登记基准使用真实source22 Actor、真实rollout23前256步/两个环境的learner历史，
完整128帧窗口，BF16路径；legacy/incremental各比较原版与候选，每组3次，报告后两次
耗时中位数。要求被采样动作完全一致、CUDA RNG完全一致、selected logp差≤0.03、
最大TV≤0.01，优化器更新数0；输入轨迹固定，不是重新执行的闭环比赛，不等于整轮PPO
吞吐或模型强度。基准输出`forced_wait_cuda_benchmark.json`，不自动启用候选、不自动
更改稳定注册；若数值不通过或超时，保留失败，继续默认关闭。

## 六卡组课程实验（2026-11-05，进行中）

Actor batch256 的 96 局开发比较未达到提名门槛，因此默认仍为有效 batch64，稳定注册仍为 generation6。本轮只改变训练卡组覆盖：从 `giant_sparky_v1`、`hog_cycle_v1`、`graveyard_control_v1` 扩为追加的 `human_popular_04`、`human_popular_07`、`human_popular_18`；Actor/Critic、学习率、奖励塑形、GAE、采样预算和历史池比例保持不变。

六卡组迁移先以真实 generation22 做 CPU 恢复验证：模型和 Critic 参数逐项一致，Actor step342、Critic step6028、Critic warmup、NumPy/Torch RNG 均保留，迁移只追加 deck pool transition，未做优化器更新。此前两个启动目录因模拟器缓存缺失和残留状态被安全拒绝，均保留失败日志，没有覆盖 checkpoint；第三次启动接回固定 crforge 缓存后通过真实 122M CUDA 审计，BF16 selected-logp 最大误差0.02191496（阈值0.03），最大TV0.00457958（阈值0.01）。

当前 Drive 运行 `122m_deck_expansion_online_20261105_r5`：恢复 generation22 后已完成 update23 rollout，新增3,943 learner samples，faulted samples=0；Critic 正在第4个 epoch结束后的 Actor 优化（有效 batch64、microbatch8、两轮 Actor epoch），GPU 利用率约96%，尚未发布 update23 checkpoint，也尚未开始固定96局评估。W&B 镜像未启动是因为该 Colab 环境没有 API 凭据；训练指标、轨迹、checkpoint 和独立核验仍写入 Drive，本地不保存完整数据。只有在两轮训练、完整 checkpoint SHA/有限数值核验和96局原生回放审计均通过后，才比较 candidate 与 batch64 control、parent22、BC、stable6；不自动晋升稳定模型。

## batch256 结案与工程基准（11月5日，最终读回）

batch256 在线两轮后的固定96局开发已全部完成，CPU独立复核55份清单文件、对手池、完整有限数值模型/优化器状态、冻结actor SHA、96份原生回放以及配对/比分/策略汇总。summary SHA为 `7e85abcb5ccf16656d51ffbd28a7d07ef6ced86e4b1cfc6bc2c74817a0223fd1`。

| 比较 | 胜–负 | 得分 | 配对bootstrap95% |
|---|---:|---:|---|
| batch256 vs batch64 control24 | 12–20 | .375 | [.25,.5] |
| batch256 vs parent22 | 8–8 | .5 | [.5,.5] |
| batch256 vs BC5096 | 8–8 | .5 | [.5,.5] |
| control24 vs BC5096 | 8–8 | .5 | [.3125,.6875] |
| batch256 vs stable6 | 7–9 | .4375 | [.3125,.5] |

未提名，因此条件确认明确 skipped_no_nominee，CPU跳过回执通过；不自动额外比赛、不改稳定注册。Actor覆盖增加没有转化为本队列强度改善，batch256不设为默认。此前冻结batch机制试验存在Adam时钟共享的公平性问题，不能以其纯batch因果解释；此次独立恢复在线对照不受该共享影响。

默认关闭的 forced-WAIT 推理候选已完成真实122M/128帧/BF16 CUDA基准：legacy 9.5403→3.5844秒、incremental 6.3564→2.1643秒，768行里558行跳过整个Actor前向。采样动作差异0、CUDA RNG一致，selected-logp最大差0.0097308、TV0.00367196，原阈值通过。仅为固定轨迹的推理局部基准，不是整轮加速或强度证据；六卡组实验保持该候选关闭以只改变一个因素。96局GUI显示副本已更新，本地索引 `artifacts/replay_viewer/latest_replay.json` 指向 batch256_development_gui.jsonl；原生归档/SHA没有修改。

## 六卡组运行工程补充与最新状态（11月5日07:06 CST）

本次启动过程中原目录先因缺少模拟器缓存而在CUDA审计前失败；r2只留下空Drive目录，r3因SSH环境不含kernel runtime字段而被拒绝，r4因本地已恢复的状态目录存在而拒绝。所有失败均发生在优化前并封存，未覆盖源22。最终r5使用全新本地状态目录完成恢复和数值审计，两轮训练始终为同一trainer108680。临时hold在同一kernel9111正常启动真实release guard115464后已停止。当前训练controller108034、wrapper108037，独立CPU worker230143；原连接watcher31836继续监督。没有读取或恢复W&B凭据。

update23完整checkpoint `d0740a0d4566ffbfa0c871f2eb592b3ffc522c9a2821db1726b68cb9ac02f944` 已由CPU重新SHA校验并载入，确认确实含6套卡组和显式3→6迁移记录、warmup2轮9局。第23轮耗时1053.03秒（优化901.87秒）；第24轮新增3809有效样本，当前Actor优化，累计learner104885，尚未最终发布24或开展96局开发，不提前声称强度改善。最终CPU还会核验全状态有限数值与逐卡组实际learner覆盖。

## 六卡组两轮训练完成（11月5日07:14 CST）

update23/24分别3943/3809有效learner样本，共7752；Actor103/113步、Critic248/240步，共216/488。两轮Actor均在第2个epoch按有效batch64统计触发原KL门限保护；不是4样本微批噪声中断。终局2/4共6场，故障样本0。第24轮起始episode steps为[297,665]，与23轮末一致，训练期间没有断局重启。单轮1053.03/928.89秒（含模型序列化前），不能把局部推理基准当成本轮整体提速。

完整generation24 checkpoint共2919969909字节，SHA `f90c971ef1ee0d3d32c690c250c4a8a1588834625b754336e69405886173062b`，wrapper已完成最终发布SHA验证；独立CPU全清单及96局核验仍待执行完成。固定开发seed12000000、sampling20261114、配对0–23，96局与原既定提名规则不变；当前不读取中间比分更改训练或队列。

只读覆盖审计发现NPZ逐行访问会重复解压整数组，已停止该只读审计231825与旧等待器230143，保留修订记录；每个obs/actions/episode数组改为一次性解压。新CPU最终核验232807、即时覆盖232808，独立于L4训练/评估。通用源码不变、训练无重启。最终确认候选若通过开发门槛，再用未参与本轮开发的pair24–39、新simulator seed13000000和sampling20261115进行96局条件确认；未提名则不做额外比赛。

## 六卡组实际采样覆盖（两轮完成后的CPU直接读轨迹）

| 卡组 | 有效learner行 | 非WAIT动作 | episode-seat数 |
|---|---:|---:|---:|
| giant_sparky_v1 | 1742 | 76 | 2 |
| hog_cycle_v1 | 0 | 0 | 0 |
| graveyard_control_v1 | 2313 | 114 | 3 |
| human_popular_04 | 727 | 35 | 1 |
| human_popular_07 | 1474 | 75 | 2 |
| human_popular_18 | 1496 | 91 | 2 |

总计7752有效行；三个新增卡组都实际进入PPO，但六套配置中本轮仅五套产生learner样本，不能把“六套配置”称作“六套均训练完成”。两轮只有6场终局，uniform随机开局不足以保证全部卡组覆盖。后续可独立比较更长且有覆盖保证的课程，但不以这个中间观察改变当前已冻结96局评估、门槛或候选权重。非WAIT计数包含技能请求，不等同于原生确认成功出牌数。

## 默认关闭的工程入口完成与学习信号只读审计（11月5日07:30 CST）

`--forced-wait-fastpath` 已接入 v2训练与冻结评估CLI，默认关闭；仅Actor/冻结历史Actor可跳过只有WAIT合法时的前向，Critic保持计算。独立CPU根 `/content/cr-rl-forced-wait-cli-review` 的227项回归通过；真实tiny模拟器各两轮优化对照（显式合成warmup bypass，仅验证优化路径）动作/状态/奖励/learner mask与Torch/NumPy RNG一致，V_old/bootstrap/最终Critic逐项一致，最终Actor最大差7.08e-6。共跳过212个Actor行；不把该tiny对照称作122M整轮速度或强度验证。通用提交 `b073e0a` 已push，当前六卡组96局评估根目录仍冻结原源码与默认关闭路径。

原始两轮轨迹只读诊断显示：update23有367/3943（9.31%）个真实选择行、3576个forced-WAIT行；update24有341/3809（8.95%）个选择行、3468个forced-WAIT行。有效batch64通常仅含约6个可产生策略梯度的状态，配置batch不等于有信息的动作数量；这不撤销先前batch256未能改善胜率的实测结论，也不在本轮中途改变采样或优化。

完整恒定behavior的4局/6个learner座位/5214相关帧：采样时V_old对实际折扣塑形回报MSE0.59259、EV0.01417，目标标准差0.77475、预测标准差0.04482。将潜势加回V_old后，原始胜负加溢出回报的EV0.10130、MSE仍0.59259。PBRS折扣恒等式最大误差4.74e-8，未发现该项实现错配。2个混合behavior座位和2个不完整座位排除，帧不作为独立对局；当前不能据此认定网络容量不足。报告 `learning_signal_audit.json` 保持原SHA，进一步GAE目标诊断另存文件。

## 目标诊断与下一步选择（11月5日07:38 CST）

同一完整纯behavior轨迹上，现有lambda=.98的GAE价值目标标准差0.1530，而实际折扣完整回报标准差0.7748；该目标对已实现MC回报的MSE0.5214、描述性EV0.1327。只读重算lambda=.995后目标标准差0.3041、MSE0.3465；lambda=1在完整终局上按定义等于MC（误差<1e-9）。这些是同一数据的目标对照，不能作为更高lambda会变强的证据；更高lambda也提高方差。PBRS恒等式正常，先不扩大网络。报告 `credit_assignment_audit.json`，原审计文件不覆盖。

另重新读取原 `122m_ppo_continue_20261104/learning_curve_summary.json`：generation10对6为13–3、对BC9–7（各16局），该版本后续从未被这次独立确认队列检验。冻结10号Actor重新核验SHA `33dd65766f35366b6101d58ed35bbcc8d7a5b51cd6729c7a7a525d3119f08b25`，原开发汇总SHA `7dbe5a67981040b818a1520a1553edd69408915ad8dfabc57a068e951126a45d`。这只是历史开发迹象，不直接挑选它作为稳定版本。

后续顺序：当前六卡组固定96局与独立CPU核验完整完成后，若原门槛提名六卡组候选，则先按已登记的新seed独立确认该候选；否则优先做另一项固定历史候选10验证，暂缓新超参训练。历史10验证不新训练，冻结10对stable6/parent22/BC各32局共96局，pair24/28/32/36、simulator14000000、sampling20261116、pairing20261103。选择理由在任何这批新比赛前记录；三项均需paired95下界>.5及sweep单侧p<=.05/3才能进入独立晋升审查，不自动写稳定注册。卡组身份/配对仍有历史复用，不称全新保留集；所有失败照实保留。

## 有界确认接续已排队（同一L4，不重启当前评估）

两条确认程序各自通过合成96局复算及Actor/回放篡改拒绝测试（各2项）。已保存 `next_confirmation_preregistration.json`，在当前开发结果完整出现前固定分支规则、三位对手、96局、新seed与family-wise门槛。CPU路由器239707只等待接续登记并做独立读回，不启动比赛。原kernel9111已排队真实接续脚本（本地观察session90422），必须等当前96局完成、原controller退出及CPU全部55文件/回放验真通过后，才启动唯一选择分支；前一步失败则保留现场，不自动重训。

接续时真实controller与release guard均存活、命令行路径核验通过后才原子交接释放保护；新确认结束后仍要求独立CPU核验96份原生回放、四份冻结Actor有限数值、GUI归档与所有比分/门槛。稳定注册全程保持6，确认结果只给资格，不自动晋升。通用源码最新为b073e0a，详细审计/轨迹均未push。

## 连接监督有界接续（11月5日07:51 CST）

原本地观察器31836仍为同一endpoint/kernel，定时180分钟预算即将到期。已启动隐藏辅助进程24340，只等待该已核验观察器真实退出（最多90分钟），随后启动新的180分钟同会话观察器，沿用history_connection.jsonl；没有并行启动第二个观察器。辅助状态在ignored artifacts的connection_successor.json，若原assignment已不存在，新观察器会正常退出而不分配新VM。当前真实工作仍为CPU验真及其后固定确认，不能保证突破Colab平台自身的最长运行限制。

## 六卡组固定96局最终结果与历史10独立验证启动（11月5日07:56 CST）

CPU最终独立核验55份不可变备份、pool、完整Actor/Critic与优化器有限数值、两轮超参/预热、六卡组迁移、全部96份原生回放SHA、seed/座位身份、汇总和提名判据通过。最终结果SHA `f66377d9811ea96e90761f421a34d2758801e3cd49b3946ece182acabaee26d3`。

| 比较 | 胜–负 | 得分 | 配对bootstrap95% |
|---|---:|---:|---|
| 六卡组candidate24 vs 原三卡组control24 | 12–20 | .375 | [.25,.46875] |
| candidate24 vs parent22 | 7–9 | .4375 | [.25,.625] |
| candidate24 vs BC5096 | 9–7 | .5625 | [.5,.6875] |
| control24 vs BC5096 | 6–10 | .375 | [.125,.625] |
| candidate24 vs stable6 | 8–8 | .5 | [.5,.5] |

按原开发门槛不提名、不晋升；不同对手/卡组配对上的结果不一致，本次短预算不能证明卡组扩展无效，更不能声称已稳定变强。学习者实际覆盖5/6、没有野猪循环样本的限制保留。最新GUI索引已切换为本轮96局显示副本 `deck6_development_gui.jsonl`，模型名明确标注curriculum6/original3/parent22/stable6/BC5096；原始归档和原生回放SHA不修改。本地仅新增约104KB压缩小报告包，不存模型或数据集。

已按事先写明的顺序转入 `122m_historical10_confirmation_20261105`：controller129042、release guard129129，原guard115464已验证两实际进程后交接；kernel9111继续监督。固定历史10 Actor SHA为33dd65766f35366b6101d58ed35bbcc8d7a5b51cd6729c7a7a525d3119f08b25，simulator14000000、sampling20261116、pair24–39，对稳定6/parent22/BC各32局共96局。只复测，不新训练。CPU路由器239707在该确认全部完成后独立复算96份原生回放、权重绑定与有限数值及family-wise门槛；通过前稳定注册继续6。

## 通用代码静态检查补齐（11月5日08:02 CST）

CPU系统Python最初未安装Ruff，因此首个静态检查未执行；已仅在 `/content/ruff-candidate` 隔离安装项目声明的开发依赖。Ruff0.17.3检查近期7个通用代码/测试文件发现5处超长行，均只断行修复；使用Python AST逐项比较三份文件与修复前HEAD，语法树完全一致。相同范围最终All checks passed，227项行为回归与真实tiny两轮对照仍适用，不重复占用GPU。最终通用格式提交7785156已push；当前历史10号确认仍用原冻结源码，未热替换。回执 `forced_wait_cli_ruff.json`。

## 历史generation10独立确认结案（11月5日08:43 CST）

固定96局全部完成，CPU独立核验96份原生回放/cohort/比分/策略/family-wise门槛、4份冻结Actor有限数值及96局GUI归档通过。结果SHA `244e18ee07abd6d7d05e5f57544f236406f9e327b054c84213ea31c002325526`。

| 比较（generation10） | 胜–负 | 得分 | 配对bootstrap95% | sweep单侧p |
|---|---:|---:|---|---:|
| 对stable6 | 18–14 | .5625 | [.4375,.6875] | .3125 |
| 对parent22 | 22–10 | .6875 | [.53125,.84375] | .03515625 |
| 对BC5096 | 17–15 | .53125 | [.40625,.65625] | .5 |

三个比较均未满足预登记资格（每项需lower95>.5且p<=.05/3）。旧16局13–3不能替代这次新seed的独立证据；不晋升，不追加挑seed到显著，不将“未证明改善”写成“证明完全无用”。最新GUI索引已切换为历史10号96局显示副本，原始归档/SHA未改；本地仅下载约82KB小报告包。稳定注册继续6。

## 条件lambda单因素试验实际启动（11月5日08:43 CST）

基于先前4局完整回报/GAE只读诊断，预登记lambda .98→.995（gamma仍.9995，指数衰减尺度约12.2秒→45.5秒），不改网络规模。两臂同源generation22，六卡组配置相同；control复用刚完成且验真的六卡组24，candidate显式lambda=.995续两轮。Actor/Critic有效batch64、microbatch8、学习率2e-5/1e-4、Critic4epochs、奖励/历史池/环境预算6144均保持。新开发seed15000000、sampling20261117、pair0–23固定96局；同一原开发提名规则，满足后才考虑独立seed16000000/sampling20261118/pair24–39确认，不自动晋升。forced-WAIT工程开关继续关闭以保持算法单因素。

入口增加 `--resume-gae-lambda`，默认None保持原lambda；有限范围[0,1]、v2/重复参数校验与未来恢复保留均有测试。228项CPU回归及Ruff通过。真实122M source22恢复验证在初版测试脚本中误将保存方法checkpoint()当返回字典，保存后断言TypeError而退出；临时输出采用原子replace，没有写回源22。修正脚本读取保存后的文件，以新目录r2验证通过：Actor/Critic全部参数及Adam342/6028、预热、初始NumPy/Torch RNG保留，lambda=.995实际写入新checkpoint，2步CPU采样0优化；源22 SHA仍db7d196b2981687516a305c819e1f56c3f528fbb4d40bdb859bcfa11270b2fe6。

当前Drive `122m_lambda995_online_20261105`，controller139918、wrapper139922、trainer140571、release guard139927、CPU独立核验251868。原历史确认guard129129已核验实际进程后交接，同一kernel9111监督。原阈值CUDA审计通过，原stored BF16 logp最大差0.019001。实际命令lambda .995、gamma .9995、2env×1536步×2updates、2Actor epochs/4Critic epochs；23轮采集3943 learner样本并进入Critic第4epoch，GPU100%/约10.4GiB。必须在发布首轮后核对与旧control全部12组数组精确一致；后续轨迹可因训练差异而分叉，不能称两轮完全相同。尚未完成本轮训练/强度评估，不声称改善。

新本地观察器23416已在原31836预算结束后真正接续，同一kernel已读到connection_healthy/busy；不是第二台VM或虚假活跃。W&B凭据仍缺，真实每步指标留在Colab/Drive。1ce1c5d通用代码已push，所有详细报告仅私有保存。

## 真实Adam状态逐项补核（11月5日08:46 CST）

使用原source22完整checkpoint与CPU恢复后保存的lambda=.995临时checkpoint逐项比较：Actor305份、Critic278份Adam状态中的全部Tensor（含一阶/二阶动量与step）及原参数组字段完全一致，0次优化器更新。`optimizer_resume_exact_cpu.json` 已写入当前lambda运行目录；这补足只看max step的证据范围。此前无效冻结batch机制的跨分支CPU step共享问题没有沿用，本轮新candidate为独立进程从原始22完整状态恢复，原源SHA已复核保持不变。

## lambda首轮正式发布与公平性核验（11月5日08:51 CST）

update23：3943有效learner样本，Actor20步后按完整64样本batch的KL守护停止，Critic独立完成248步/4epochs，2正常终局、0故障，790.88秒。完整checkpoint2919969909字节，SHA `a1f6274fcaa93e19a4af4e50ecba7f116f4380556b1007d698f4972cbf3215dd`，已发布；完整CPU最终状态读回仍在两轮/96局结束后执行。

首轮CPU关键公平性核对通过：obs、masks、actions、training_rewards、old_values、old_log_probs、policy_versions、learner_mask、valid、seeds、episode_ids、bootstrap_values共12组数组，与旧六卡组lambda=.98 control23逐元素精确一致；两侧轨迹JSON分别明确lambda .995/.98。因此本轮新GAE与优化之前的真实输入相同，V_old保留；不从指标近似推断相同。后续轨迹可因新策略/优化RNG消耗而分叉，第二轮3799样本与旧control3809不同，环境决策预算仍相同。

20个Actor步骤少于旧control103，是lambda改变后的梯度/信赖域守护结果，不称为工程加速或更强；本轮仍保留同样2个Actor epoch上限、Critic4epochs和KL门槛，不为追求覆盖而放宽守护。不同lambda价值目标的原始value_loss不可直接当作价值质量改善，仍需要完整回报诊断和固定胜率验证。CPU回执 `first_rollout_comparison.json`，训练子进程140571没有重启。

## lambda两轮完成与条件确认入口（11月5日09:28 CST）

update24：3,799有效learner样本，Actor24步、Critic240步，4正常终局、0故障，775.00秒；两轮总计7,742样本、Actor44/Critic488步。24起始episode steps为[297,665]，与23结束一致；末尾[349,727]。两轮最初logp误差分别0.000139/0.003929，仍通过原有数值保护。24模型已由正常wrapper发布，最终独立CPU全清单读回待全部固定96局结束。Actor/Critic Adam预期为386/6516，不在验真前称已逐项核验。

固定96局按原顺序执行：candidate对control为15–17（32局），candidate对parent22为9–7（16局），candidate对BC为9–7（16局），后续control对BC与candidate对stable6各16局尚未完成。直接比较已低于原55%提名门槛，因此即便后续较好也不提名、不追加找seed；保留剩余比较完整记录。不能把高lambda导致更早KL停止及较少Actor步骤解释成有益加速或强度改进。

条件确认入口采用新simulator16000000、sampling20261118、pair24–39，对stable6/parent22/BC各32局，合格后才执行，无旧历史候选fallback。合成完整96局复算及Actor/回放篡改拒绝2项测试通过。首个pytest调用没有指定项目PYTHONPATH，导入错误而未执行测试；用该CPU根src/scripts路径修正后2 passed（0.42秒），没有修改训练代码或回放。

新只读CPU等待器260504已启动；同一kernel UUID3f55b42f-3e02-4081-8225-07d08366e095的接续观察session61844已排队，等待开发完成、controller退出、CPU回执SHA一致才检查提名。不合格会记录skip并返回，原release guard继续负责已核验后900秒宽限释放；不会用空hold延长GPU。局部诊断和文档仅本地ignored artifacts/私有Drive，不push、不上传W&B。用户侧一次进度消息误将parent比较写成32局，已明确纠正为16局；预登记代码和实际队列未改。

## lambda=.995固定96局结案（11月5日09:46 CST）

CPU独立核验55份不可变备份、pool、完整Actor/Critic及优化器有限数值、两轮超参/预热/跨轮连续性、96份原生回放SHA/配对/结果/策略汇总及提名判据通过。最终24号完整checkpoint2919969909字节，SHA `7e43918ccd8a1e4f619269c7c8719ac0203643c8b925d7c7ef96cf9d90ee26b6`，结果SHA `d23f94e4f0acf6ccaab795e0c9e50b0d3a21b0357fdfbcc8afeb64e61118ae90`。

| 比较 | 胜–负 | 配对bootstrap95% |
|---|---:|---|
| candidate .995 vs control .98 | 15–17 | [.375,.5625] |
| candidate vs parent22 | 9–7 | [.375,.75] |
| candidate vs BC5096 | 9–7 | [.5,.6875] |
| control vs BC5096 | 7–9 | [.3125,.5] |
| candidate vs stable6 | 8–8 | [.25,.75] |

不满足原提名门槛，条件确认和CPU确认均明确skip，不晋升稳定注册，不把未通过写成显著劣化证明。训练实际learner仍仅5/6卡组，野猪循环0行；不能称通用六卡组充分训练。最新GUI `lambda995_development_gui.jsonl`共96局，显示名区分PPO24 lambda .995/.98、parent22、stable6、BC5096，原归档SHA未改。本地仅下载约100KB压缩小报告和GUI（原GUI约1.14MB），没有下载模型或数据集。

只读实际回报诊断：4完整恒定behavior对局/5座位/4465相关帧，V_old对塑形MC的EV=-.03094、MSE=.51732、预测std=.07326/目标std=.67692；加回已知potential后的原始含overflow回报EV=.22684。PBRS折扣恒等误差5.55e-8。该小样本不能证明容量不足，也不是新旧critic强弱的独立验证；数据已因策略变化而不同。报告credit_assignment_audit.json保持单独保存。

## 微基准注册失败与安全退出

原计划在固定评估/CPU验真之后运行450秒零LR Critic微批量基准；但多个独立客户端排队执行不构成可靠依赖顺序，基准在confirmation_skip.json写入之前到达该前置断言，尚未写protocol、未创建GPU子进程即退出。已读取kernel保留异常核对并写skipped_registration_order；不伪造速度结果、不在剩余释放宽限里自动重试。后续真实试验入口显式验证依赖文件及回执，不依赖排队先后顺序。此前局部吞吐结果仍仅适用各自既定配置。

## 新单因素：只提高Critic目标lambda（11月5日09:58 CST）

理由：直接同时提高Actor/Critic lambda到.995没有通过强度开发，早期Actor KL停止明显提前；只读完整回报诊断又显示价值目标严重依赖较弱V_old。下一候选将两者解耦：Actor advantage仍使用原lambda=.98，只有Critic回归目标使用TD(lambda=1)。完整终局后缀等于折扣已实现回报；未完成rollout尾巴/时间截断仍使用采样时bootstrap，不能把所有样本都称为纯MC或给半局假终局0。

通用入口新增默认None的critic_gae_lambda与显式续训覆盖；默认完全保持旧GAE，checkpoint和rollout元数据保存独立值，恢复继续沿用。Actor advantage和Critic target都在优化前由同一采样时V_old/next_values固定；Critic学习不会回写old values、old logp或本轮advantage。237项Linux回归和Ruff通过；新增测试覆盖默认精确一致、Actor目标不变、终局切断、截断bootstrap、旧值后续变更不影响目标及非法lambda/恢复参数拒绝。

真实122M CPU恢复source22，两步环境采样、0优化：Actor/Critic参数逐项一致，全部Adam动量/step/参数组完全一致（Actor342/Critic6028），预热2轮9局及初始Torch/NumPy RNG保留，Actor lambda=.98、Critic lambda=1确实写入新checkpoint，原源SHA仍db7d196b2981687516a305c819e1f56c3f528fbb4d40bdb859bcfa11270b2fe6。

当前Drive122m_critic_lambda1_online_20261105，从相同完整22恢复、同六卡组、两轮2env×1536、有效batch64/microbatch8、LR2e-5/1e-4、2Actor/4Critic epochs、奖励与历史池不变；forced-WAIT开关继续关闭。原CUDA多路径阈值通过，stored BF16 logp差.019001。controller161017、wrapper161021、实际trainer161667、release guard161026；原139927已验证两活进程后交接。09:55时23轮已采3943 learner样本并进行Critic epoch1，GPU100%/约9.45GiB。CPU271290会先复核首轮12数组完全一致及首轮更新Actor是否保持，再做最终全量验真；尚未取得这些运行回执，不预先声称已通过。

新开发固定simulator17000000/sampling20261119、pair0–23共96局，提名规则与之前相同；只有完整验真且提名后才按simulator18000000/sampling20261120、pair24–39对stable6/parent22/BC各32局确认，仍需三项lower95>.5及p<=.05/3，不自动晋升。确认验证器2项合成96局/篡改测试通过，独立CPU272862等待；新确认kernel观察session80183已排队。卡组身份仍有历史复用，不声称全新最终测试集。

通用代码本地提交1dddc65，分支codex/independent-critic-targets。main推送先被自动审批以默认分支授权不足拒绝；按建议改独立分支后仍拒绝，以公开仓库具体代码发布未获明确确认作为理由。已用GitHub连接器核实origin为Light-yzc/cr_rl、public、用户有admin/push权限，且origin/main仍1ce1c5d；核实后再次申请仍被拒绝。没有绕过工具发布，已向用户请求明确确认公开仓库/分支/六个通用文件提交。代码已全部可审查，详细traj/plan和实验报告始终未暂存；当前GitHub尚未更新。


## 真实恢复补核（11月5日10:00 CST）

在实际CPU保存的完整122M checkpoint上重新运行训练CLI，刻意省略--critic-gae-lambda：CLI仍恢复Actor lambda=.98、Critic lambda=1，未静默退回旧目标。此项只解析真实保存状态，0次采样/优化；回执critic_lambda_saved_default_recovery.json保存在当前Drive运行。新条件确认预登记也已在任何本轮对战结果前写入，seed18000000/sampling20261120、96局与原门槛固定。


## Critic-only首轮公平性失败（11月5日10:25 CST）

23轮完成：3943 learner样本，Actor103/Critic248步、2终局、0故障，1054.49秒（优化901.90秒），完整checkpoint已发布；24轮起始episode steps与23末[297,665]一致。长SSH观测连接断开后已先检查sessions、实际PID161667和进度，原trainer仍运行，未因观测断开重启。

CPU271290在更新Actor参数检查失败后真实退出；原始堆栈保留。只读复算12组首轮数组全部精确一致；但302个Actor Tensor不同，最大差0.0008416175842285156（row embedding），超过预登记1e-5。Actor两侧均103步，policy_loss分别.00812772/.00843455。随后完整上下文复核NumPy/Torch/CUDA RNG、Actor/Critic配置、reward、role mixture/draws、next_episode、unfinished steps、gamma、Actor lambda和warmup全部一致。参数差异原因仍未确定，不能擅自称为可忽略舍入或Critic改善。

已明确保存first_rollout_comparison.json passed=false，附12数组true、实际差值/原阈值及禁止评估理由；first_actor_discrepancy.json和first_actor_context_comparison.json分别保存量化与上下文结果。未放宽门槛、不伪造通过回执、不运行该96局来转移检查失败。现有有界trainer继续完成第二轮落盘，controller之后会按原门控失败；条件确认也不得启动。

计划在原trainer退出后，以同一固定122M Actor和64个真实历史/动作，分别普通与torch deterministic模式重复三次完全相同的加权log-prob backward，不执行optimizer step，验证权重和采样RNG保持。诊断仅检验GPU梯度重复性，不能替代完整103步Adam轨迹复现，也不能自动撤销原公平性失败。其后只有CPU核验最终24号全部备份/pool/有限数值/跨轮连续性通过才释放L4；备份完整性与强度/公平性验收分开报告。


## Critic-only训练备份结案与GPU释放（11月5日10:33 CST）

两轮最终新增3943+4006=7949 learner样本，Actor103+33=136步、Critic248+252=500步，6正常终局、0故障；累计105082。24起始episode steps[297,665]与23末完全连续，末尾[479,530]。第24轮Actor在第1epoch33步后由原完整64样本KL门槛停止（该拒绝batch reference KL=.075998，未再step），Critic仍独立完成4epochs。新目标value_loss不能与旧lambda直接作价值质量排名。

独立CPU280165核验55份最终清单文件、pool、完整模型及两个优化器有限数值、Adam总步342+136/6028+500、超参/预热/跨轮连续性通过；另在真实首轮记录上重新计算，旧compute_gae与新helper的Actor advantage逐元素完全一致。最终24号checkpoint SHA `1ab647d49456c1d8437193ffd42a23d12c72b1622acf74cd3103fb4a634013fb`。回执cpu_training_integrity.json明确training_backup_verified=true、actor_equality_gate_passed=false、strength_evaluation_games=0，不能当作公平性或强度通过。controller按原门控真实失败，原确认和CPU确认均因开发失败停止，无额外比赛。

后续零更新探针122m_actor_gradient_repeatability_20261105完成，Actor122326278参数、128历史、FP32 frame/BF16其余路径，三次普通/三次deterministic backward，权重与采样RNG均保持。计划64窗口但默认分组选择器仅取得48，实际report.histories=48；原protocol保留64计划，不修改成事前吻合，测量只能按48窗口的描述性诊断解释。普通模式第二/三次最大梯度差4.66e-9/5.59e-9，L2相对约5.2e-9；确定性模式三次所有梯度精确一致。耗时普通稳态约1.74秒、确定性约1.99秒，仅单个48窗口梯度探针，不能当作整轮速度或103步Adam重现。单次梯度微差的存在不足以认定它解释了.0008416参数差，未撤销首轮失败。

同一kernel9111在探针结束、独立CPU完整训练备份验证及最终SHA复核后调用正常runtime release；release_verified.json已落Drive。随后colab sessions明确清理原L4，当前仅drive-reader-cpu，连接观察器会因原assignment消失退出而不新分配VM。本地仅下载约6KB失败/诊断报告包并安全解包，完整模型和数据未下载。GUI latest继续指向上一项lambda=.995的真实96局，未捏造本失败试验对战回放。

接续优先级：先在固定输入/独立Adam状态上重放完整103步优化，定位首次分叉，比较普通与确定性路径，并核查历史窗口和batch顺序；在原因明确且重新预登记前，不把现有失败试验当作Critic-only的因果结论或晋升候选。当前没有新稳定版本，目标继续有效。通用代码公开发布仍需等待已发出的明确确认，不绕过自动审批。


## 用户新方案接续、发布与CPU重建（11月5日12:42 CST）

上一goal turn归类为进展：完成了受控训练、原生回放验真和失败诊断，当前继续保留完整目标而非把通过测试视为完成。用户明确授权1dddc65的六份通用文件发布到Light-yzc/cr_rl的codex/independent-critic-targets；已成功push并用ls-remote核对远端精确SHA1dddc65f27d64f3d084f33c218fee20027a20d2f，私有traj/plan/报告未暂存。

在原Drive轨迹上复原首轮开局/重置/对手池EMA更新和每次NumPy抽样，核对3,943 learner行、4次reset、所有实际seed/deck/version与末尾RNG/pool一致；复原4遍Critic、2遍Actor确切排列和原始规范化历史输入，prepared bytes=66,872,624匹配旧日志。该输入重建不是原始GPU内存捕获，不能自行解释参数差。进一步重放入口对独立Adam做deepcopy避免CPU step共享；首次小夹具前16帧全WAIT、零梯度，被KL测试正确拒绝。换为含4个choice、3个非WAIT标签的16帧后，两模式各两次完整103步，全部模型/Adam逐项一致，KL测试在5步真实早停，源文件不变。此为134,613参数CPU夹具，不代替真实122M CUDA103步复现；按用户新优先级，暂不为旧参数相等检查再消耗GPU。

用户随后提供两份具体修复方案，执行范围改为：Actor只取有选择状态、Critic保留全部有效状态、card/tile KL独立守护、与推理一致的低费蓄水探索，从generation10完整状态连续8轮。采纳其改动但纠正证据表述：10号未被证明最强；Critic-only此前已完成两轮；不可把.0008416差直接定性为已证明正常；32局配对符号检验不等价于固定23胜二项门槛。新阶段将跨分支参数相等保留为诊断，不作为强度验收硬门槛，旧失败记录原样保留。

额外直接轨迹审计：3,943 learner/367 choice/3,576 forced WAIT，其中粗略866 forced帧仍有费用可负担牌；不能说91%全因没费。新指标扣除reserved费用、排除pending手牌，分别记录choice、forced中可负担未排队牌、可用圣水>=5、请求成本/核心牌请求；请求不冒充原生成功出牌。crforge Schema已核对角色kind1、敌side1、己后场y0。

## 组合候选实现与验证

Actor只在eligible且legal_count>1的行优化和归一化，保留有选择WAIT/技能；Critic全部eligible，GAE沿全部时间计算；零choice安全跳过Actor。KL按链式法则拆card与新卡槽概率加权conditional tile，整个有效batch统计，门限.015/.06，超限待梯度清零且Critic独立完成；不声称维数本身必然产生大KL。新old-logp初始保护.05在step前检查，数值/合法/回放/备份保护保留。

固定先验在ModelConfig和Actor.forward内实现，而非采样器偷改概率：手中>=4费未排队不可负担牌、附近可见敌方角色无紧急威胁、未近满费时，按未解锁高费牌比例加WAIT logit最多3；训练/rollout/reference/旧与增量缓存/导出一致，历史模型使用其原配置，不保证某个高费覆盖比例。默认strength0保持旧行为。Actor lambda.98、Critic .995，两目标在优化前由V_old固定；尾段正确bootstrap。

首个定向19项出现2项测试schema不完整失败；修正夹具为完整费用/HP字段后又检出测试错误的位置参数，以及实际forward中原始实体引用不应覆盖encoded实体的缺陷，已修复。完整248项通过；加完整累积KL、旧概率拒绝和新行为覆盖日志后251项/Ruff通过。格式只调整import/断行，已按非import AST与import集合核对（去掉一个unused torch）；未因格式失败重跑GPU。源10重新分支还需避免旧14快照重号，新增独立snapshot ID与冻结轨迹version，旧ID空闲时沿用，历史保留/恢复测试通过；最终252项Linux回归通过，L4新增相关18项通过。

真实模拟器小模型8轮共42 Actor/146 Critic步、8终局、0故障，跨轮连续，6轮Actor激活，全部Actor样本数等于choice数，Critic覆盖全部eligible，old-logp误差<=.001、合法性和实际CLI恢复配置通过；模型宽16/32/context16，报告明确非122M强度或速度证明。

真实generation10完整恢复首轮脚本找错6号manifest路径而未进入模型优化，失败保留；改用stable_policy.json实际路径/SHA，在新r2目录完成122M恢复和2步CPU采样/0优化：模型和Critic逐项一致，两个Adam全部moments/step/groups一致112/2892，warmup2轮9局、初始RNG保留；Actor lambda.98、Critic.995、prior3、FP32frame、历史6/10/14/22与30/50/20均保存。源SHA1190dcb619f7d4c16f1c130e2e47c6dd954c41f3e6f36c951cbba58fa376abd6未改。

## 新L4正式入口与当前状态

新会话ppo-choice-l4，endpoint gpu-l4-s-ass1a1-1sn7lwnm1n48s，用户已授权Drive。首次非TTY挂载空stdin提前提交失败，交互TTY重新请求并在用户完成后成功挂载；后续新挂载应使用交互TTY。固定引擎构建和251项L4回归通过，身份修复相关18项补测通过。初次自建监督器因CLI Drive自动挂载不登记kernel而拒绝；标准exec为正式kernel登记后首个工作目录请求连接断开，已确认试验目录/进程均不存在才复用该已登记kernel启动，未重开VM或重复trainer。当前kernel UUIDf17e1064-3153-49ad-89eb-1623ae72f7a8、PID8027。

Drive122m_choice_gen10_8updates_20261105：controller8712、guard8963、独立CPU311263，本地连接watcher24624（先前9640因登记竞态已真实退出后才替换）。正式controller已校验冻结parent10/stable6/BC、登记8轮和192局协议，完整10恢复后正在新行为512步GPU采样与数值验收。此验收使用prior3/FP32frame实际模型自身的old概率，不拿行为已改变的历史old概率当新PPO比率。采样验收0优化，不计入正式环境预算；通过后从同一未修改本地10状态真正连续8轮至18，2env*1536*8，目标/对手/模型固定；192局parent10/stable6/BC各64，sim19000000/sampling20261121/pair0-31，开发只提名不自动晋升。

controller和发布/释放入口均有真实子进程/有限时间和独立CPU回执约束；当前还没有正式梯度更新或强度结果，稳定版本仍6。通用实现本地提交894857e（14文件），新分支codex/choice-ppo-recovery未公开，之前授权的1dddc65分支已发布。新计划docs/CHOICE_PPO_RECOVERY_20261105.md与本轨迹均私有保存。

## 正式CUDA验收与第11轮启动（11月5日12:50 CST）

新行为512步采样验收完成，515 choice行、32非WAIT请求；验收未执行优化，正式恢复checkpoint与
pool SHA保持。CUDA原阈值通过：BF16 selected-logp差约.0139813（限.03）、TV约.00147364
（限.01）、储存old-logp差.01390314（限.05），FP32差约3.6e-5（限.0005）；没有修改阈值。

实际trainer10256随后从相同完整generation10状态进入正式固定8轮。第11轮1536步采样约
158.91秒，3072 learner行，4正常终局、0故障。12:50时Critic实际完成128步（Adam2892→3020），
进入第3epoch，有限数值；此时Actor尚未开始、首个新checkpoint尚未发布。
独立CPU311263继续等待最终18号全量核验；另启动首轮CPU只读核验，发布后检查完整SHA/
模型及两个Adam有限数值、真实choice行/全部Critic行和采样动作合法性。它不会新增训练或评估。

恢复generation10时继承的metrics.jsonl还含原源11–14历史记录；本轮进度只按新增
actor_choice_only=true记录辨认，最终独立核验仅取末尾11–18八条，不把旧训练数重复累计。
完整模型和数据只在Colab/Drive，文档私有保存；稳定模型仍6，无本轮强度结论。

## 首轮正式发布与独立CPU验真（11月5日13:01 CST）

第11轮完成：3072 learner行、1556 choice行，Actor24步、Critic192步，4正常终局、0故障。
Actor第1epoch尾部20样本的完整剩余batch触发card KL=.0169701>.015，tile=.00077675；
待梯度丢弃，未执行该尾batch或第二epoch。初始old-logp最大差.00662374496<.05。
本轮1556个Actor选中状态均有真实选择；全部3072个eligible状态仍用于Critic，不将
选中行数冒充实际已执行的optimizer batch样本。请求不是已成功落牌，仍等原生评估验证。

566/3072帧可用圣水>=5（18.4%），110次出牌请求，平均请求费用3.12727，其中7次>=5费；
169个forced-WAIT状态仍有可负担未排队卡，不能把全部forced-WAIT归因于费用不足。
首轮角色抽样为historical3/anchor3、没有current，属于6次实际开局的抽样结果，
没有改30/50/20配置。历史池5条（BC与6/10/14/22）实际载入；不能用本轮单独证明全对手泛化。

含完整发布699.2356秒：采样158.958、优化470.660、序列化12.157、Drive发布56.349；
3,072有效learner样本吞吐4.393/s。此次第一轮无双座当前策略额外样本，且行为/训练选择
已变化，不拿它直接宣称与旧8.35/s统计构成受控速度对比。峰值allocated约9.126GiB。

首个新full checkpoint SHA为78d00f3bf8ab054ba6f58a555081a63426dd1df1048df9c938ad83415b85af7a。
独立CPU只读核验29份文件、合计约5.51GiB，完整SHA/pool、模型与两个Adam有限数值、
Adam112+24/2892+192、全部动作合法、policy version10、choice/critic行选择和新配置通过。
回执cpu_first_update_readback.json passed=true已从Drive独立读取；不是只看训练进程存活。
只读核验的本地SSH观察连接没有正常返回输出，确认原子回执后关闭本地observer5792，
未关闭trainer10256或长期CPU311263，未把连接中断解释成训练失败而重启。

第12轮已连续运行，并完成采样进入Critic优化；继续固定到新18后192局评估，稳定版本仍6。
实际基础源码包、最终身份补丁和控制脚本共约407KiB已校验备份到本轮execution_source，
私有manifest关联本地提交894857e。文档同步包含todo入口，避免旧LR/历史池状态误导接手。

## 连续三轮进度与条件确认登记（11月5日13:22 CST）

本轮11、12、13均已完整发布，trainer10256/controller8712/guard8963仍为同一组真实进程。
新增learner3072+3072+3645=9789，Actor24+4+27=55步，Critic192+192+228=612步，
终局4+4+3=11、0故障。12起始[328,82]=11末，13起始[348,144]=12末；当前14采样中，
其起始继续13末[410,573]。初始old-logp最大分别.006624/.002340/.014593，均<.05。

行为并未持续改善：可用圣水>=5费帧566/3072=18.4%、439/3072=14.3%、197/3645=5.4%；
平均请求费用3.1273/2.9606/2.5326，>=5费请求7/9/4。第13轮包含current自博弈并变化了
对手/卡组/局面分布，不能单凭这些相关观测认定先验因果失效或成功；保留固定8轮及
最终192局协议，不根据中间行为挑checkpoint或改bias。三轮均正常只证明工程健康，
训练内胜负和低value_loss不替代强度评估。第12/13含发布699.65/799.04秒，吞吐4.39/4.56行每秒。

在任何本轮开发对局结果前登记条件确认目录122m_choice_confirmation384_20261105：
最终18固定对原10/stable6/BC各128局，总384；每参考32组卡组、双向换边、2组seed重复。
simulator20000000/21000000、sampling20261122/20261123、pair0–31、pairing20261103固定。
重复seed仍聚类为32组卡组，不当成64组独立卡组；三比较均需deck-pair cluster lower95>.5
且按重复后每组卡组均值的单侧符号检验p<=.05/3，完整原生回放及CPU独立验真后才可审查。
这是事前明确的重复seed确认方法，不把32局固定胜数误作符号检验门槛；卡组身份历史复用限制保留。

私有choice_confirmation.py负责登记/完整128局聚类/三参考门槛/回放SHA与权重绑定，
run_choice_confirmation.py严格等待192局和CPU提名；未提名明确skip，失败保留现场。
readback_choice_confirmation_cpu.py独立验证384局或0局条件跳过；当前CPU322165真实存活等待。
start_choice_confirmation_kernel.py已通过同一已挂载kernel排队（本地exec2206），
必须原controller真实结束且CPU完整验真后才启动确认；需要交接时用两真实controller/guard
与live_handoff校验，不写假占用。当前仍registered_no_games，未启动确认GPU对局或新增VM。

6项Linux检查通过：真实卡组范围与128预算、三比较纠正阈值、完整128局只计32聚类、
回放篡改/漏块拒绝、无提名不启动、失败CPU回执阻止及独立skip复核；本地AST语法检查通过。
当前CPU无Ruff模块，未冒称本批私有脚本通过Ruff；此前252项正式训练回归与Ruff结果保持。
协议与源码/测试回执均私有备份；实际训练未热替换，本地完整数据/模型仍未下载。

## 四轮训练及只读探索/完整回报审计（11月5日13:40 CST）

第14轮完整发布：3986 learner/1466 choice，Actor22/Critic252步，3正常终局0故障，
old-logp最大.0080123。11–14累计13775新learner，Actor77/Critic864步、14终局0故障；
所有update边界连续，14起始[410,573]、结束[1234,252]。15轮已采样并进入Critic第2epoch，
trainer10256/controller8712/guard8963实际存活，未因连接偶发ReadTimeout重新启动。
14轮含发布805.14秒（采样139.38/优化618.21/序列化12.67/备份32.46），4.951样本每秒。

独立CPU对immutable rollout及model-config provenance核SHA，直接调用实际保存配置的
ElixirHoldingPrior forward，按卡组/对手/时段/是否choice拆分11–14轨迹；未读取/修改模型
参数或优化器，未新增对局。报告exploration_stratified_11_14.json及脚本在本轮Drive保存。
第13轮hog占2109/3645（11轮仅82/3072），但同卡组电车>=5费覆盖27.4%→14.8%→9.8%，
墓园16.0%→18.4%→7.6%，所以不能只用卡组比例解释总体回落；样本/对手仍不同，非因果对照。

先验并非未执行：11–13有choice状态的启用率36.7%/38.3%/44.5%，其高费覆盖反而下降。
该启用率也会受高费牌重新锁定影响，不应解释为更强先验导致劣化。13轮>=120秒时仅1.58%
帧可用圣水>=5，78.5%高费牌未解锁状态被近半场敌方角色条件关闭先验；14轮全局启用率
23.3%，锁定状态74.5%被威胁关闭，>=120秒为78.6%。这支持继续检查紧急威胁门控是否过宽，
不证明取消防守门控有利。任何下一改动须等固定192局后另行预登记，不热改当前bias或阈值。

14轮>=5费243/3986=6.10%，其中>=120秒22/2452=.90%；开局30秒142/480=29.6%。
268次出牌请求平均2.3433费，6次>=5费；hog2558帧有218请求、均价2.133，仍有低费循环风险。
11–14满费且合法出牌时仍WAIT总计.25/.75/0/0 seat秒，只是训练请求轨迹，不代替原生成功
出牌或固定评估的满费等待审计。不能把“没有大段挂机”当作“已经学会攒费或已经变强”。

另用采样时V_old与完整折扣实际回报做价值诊断，核验rollout和目标provenance SHA：
8场完整恒定learner行为对局（各轮4/2/1/1），排除7个混合行为座位、3个不完整座位。
PBRS折扣恒等最大差4.7685e-8，未发现奖励潜势计算错误。前四轮shaped MC MSE分别
.61294/.34289/.70309/1.37807；11轮预测std=.1045、目标std=.6747，EV=-.1801。
12轮两场都是赢、13/14各一场输，目标std小，不能以EV=-1.15/-5.83/-.64给跨轮强弱排序。
14轮该单局预测均值+.418而实际回报均值-.737，提示校准仍弱，但实际回报含随机性且样本
极少，既不能声称Critic已成熟，也不能据此断言Transformer容量不足或某lambda因果失效。

报告credit_assignment_11_14.json与补充均值/终局构成的credit_assignment_11_14_v2.json
均保存，原脚本和v2分别归档，未改原失败/诊断报告。下一轮分析应在最终8轮/固定192局后
综合判断价值目标与先验门控，而非继续仅按训练loss或扩大网络选方案。稳定注册继续6。

## 用户最新接续：完成固定评估再讨论威胁门控（11月5日）

用户要求先保持当前配置跑完15–18和完整192局，然后给出实际换边成绩与出牌审计，不中途
打断或热改。已接受此顺序；用户提出的`y<=.45`敌方角色总HP>.12（约1200原始HP）紧急
威胁门控仅作为192局结束且后半程仍刷低费牌时的候选，当前未改holding_prior.py，未开新试验。
若后续采用，不能当作保证有效的一行修复，需保存行为配置并独立验证及预登记。

同时纠正用户引用中的两点：项目EV实际是1-Var(G-V)/Var(G)，而非1-MSE/Var(G)；第11轮
V_old来自该轮优化之前，因此std=.1045不能作为lambda=.995已经拉大价值幅度的因果证据。
同理choice占比提升属于组合改动后观察，不单独证明过滤或KL改动造成占比上升；实际样本
分离、KL组件约束和Critic独立拟合已由代码/真实运行核验，强度仍待完整固定评估。

## 第15轮完整发布（11月5日13:47 CST）

15轮新增3547 learner/1135 choice，Actor11/Critic224步，4正常终局0故障，old-logp最大
.0124425<.05。11–15累计17322新learner、Actor88/Critic1088步、18终局0故障，跨轮连续。
15起始[1234,252]与14末一致，末[760,136]；16已开始并观测到正常终局，未重启trainer。
15轮可用圣水>=5为211/3547=5.95%，164次请求均费2.7683、2次>=5费，未当作强度结论。
用户最新顺序继续保持：固定到18、完整192局，然后报告；新威胁门控只记候选，未改实现。

13:51连接交接：为覆盖训练及条件确认，将旧240分钟观察器24624替换为新480分钟只读
观察器26328（choice_connection_r2.jsonl）。先验证新进程已连接同一kernel且实际busy，
再核对旧进程命令行并停止旧观察器；没有停止trainer或重启kernel/环境。此观察器不执行
虚拟保活任务，不分配新VM；原assignment释放时会退出。源holding_prior.py哈希仍为
edf8f90da31166df45c9bd90af66fd35474aa225e0ec43e093a83f239708cd86，未实施用户后续门控候选。

## 用户只扩充训练：至少300场完整对局（11月5日14:35 CST）

用户询问扩到几百场后明确选择“只扩充训练对局”。已将本轮从generation10恢复以来目标
设为至少300场有效完整训练对局；原11–18的有效完整局数计入，不包含之前BC/消融/评估局。
评估保持192局，移到达到目标后的最终checkpoint；不再给中间18另跑192或旧条件384局。
自动优化goal已由用户设为paused并实读确认，本次仅执行该新授权的有限预算扩容。

实现TrainingGameCounter及--min-training-games，记录全局episode ID/已见开局/完整终局集合：
两座位同一局计一次，跨update/恢复不重复，故障、截断、不完整终局片段和非learner局不计。
每轮优化和完整checkpoint后才停，保存counter origin和目标，恢复省略参数仍按保存目标继续。
新阶段以完整18为origin，目标=300减原11–18独立核算的完整局数；max160追加updates作为安全上限。
真实对局可跨权重更新边界，计数不冒充300场整局同一冻结策略的MC审计样本。

261项Linux回归及Ruff通过；同一L4隔离目录9项通过。真实CPU小模型目标2局在update2停止，
实际CLI省略目标恢复且已满足时文件SHA不变；再显式扩展到3局，update4累计4局停止（允许
末轮越过下限），逐份rollout重算episode ID与checkpoint集合精确一致，0故障。该fixture宽
16/32/context16，不能当122M强度或吞吐证明。CPU观察SSH仍活跃时一次下载被429拒绝，
已等待原桥完成后重试成功，未重启任何训练。验证回执存training_games_tiny_validation.json。

正式新代码目录/content/cr-rl-choice-train300与原GPU目录隔离，复用已固定模拟器和系统依赖。
保持原2env×1536、Actor/Critic8×8、2/4epochs、2e-5/1e-4、lambda .98/.995、gamma .9995、
choice-only/split KL/FP32/prior3/三卡组/历史池配置，只增加游戏计数及停止预算。当前holding
prior未改HP/位置规则。完整模型和数据仍只在Colab/Drive，本地仅代码及小报告。

控制器34811已启动等待完整18，确认原trainer10256实际退出后冻结并停止旧编排8712，
仅取消其旧评估子任务，未中断learner更新。源status记为superseded_training_extended，
原source protocol和权重未篡改。旧私有确认helper重命名保留以取消已排队入口；旧确认CPU322165
明确取消并保留0局回执。原前台监督返回error为本次有意停止旧编排，不是训练或数值失败。
新前台exec79236已在同一挂载kernel执行，release36910是真实worker；原guard将按真实双进程
handoff退出，不用假占用进程保留GPU。当前等待原8轮CPU311263校验；其实际状态D正在读盘，
不是失效。新真实恢复CPU337263和最终验真CPU337269均在等待各自前置回执。

后续必须通过真实18模型/两个Adam/RNG逐项恢复、原CUDA阈值后才开始扩容梯度更新。每轮
完整备份SHA，最终独立重算300局ID及全部192原生回放，不自动改stable6或继续其它实验。
私有Drive运行122m_choice_train300_20261105的training_game_progress.json给出已发布计数，
traj.md按已发布轮次追加。按前段速度剩余训练约十几小时，实际以完整局数而非步数/时长宣称完成。

## 300局接续正式启动与真实恢复验真（11月5日14:45 CST）

原11–18八轮训练及备份经独立CPU验证：30709新learner、Actor131/Critic1928步、30场正常
完整训练对局、0故障、跨update连续。18号完整checkpoint SHA为
df86da31c869397dc3cba0b005bb0d0d6d4f9eb9c9b49cb325b60fd0b51419a6。
按原始rollout逐步重算全局episode ID53–82共30完整局，83/84未结束不计；不计历史其它训练或评估。
扩容追加目标因此为270局，总预算至少300，source18为counter origin，最后完整update可略超下限。

真实122M独立CPU恢复核验通过：Actor/Critic所有参数与两个Adam moments/steps/groups
完全一致，源Actor Adam243、Critic4820、learner78012，预热2轮9局、role mixture及RNG首次
reset前完全保持。源checkpoint SHA不变，恢复只新增counter0/origin18及停止目标270。
回执full_restore_readback.json；原CPU311263在完整回执保存后停止，新恢复337263正常完成。

同一L4从18号实际配置收集512步零优化验收并检查CPU/CUDA RNG哈希；原正式恢复文件及pool
不变，随后原CUDA多路径阈值通过：48个实际历史，最大BF16 selected-logp差.0122781<.03，
TV最大.00277820<.01，储存BF16 old-logp差.0101805<.05；FP32最大差7.4863e-5<.0005。
不修改数值阈值，不以不同行为的旧概率验新行为。此额外采样是零优化验收，不计入270局预算。

正式trainer39195已开始新19轮，1536步采样产生4513有效learner，开始实际Critic更新；
14:42已观测Adam4825、有限loss/grad。controller34811、release36910实际存活，原guard8963
已按真实handoff退出；kernel8027的独立前台监督确认controller_live=true。尚未发布首个新19号，
不得把已采样但未发布的游戏数当作完成预算，机器进度只按完整发布checkpoint统计。

通用计数代码四文件本地提交ca56c4f，分支codex/training-game-budget；私有轨迹/计划/报告
未暂存、未公开push。源码包和执行脚本在本轮Drive source目录。连接观察从旧26328安全
换成20656（train300_connection.jsonl，30小时），先核验同一kernel healthy/busy再停止旧进程；
不执行伪造保活、不分配新VM。进度记录器只读已发布计数，追加到本轮私有traj.md；最终
192局及独立CPU验真后再追加完整结果。自动优化goal继续paused，不开其它试验/不自动晋升。

## 扩容首份checkpoint与计数验真（11月5日15:00 CST）

19号已完整发布，新增4513 learner，Actor11/Critic284步、4完整训练局、0故障；初始old-logp
最大差.0071411<.05。现在累计30基线+4扩容=34/300，保存的追加目标270及counter origin18正确。
source19 full SHA为f5fa3196a3ec6a0ad3229cdf3dabd9cf22f899854effa47b424758ad9112f2e4。
独立CPU核验rollout SHA、所有完成episode ID集合与checkpoint counter逐项一致、与原基线ID
不重叠，Actor Adam243+11、Critic4820+284正确，回执first_update_game_counter_readback.json。
该回执范围为首轮预算/counter/rollout及Adam步数，最终所有模型/优化器有限数值、全文件SHA
及192份原生回放的独立验真仍由CPU337269在完成后执行，不冒称已跑完300或已提升强度。

20轮已在同一进程继续。只读进度记录器341485实际存活，Drive training_game_progress.json
已显示34/300并将已发布update追加到云盘traj.md；本地文档为当前快照，不能覆盖后续云盘自动追加。
稳定注册仍6，goal保持paused；本次有限任务继续达到至少300场，再执行原192局评估并验真。

## 用户审查补充：机制验证与因果结论分开（11月5日）

接受用户对“已破案/改一行可解决”等表述的限制。Actor只用choice与分解KL已由真实运行
验证，但choice占比还受起点22→10、先验、对手和局面分布变化影响；过滤并不直接增加合法
动作，Card KL先超限也不直接证明选牌/等待变好。威胁门控关闭约78%只是可检验机制线索，
更大进攻压力可能同时导致关闭先验和难以攒费，不能用相关观测完成因果归因。

1200HP/y=.45不作为已确定的修复：低血贴塔气球可能紧急，高血坦克被己方拦截则未必
紧急，位置线后移还可能损失反应时间。后续候选至少保留近塔/即将造成伤害时关闭攒费的
兜底；参数与行为须保存版本，旧历史模型保持原定义。当前holding_prior.py源SHA仍为
edf8f90da31166df45c9bd90af66fd35474aa225e0ec43e093a83f239708cd86，未热改。

EV公式保持1-Var(G-V)/Var(G)，另列MSE与平均偏差。单局可计算EV但不支持跨轮价值能力
比较；第11轮V_old来自优化前，不以其std=.1045推断本轮lambda=.995的因果效果。

若最终仍有后期低费循环，新旧门控应从同一完整checkpoint/两个Adam/RNG/初始对手池
做受控对照，保持其余配置及明确的事前预算一致；报告实际完整局数、环境步、有效样本和
耗时。以固定换边胜负、明确原生口径的塔伤与成功出牌为主，高费覆盖仅作诊断；同时核查
紧急防守与满费等待。当前仅记录要求，不新增试验或改训练参数。

预算解释保持此前用户明确的“只扩充训练”：原8轮已经完成，当前累计目标至少300完整
训练对局，192局移到扩容后的最终模型。此次审查理解为证据边界与未来对照约束，未撤销
300场任务。只读状态核对时trainer39195确在20轮Actor优化，已发布计数34/300；不以此
训练内进度宣称强度提升，自动优化goal仍paused。

## 新L4恢复update22并落实300训练/48评估（2026-11-05 17:58 CST）

旧L4消失后，Drive最后完整发布为22号：原11–18共30局+19–22共16局=46/300；有效learner
累计93041，19–22均0故障。旧status中的trainer39195/controller34811仅为失效历史，不是存活证据。

新L4 session ppo-choice-train300-r2已重新授权Drive，使用原train300_source.tar.gz（SHA
81d8ef0c5d916c91c0c763a6737a9db9e0d14e19ab2001864fc073e94c75baea）原样恢复训练代码；
Java21和固定crforge7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6/data16.635.3构建成功。
原归档一个测试闭包触发新版Ruff B023，仅绑定测试的默认参数；完整261回归10.54秒、Ruff通过。

独立CPU从22号真实122M状态零优化恢复通过：Actor/Critic所有参数、两个Adam完全一致；
首次reset前Torch/NumPy RNG一致。Adam300/5764，预热2轮9局，counter16/origin18，追加停止
目标270保持；22号SHA c9f7b9e71c94248a9f0e4a1bbc05d18adfefdd883017cf91f165b4692578d14c。
随后同一L4零优化512步产生221choice/42非WAIT请求，CPU/CUDA RNG检查通过，原CUDA阈值
未修改且已通过（stored BF16 old-logp最大.00377667<.05）；原恢复checkpoint及pool保持不变。

控制器、protocol、最终CPU验真、GUI48份归档、进度和释放条件均已切换为48局：parent10/
stable6/BC各16，pair starts=[0,4]；不自动追加、不晋升。旧协议/日志/命令归档到recovery_0001/
previous_control；原18号full_restore_readback.json保留，新22号回执另存recovery_0001。
resume_boundaries.json登记22→23，丢弃原[406,598]步两局未完成片段，不计完整场数；同进程后续
跨轮保持环境连续。新命令只更新恢复路径及剩余上限156，其余训练选项逐项一致；未改holding prior。

正式controller24569/trainer26406已启动23轮；guard24593、最终CPU24594、记录器24595、首轮
CPU26200存活。kernel676真正监督恢复阶段，观察连接曾中断但通过独立SSH确认任务存活，未重启。
首份23号发布后自动独立核验所有SHA、有限模型/Adam、计数/episode ID、合法动作与策略版本。
本条仅确认恢复与正式训练启动，尚未把23轮采样或额外验收计入46场，不宣称强度提升。

后续按已发布checkpoint追加机器进度及云盘traj；首次累计>=300的完整update冻结，再严格48局
及原生回放独立CPU验真。release guard已为Drive短暂读失败增加有界重试，并移除其它试验交接。
保留15分钟故障/完成检查窗口，完整备份核验后释放。私有运行文档与操作脚本未公开发布。

## 首份恢复23号发布及独立CPU验真（2026-11-05 18:11 CST）

23号已完整发布并独立CPU验真：新增3场有效完整训练对局，累计49/300
（原11–18基线30+扩容19），本轮3871有效learner、
1046choice Actor行、Actor17/Critic244步，
0故障，initial old-logp最大差0.00380135<.05。采样144.39秒。

独立CPU逐项核验55个manifest文件及pool SHA；全部Actor/Critic/两个Adam
数值有限；两个Adam步数与真实22号恢复回执加本轮更新量一致；唯一episode集合重算、合法动作、
choice行数、learner策略版本22、counter origin18及目标270均一致。22→23重置边界已声明，
未完成片段不计。回执recovery_0001/first_update_readback.json；不能据训练loss推断实战提升。

同一trainer26406继续24轮，无再次重启；其episode start与23号结束的连续性由最终CPU验真复核。
完整机器进度及Drive traj继续按已发布轮次更新。评估仍严格48局（三对手各16），只在首次达到
累计>=300的完整update之后执行，不自动追加或晋升。固定三对手文件及SHA已预检通过。
本地plan/todo/300局文档已更新；仅追加云盘轨迹，不覆盖更新中的历史文档。

## 用户批准Critic2遍与12次分段短测（2026-11-05 18:36 CST）

用户明确接受：训练仍300场，评估总预算48场，拆成12次各4场换边短测。轮换stable6、
generation10、BC，各16场来自不同训练快照；不额外追加，不合并宣称最终单模型48场胜率。
阈值固定为[56, 78, 100, 123, 145, 167, 189, 211, 233, 256, 278, 300]，首次56场立即测，此后通常每新增22–23场训练再测。

25号完整发布后安全暂停旧controller24569/trainer26406，累计56/300、learner103951、0故障。
新控制器35038从完整25状态切换Critic epochs4→2，剩余update上限153、counter26/origin18、
追加停止目标270均保持。真实CPU零优化恢复逐参数/两个Adam/RNG精确一致，Adam327/6452，
25号SHA471e225b939079c6afc895fdacec795744c9c35651bb32a03b0555373b9f491a。25→26一次模拟器
重置已登记，未完成片段不计。其余训练配置未变；未改holding prior或模型架构。

新trainer35832已实际加载critic_epochs2、actor_epochs2并开始26轮；正式包装在同一
PersistentSelfplay进程每轮完整发布后按阈值执行冻结权重短测，评估子进程不推进父模拟器或修改
其随机数。后续无需为每次短测重启训练。每轮optimization_configs侧车绑定checkpoint SHA及
实际2遍步数，首轮CPU35278与最终CPU35276独立核验。guard35093、
记录器35277及kernel676监督均已切换；旧进程已停止，不能按旧PID再启动任务。

控制验证：6组不同起点的阈值/预算/每参考16场检查通过，正常换边报告通过，错误权重SHA、
种子、局数、pair_count、非法局、重复对局等7种错误均拒绝。训练底层实现未改，未新增架构实验。

首轮短测为切换前25号对stable6：1胜3负0平，pair0
两局均负，pair1一胜一负。4份原生对局结果及replay SHA核验通过。成功出牌总92、场均23，
出牌均费2.6304，出牌前圣水均值3.9491，
>=5费占比7.6087%，满费合法WAIT均值0秒。仅4场，未证明优于stable6；
这是2遍配置启动前基线，不能归因于Critic2遍。下一次阈值78场，对parent10，pair0–1。

回放归档progress_evaluation/probe_00_update_000025_stable/all_matches.jsonl及本轮总all_matches.jsonl
已生成。latest_probe.json/进度评估账本逐次更新，traj自动追加，不用旧PC文件覆盖云盘记录。
旧最终48场控制协议归档在fast_probes_0001/previous_control；新的results.json将按每快照逐行
报告12次结果。最终完整300场训练+48场原生短测CPU验真后释放，不自动晋升或启动其它试验。

## 首轮Critic2遍发布与实测（2026-11-05 18:43 CST）

26号已完整发布并CPU独立验真：新增3场完整训练局，累计59/300，
3861有效learner、1204choice Actor行，Actor4/
Critic122步；Critic严格等于2×ceil(learner/64)，未执行第3/4遍。0故障，
initial old-logp最大差0.00469649<.05。61文件/pool SHA、有限模型和
两个Adam、各自步数、唯一episode ID/目标270/origin18、合法动作/choice/行为版本25全部通过。

整轮含发布412.66秒（6.88分钟）：采样148.08秒，
优化220.22秒，序列化11.19秒，备份30.69秒。
此前23号4遍为775.95秒，3871learner；本轮样本和轨迹不同，此比较为实际进度观察，不能称严格
同轨迹对照或保证固定提速倍数。独立回执fast_probes_0001/first_update_readback.json，逐轮2遍配置
侧车在candidate/optimization_configs/update_000026.json并绑定checkpoint SHA。

同一trainer35832已继续27轮；频繁短测回调保留。首轮基线25号对stable6为1胜3负/4份原生
回放验真，预算已用4/48；下一次在训练累计>=78场的首个完整update，对parent10测4场。
各次结果单独解释，不把不同模型合并胜率，不自动追加/晋升。GUI档案已下载并SHA核验到本地
artifacts/train300_fast_probes_20261105/probe_00_update_000025_stable/all_matches.jsonl。

## 78场阈值短测与6套卡组正式接续（2026-11-05 19:33 CST）

用户确认保留48场评估预算，先分阶段扩卡组；随后再次明确不增加Actor/Critic LR、Buffer或
MC目标改动，max_grad_norm=.5、Critic2遍保持。原72+96场候选不执行。3→6在78场短测后，
6→12在145场短测后采用同进程append-only扩展；不凭4场输赢自动调参或晋升。

31号已发布79/300完整训练局、learner125691，0故障。对parent10的4场短测为2胜2负，两组
换边均1–1；4份原生结果/replay SHA验真、GUI归档通过。成功出牌115次，均费2.5739，满费合法
WAIT均值0秒；暂未显示明确优于parent10，不把1–1自动归因卡组克制。评估累计8/48，下一次
阈值100对BC。25号对stable6的1–3是另一参考、另一版本，不能直接与本次2–2比较成强度走势。

旧fast控制器35038/trainer35832在31号完整发布和短测后安全停止，新controller49727/
trainer49956通过--expand-decks接入6套。真实完整31号CPU零优化恢复逐参数、两个Adam、
预热、counter和首次reset前RNG精确一致：Adam437/7138，counter49/origin18/target270，
源SHA1453e77bf1316711cabe6ba77ee990cc7e8536b12b1891fa6e83b47aca1ff0e5。31→32实际模拟器
重置及未完成片段已登记；其后轮间连续，145场在线追加12套保留旧索引、trace和RNG。

新6套包括原3套及human_popular_04/07/18；04实际无天狗，不误标天狗体系。后6套来自指定
训练配置，以普通槽位接入，SkeletonKing/ArcherQueen本身技能保留，无需放开所有特殊形态。
候选12套与留出名称/完整8卡集合零重叠。CPU接口12×256步、0故障0非法、14合法技能请求
通过；该工程验收不计训练或48场策略评估，不作强度证明。在线追加与规范恢复命令测试通过。

32轮已实际初始化6套/2遍。CPU49955等待首份新checkpoint；最终CPU49953、
guard49777、tracker49954和kernel676监督均接到curriculum_0001。每份新卡组
侧车绑定checkpoint SHA；原始rollout按卡组Learner曝光独立重算，排除冻结对手座位。29号已
用4595行验证解码正确，不再把600个总座位当600个Learner或承诺每套50局。

对新的Critic曲线建议仅只读核查，未叠加新实验。26–30轮现有数据筛出12完整固定行为局、
15 Learner座位，V_old std .2635、MC MSE .7071、EV−.1627，价值预测仍弱，但不是优化后验证
或容量判决。旧审计加回Phi前后MSE几乎一样，EV变化不证明新学到信息；梯度日志是裁剪前。
详见docs/CRITIC_REVIEW_20261105.md，不把历史窗口无修正混为当前on-policy价值目标。

本地及Drive轨迹继续按已发布轮次和短测更新。未来中断恢复须新增真实完整回执和模拟器边界，
不直接绕过当前初始化检查。源码/holding prior保持；仅配置和私有操作包装新增，不公开push。

## 首份6套32号发布及实际曝光验真（2026-11-05 19:41 CST）

32号已完整发布：新增4完整训练局，累计83/300；3942有效Learner，
1438choice行，Actor15/Critic124步。Critic仍严格2遍，
LR2e-5/1e-4和max_grad_norm=.5保持，0故障，initial old-logp误差0.00776005<.05。
独立CPU核验74文件/pool SHA、有限Actor/Critic及两个Adam、各自步数、唯一完整
episode集合、保存目标270/origin18、行为版本31及采样合法性；checkpoint中的训练league确为6套。

按原始obs.deck、词表与valid Learner mask重建曝光，排除冻结对手：
- hog_cycle_v1：862有效Learner行，424choice行，1完成Learner座位对局。
- graveyard_control_v1：737有效Learner行，168choice行，1完成Learner座位对局。
- human_popular_07：940有效Learner行，259choice行，1完成Learner座位对局。
- human_popular_18：1403有效Learner行，587choice行，2完成Learner座位对局。

本轮giant_sparky_v1与human_popular_04未抽到Learner样本；不把卡组池已包含6套说成单轮已经
覆盖全部6套。新飞桶/X连弩已实际进入训练。完成Learner座位对局可因双方均为Learner而大于
唯一游戏数；本轮4个唯一游戏仍只计4场。最终会汇总每套实际曝光，不承诺每套50局。

本轮含发布508.20秒（8.47分钟），采样147.42秒、优化
293.88秒、备份52.39秒。不同牌组、Actor步数和实体数会改变耗时，不保证
每轮固定6.8分钟。训练请求均费2.333、>=5费请求0仅作诊断，按用户确认不因此热改LR/Buffer。

同一trainer49956正在33轮；下一次短测仍累计>=100场对BC。已用8/48评估场，78阈值实际
79场/31号对parent10为2–2，每组换边1–1；56场/25号对stable6为1–3。两次参考不同，不合并
宣称强度改善。两批各4场原生回放/GUI档案已下载并本地SHA验真。

145场短测后将在同进程6→12追加，保持现存对局/旧索引/RNG；没有新增Actor/Critic学习率、
价值Buffer或MC混合目标变更。所有新证据在curriculum_0001；仍完成300场及48场CPU验真后释放，
不自动追加或晋升。plan/todo/300局文档和多卡组计划更新，私有Drive仅追加轨迹，不覆盖旧记录。

## 近期训练回放导出与迁移备忘（2026-11-05 20:20 CST）

最新已发布36号，97/300完整训练局、learner145649；6套继续，100场短测尚待触发，最新79场/31号对parent10为2–2，已用8/48评估场。

按最近完成顺序从32–34号已保存轨迹选择episode148–151四局，覆盖飞桶镜像、矿工气球对巨人电车、矿工气球对毒墓、X连弩对矿工气球，未按胜负筛选。根据原seed/cards/actions重建，6088次完整原始obs比对、所有mask/reward/terminal及独立原生重演全部相同。原训练arena为空导致第一次原生导出失败；新副本仅补Arena_L元数据后逐元素观测仍全相同，训练进程未修改。部分对局跨update，标签标PPO TRAIN v32–33，不冒称固定最新checkpoint测评。

与已有31号对parent10四场合并为本地8场GUI档案artifacts/current_replays_20261105_update34/all_matches.jsonl，SHA/原生结果核验通过。viewer25360已打开，窗口确认8/8可读，默认入口已更新。确定性重建不新增训练/评估对局。

用户要求将14600K、项目16GB RAM/100GB HDD、2×V10016GB的迁移考虑记到loc_transfer，已写loc_transfer/README.md。当前仅规划，未迁移：记录PPO最小恢复约7GiB、trainer峰值约6GiB、本轮历史归档约58GiB，列出V100无BF16/现有代码FP32回退、FP16/GradScaler待适配、单GPU基线、双卡/DDP未实现、HDD保留策略与完整状态迁移验证。完整BC数据尚未盘点，不宣称100GB能容纳全部BC。云端既定参数及300/48预算保持。


## 分阶段性能分析（2026-11-05 20:52 CST）

按用户要求完成当前运行profile，不开subagent、不改训练参数或增加策略评估。基线为连续6卡组第32–38轮，7轮26场完整训练局、26883个有效Learner样本，全部发生在本次观察器启动前。平均每轮523.43秒：Critic271.75（51.92%）、采样151.24（28.89%）、Actor36.49（6.97%）、Drive38.47（7.35%）、本地序列化12.05、历史候选准入均摊9.78、日志2.36、优化准备1.29秒；有效新样本7.34/秒。第33轮单次候选检查68.49秒单列；37→38之间4场短测及审计回调151.88秒不混入训练耗时。

20:45:05–20:52:05 CST对现存trainer49956完成420秒49Hz非阻塞py-spy与GPU采样。19798个成功样本、680次错误（3.32%），2812个空/缺失调用链单列。该次观察的GPU利用率：采样32.70%、Critic95.59%、Actor95.10%。Critic大量栈停于loss回读/非有限检查，包含等待GPU，不能声称这些比例都是可省CPU耗时。没有启动第二个GPU训练/基准进程。

CPU审计已保存7轮mask：Learner强制WAIT行76.04%；按当前调用分组，整批仅WAIT的当前Actor调用52.85%、冻结对手调用43.30%。现有forced_wait_fastpath仍关闭。另发现推理每步计算entropy/max-prob，当前模型统计每64步才使用、对手统计直接丢弃。候选顺序：验证WAIT快路径与按需统计；隔离测试当前FP32帧编码的Critic microbatch8/16/32（有效64、2遍保持）和GPU算子；再考虑不可变快照的有界后台Drive发布。ref缓存/标量延迟/4环境低优先，不承诺加速倍数。

本地报告docs/PPO_PROFILE_20261105.md，原始证据、脚本、图与计算结果artifacts/perf_profile_20261105/。100场阈值实际37号102场完成对BC短测2–2，评估已用12/48；20:52读取的最新已发布39号109/300场，40轮优化结束等待发布。此状态为当时快照，后续按Drive机器进度确认。

同时回应用户的BC时机顾虑：当前WAIT margin仅剔除动作前1秒/后0.5秒的WAIT监督，保留历史和正动作原始时刻，不移动/复制法术标签。仍可能弱化精确邻近时机监督。PPO不应用该过滤，但继承BC初始化偏差；当前决策频率0.25秒也是精细时机限制。本次不改数据或频率。

性能报告与17份证据已私有归档Drive profiling_20261105，SHA核验完成；20:57重新读回已发布40号113/300场、160602累计Learner样本，trainer49956仍存活。300/48预算保持。


## 推理提速接入与首轮独立验真（2026-11-05 21:32 CST）

用户同意后已实施强制WAIT fastpath和按需推理统计；源代码仅改inference.py/selfplay_v2.py，旧调用默认仍返回统计。265项Linux回归及Ruff通过，测试覆盖动作/logp/RNG保持。第43轮完成原定123场阈值短测（实际124场，对stable6为2–2）后，隔离L4基准验证真实122M/128帧：384步动作与CUDA RNG一致，logp最大差.003677、TV.001867，通过原.03/.01门限。固定256步推理中位7.003→1.848秒（约3.79×）；样本强制WAIT偏多，仅Actor路径，不能外推整轮。

Critic同一真实权重/128个完整历史状态/有效64/两遍/零LR基准：batch8中位7.802秒，16为8.224秒；32在旧trainer驻留的共存显存条件下OOM。没有改Critic batch。首个基准因隔离副本缺模拟器路径，在GPU工作前失败并自动恢复，停放3.35秒；第二次175.23秒成功后同样恢复。失败与停放回执保留。之后从43号完整checkpoint做零更新CPU恢复核对，模型、两个Adam、torch/NumPy RNG、warmup/counter精确一致，再声明一次43→44模拟器恢复边界。未完成局[88,15]和刚开始的未发布44片段丢弃、不计完整局数。

新controller78693、trainer78728、最终CPU78692，guard49777/tracker49954与kernel676保持。入口/content/crrl_speed_20261105/runtime_train.py；原curriculum短测/在线12套回调保持。恢复包装初始校验绑定43，运行时中断需重新生成最新恢复回执及边界，不能直接重跑。协议/counter仍origin18、270加原30=300，评估仍12×4/48，不追加或晋升。

第44轮已发布并独立CPU验真98份文件及SHA、模型/两个Adam有限值、合法on-policy动作、choice-only、两个optimizer步数、LR和原old-logp阈值通过。4144 Learner/735 choice，Actor24/Critic130步，0故障，old-logp最大差.00560474<.05，跳过当前Actor强制WAIT行2571。采样76.55秒、全部优化342.01秒（Critic按事件270.56）、本地保存12.53、日志1.18、Drive36.38，总468.66秒（7.81分钟）、8.84有效新样本/秒。第45轮连续采样87.66秒后继续优化，未额外重置。对比旧32–38平均523.43秒/7.34样本秒只是生产观察：样本数、局面和Actor步数不同，不作受控整轮倍数结论。

最新已发布44号126/300完整训练局、175553累计Learner；已用16/48评估，下一次>=145场对parent10，随后同进程6→12。LR/max_grad_norm/Critic2遍/有效batch64/奖励/历史/holding prior均保持。后台Drive发布涉及队列、不可变快照和短测发布屏障，本阶段未开启；保存与校验仍同步。报告docs/PPO_SPEED_IMPLEMENTATION_20261105.md，私有Drive speed_implementation_0001及source归档，本地artifacts/speed_implementation_20261105；未公开push。


## BC邻近WAIT与PPO时机只读核查（2026-11-05）

用户再次关注动作之间padding对精确法术时机的影响。本次核对BC过滤/历史padding/原生标签量化、PPO采样/GAE/choice-only/参考KL与模拟器命令延迟，没有改训练或新增评估。BC isolated_wait_rows只剔除训练WAIT候选，历史保留；PPO不调用，先在完整时间轴计算GAE后才筛Actor choice行。固定BC只是对手，KL参考为每轮当前Actor副本，不能说BC偏差绝无继承，也不保证PPO已修复。

只读44–47四轮12,288个环境transition，12,266次连续next_obs对下一obs逐元素比对零错位，所有非终局推进5tick、动作合法。1,376个动作前1秒内的WAIT仍属于Actor候选（不是保证KL早停前全部实际更新），未复用BC剔除。轨迹play_delay_ticks均20，即1秒固定原生命令排队；与BC padding不同。

补充此前“正动作原始时刻没动”的边界：WAIT过滤本身确实不移动标签，但BC导入原本按5tick窗口归类人类givenTick，窗口内0–4tick即0–0.20秒偏移不是动作输出；PPO4Hz也不支持子步偏移。是否法术打晚/打空还需区分请求tick、执行tick及目标实际位置，不以本审计归因或追加48场预算。报告docs/PPO_BC_TIMING_AUDIT_20261105.md及artifacts/ppo_timing_audit_20261105/report.json。


## 学习信号、异常值与修复候选审计（2026-11-05）

用户连续询问梯度/loss/reward、自适应降低LR及全量异常清单。本轮仅只读CPU审计，不改训练配置、不增加评估或开subagent。日志38–49共12轮Actor140步/Critic1354步，范数全有限非零；Actor裁剪前范数中位9.06、最大86.23，140次均clip(.5)，不等于未保护梯度爆炸。9/12轮KL早停（8 Card/1 Tile），实际样本呈现8600/名义两遍18006=47.76%，仅5轮跑完首遍。44→49完整checkpoint的WAIT/选牌/落点/主干及Critic参数和Adam矩证明实际更新。技能头动量全0，但旧6卡组技能合法行也全0，不能认定断梯度。

44–49六轮20643 Learner/4375 choice：奖励重算与保存逐元素一致、终局胜负无反号、HP/皇冠零和；overflow全0，不能把低费行为归因于该惩罚。原始优势std .131，奖励/价值部分std .115/.075，与总优势相关.823/.485，没有优势全零或完全自举无奖励证据。真实HP变化1811次/非终局8.78%，大部分其它非零shaping是折扣势能漂移。on-policy版本、old logp/value有限性均通过，无固定对手轨迹混入。

13局完整同版本/15 Learner座位（帧相关）采样时Critic对完整shaped MC的MSE .59390、EV−.09252，零预测基线MSE .53893，乐观偏差约.124；不能用低自举训练loss称其成熟。20643帧实际critic lambda=.995目标的prefit EV−.06646。旧rollout_value_diagnostics仍用Actor lambda=.98，当前run无该W&B镜像；若用它看Critic目标会误导，需拆目标标注。汇总policy_metrics还包含拒绝batch，需区分accepted/rejected；这些不改变当前梯度。

高费探索稀少：≥5可用费629/20643=3.05%，高费合法choice33帧，普通牌请求1078次仅5次≥5费。后期高费未解锁状态的威胁关门控全状态72.42%，排除强制WAIT只看真实choice后49.79%；不能把关闭率直接归因为卖费失败。Actor GAE有效指数尺度约12.2秒，20秒后单奖励直接权重.191，仍可由可靠Critic接力，不是硬截断。

用户提出早停降低LR：当前KL在step前检查，已超限的现有参数不会因改LR立即回退。候选分为下一轮调Actor LR（无法追回本轮未用数据）与完整Actor/Adam/RNG起点回退重放（可重用本轮，但须限制重试及算力，参考/优势保持冻结）。建议先修监控，再单因素Actor LR有界对照，KL/Clip/Critic2遍保持；12卡组曝光先观察，Critic目标/Buffer/MC/架构不同时改。本轮没有执行候选调参。

已确认49号147场对generation10短测2–2，20/48预算；之后同进程6→12，50号149/300完整发布，51号优化继续。当前数据主要在12套切换前，不能当12卡组效果判断。详细报告docs/PPO_LEARNING_AUDIT_20261105.md，证据artifacts/ppo_learning_audit_20261105及Drive learning_audit_20261105。


- 学习审计补查12卡组50–52号：3579/4608/4941 Learner，Actor6/10/38步，52号完整两遍；均0故障、合法且on-policy，old-logp最大.00939<.05。技能合法行57、请求7，52号技能头Adam矩已非零，证实链路在工作。高费请求7/778=.90%仍少，不用3轮证明扩容成功或失败。补查时52号157/300、53号继续；评估20/48。修复候选仍只列报告，未改参数。


## Actor LR与监控修复正式接入（2026-11-05 23:38 CST）

用户“给我干”后实施监控与单因素Actor LR修复，不开subagent。267项Linux回归/Ruff通过；新旧separate在deferred true/false、接受/拒绝四组合中Actor/Critic参数及完整Adam逐位一致。telemetry按Actor .98/Critic .995分别重建目标，按实际batch样本数汇总loss，分开已提交/拒绝指标与样本呈现/唯一覆盖/裁剪率；真实44–49轨迹诊断与独立float64审计误差<2e-5。Critic、奖励、门控、max_grad_norm、KL和预算不改。

真实12卡组源50完整权重/Adam、冻结51轨迹加50历史prefix、同old-logp/优势/reference、两shuffle53/307，对照2e-5/1e-5/5e-6。首轮开发发现CPU Adam step可能共享，整体标无效并保留日志，未回写生产；停放182.74秒后原trainer恢复。修正完整独立Adam副本并断言源step不变后重跑，正式停放698.54秒无模拟器重置。原2e-5两组提交8步512次、38步2432次，共2944/4864=60.53%；1e-5和5e-6两组均38步2432次/完整两遍，共4864，提升65.22%。固定128状态Card/Tile KL均在原.015/.06内；按预声明选择最高合格LR1e-5。该对照证明开发数据利用与KL合格，不是胜率提升。

从最新完整56号174/300场交接，源SHA ac521e8a838e1759bdba6f24e81d62d3778abd6c59b0fdc48f1beece2a7ff828。CPU核对模型/Adam矩及step/torch-NumPy RNG/warmup/counter精确一致，仅Actor LR显式2e-5→1e-5。源Actor779/Critic10158步、221820累计Learner、扩展144局。新增56→57恢复边界，未完成局[7,25]和未发布57初段不计局数。新controller110678/trainer110701，最终CPU110677、首轮CPU110702已完成；guard49777/tracker49954保持。入口/content/crrl_actor_lr_20261105/runtime_train.py绑定56，旧43/31入口不再裸启动；中断后须新回执/边界再恢复。

57号完整发布并CPU独立核验124份manifest/SHA、模型/两个Adam有限值、合法on-policy动作、choice-only、步数与LR通过：655 choice两遍全部提交1310/1310、唯一655、拒绝0，Actor22/Critic126步，7970 Critic样本呈现，old-logp最大.005713<.05、0故障。当前178/300完整局、225805 Learner、58继续。57轮含发布542.67秒/9.04分钟（采样90.07、优化402.29、保存13.69、日志1.62、Drive35.00），更多优化有计算成本，不称提速。22次Actor仍裁剪，意味着本轮没有调整clip；不能把裁剪完全消除当LR修复目标。

发现早前kernel监督仍绑定旧PID、心跳过期，已在同一kernel676以真实训练监督恢复，exec本地session54502、controller110678新鲜心跳。该监督不另开训练，下一次控制器变化仍需认证/更新。实际进程已写回curriculum_0001/processes.json。12套已在49号147场短测后在线追加，55号168场对BC短测完成，已用24/48，下一次>=189场对stable6，300场训练/48评估不变，不晋升。Critic1e-4/2遍、有效64、lambda .98/.995、奖励/holding prior保持；本阶段固定Actor1e-5，无每次早停自动减半/同轮回退调度。报告docs/ACTOR_LR_REPAIR_20261105.md，证据artifacts/actor_lr_repair_20261105和Drive actor_lr_repair_0001。

## 攒费与信用分配组合阶段接入（2026-11-06）

用户先要求暂停讨论，旧controller110678/trainer110701停在完整66号216/300场；恢复watchdog132800已终止。随后用户明确同意组合修复并要求继续，旧bias-only未上线。已实现版本化prior2：只对指定>=4费核心牌攒费，安全WAIT+5/普通压力+2/紧急0；按存活塔实际距离、移动方向、攻击目标及后场兜底判断紧急，避免只凭低HP放行。合法核心slot+.7×压力系数，不改mask或tile，不给所有贵牌加分；Actor lambda .98→.993，Critic .995/1e-4/2遍、Actor1e-5、有效64、clip.5、KL .015/.06及奖励保持。

269项隔离Linux回归及Ruff通过。源66 SHA bd38c9cb730a92a891f71f4f568518d5a1df8d3c0dc725a5e20e4cf1771c298a；CPU两个模型/两Adam完整状态、warmup/counter/RNG精确恢复，Actor951/Critic11312步/Learner258561。新GPU运行首reset前torch/NumPy/CUDA RNG精确核对通过。登记66→67边界，未完成局[333,257]和旧未发布67优化片段丢弃不计局数。controller143453/trainer143476/最终CPU143452/首轮CPU143481，guard49777/tracker49954保持；kernel676新监督session61582。

入口/content/crrl_holding_combo_20261106/runtime_train.py绑定66号。新数据/behavior/PPO重算/导出同prior2，无旧轨迹换先验训练。总300/48预算保持，已用32评估；>=233场对BC四场短测后自动stage_review_wait，等复盘再继续，不自动新增对局或晋升。67首轮发布与CPU验真尚待确认，初始化通过不冒称强度改善。详细文档docs/HOLDING_COMBINATION_20261106.md，本地证据artifacts/holding_combo_20261106/，私有Drive holding_combo_0001。
- 组合阶段67号首轮发布：220/300场、Learner263115；4554有效/1590choice，Actor50步两遍3180/3180、无KL早停；Critic144步，0故障，old-logp最大.009553<.05。238普通请求均费2.2731，>=5费5次（2.10%）；核心合法39帧，核心12/WAIT8/其它牌19；连弩3、钻机8、皇家巨人1原生命令接收，非部署/胜率证明。巨人合法8次仍0请求，电车0合法，低费循环尚未解决。满费合法WAIT0秒。首轮CPU回读待确认，下一预定233场对BC四场，之后暂停复盘。24份源码/操作/文档私有备份经用户明确授权已完成SHA验真，未公开push。
- 67号首轮独立CPU回读已通过144文件/尺寸/SHA、两模型/两Adam有限值、合法on-policy动作、choice-only、optimizer连续和prior2/lambda.993配置验真；首轮CPU143481正常退出，68继续。67完整墙钟717.41秒=11.96分钟，采样107.79/优化503.39/历史准入44.84/保存13.22/日志1.87/Drive46.30；有效6.35样本秒。更多choice增加优化算力，不宣称提速。233场对BC短测尚未发生，修复胜率效果待测。
## 组合修复233场短测与阶段暂停（2026-11-06）

完整70号/233场、Learner276044，67–70共4轮17场；Actor四轮全部两遍，无KL早停/0故障。普通请求797次>=5费28（3.51%），均费2.5696，可用费>=5为1169/17483（6.69%）；卡组/对手不同，不能当因果或胜率改善。70号对BC5096预定四场为1胜3负，pair4双边全负、pair5为1–1。成功牌90次、均费2.8222、>=5费8次8.89%、满费合法WAIT0秒；电车1/巨人3/石头人1/野猪2/攻城槌4。原生验真及4局GUI归档完成，评估36/48，剩12。已stage_review_wait，无新增采样，剩训练67场，运行时仍在线。

70号Critic第1遍第15个有效batch出现loss143.501686、clip前grad_norm23640.5，clip.5生效，前后约.02；整轮均值1.008216，其它三轮.0036–.0083。旧值/奖励均正常有限，不能称完全健康。已从69号numpy RNG和5次reset精确重建70全部shuffle，终态与checkpoint70一致，定位64样本；仅在独立副本重放前15个Critic batch，不回写生产、不增加对局，根因尚待确认。详见docs/HOLDING_COMBINATION_20261106.md末节，证据artifacts/holding_combo_20261106/与Drive holding_combo_0001。
- 233场复盘补查完成：独立副本按精确重建shuffle重放前15个Critic batch，39.72秒；第15批loss.01125/grad2.911，未复现原143.5/23640尖峰。但小量浮点/梯度差异逐步累积，并非逐位轨迹，不能排除原异常或认定硬件故障。原尖峰贡献整轮loss约99.6%，其余批均值约.0043。70号Actor/Critic/两个Adam全量有限值检查通过；无生产回写或新增对局。stage_review_wait保持，根因仍未确定，不自动续剩余67场。证据review_233_integrity_and_replay.json及Drive critic_spike_replay.json。
## 233场四局战术只读复盘（2026-11-06）

按用户“你看看”完成保存输入的原生重演，新增单位/手牌/增益/实际治疗观察；没有策略推理/新比赛/训练。四场与原审计结束tick、胜者、皇冠及双方支出一致。原回放及新证据下载本地SHA验证通过，目录artifacts/tactical_review_233_20261106/replays/；默认GUI仍旧8场，显式view-replay.cmd -Replay该目录/all_matches.jsonl可看新4场。

第一场5狂暴花10费，3次无增益目标、1次只增益国王塔、1次增益巨人。5.5秒空场狂暴时手牌已有电车6费且有7.958费；不是为了抽电车，但抽巨人/过牌价值不能排除。Heal在表中是治疗精灵，不是旧治疗法术；第一场4个精灵实回血0，第三场4个实回血500，不能把0回血当0总价值。首个巨人落下时电车已走到2.28格附近，纠正先前仅看初始坐标推测分路的说法。

我这版prior的tower-only威胁门控也暴露缺口：20.75秒电车位于河道(14.149,16.339)1149HP，对岸双龙y约19.3–19.5，23.5秒电车218HP；敌军远离我塔且y>.55，按执行快照规则判安全，可能继续偏向攒费/核心而压制前线支援。并非完整复现提前1秒决策的反事实，不直接证明此项导致输局。优先考虑前线主力受威胁时放开支援，以及法术/支援牌目标落点学习，而非继续抬贵牌率。训练233暂停、评估36/48保持；没有热改。详见docs/TACTICAL_REVIEW_233_20261106.md。

## 2026-11-06：233场关闭整套人工战术先验

用户授权“撤掉试试”。从完整70号启动no_prior_0001，WAIT/slot/名单/威胁门控整体关闭，保留旧模型兼容。270回归/Ruff、CPU两模型两Adam及RNG精确恢复、GPU首reset前RNG检查通过。登记70→71重置，未完成[159,213]不计局；其余奖励/LR/GAE/对手均保持。预算300/48，交接233/36，256场既定短测后暂停。首轮发布及完整验真待完成；不宣称已变强。详见docs/NO_PRIOR_TRIAL_20261106.md。

- 无prior首轮71号发布并独立CPU验真152文件通过：236/300，评估36/48；3958 Learner/521 choice，Actor18步全部两遍，Critic124步，0故障，old-logp max .007536。训练混合对手终局2胜1负，不能当固定测评；Critic max loss .025214/裁剪前grad7.075，Actor max preclip grad64.02且18/18 clip，未宣称梯度理想或旧异常已解。72号继续，256场既定短测后暂停。


## 2026-11-06：无prior阶段性能复核

71/72正常轮次485.77/552.35秒，71 Critic287.74/Actor53.26/采样91.60/本地保存14.56/Drive36.10秒。真实122M隔离测试：TF32局部省14.9%但三步参数差.000489，融合Adam省1.5%但参数差.0003818，均不通过事先1e-5门限，不接入。reference缓存+延迟指标Actor省12.8%，128 choice两遍4更新两重复参数逐位相同，估算整轮省约6-7秒；已有开关留至下一阶段切换，本轮未改生产。73号两次带watchdog停放共106.84秒，已恢复同一trainer168013，不能用73号声称吞吐变化。正式评估仍36/48，256场短测后暂停保持。详见docs/PPO_SPEED_ROUND2_20261106.md。


## 2026-11-06：集中工程提速及无人值守续训

74号248场停放后，283回归/Ruff、原生两轮实际Actor/Critic优化/异步重叠/restore与完整74号模型两Adam/RNG核对通过。启用cache-reference、defer-scalar-metrics、async-publication并去掉每轮empty_cache。帧去重局部省17.5%但参数差9.44e-5未过1e-5门限，保持关闭；TF32/融合Adam亦不启用。74→75重置登记[112,698]+旧75第1步不计局。controller184501/trainer184528/CPU184500/184529，kernel676监督正常。用户要求无人值守到原300/48并验收后本地关机，不在256人工暂停；三次剩余短测保留。首轮75发布/CPU验真待完成。详见docs/PPO_EXTREME_SPEED_20261106.md。

- 优化首轮75正式验收通过：252/300，5345 Learner/702 choice，Critic168步，Actor19步；Card KL .017775触发原门限，拒绝64行，未放宽保护。160文件/两模型两Adam有限/合法on-policy/步数连续通过，old-logp max .004441。主线程550.74秒、后台42.13秒，76 collect比75发布早42.71秒开始，整个上传已隐藏，快照.046秒/背压约2微秒。样本比71多，不能直接比轮墙钟；36/48评估保持，云端自动继续300/48及最终验真/释放。关机前源码快照完成后执行本机关闭。


## 2026-11-06：CPU读回与学习审计结论

新CPU cr_cpu_review_20261106挂载Drive成功。旧L4已不存在；最新完整84号288/300、329517 Learner、44/48评估；尾日志85轮769步，无完成/最终CPU/release回执。84号完整SHA/两模型有限/两Adam矩有限通过。最新259场stable2–2、280场parent1–3。只读71–84审计53473行/7296 choice，高费合法88/高费请求24；奖励重算7.15e-8，无mask/版本/非有限错误，Actor183步且参数/Adam确实更新。31完整同版本局/42 Learner座位MC EV+.214（不是旧负值），自举目标EV+.504；≤2费动作平均归一化优势−.117，WAIT+.013，不支持奖励始终鼓励低费。报告docs/PPO_LEARNING_FINDINGS_20261106.md；本次无训练/新对局/参数改动。


## 2026-11-06：新增1000场采样优先实验

用户授权新增1000/独立48评估。新L4 ppo_long_l4_20261106恢复完整84，2环境×3072步，Actor最多2遍/Critic1遍，旧LR/GAE/KL/clip/奖励/历史128/12卡组/角色比例保持。原实验288/300仍未完成，新counter258→1258。283回归及完整模型两Adam/RNG、GPU首reset前CUDA检查通过，controller6836/trainer6837/kernel3538，独立CPU12907；85采样中，首轮完整验真待完成。过程20/60/100等9次各4局，最终同一模型12局，总48。此为预算配置实验，不保证收敛加速。详见docs/PPO_LONG1000_20261106.md。

- 新长训85首轮计算结束：8/1000新局（全局296），8728 Learner/1076 choice；Actor34步2152/2152呈现全部两遍，无KL早停；Critic137步8728呈现单遍，0故障，old-logp .007068。summary593.41秒，优化410.77秒，首轮14.71有效样本/秒仅描述，不据单轮宣称收敛提速。86采样继续，首代需要上传继承历史池，完整发布/CPU验真待回执。

- 长训首轮85完整发布及独立CPU验真通过：12文件SHA、Actor/Critic/两Adam有限值与步数连续、3072采样/合法on-policy/choice-only核对通过，new8/1000。前台593.53秒、初代后台77.83秒已与86计算重叠；GPU kernel与CPU心跳正常，尚无新阶段20场短测。源恢复/预算/首轮回执保存在artifacts/long1000_20261106/first_update_snapshot.json。

- 新长训24/1000、完整87后首次对84短测实际2–2。审计部署包漏scripts/java/ReplayExecutionAudit.java导致退出，属于执行遗漏；已补入源码并在原4场上完成原生审计/GUI归档，未重打、ledger4/48。补齐全资产打包与启动javac预检。完整87 SHA/两模型两Adam/CPU RNG核对、GPU首reset前CUDA核对通过，恢复边界87→88丢弃未完[650,1111]，controller17752/trainer17780，kernel3538新监督96945，CPU12907。旧失败日志保留。前三轮Actor100步全部两遍/无KL早停，Critic EV.259/.737/.699；前台14.01有效样本/秒排除停机，短测仍未证明变强。继续按用户要求监控88、89。


## 2026-11-06：恢复后两轮监控通过与连接维护

完整88号新增31场，89号新增41/1000场（全局329）；90号正在Critic优化。
88/89分别9606/8370 Learner，Actor38/30步，Critic151/131步；0故障、无KL早停，Actor两遍全部提交。
独立CPU回执monitor_completed.json为passed：完整SHA、Actor/Critic与两Adam有限、优化器连续、合法on-policy及模拟器边界通过。
CPU首次读88目录遇到Drive跨VM可见性延迟，加入最多180秒只读重试后通过；不修改SHA或数值门限，训练未因此停机。
Critic fit loss .006311/.004342，采样时自举目标EV .4526/.7618；这些不是完整MC或独立实力证明。
正式评估仍4/48，对起点84号2–2，两对均1–1；下一新增60场对stable6，未追加对局。

连接维护：L4 watcher26288、CPU25548每300秒刷新连接凭据，最多2700秒轮换真实WebSocket；不分配新VM、不重启trainer、不执行假负载。
两kernel均有真实训练/验真前台监督，CPU监督43902已确认verifier12907存活。
本机临时防休眠进程6284仅在watcher存活时请求系统保持唤醒，最长24小时，不改永久电源配置、不阻止手动关机。
当前连接健康；尚未观察到本轮首次连接轮换，不能宣称长时间不掉线已验证。
官方动态回收/运行时上限仍适用，完整checkpoint与独立验真才是被回收后的恢复保障。
详见 docs/COLAB_LIVENESS_20261106.md；证据 artifacts/long1000_20261106/monitor_completion_summary.txt。


## 2026-11-06：用户授权奖励阶段v2

完整92号62/1000新局（全局350）；新增60场短测对stable6为2–2，累计评估8/48。
92之后开始奖励阶段v2：公主塔HP .15→.05、国王塔HP .20→.05、皇冠.15保持、overflow .001→0。
终局±1、gamma .9995、LR/GAE/KL、采样/训练配比、对手与12卡组、1000/48预算保持。
奖励4项语义回归通过；完整92的Actor/Critic、两Adam、torch/NumPy零优化恢复核对通过；GPU首reset前CUDA检查仍保留。
首次启动因发布对手池相对路径未映射而退出，未完成93、无新评估；已保留失败证据并逐个验证本地历史对手SHA，再从同一92恢复。
不更改通用resume奖励一致性门限；阶段入口显式先按旧reward恢复、再切换新reward，旧checkpoint不被改写。
Critic目标尺度改变，前后loss不直接比较。首轮新rollout的实际reward重算监控已启动，不额外造局。
执行详情与回执见docs/PPO_REWARD_STAGE_V2_20261106.md、Drive reward_stage_v2/。
