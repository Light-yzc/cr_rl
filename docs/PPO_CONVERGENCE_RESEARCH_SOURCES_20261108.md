# PPO 收敛与样本效率：核实来源及本项目适用性（2026-11-08）

本轮只做研究，没有修改训练参数或加入 BC。用户最新选择是先完成提速与对战测试、暂不混入外部 BC。以下“优先级”是拟做受控对照的顺序，不是已证实能提高本项目胜率的承诺。

## 当前证据及已有能力

- 第 179 轮冻结模型对迁移包 BC 锚点 128 局为 77–51；两组种子分别 42–22、35–29。按物理卡组配对聚类、同时保留换边和两组种子的 bootstrap 区间约 [0.53125, 0.671875]。这是对该锚点、这组卡组的证据，不是对所有 BC 或对人类的强度结论。原始 BC5096 与锚点导出权重尚未逐参数比对。
- 原生回放全部 128 局 outcome parity 通过。实际出牌均费 learner 2.3496 / BC 锚点 2.9654，≥5 费占比约 0.516% / 10.739%；BC 锚点满费合法可出但等待平均约 32.61 秒/局。赢过该锚点不等于高费策略已学会。
- 169–182 共 14 轮：15,936 新 choice，11,461 unique choice 提交；17,755 presentations 提交 / 31,872 计划 = 55.7%；8/14 轮 Actor 发生 KL 早停。加 epochs 不一定能让更多样本真正提交。
- 181–182 保存 rollout 的完整恒定 behavior episode 审计：25 games、30 seat episodes、22,042 rows；MC EV≈0.4511，MSE≈0.3001，target std≈0.735、value std≈0.516、bias≈−0.0588。相关小样本，不支持“Critic 完全失效”的说法。
- 现有代码已做 choice-only Actor、choice 集合上的 advantage normalization、独立 Critic、GAE、Card/Tile KL 保护、梯度裁剪、历史 PFSP。不要把这些再次列为新加入的 trick。
- `rollout_value_diagnostics()` 已存在于 `src/cr_rl/training/telemetry.py`，可重建冻结 rollout 的 GAE target、EV、MSE、bias；这是 rollout 时 Critic 的诊断，不是优化后泛化效果。

## 优先级 1：保留和发现能克制低费裸下的对手，改善探索覆盖

**依据。** [AlphaStar，Nature 2019](https://www.nature.com/articles/s41586-019-1724-z) 的原文摘要强调适应策略与反策略的 league；Extended Data Fig. 9 的 PFSP 对照说明 PFSP 群体在其任务中更强、更不易被利用。论文全文主体未全部取到，已核实的是摘要及扩展图说明，不把其精确采样权重伪称为已读。

**本项目实验。** PFSP 已有，新增的应是保护对不同卡组、不同费用节奏有区分的历史策略，按 learner–opponent 卡组配对胜率检查是否只有“BC 满费空等”这种弱参照在提供学习信号。先诊断池覆盖和遗忘，再对池保留策略/role 比例做有边界的对照。需要同时报告新高费合法机会、攒费时长及胜率。

**边界。** 不是把对手全换成当前最强，也不是奖励“出贵牌”。高费不可负担时，单纯增大动作熵不能给被 mask 的高费动作概率；需要让探索覆盖攒费前的 WAIT 决策。对手、课程改变会改变训练分布，必须固定独立评估。

## 优先级 2：先研究 KL 控制下的有效更新利用率

**依据。** [Implementation Matters](https://arxiv.org/abs/2005.12729) 说明代码层面的优化和超参数能显著影响 PPO 的实际行为；[官方 MadryLab 实现](https://github.com/MadryLab/implementation-matters) 可逐项切换这些配置。[rl_games 官方项目](https://github.com/Denys88/rl_games) 的 README 配置表明确提供 `lr_schedule=adaptive`：按每个 mini-epoch 的 KL 调学习率，KL 小于阈值的一半时乘 1.5，反向条件则降低。这里是项目实践，不是证明该系数适用于本项目。

**本项目实验。** 保持 Card/Tile KL 硬上限及回滚/丢弃规则，比较固定 Actor LR 与小范围的 KL-aware LR controller；统计每轮实际 unique choice 提交、计划/提交比例、guard 拒绝的 Card/Tile KL 分布、相同新样本和相同时间下的固定对战得分。控制器应看到被拒绝批次，不能仅看通过批次的平均 KL 而误以为步长很小。

**边界。** 降低 LR 可能只增加提交次数，却减慢净策略进步；提高 LR 可能更早撞到 KL。不能用“更多 optimizer steps”当成功指标，也不能为了提交更多批次直接放宽原 KL 门限。

## 优先级 3：PPG 辅助 value 相位，向 Actor 表示传递价值学习

**依据。** [Phasic Policy Gradient，2020/ICML 2021](https://arxiv.org/abs/2009.04416)，正文 Sec. 2、Algorithm 1；[作者官方代码](https://github.com/openai/phasic-policy-gradient)，特别是 `ppg.py` 的 `aux_train()`。PPG 使用独立 policy/value 网络以减少干扰，再在辅助相位给 policy 的共享表示增加 auxiliary value head，用 value loss 学表示，同时用 KL 保持辅助相位开始前的自身策略。

这项自策略 KL 在论文中叫 behavioral cloning loss，**不等于重新模仿外部 BC 教师或把权重退回 BC**。与用户“暂不混 BC”的要求可以区分，但仍是算法改动，当前未启用。

**为什么契合。** 当前 Actor 只有约 7% 原始有效帧有选择并直接获得 PPO 梯度，独立 Critic 在其余状态学习到的表示不会自动传给 Actor。PPG 专门处理“独立网络避免干扰，但失去表示共享”的取舍。PPG 原文证明的样本效率提升发生在 Procgen，不保证在本项目重复。

**实现/风险。** 小型 Actor auxiliary value head、有界 CPU 原始轨迹缓存、按相位冻结自身策略分布，保持 old-logp 与 PPO 行为策略语义；辅助损失结束后仍走原 Card/Tile KL 与对战检查。不要无界保留多轮 GPU 激活或历史数据，避免再次 RAM 爆。额外优化会花更多计算；即使每条样本学得更好，也可能让每秒进展变慢。

## 优先级 4：更长延迟信用的 GAE / decision-time 小实验

**依据。** [GAE 原文](https://arxiv.org/abs/1506.02438) 明确将 λ 定位为偏差/方差的取舍；较小 trace 衰减降低方差，也削弱远期信号。当前 γ=.9995、Actor λ=.993，在每步 .25 秒时，(γλ)^t 的 e-fold 时间约 33 秒；这不是硬截止，自举仍可传递更远收益。

**本项目实验。** 与“攒费—部署—进攻结果”延迟相配的小范围 λ 对照，或者基于 choice 决策时间的 SMDP target 研究，保持相同新增 choice 和冻结胜率评估。MC EV≈.45 表示 Critic 有可用信号，但尚非完美，因此长 trace 可能减少自举偏差，也可能增加方差。

**关键约束。** 当前沿物理时间计算 GAE、只对 choice 取 Actor 梯度是合法基线，不能称其实现有错。若跳过强制 WAIT，必须聚合期间的折扣奖励、用 γ^Δt bootstrap，并正确处理终局/故障/截断；不可筛选 choice 后直接把不等间隔行当等间隔 GAE。沿“决策次数”使用同一个 λ 也会大幅改变实际秒数对应的 trace。

时间抽象的经典参考是 Sutton、Precup、Singh 1999 [Between MDPs and semi-MDPs](https://doi.org/10.1016/S0004-3702(99)00052-1)。本次只通过 Crossref 核实该 DOI 和书目信息；作者 PDF 因证书、其他镜像因 404 未读到，SMDP 项目适配是这里的研究推断，不能说论文已经验证可提高本任务 PPO 收敛。

## 条件候选：ValueNorm / PopArt，而非无条件必加

**依据。** [Learning values across many orders of magnitude，PopArt 2016](https://arxiv.org/abs/1602.07714) 正文提出 adaptive target normalization 和保持未归一化输出的参数修正。[What Matters In On-Policy RL?](https://arxiv.org/abs/2006.05990)，Sec. 3.3 明确报告 value normalization 在部分 MuJoCo 任务有益，在 Walker2d 明显有害；不能从论文推出“总会更快”。Sec. 3.5 还说明数据复用、minibatch、rollout 总量和并行环境数量要联合考虑，更多环境可能更快但样本效率变差。

当前 return std≈.735，并未看到跨数量级的 target 失配；因此先按卡组/阶段诊断 EV、bias 和梯度，而不是因一次 FP16 overflow 就加入 PopArt。若启用，必须保存归一化统计、保持 value/bootstrap 的未归一化语义并处理继续训练的一致性；不能直接改 reward clipping 来凑数值尺度。

## 系统候选：Sample Factory 解决墙钟时间，不自动解决策略收敛

[Sample Factory，ICML 2020](https://arxiv.org/abs/2006.11751)，[作者官方项目](https://github.com/alex-petrenko/sample-factory)，利用单机异步采样、GPU 推理及 off-policy correction 提高吞吐，并支持 self-play/PBT。这对 actor/learner 调度有借鉴价值，但其 >10^5 frames/s 来自论文中具体环境，不能套到当前 122M、128 帧注意力与双 V100。

当前同步 PPO 的 old-logp 和冻结行为策略检查是重要约束。若让采样与 learner 异步并行，必须记录策略版本、限制 staleness 并验证 importance correction；不能把旧轨迹直接当作新 on-policy PPO 数据。这项系统重构也会增加 RAM 压力，优先级低于已在做的局部等价工程提速。

## 如何比较实验

分层 entropy 的小范围 A/B 可以作为探索候选，但现有 Card/Tile 两层都用系数 .001，并不意味着二者需要相同原始熵值。不能仅从 card entropy≈.6、tile entropy≈3.8 判定卡牌层已坍缩或落点层探索过多：合法支持数不同。只读重算保存 rollout 后，181 的 956 choice 中有 815 帧（85.3%）只有两个合法大类别，182 的 1135 choice 中有 849 帧（74.8%）只有两个大类别；choice 平均最大可能 card entropy 分别只有 0.7813 / 0.8161。应先报告按合法支持数归一化的熵、WAIT/PLAY 门控和各层熵梯度范数，再决定是否单独调整 card entropy 系数。增加熵不能给被 mask 的高费动作概率，也不能保证形成连续攒费。该 A/B 是项目假设；没有已核实的论文直接证明其能修复当前高费策略，SAC 的自动温度也不能直接当成已验证的 PPO 改进。[On-policy study](https://arxiv.org/abs/2006.05990) Sec. 3.8 在其连续控制实验中反而没有看到额外 entropy regularization 的明确收益，需按本任务验证。

固定新增有效 choice 数、对局来源和独立对战种子，同时报告相同墙钟预算下的得分；训练集里的高费请求次数、loss 下降和 optimizer step 数只作诊断。至少比较多个种子，保持同一卡组/牌序换边，按配对聚类估计不确定性。不要根据中期有利结果才补种子或更改胜率门槛。一次只改一个因素，不同时改奖励、课程和优化器。

## 资料获取记录

2026-11-08 实际通过公开网络获取：PPG/GAE/PopArt/On-policy study 的 arXiv PDF 并用 pdftotext 读取正文；Implementation Matters 的 arXiv 摘要和 MadryLab README；PPG README、`train.py`、`ppg.py`；rl_games README；Sample Factory 摘要和官方 README；CleanRL `ppo_atari_lstm.py`；[ICLR PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)；Nature AlphaStar 摘要及扩展图说明。临时原始资料位于 `/tmp/ppo_research/`，没有下载数据集或修改运行中的训练。
