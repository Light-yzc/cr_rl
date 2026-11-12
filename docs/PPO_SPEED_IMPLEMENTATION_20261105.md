# PPO 工程提速实施记录（2026-11-05）

用户在阶段profile后同意按顺序实施。基线与热点见[PPO_PROFILE_20261105.md](PPO_PROFILE_20261105.md)。
本次不改300场训练/48场短测预算、LR、有效batch64、Critic2遍、历史128帧、奖励和攒费门控。
不开subagent，不公开推送运行细节。

## 代码与回归

`FrozenInference.sample()`及`sample_incremental()`增加可选`collect_stats=True`参数；
默认继续返回完整诊断，保证已有评估/脚本调用兼容。PPO采样仅在每64步和最后一步为当前Actor
计算entropy/max-prob，冻结对手不计算这两项未使用诊断。动作与selected log-prob仍正常采样和保存。

已有`--forced-wait-fastpath`仅跳过整批动作mask只有WAIT合法时的Actor前向；保留完整形状的
Categorical抽样以维持随机数消耗。历史原始帧仍逐帧记录，下一次真实选择会补齐帧编码。
Critic价值保持逐步计算。候选不会将强制WAIT行混回Actor优化。

四项新增组合测试覆盖增量/旧窗口、真实选择/强制WAIT，要求动作、selected log-prob和抽样后RNG
逐位相同，并明确禁止quiet路径调用entropy。Linux全套265项通过，改动文件Ruff通过。

## 真实L4隔离验收

独立目录`/content/crrl_speed_20261105`，生产目录保持`/content/cr-rl-choice-train300`。
在同一trainer的采样初期使用pidfd短暂停放；独立GPU进程结束或失败后`finally`恢复同一trainer。
停放时间逐次写回执，不能拿含停放的轮次声称生产提速。

首次尝试在读取模拟器时失败：独立副本没有`.cache/vendor/crforge`，尚未进行GPU推理或优化；
trainer自动恢复，停放3.35秒。失败日志和回执保留，第二次通过只读缓存路径修复重试。

第二次检查：真实122M权重、连续384帧原始状态/合法mask、128帧历史，比较原路径与fastpath的
完整概率、selected log-prob、CUDA RNG、实际采样动作与跨局缓存补齐。沿用BF16验收阈值
logp<=.03、TV<=.01，要求抽样后CUDA RNG精确一致；不放宽生产old-logp<=.05保护。

性能对比采用相同128帧warmup后256个测量步，三种路径各3次交替顺序：原路径带统计、
原路径不计算统计、WAIT fastpath且不计算统计。它只度量冻结Actor推理，不冒充整轮吞吐。

Critic使用同一真实权重、按实体密度选出的128个真实满历史状态、固定目标残差、零LR，
测microbatch8/16/32，均保持有效batch64、两遍、固定样本顺序和4次optimizer step。
记录显存、输出差和OOM；仅吞吐测试不够支持迁移非零LR优化数值，生产先保留8。

## 接入与恢复边界

通过上述真实验收后，在新一轮刚开始、父checkpoint和所有应有短测已落盘时停放旧控制器/训练器。
从完整父checkpoint精确核对Actor、Critic、两个Adam、warmup、torch/NumPy RNG和局数counter，
再恢复同一训练目录。此次进程重启产生一次明确的模拟器重置；父checkpoint未完成局和下一轮
刚开始的未发布片段不计为完整对局。保留所有已经发布的训练和评估证据。

新的包装仍调用原短测/6→12扩容回调，保留release guard与progress tracker；最终CPU验真器增加
经过独立完整恢复回执认证的这一处边界，不放松其它连续性检查。

首个新checkpoint另有独立CPU验真：全模型/Adam有限值、完整manifest/SHA、合法on-policy动作、
choice-only Actor、Critic2遍/有效64、两个optimizer步数连续、LR不变、old-logp原阈值和局数counter。

## 后续范围

先观察这两项推理改动的真实采样耗时。Critic如要改microbatch，须补非零LR数值与完整update验收。
后台Drive发布需要不可变快照、有界队列、与固定短测checkpoint绑定的发布屏障和失败恢复，
现有控制器假定每轮同步发布，不能只启动上传线程就启用。本阶段先保留同步完整发布，后续再独立落地。

本地脚本与证据：`artifacts/speed_implementation_20261105/`。
私有Drive：`runs/122m_choice_train300_20261105/speed_implementation_0001/`。

## 第二次真实验收结果与接入

| 固定256步Actor推理 | 三次中位耗时 |
|---|---:|
| 原路径，带统计 | 7.0034秒 |
| 原路径，关闭统计 | 6.9894秒 |
| WAIT快路径，关闭统计 | 1.8478秒 |

这批推理约3.79倍，单独按需统计约0.2%的差异接近噪声。该片段384步中有310步整批强制WAIT，
比生产7轮调用比例更高，而且不包含Critic/模拟器/Drive，所以不能外推整轮3.79倍。
384步中的采样动作和CUDA RNG均相同；最大selected log-prob差0.0036774、TV0.0018673，
通过原0.03/0.01门限。这不是全状态概率逐位相同的证明。

| Critic microbatch | 固定128样本/两遍中位秒 | 本基准进程峰值显存 |
|---|---:|---:|
| 8 | 7.8019 | 6.13GiB |
| 16 | 8.2237 | 10.15GiB |
| 32 | 三次OOM | 未完成 |

16比8慢约5.4%；32在驻留原trainer的共存显存条件下OOM，不能说单独生产进程必然OOM。
因此没有改变Critic batch。该隔离基准停放原trainer175.23秒后自动恢复，另有首次失败3.35秒，
均在第43轮完整发布后的第44轮片段；旧片段未用于新生产轮耗时或局数。

21:20–21:21 CST从43号完整checkpoint接入新路径； source43 SHA：
`ef1297a8e5ba898854446874564ea290333b870302c3c592d0844cdfbbfdf0ac`。
完整CPU恢复验证Actor/Critic、两个Adam、torch/NumPy RNG、warmup、counter精确一致；
恢复时Actor562步、Critic8574步、累计Learner171409、扩容阶段94局加原30局=124/300。
第43轮123场阈值实际124场，对stable6短测2胜2负；累计16/48评估，下一次145场对parent10。

新增明确恢复边界43→44；原checkpoint未完成局步数[88,15]及下一轮未发布初段丢弃，未计入局数。
控制器78693、trainer78728、最终CPU验真78692、首轮CPU验真78694；guard49777/tracker49954保持。
包装入口`/content/crrl_speed_20261105/runtime_train.py`；仍使用原curriculum_0001的短测和在线扩容回调。

恢复包装的初始校验绑定43号，本轮若中断不能盲目重跑；需针对最新完整checkpoint生成新恢复回执、
登记实际模拟器边界，再调整包装初始校验。原`/content/train300_curriculum/runtime_train.py`仍绑定31号，
不再是当前入口；不要裸启动旧命令或重复启动第二个trainer。

## 首轮生产结果与独立验真

第44轮已发布，21:31 CST独立CPU验真通过：98份manifest文件/SHA、模型/两个Adam有限值、
合法on-policy动作、choice-only Actor、优化器步数连续及原old-logp保护均通过。
该轮跳过2,571个当前Actor强制WAIT行，4,144个有效Learner状态全部继续训练Critic；
Actor735个choice样本，实际24步；Critic130步，恰好两遍、有效batch64。
old-logp最大差0.00560474，原阈值0.05未改；0故障。

| 第44轮实际阶段 | 秒 |
|---|---:|
| 采样，含最后bootstrap等收尾 | 76.55 |
| Critic，按事件边界近似 | 270.56 |
| 全部优化，含Actor、准备和汇总 | 342.01 |
| 本地序列化 | 12.53 |
| 日志与局面统计 | 1.18 |
| Drive同步发布 | 36.38 |
| **整轮，含发布** | **468.66（7.81分钟）** |

有效新Learner样本8.84/秒。此前32–38轮平均523.43秒、7.34样本/秒，但两者局面、
对手、自博弈座位比例和Actor步数不同；这些只是生产观察，不能作为同轨迹受控加速比。
第44轮Actor实际24步，旧基线平均约12步；没有靠减少优化遍数或步数获取此次采样提速。

第45轮未重启模拟器的连续采样为87.66秒（不含最后少量收尾），进一步说明改善不限于首次恢复。
仍需更多轮次观察，不能保证每轮稳定某个加速倍数。最新已发布44号，126/300完整训练局，
累计175553 Learner；第45轮优化继续，评估已用16/48，下一次>=145场对generation10。
该次短测后仍按原计划同进程6→12扩容。
