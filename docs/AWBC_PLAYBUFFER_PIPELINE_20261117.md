# 实战 playbuffer 接入 PPO

## Value 的含义与更新

PPO Critic继续仅用当前策略的模拟器rollout训练，参数和优化器不会被实战MC直接覆盖。
AWBC每次从当前Critic提取冻结的因果时序特征，只深拷贝MLP头，拟合实战终局回报。
这组影子头估计采集策略的V_mu；不是“已经校正到当前策略”的V_pi。

每个采集Actor指纹单独做3折完整对局交叉拟合；拟合、内部早停、OOF预测和外层验证局
相互隔离。每版本训练局不足9条、终局结果只有一种、OOF MSE未优于同折训练卡组常数
基线2%，或没有同版本外层验证局，该版本Actor权重为0并跳过；不再退回普通自模仿BC。
不因数据来自旧模型而拒绝，也不跨策略版本拟合一个混合价值头。2%只是工程门槛，
不是显著性检验；9条也不是可靠价值估计的样本量保证。

回报只有已验证终局+1/-1/0，以当前PPO gamma按实际tick差折扣，8 ticks为参考周期。
固定A=G-V_mu、w=1+0.5*(clip(exp(A/0.5),0.25,4)-1)，权重范围0.625–2.5。
这些是保守的行为相对回报权重，不宣称是当前Actor无偏的优势估计；它们不能保证纠正规划。
每批重算影子特征/权重，批内固定，不将拟合对象的in-sample残差拿来赋权。
外层验证权重由仅用训练局拟合的折模型集成预测得到；验证回报只用于计算固定的
验证目标，不参与拟合、内部早停或OOF价值门槛。Actor以固定加权验证NLL选择候选，
普通未加权NLL只作诊断，避免把更像旧策略误当成纠偏成功。

## 回放缓冲区

入口`src/cr_rl/training/awbc_playbuffer.py`，只索引已有实战会话，不操控或中断本地采集。
只接收完整、带可信result.json、通过load_episode的policy_sample提议轨迹。
更新：使用load_episode(...,purpose='awbc')的不规则时间合同；不超过10秒的真实间隔
保留原tick，间断边界Actor标签屏蔽。严格PPO导入仍使用0.8秒门槛，两者不混用。
完整历史始终保留，已证实未执行的开场动作、stale_action_skipped和spell veto不作Actor标签。
stale提议保留原动作和logp，不改写成WAIT。非WAIT动作必须有明确的提交/确认链，或
可校验的未执行记录；孤立、冲突、重复事件及无解释的未提交出牌会拒收。
不把实战旧logp传入PPO ratio，也不拿模拟器BC数据伪装成实战回报。

- 每50次PPO更新检查一次；默认最少12条未使用的训练局、至少2条验证局。
- 按采集版本组合可拟合批次：每个选中版本至少9条训练局，并有同版本验证局；
  优先补入上次跳过批次之外的新数据。Value和有效标签过滤之后，仍须有至少12条
  新训练局与至少2条可信验证局才能进入Actor更新，不能用旧局或零权重局凑数。
- 每批最多24局，训练局每局最多参与2个批次；同一批最多2个Actor epochs，lr=1e-6。
- 默认活跃索引为最近256个有结果的会话，只逐局检查和按批载入数组。
- 不复制原始日志；缓存仅为小索引、当前批次特征和更新报告。特征计算后释放mmap并回收。
- 以行为指纹、原始观测和动作内容去重，整局hash固定80/20划分；验证局不转为训练局。
- 已消费ID保留小型tombstone防止恢复后重复入库，旧日志本体由采集侧管理，不主动删除。
- 消费次数/最后处理update写入完整PPO checkpoint，以它为恢复依据；sidecar报告不是提交凭证。
- 迁移后refresh会按内容匹配重新绑定数据路径；修改已收录局的结果必须另外审计。

## PPO 中的更新边界

完整PPO优化结束后、历史快照/本轮checkpoint之前调用AWBC，不在rollout中途改权重。
使用现有Actor原位更新WAIT/slot/ability/wait_hand_gate读出；实体、时序和落点骨干冻结。
PPO当前Actor、已采集实战策略不需要相同；初始参考概率从当前Actor重新计算。

每个候选epoch检查实战验证局KL，并检查本轮PPO最多4条轨迹的KL，均不超过0.01；
PPO探针扫描全部segment，优先覆盖不同对局、对手角色、卡组和动作类型，避开只含
强制WAIT的片段，不再固定取开头两条。
对连续局prefix仅推进编码器，不计标签/KL。没有可用PPO选择状态时本次不更新。
固定加权验证NLL未改善至少1e-4或KL超限就结束，回滚到本批最佳或epoch0；
异常也回滚读出与requires_grad。
这些是离线漂移守卫，不是胜率验收，不能替代后续独立对局。

接受后仅清理改动读出的PPO Adam动量；保留其他Actor和全部Critic优化器状态。
同步behavior和双卡Actor副本，标记记忆失效，下一轮从完整前缀重新burn-in。
即使候选被回滚，也同步/重建已变化的memory generation。
实际提交过Actor优化步骤才消费参与训练/验证的局；零权重、门槛不足或零优化步骤
不消耗训练复用额度。实际尝试后即使被KL/验证拒绝，也会记账，避免反复试同批数据。
记录数据门槛跳过的批次和当时的活跃数据ID，数据未变时不反复拟合影子头。
实际尝试或新增跳过状态均强制保存完整PPO checkpoint，使模型与账本同时提交。
当场W&B记录awbc/*，完整报告awbc_latest_report.json包含Value分组/OOF与候选epoch指标。
报告另含实际训练/验证索引、WAIT/出牌/技能标签数、终局分布、权重范围和frame ESS。
frame ESS的单位是相关帧，不是独立对局数。

支持单进程单GPU及单进程双GPU（包括后续V100）；暂不支持torchrun多进程AWBC hook。

## 开启

在原PPO启动/恢复命令后增加：

```bash
--awbc-sessions /content/drive/MyDrive/cr_rl_ppo/sessions \
--awbc-interval 50 --awbc-min-new-games 12 --awbc-batch-games 24 \
--awbc-epochs 2 --awbc-value-epochs 20 --awbc-learning-rate 1e-6 --awbc-max-kl 0.01
```

已有playbuffer的checkpoint恢复时默认继承其AWBC设置；显式CLI值优先。
目录缺失/新局不足时继续PPO，不启动单独的AWBC进程，不等待正在打的对局。
独立`train_live_awbc.py`仍可用，但输出不能直接当完整PPO恢复文件覆盖。

## GPU smoke 的严格含义

`launch_prepared_pipeline.py --stage ppo-smoke --awbc-playbuffer`现在会显式覆盖为
2次PPO更新、AWBC interval=1，并传入`--require-awbc-smoke`。必须真的进入Actor
优化（接受或回滚均可），随后再成功完成一次有Actor优化的PPO更新，才算通过。
结果写入`awbc_smoke_report.json`；缺数据或Value门槛跳过会失败，不能冒充hook验收。
这个频率仅用于smoke，正式训练仍以50为默认；数据不合格时先做纯PPO smoke。

## 验证界限

本地小模型验证：旧策略数据训练当前Actor、按版本交叉拟合隔离、Critic参数不变、
Actor骨干不变、KL拒绝/异常回滚、PPO hook同步/动量清理、重复调用防护、buffer恢复、
prefix掩码不影响后续记忆、W&B字段。尚未在真实大模型GPU上完成整个hook吞吐/数值验收，
也未因本次改动启动正式训练或修改当前实战采集策略。

修复前的历史验收：Colab CPU运行上述playbuffer/独立AWBC/遥测/备份，以及PPO跨更新burn-in与
behavior同步回归，共20项通过（46.90秒）。测试日志已保存到Drive
`MyDrive/cr_rl_ppo/preparation/awbc_playbuffer_validation/`。

2026-11-17本地复查及修复后的测试结果、文件清单和实际数据门槛，见
`PPO_AUX_REVIEW_AND_FIXES_20261117.md`。本地修改不等于已同步到正在运行的Colab。
