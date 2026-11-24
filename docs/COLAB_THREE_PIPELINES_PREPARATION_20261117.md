# 单 checkpoint 云端准备与三条训练管线

> 2026-11-17 划分更新：52 片筛选仍按原始约 20% Validation 生产 v1 分片；
> **全量完成后自动将可重新划分的约一半 Validation 转为 Train**，最终非 Heldout
> 数据约 90% Train / 10% Validation。固定 Heldout、历史 registry 不变；
> v1 全局分区快照归档，顶层 split_registry.json 切到 v2。
> 25/52 片同批对局的预期变化：Train 1,115→1,279、Validation 310→146、
> Heldout 161 不变。无需重跑源扫描或原生重建，且不改变 training_ready=false。
> 详见 QUALITY_BC_VALIDATION_REBALANCE_20261117.md。

## 当前实例与文件

- CPU session：`crrl-cpu-prep-1117`，本次已通过 Google 授权挂载 Drive。
- Drive：`MyDrive/cr_rl_ppo/ppo_latest.pt`，4,375,624,553 bytes。
- 核验 SHA256：`b611b9af9d7a9f28bfc4272097976653975ebd28f37267308c4ac9bab710b124`。
- 确认 U929、architecture-v3、8 ticks；Actor 指纹与本地实战采集起点一致。
- 云端工作区 `/content/crrl_training`；代码备份 `MyDrive/cr_rl_ppo/preparation/crrl_training_code.zip`。
- 本地 notebook：`notebooks/colab_three_pipelines.ipynb`。
- 本轮只准备 CPU，未开 GPU、未启动正式训练、未更换 Torch。

## PPO：先恢复基线，再加入辅助 BC

`prepare_single_ppo_checkpoint.py` 从完整 checkpoint 提取一份固定 Actor，锚点明确标为
“起点 PPO”，不是之前 BC 专家。完整 checkpoint 保留原 Actor、Critic 和优化器。
新运行目录创建新对手池，初始没有历史。请求比例为 current 50%、history 25%、anchor 25%；
历史为空时按现有 RoleMixture 逻辑退回锚点，因此实际初始为 current 50%、anchor 50%。
随后每 20,000 learner samples 积累历史，池容量先设 4，保留锚点并回收旧 Actor。
不把同一 Actor 的复制品伪称为过去历史，不随机破坏权重。

已生成 GPU smoke argv；它不是经过吞吐调优的正式配置。保留原 checkpoint 的 FP32
Actor 稳定性选项与奖励系数，不在迁移时悄悄改变精度语义或 reward。
单 GPU 映射 Actor/Critic/opponent 到 cuda:0；V100 双卡另行做设备与吞吐验收。

下列部分仍需后续实现/验收，不能当作已启用：

- 仅对冻结对手应用温度变体，Learner 保持原采样；变体须有独立行为身份与日志。
- 每约 10 次 PPO 更新插入 8–16 条完整座位序列的小剂量 BC。
- BC 位于全部 PPO 优化后、下一次 rollout 前；重建受权重变化影响的时序状态、同步
  behavior replica；不得用 BC 改权重后继续优化旧 rollout。
- 独立 BC optimizer、计数/RNG/断点恢复、KL 守卫与 W&B 汇总，先小规模对照。

以上数字是待验收起点，不是已证明最优。辅助 BC 的数据目前不够，正式混合训练应关闭。

## 高质量 BC：服务器下载与原生重建

固定来源 `VanguardX101/IL_Replay` revision `059d43a02138a34b1b3009cc2acc7630fb99a638`。
源 manifest 有 52 个分片；本次只在 CPU 实例下载首个 17,432,059 byte 分片，SHA256
`9f6127099b57d91eef64da308873a2635f72b27ae8fd9d802fc7d1767363b656`。
不要求用户上传 BC 数据、不下载到 Windows。

`build_quality_bc_native.py` 逐批读取 parquet，逐局在 8 tick backend 中重建，不把旧
0.25 秒数组伪称为新原生轨迹。推断初始轮转只是兼容候选，不声称恢复真实手牌。
禁止手牌/落点修改、额外等费、遗漏、未确认执行；要求整局源动作保留且至少 20 次。
允许 0–7 tick 的向后网格对齐（最大 0.35 秒），因为 0.4 秒原生网格本身存在量化；
不能把旧 5 tick 的 0.2 秒门槛机械套到新网格。额外 FIFO 延迟仍拒绝。
源动作结束后的 pending drain 保留为时序上下文，但标 GENERATED_WAIT_TAIL，不能当 WAIT 示范。

按 canonical identity 去重；保留传入的原 split registry（`--prior-split`）。没有旧 registry
时按固定 hash 划分开发验证，并将配置中的 eval decks 整局留出。新划分不能冒充旧训练
从未见过的独立测试集；若恢复旧完整 registry，必须先检查冲突并重新发布数据版本。

64 条源记录实测只得到 1 局通过，且为 validation，不能转成 train 凑数。
扩大到 512 条的同源扫描用于估计可用量，不自动放宽质量标准。
该试验最终通过 8 局（train 6、validation 2），共 6,476 帧。
用户随后要求重新拉完整来源，已启动 `colab_full_quality_bc.py`：先拉齐全部 52 分片并逐个
校验源 SHA，再逐片扫描所有源记录、原生重建及质量筛选。这个任务不止扫描 512 条，也
不以试验中的 8 局为最终训练池。状态写入 `public_bc_full/state.json` 并同步 Drive。
源文件只放 CPU VM 临时盘；每片完成即持久化候选与审计，停止/重启支持按已完成分片续跑。
生成 dataset manifest、逐局审计、split、拒绝原因；默认 training_ready=false。
原局状态一致性、玩家水平和示范战术质量尚未证明。必须完成模型输入契约核验与人工/回放
抽查后再发布训练池；不能靠胜局、Ranked 或低重建偏移自动等同专家。

## AWBC：独立实战纠偏分支

更新：现已实现PPO原位playbuffer hook，详见 `AWBC_PLAYBUFFER_PIPELINE_20261117.md`。
它接收旧采集策略的数据，影子Value按采集版本交叉拟合，PPO Critic不覆盖；达到新增局数
门槛后在更新边界训练读出，检查实战/当前PPO状态KL，随后同步副本并写完整checkpoint。
当前尚未完成大模型GPU hook验收。专家BC hook和温度变体仍未启用，勿混为一项。

已有 `train_live_awbc.py`、`run_live_awbc_colab.py`、`colab_live_awbc.ipynb`，详见 LIVE_AWBC_COLAB.md。
使用当前 Actor 采集、有真实结果且通过完整性校验的实战局，至少 12 局；仅有 checkpoint
不能构造这条数据。新账号目前只有模型，没有 sessions，因此 AWBC 暂不能开训。

冻结 Critic 骨干、交叉拟合价值头；样本/结果支持不足、OOF 检查不通过或缺同版本验证局
时跳过该版本Actor更新，不再退回普通自模仿BC。Actor保守更新读出，以固定优势加权的
留出NLL/KL早停。AWBC独立入口输出不是包含原Critic/optimizer的完整PPO checkpoint，不覆盖
`ppo_latest.pt`；回流 PPO 必须另做明确的 Actor 权重迁移与 optimizer/behavior 状态处理。
不拿专家回报直接覆盖当前 PPO Critic。

## 验收和空间

本地质量筛选、轮转、AWBC/恢复/备份相关 13 项测试通过；云端 checkpoint 与真实原生
BC 重建已执行。GPU 训练吞吐、PPO 单卡更新及正式辅助 BC hook 尚未验收。
临时原始 parquet/缓存留 `/content`；Drive 持久化代码、最终数据分片/审计和状态。
只有一个最新完整 PPO checkpoint；对手保存 Actor，容量有界；不持久化所有 rollout。
预热/筛选不启动 GPU，后续切换 GPU 后重用 Drive 代码包与 checkpoint。

## 当前追加验收

2026-11-17：全部52源分片已下载并校验（252,238条记录，825,394,311 bytes）。
11个已完成分片的增量catalog共有723个唯一候选，590,082帧，划分为train497、
validation148、heldout78。该数量不是最终全量结果。
字段顺序、词表names计数及嵌套词表中的data_version/content_sha与当前Actor一致。
两条训练座位各取从开局连续32帧做CPU前向，输出[1,32,2307]且合法动作logits全为有限值。
报告位于Drive preparation/full_quality_bc/model_contract_audit.json。
catalog只消费producer state.json已提交的分片，避免同步中的半成品；相关回归通过。
AWBC playbuffer另有20项Colab CPU回归通过。正式GPU训练未启动。

## 2026-11-17 本地代码复查追加

上述云端数据计数和20项测试为当时的运行记录，不代表已部署后续修复。
本地已修复实战执行标签、Value失败后的自模仿、加权验收、按版本组织批次、实际消费
记账、PPO探针和恢复时对手池参数漂移；详见`PPO_AUX_REVIEW_AND_FIXES_20261117.md`。
带`--awbc-playbuffer`的prepared smoke现要求2次PPO更新、AWBC interval=1、实际Actor尝试
以及随后一次PPO更新，不能再以未触发AWBC的1次PPO更新作验收。
训练前须同步这些本地代码改动；专家BC hook仍待完成数据质量验收后接入。

同日经DevSpace调用Colab CLI再次只读确认：`crrl-cpu-prep-1117`仍为CPU/Standard，
后台筛选在运行，进度为20/52片、1,211局跨片去重前候选；VM和Drive state一致。
当前配置的Drive/VM实战sessions目录均不存在，尚不能执行严格PPO＋AWBC smoke。
没有申请新GPU、上传覆盖源码或中断筛选；完整元数据保存于本地
`artifacts/ppo_aux_review_20261117/colab_status.json`。
