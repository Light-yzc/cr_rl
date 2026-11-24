# PPO＋专家 BC＋实战 AWBC：复查与修复记录

日期：2026-11-17（北京时间）
项目：`G:\Code\cr_rl`

## 结论与当前状态

这条路线可以继续：PPO负责当前策略的模拟器收益，严格验收的人工示范补充动作监督，
实战AWBC只做小剂量的行为倾向纠偏。模型架构保持architecture-v3，0.4秒/8 ticks契约不变。
本次修复了会影响辅助训练有效性、数据消费和验收可信度的具体问题。

- 本地代码修复已落盘，受支持的Python 3.13环境下联合CPU回归 **78项通过**。
- 本机5局带结果实战记录重新校验后只有1局合格，尚未满足playbuffer的12条新训练局＋2条验证局门槛。
- 大模型GPU上的PPO＋AWBC完整衔接仍须单独验收；CPU回归没有证明实战胜率提升。
- 专家BC hook仍未接入，数据筛选候选不能直接视为高质量专家训练池。
- 本地修改不会自动更新已有Colab工作区或Drive代码包，云端试验前须同步版本。

## 一、本次具体改了什么

| 问题 | 修复后的行为 | 主要文件 |
|---|---|---|
| 明确因状态过时而未发送的出牌仍可能成为Actor标签 | 验证stale事件与决策的关系，排除其策略loss，保留原动作、logp和全部时序上下文 | `src/cr_rl/client/rl_recording.py` |
| 无解释的未提交动作和重复决策事件可能漏过 | 每个非WAIT执行动作须有完整提交/回执或明确未执行证据；正整数decision ID必须唯一，冲突/孤立/重复事件拒收 | 同上 |
| 价值估计失败后退回普通BC，继续模仿自己的旧行为 | 不可靠版本Actor权重为0并跳过，不再使用无条件自模仿fallback | `src/cr_rl/training/live_awbc.py` |
| 用未加权旧行为NLL选模型，可能拒绝有效的相对纠偏 | 使用批内固定的优势加权holdout NLL，并保留实战/PPO状态的KL守卫；普通NLL仅诊断 | 同上、`awbc_playbuffer.py`、`scripts/train_live_awbc.py` |
| 全局抽样将多个版本拆散，各自都不够拟合 | 按版本组织批次；默认24局预算枚举所有单/双版本组合，保留新数据优先与复用上限 | `src/cr_rl/training/awbc_playbuffer.py` |
| 初选12条新局，Value过滤后真正可信的新局可能不足12 | Actor更新前再次检查最终可信集合的新训练局数与holdout数，不用旧局、零权重局凑数 | 同上、`state_selfplay.py` |
| 跳过的局被算作训练消耗，或旧失败批次反复重试 | 精确记录实际优化的轨迹；数据门槛跳过不消费，活跃数据ID未变不反复拟合 | `src/cr_rl/training/state_selfplay.py` |
| 零优化步后的防御性恢复仍可能改变memory generation | 单独返回memory_invalidated；即使不消费数据，也按需同步副本并重建记忆 | 同上、`awbc_playbuffer.py` |
| PPO KL探针固定取前两段，可能全部是强制WAIT或同一局 | 扫描全部segment，最多选4条覆盖不同对局、角色、卡组和动作类型的轨迹，完整prefix保留 | `src/cr_rl/training/state_selfplay.py` |
| 单次PPO smoke可能完全没触发interval=50的AWBC，却被当成验收 | AWBC smoke强制2次PPO更新、interval=1，必须实际尝试Actor优化并再完成一次PPO更新 | `scripts/launch_prepared_pipeline.py`、`scripts/train_selfplay.py`、`state_selfplay.py`、`scripts/prepare_single_ppo_checkpoint.py` |
| 普通resume可能用CLI默认值覆盖已配置对手池 | 恢复role_league、50/25/25等已保存比例、容量和快照间隔；显式CLI优先 | `scripts/train_selfplay.py`、`state_selfplay.py` |
| 只读审计仍使用旧的“没有导入器”等判断 | 以当前load_episode为准，区分记录适用性与on-policy PPO资格；坏文件以拒收原因报告 | `scripts/audit_nulls_rl_readiness.py` |

### 1. 执行标签的严格语义

`stale_action_skipped`表示提议因状态过时没有发送。它不是成功出牌，也不是模型选择WAIT。
因此保留原提议和对应概率，只把该帧的`policy_loss_mask`设为false；记忆仍按完整观测推进。
有正常多动作选择的主动WAIT继续保留。已有spell veto和可证实未执行的开场超时规则继续生效。

新校验覆盖：stale指向不存在/尚未发生的decision、动作不匹配、重复stale、stale与提交/回执冲突、
未解释的非WAIT未提交、重复decision事件以及bool/None/list等非法decision ID。
同一ID的完全相同重复事件也拒收，不能让字典覆盖掩盖事件数量。

在这次实际检查的日志里，有stale事件的局已经先被观测缺口规则拒收。
这证明存在需要修复的代码路径，不证明已有合格缓冲区实际被这些stale标签污染。

### 2. Value门槛失败时跳过Actor

每个采集Actor版本独立处理，不跨版本混合拟合一个价值头。以下情况不产生Actor监督权重：

- 该版本训练局少于9条；
- 训练局只有一种终局结果；
- OOF MSE未比仅由相应训练折计算的常数基线改善至少2%；
- 没有同采集版本的外层holdout。

最终Actor集合还必须有至少2条可信holdout。PPO playbuffer另外要求价值和标签过滤后
仍有至少12条从未参与过Actor训练的新训练局。整个集合不足时跳过，不为了触发hook降低门槛。
独立AWBC入口原有的“至少12条总输入局”是另一个计数，不能与playbuffer的12条新训练局混淆。

影子头继续只拟合冻结Critic的时序特征，估计采集策略的MC V_mu，不冒充当前策略V_pi。
它不覆盖PPO Critic。实际tick间隔折扣保持`gamma ** (step_ticks / 8)`，默认gamma=0.9995。

可信版本仍使用原有保守权重：

```text
A = G - V_mu
w = 1 + 0.5 * (clip(exp(A / 0.5), 0.25, 4) - 1)
w范围：0.625至2.5
```

这些权重始终非负。低回报动作只是相对减轻监督，并不等于已经获得可靠的“这个动作是错的”标签。
终局输赢仍有方差和信用分配问题；9条样本、2%门槛、较低学习率都不是统计可靠性的保证。

### 3. 加权验证目标与回滚

训练局的权重来自严格OOF预测。外层holdout由仅使用该版本训练局的折模型集成预测，
其结果标签不参与拟合、内部早停或OOF门槛，只用于形成固定的外层加权验证目标。
权重在本批Actor训练期间固定，不随每个epoch重新拟合。

PPO hook每个候选epoch都要求固定加权holdout NLL相对最佳候选至少改善1e-4，
同时实战验证状态和当前PPO探针上的完整合法动作分布前向KL均不超过配置上限，默认0.01。
未加权NLL可以上升，仍会记录，但不再单独否决加权目标的改善。
没有通过的候选恢复到本批最佳或epoch0；异常恢复基线和requires_grad。

这些都是开发选择与漂移指标。4条PPO探针不覆盖整个状态分布，平均KL也不是逐状态的最坏情况保证。
独立入口的默认8 epochs、KL 0.02与PPO hook的2 epochs、KL 0.01保持各自语义，不混为同一配置。

### 4. 批次、实际消费与checkpoint

默认最多24局，训练轨迹最多使用2批，Actor lr=1e-6、最多2 epochs。
选择器要求每个被选版本至少9条可复用训练局且有同版本holdout；全局至少2条holdout。
默认容量256下，全枚举单/双版本组合的数量有界；自定义更大预算保留有界多版本候选。

新增反例覆盖：A有1条新局＋8条已用训练局、B/C各10条新局；A被优先排序时，
不能因A+B/A+C各只有11条新局就遗漏可行的B+C组合。

`actor_attempted`只表示确实提交了优化步骤；`actor_train_indices`只列真正优化的训练轨迹。
实际尝试后即使因KL或验证目标拒绝，也会计入使用，避免重复试探同一批数据。
数据/Value门槛跳过或所有优化步因数值条件跳过，不消耗训练复用额度。

跳过记录保存该时刻全部活跃数据ID，数据未改变时不反复拟合；有新合格训练/验证数据后可重试。
实际训练消费或新增跳过状态会触发完整PPO checkpoint提交。
接受更新后只清除已修改读出的PPO Adam状态，其他Actor及全部Critic优化器状态保留。
策略副本同步和记忆重建与是否消耗数据分开处理，避免零步回滚后的generation不一致。

报告增加实际训练/holdout索引、跳过原因、终局分布、WAIT/部署/技能标签数、权重范围与frame ESS。
ESS明确标记为相关帧，不是独立对局数。

## 二、保留的主训练设计

本次没有改变模型架构、U929初始化方案、PPO收益定义、0.4秒采样契约或现有实机采集控制器。
已有正确边界继续保留：PPO的old logp和参考价值在优化前固定，辅助更新位于全部PPO优化结束后、
下一轮采样前；实战旧策略概率不进入当前PPO ratio；完整历史/prefix用于记忆重建。

对手池继续使用固定U929起点锚点、历史回收和空历史回退。容量4包含锚点，因此最多约3个历史槽位。
锚点仍称“起点PPO”，没有改称专家。温度变体没有实现。

50/25/25是对局抽样比例。当前策略自对弈双方都产生learner样本，对历史/锚点对局通常只有一侧产生
learner样本；在长度、有效帧数量相近时，训练样本占比约为66.7/16.7/16.7。
这不是自动需要修改的bug，应记录实际有效样本贡献再调比例。本次没有擅自改50/25/25。
初始当前策略与固定锚点同为U929，空历史也回到锚点，池配置本身不会立即创造行为多样性。

## 三、本机真实记录复查

范围：本次检查时`artifacts/nulls_live`中77份manifest，60份标记采集完成，
其中5局同时有结果记录。它们不能代表用户在其他设备或云端的全部数据。
使用修复后的实际`load_episode`与readiness审计器重新验证，结果如下：

| 会话 | 帧数 | 修后结果 | 原因/可用标签 |
|---|---:|---|---|
| `play_20261117_001127_001` | 68 | 拒收 | 观测间隔超过0.8秒 |
| `play_20261117_001251_060` | 98 | 拒收 | 未确认或不明确的动作执行 |
| `play_20261117_002235_692` | 608 | 拒收 | 观测间隔超过0.8秒 |
| `play_20261117_002756_739` | 146 | 通过记录契约 | 26个Actor选择标签；4帧已证实未执行的开场动作被排除 |
| `preset_pilot_20261117/20261117_005354_slot2_aa82ad` | 441 | 拒收 | 观测间隔超过0.8秒 |

完整报告：`artifacts/ppo_aux_review_20261117/live_recording_readiness.json`。
这1局通过只代表输入可被AWBC读取，不代表一个训练批次已成立，也不代表可直接用于当前on-policy PPO。

时间日志提示两个不同问题：有1.3–1.6秒的推理尖峰，也有读入/推理较快但状态年龄达到约4.09秒的
记录；最大的相邻观测缺口为81 ticks，即4.05秒。另有较低状态年龄但仍出现1秒间隔的记录。
因此不能把全部缺口直接归因于GPU推理。下一步应围绕读入、推理、提交/确认、调度和日志写入的
实际时间点定位延迟来源；当前采集器已有预热，不应把“缺少预热”当成已证明根因。
本次没有放宽0.8秒门槛，也没有把缺失的轨迹伪造补齐。

## 四、验收记录

### 本地CPU联合回归

实际环境：Python 3.13.1；Torch 2.7.0.dev20250119+cu126；pytest 8.4.2。
项目要求Python >=3.12,<3.14，本次最终验收使用受支持的3.13环境。
pytest、JPype、Gymnasium、PettingZoo等缺失的测试导入依赖安装到
`artifacts/aux_review_testdeps_20261117`，没有替换生产Torch或修改项目依赖锁定。

| 测试范围 | 用例数 | 结果 |
|---|---:|---|
| 实战记录契约 | 22 | 通过 |
| readiness审计 | 12 | 通过 |
| 既有Nulls监督导出 | 3 | 通过 |
| AWBC权重/事务/缓冲区/独立入口 | 29 | 通过 |
| PPO集成、恢复参数、smoke严格条件 | 11 | 通过 |
| 既有Colab备份完整性 | 1 | 通过 |
| **合计** | **78** | **26.39秒，退出码0** |

日志：`artifacts/ppo_aux_review_20261117/cpu_regression.txt`。
9个修改的生产Python文件通过AST语法解析；相关文件`git diff --check`通过。
测试使用CPU小模型和合成完整记录；价值门槛两条CLI路径在fixture中受控触发，
独立测试另外验证OOF隔离、真实梯度更新、回滚和加权目标，不把fixture的通过当成真实数据有可靠优势。
测试没有运行GPU训练、JVM对局或ADB动作。

可在项目目录复跑：

```powershell
@'
import os, sys
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
sys.path[:0] = ['artifacts/aux_review_testdeps_20261117', 'src', '.cache/vendor/crforge/python']
import torch, pytest
torch.set_num_threads(1)
raise SystemExit(pytest.main([
    'tests/test_live_rl_recording.py', 'tests/test_live_readiness.py',
    'tests/test_nulls_training_export.py', 'tests/test_live_awbc.py',
    'tests/test_awbc_playbuffer.py', 'tests/test_live_awbc_cli.py',
    'tests/test_ppo_auxiliary_integration.py', 'tests/test_awbc_colab_backup.py',
    '-q', '-p', 'no:cacheprovider',
]))
'@ | & 'F:/python/python.exe' -B -
```

### GPU smoke必须证明什么

只启用AWBC目录、成功退出一次PPO更新，不足以证明hook能工作。
现在带`--awbc-playbuffer`的prepared smoke要求：

1. 实际执行一次AWBC Actor优化尝试；接受或拒绝回滚都可，但纯数据/Value跳过不算。
2. 此后至少再完成一次有Actor优化步骤的PPO更新，初始logp检查为有限值。
3. `awbc_smoke_report.json`明确写出通过与实际触发的update，是否接受策略改动单独记录。

检查命令（只打印计划）：

```bash
python scripts/launch_prepared_pipeline.py \
  --root /content/crrl_training --stage ppo-smoke \
  --sessions /content/drive/MyDrive/cr_rl_ppo/sessions \
  --awbc-playbuffer --plan-only
```

确认数据和GPU可用后才执行实际smoke。此入口会覆盖为2次更新和interval=1，
这是验收配置；正式训练仍使用interval=50等生产设置。
`accepted=false`但smoke通过只证明真实尝试、回滚及后续PPO路径跑通，不证明已得到更强策略。

## 五、尚未完成的工作与建议顺序

### Colab CLI追加实读

用户授权后，通过DevSpace调用现有Colab CLI 0.7.5，只读检查了
`crrl-cpu-prep-1117`；观察时间由远端时间戳记录在
`artifacts/ppo_aux_review_20261117/colab_status.json`。

- 现有会话是CPU/Standard，没有已分配GPU；kernel显示IDLE，但后台筛选进程仍在运行。
- 52个源分片已下载，252,238条源记录；VM与Drive状态一致，完成 **20/52片**，
  共 **1,211局去重前候选**，正在重建`part-000020.parquet`。
- 当前配置的`/content/drive/MyDrive/cr_rl_ppo/sessions`和`/content/crrl_training/sessions`
  均不存在，未在其他无关目录推测或拼凑实战数据。
- 已准备U929完整checkpoint存在，大小4,375,624,553 bytes；既有准备报告仍为
  `generation=929`、`gpu_smoke_verified=false`、`training_started=false`。
  本次未为此重新计算整个大文件SHA，报告里的SHA是此前准备时留下的记录。

因此这次没有运行无法触发真实AWBC的GPU验收，也没有为了切GPU中断CPU筛选。
没有上传/覆盖云端源码、停止或重启worker；现有筛选继续运行。
本次未申请GPU，所以“没有已分配GPU”不等于账户无法申请GPU。
后续必要时已获用户授权通过Colab CLI新建GPU会话；若Drive挂载需要用户授权，
提供CLI实际生成的挂载链接。现有CPU筛选会话继续保留，数据不足不作为申请GPU的理由。

更新后的下一步顺序仍是：合格数据准备与输入验收 → GPU衔接验收 → 专家BC hook → 小规模对照。

### 1. 先解决数据可用性

继续保持严格执行、完整历史和0.4秒原生契约。优先提高“完成→有结果→通过契约→
同版本可拟合→可信新局”的转化率。每个阶段分别统计，不用完成录制数代替训练局数。
当前行为分组仍主要以Actor指纹为单位；若控制器、动作屏蔽或观察适配语义改变，应把这些
行为版本纳入后续provenance，不能仅凭相同权重就认为是完全相同的采集策略。本次未改造该格式。

### 2. 专家数据验收后再接BC hook

零手牌/落点修复、零额外等费与完整执行证明重建可用性；它们本身不证明玩家水平、战术质量，
也不证明模拟器重建状态与原局一致。应抽查原局与重建中的轮转、圣水、单位/塔状态和关键动作，
并确认跨分片去重以及同一对局双方、重建变体不跨train/validation/test泄漏。

每10次PPO更新、8–16条完整座位序列可作为小规模试验起点。BC接入时应有独立优化器、
消耗/RNG/恢复记录、完整prefix和WAIT语义，并以固定的PPO状态探针限制策略漂移。
同一次PPO边界若连续运行BC和AWBC，要从辅助更新前的同一个策略参考检查累计KL；
只让两条支路各自检查相对自己的起点，不能保证合起来仍在总预算内。
这些是下一步要求，本次没有声称已实现专家BC hook或统一辅助KL预算。

### 3. 做小规模对照再正式训练

保留U929固定评估基线。至少对照PPO-only、PPO＋专家BC以及再加入AWBC，
尽量对齐learner样本、评估牌组、随机种子和训练预算；把开发验证与最终未参与挑选的评估分开。
记录每类动作选择、空费/等待、错误落点、资源交换以及分牌组胜负，配合失败回放定位问题。
不要仅凭少量总胜率、候选数量、训练loss或加权holdout NLL决定上线。

当前AWBC只更新WAIT/卡槽/技能/wait_hand_gate读出。落点、实体编码和时序骨干冻结，
所以它适合小幅调整已有表征下的动作倾向，不能直接保证修复错误落点、缺失信息或长期规划。
若实战主要错误来自这些部分，应先用回放确认，再设计单独对照实验，而不是提高AWBC频率硬压。

## 六、文档与同步清单

新增本文件；同步更新：

- `docs/AWBC_PLAYBUFFER_PIPELINE_20261117.md`
- `docs/LIVE_AWBC_COLAB.md`
- `docs/COLAB_THREE_PIPELINES_PREPARATION_20261117.md`
- `docs/NULLS_RL_LOG_READINESS_20261116.md`

生产代码共9个文件，见第一节；新增/更新的回归文件为
`tests/test_live_rl_recording.py`、`tests/test_live_readiness.py`、`tests/test_live_awbc.py`、
`tests/test_awbc_playbuffer.py`、`tests/test_live_awbc_cli.py`、`tests/test_ppo_auxiliary_integration.py`。
同步到Colab时应同时带上这些互相依赖的代码，不能只替换单个hook。
独立AWBC旧版未加权NLL实验的latest会被新入口拒绝恢复，应使用新的输出目录保留对照。

## 方法背景

以下原始论文提供方法背景，不是本项目实际性能的证据：

- [Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177)：回报相对价值加权的行为回归。
- [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416)：辅助训练阶段约束策略变化的设计参考。
- [DAgger](https://arxiv.org/abs/1011.0686)：模仿学习中训练状态与实际访问状态分布的差异。

