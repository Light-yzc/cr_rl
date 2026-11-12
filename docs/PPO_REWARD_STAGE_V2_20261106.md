# 降低塔血塑形与移除满费WAIT惩罚（2026-11-06）

用户明确要求调低塔血权重、去除满费等待惩罚。本次仅改奖励配置，不增加指定牌、攒费、护兵或命中奖励。

|系数|旧阶段|新阶段|
|---|---:|---:|
|公主塔HP比例，每座|0.15|0.05|
|国王塔HP比例|0.20|0.05|
|皇冠差|0.15|0.15|
|满费WAIT，每决策步|−0.001|0|
|终局胜负|+1/−1/0|保持|
|gamma|0.9995|保持|

模拟器LadderMatch先按皇冠判胜；皇冠相同进入决胜时使用最低塔血，不是全部塔血之和。上述HP比例仅为学习用塑形，不替代真实胜负规则。

每步仍为 raw + gamma*Phi(next) - Phi(current)。真正终局next势能为0，时间截断保留末状态势能；overflow设0后塔血/皇冠塑形为零和。符合势能塑形条件时，降低HP权重主要改变学习信号与有限训练过程，不代表从理论上把之前的最终优化目标换成皇冠优先。

去除WAIT惩罚允许领先时省费等待；它不等于奖励等待。是否值得等待由后续塔伤与胜负估计。无直接高费奖励，不能承诺本改动解决低费循环。

## 执行与验收

交接worker等待已完整发布的>=92号，完成已到期短测，并确认下一轮仍在采样阶段。只在完整checkpoint上交接，不混用单轮奖励。

保存源完整manifest/summary/旧command；CPU零优化核对Actor/Critic、两Adam、RNG、预算与模拟器配置；使用原奖励通过严格恢复检查后，再显式切到新奖励。不会改写旧checkpoint的reward_config或放开通用resume保护。

记录resume_boundaries，未完对局及未提交采样不计训练完成；新增1000场与48场评估目标、卡组、对手、LR/GAE/KL、训练遍数保持。

Critic沿用原权重和Adam，新塑形会改变目标尺度；前后loss不可直接比较。重点检查首轮新rollout元数据、奖励重算、overflow精确为0、合法动作与有限数值。

部署证据位于Drive run的reward_stage_v2/；实时有效奖励另存reward_stage_runtime.json。本地脚本artifacts/long1000_20261106/reward_stage_worker.py、runtime_train.py、validate_reward_stage.py。

状态：已从完整92交接并修复对手路径，controller32534运行93号采样，已观察449→1921/3072推进；首轮新rollout最终验收待发布。

## 完整92交接回执

源92，新增62场，counter320，Actor Adam1550步、Critic14610步、Learner397444；SHA 946c78bc42061f09556ee06d39c44c9cd106dfdcf11099abfc79ed4a094ea0cc。
零优化恢复Actor/Critic/两Adam与torch/NumPy通过，GPU首reset前CUDA原检查仍启用。已到期4场stable短测2–2，评估累计8/48。

首次阶段启动在历史对手加载处失败：发布pool中的../opponents路径被直接复制到本地state目录，解析位置错误。修复为从冻结92的pool保留全部统计，仅将8个checkpoint路径映射到现有本地文件并逐个核验源manifest SHA；保留原始失败状态/日志，不改旧checkpoint。重启仍来自92，未完成的93不计局、评估不重打。

实际reward_stage_runtime.json已读回新系数；首轮真实rollout重算监控在后台等待93发布，尚未宣称首轮完成或胜率改善。
