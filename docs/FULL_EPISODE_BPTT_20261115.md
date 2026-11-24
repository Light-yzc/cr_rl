# Architecture-v3 整局 BPTT

## 行为与梯度范围

`--full-episode-bptt` 让每个采样批次的所有对局从 reset 开始，到终止或模拟器截断结束，期间不更新 Actor。每条轨迹标记 `behavior_version`，优化器拒绝混合版本、不完整前缀、非空 Actor 初始 memory、没有真实结束边界的轨迹。不能把此前跨 update 的尾段拼成整局；切换时从新的环境开始，沿用模型/优化器权重。

current-vs-current 两个座位的轨迹分别保存。PFSP 历史池、BC 固定锚点、角色比例、焦点卡组、随机等级、奖励、独立 Critic、reference/anchor KL、choice-only Actor、AMP、KL 拒绝保护和自适应学习率均沿用现有实现。

整局模式以“一批完整对局”代替 `--steps` 固定窗口。结束的泳道暂停；只对活跃泳道运行模拟器和模型推理。所有对局结束后才更新权重。吞吐会受最长对局和收尾阶段批量缩小影响。

Actor/Critic 的优化调度都将整条序列作为一个不可切分的反传单元；更新前从空 memory 计算整局。Actor 最后一个有损失的决策之后的无损失帧可以省略，之前所有真实帧（包括强制 WAIT）均参与时序梯度。有效 batch 仍按实际 eligible 样本累计，不能为了凑固定 batch 在一局中途更新参数。GAE 参数与 BPTT 范围相互独立，现有长 GAE 不变。

`--tbptt-chunk` 在整局模式下继续用于无梯度参考缓存和价值评估，**不再截断训练梯度**。KDA 内部 32 帧的三角求解块也不是 BPTT 截断。

## 不改变权重形状的内存优化

- 原有空间编码器与落点头激活重计算保留。
- KDA/GQA 时序 block 使用 non-reentrant activation checkpoint。
- `--temporal-checkpoint-mode nested` 对 KDA 内部求解块再做分块激活重计算，避免反传时同时保留全局所有五维衰减矩阵；`block` 仅重算整个时序层，用更多显存减少重复计算。两者均保留完整梯度。
- 空历史 GQA prefill 直接对实际 token 长度计算 attention，避免把 streaming KV cache 的二次幂空容量也放入 attention。输出仍保留可继续 step 的等价 KV memory；掩码支持右侧 padding 和中间无效帧。
- 不修改 KDA/GQA 权重结构，可继续加载已有 v3 checkpoint。

激活重计算通过额外计算降低显存占用；完整 BPTT 增加梯度路径，不保证收敛一定更快，也不保证单位时间样本吞吐更高。

## 检查与监控

`tests/test_full_episode_bptt.py` 验证：末尾损失能传播到 128 步之前；重计算梯度与普通完整反传相符；调度不在序列内拆 optimizer step；完整 PPO Actor/Critic 更新、初始 log_prob 校验、参考与锚点 KL 保持可用。

`tests/test_state_ppo_v3.py` 的采集回归覆盖 current 双边轨迹、不同时间结束、结束泳道暂停收集和同一策略版本标记。`tests/test_state_temporal.py` 覆盖 full/step/chunk 结果与梯度、padding、memory reset。

TensorBoard 新增 `performance/full_episode_bptt` 与 `performance/max_episode_frames`。仍查看 `performance/*samples_per_second`、双卡峰值显存、`initial_logp_max_error`、overflow、KL 停止、`train_traj/*`；不启用额外 validation。

完整规模 GPU 验证输出位于 `artifacts/ppo_full_episode_smoke/`，与生产目录隔离，不发布 checkpoint 或联赛快照。目标仍为 250000 updates，正常 checkpoint 间隔仍为 800。

## 实测与长训配置

已完成 4 次完整规模 GPU 更新：nested 微批次 1 一次，nested 微批次 4 两次，block 微批次 4 一次。nested 微批次 4 两轮吞吐为 137.1/133.3 learner frames/s；block 微批次 4 为 143.1 frames/s，Actor/Critic 更新耗时 92.6/57.6 秒。采样批次不同，因此这些是实际运行吞吐，不是相同对局的严格 A/B 基准。

block 测试最长 773 步，双卡峰值 allocated 显存 6.75/7.63 GiB，进程峰值 RSS 5.22 GiB，初始 log_prob 最大误差 0.00370；所有 4 次更新均无 Actor/Critic AMP overflow、无 KL 停止/拒绝、指标有限。nested 两轮中的最长对局为 771 步，另一个微批次 1 测试最长 782 步。CPU 回归 145 项通过，另在两张 V100 上运行的 7 项 GPU 回归全部通过。

生产启动器默认 `FULL_EPISODE_BPTT=1`、`TEMPORAL_CHECKPOINT_MODE=block`，Actor/Critic 微批次保持 4/4、36 环境。显存更紧张时可选 `TEMPORAL_CHECKPOINT_MODE=nested`；显式 `FULL_EPISODE_BPTT=0` 可恢复截断训练。整局版本仍慢于此前约 180–200 frames/s 的截断版本；改变的是梯度范围，不把吞吐下降描述为提速。

通过验证后已从生产 u708 权重与优化器状态恢复 `crrl-v100-v3.service`，使用物理 GPU 1/2（V100），目标 u250000、每 800 update 保存，validation 关闭。4090 不参与训练。停止宽限延长至 600 秒，允许当前完整 update 和退出 checkpoint 完成。详细机器可读验证记录位于 `artifacts/ppo_full_episode_block_smoke/verification_summary.json`。
