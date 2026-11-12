# Actor 延迟信用分配审计（2026-11-08）

现有数据支持把 Actor GAE λ 从 `0.993` 增至 `0.997` 做独立小对照，尚不能证明它能提高胜率或加快收敛。`0.999` 对优势方差和更新方向的影响更大，暂不建议直接应用。Critic λ 固定为 `0.995`。

## 数据与计算方式

输入为现有 `rollout_000181.npz` 和 `rollout_000182.npz`，未加载模型、采集新对局或进行梯度更新。复用训练中的 `compute_actor_critic_targets` 和 choice 样本优势归一化；先在原始 250 ms 时间网格上重算 GAE，再汇总 choice 子集，forced-WAIT 帧仍然参与递推。

重算保留记录的 next-state value、true-terminal mask 和 episode-end mask。因此真实终局清除 bootstrap，截断保留 final observation 的 bootstrap，同时阻断 reset 后的新局向前泄漏。各 Actor λ 的 Critic targets 均与基线逐位相同。

完整 MC 参考仅覆盖起点与终点齐全、步数连续、无 fault/截断、同一 learner behavior version 的 episode-seat。符合条件的共有 **25 局、30 个 episode-seat、22,042 行**。相邻 rollout 允许拼接检查完整局，Actor 训练目标仍按各自 rollout 边界重建。

数据覆盖如下。高费定义为费用至少 5，手中高费统计排除已排队的牌；高费请求统计模型请求，不等同于实际成功出牌。

| 子集 | 行数 |
| --- | ---: |
| 有效 learner | 30,703 |
| choice | 2,091 |
| 高费合法的 choice | 43 |
| 手中有高费但无法负担的 choice | 1,686 |
| 可用圣水低于 2 的 choice | 370 |
| 可用圣水至少 5 的 choice | 72 |
| WAIT choice | 368 |
| 高费请求 | 10 |

## λ 对记录目标的影响

`γ = 0.9995`。下表的时间常数为 GAE 残差权重衰减到 `1/e` 的秒数，是指数衰减尺度，并非硬性截止时间。优势标准差取全部 choice；MC MSE 仅取完整 MC 覆盖中的 choice。符号翻转比较实际 choice 归一化后的优势。

| Actor λ | 衰减时间常数（秒） | 60 秒后的残差权重 | choice 优势标准差 | Actor return 对 MC 的 MSE | 相对基线的优势符号翻转 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.993 | 33.22 | 0.1643 | 0.2161 | 0.18728 | 0% |
| 0.997 | 71.33 | 0.4312 | 0.3103 | 0.10623 | 10.86% |
| 0.999 | 166.60 | 0.6976 | 0.4217 | 0.02953 | 18.32% |

高费合法子集仅 43 行，其中 31 行有完整 MC 参考。该子集的优势标准差依次为 `0.2262 / 0.2813 / 0.3902`，相对基线的符号翻转为 `0% / 16.28% / 20.93%`。样本来自相关帧，不能把行数当作独立对局数，也不能据此推断高费策略的效果。

较大的 λ 在数学上更接近 logged MC，所以 MC MSE 下降本身不构成算法改进的证据；同时还会增加方差。它也不会创造尚不可负担的高费动作或缺失的成功攒费经验。现有审计未发现 GAE 实现错误。

建议从同一 checkpoint 单独比较 `0.993` 与 `0.997`，固定 Critic λ、对手与其他超参数，按相同采样预算及相同墙钟时间分别评估。对战测试应使用固定种子、换边和相同对手，观察胜率、实际出牌费用、满费 WAIT、KL 拒绝及吞吐；只有训练目标离线变近不能作为启用依据。

## 复现

```bash
.venv-v100/bin/python loc_transfer/audit_credit_assignment.py \
  --rollouts artifacts/local_dual_v100/rollouts/rollout_000181.npz \
             artifacts/local_dual_v100/rollouts/rollout_000182.npz \
  --out artifacts/local_dual_v100_checks/credit_assignment_181_182.json

.venv-v100/bin/python -m pytest -q \
  tests/test_credit_assignment_audit.py tests/test_rollout_value_diagnostics.py
```

JSON 包含输入文件与脚本 SHA-256、完整 episode-seat 清单、各子集统计、Critic 目标一致性及原有 rollout value diagnostics。边界测试覆盖缺失起点、fault、截断、混合版本、真实 final-observation bootstrap、reset 隔离和 forced-WAIT 递推。
