# 对手多样性保留

`OpponentPool` 可以保留经过独立换边评估、能给当前 learner 造成困难的历史对手。它只改变有界池的剪枝，不改变 current / historical / anchor 的训练采样占比，也不写入 PFSP 在线胜率。

启用时调用：

```python
pool.configure_diverse_retention("experts.json", learner_version=182)
evicted = pool.prune(protected_ids=live_ids, learner_version=182)
```

清单示例：

```json
{
  "schema_version": 1,
  "evaluated_against_version": 182,
  "candidate_sha256": "<learner checkpoint 的完整 SHA256>",
  "candidate_checkpoint": "actor_export.pt",
  "max_age_versions": 20,
  "recent_slots": 2,
  "expert_slots": 3,
  "min_games": 32,
  "min_opponent_score": 0.55,
  "experts": [
    {
      "policy_id": 1073741963,
      "specialty": "counterpush",
      "reports": ["confirmation/evaluation.json"]
    }
  ]
}
```

路径相对清单所在目录解析。使用字典传入清单时，相对路径以当前工作目录为基准。`candidate_checkpoint` 可省略；提供时重算该文件的 SHA，不提供时由调用方确保清单 SHA 对应所声明版本的冻结 learner。清单中的专家必须已经是池内的 `historical` 对手；标签应来自打法诊断，不能仅凭出高费牌就命名为克制低费策略。

报告使用 `scripts/evaluate_policy.py` 的格式，当前 learner 必须是 `--policy`，拟保留专家是 `--opponent`，两者的额外 WAIT logit bias 都须为 0。代码验证固定的换边协议、统一的牌组文件 SHA、同一专家报告中一致且运行前后不变的评估脚本 SHA，以及每组物理牌组、种子都有 learner 两个座位的完整结果。逐局 `valid` 必须为布尔真，`fault`、`truncated`、`illegal_0`、`illegal_1` 必须齐全且为 0；代码还核对逐局结果与汇总，并拒绝重复游戏身份。所有报告须对应同一个 learner SHA 和该专家实际 checkpoint 的 SHA；报告文件的 SHA 也写入池元数据。

默认准入条件是至少 32 局、对手得分至少 0.55，平局计半分。32 局是保留筛选门槛，不代表统计显著或通用实力结论；正式对照仍需固定、独立的更多评估。可以提高门槛，不能将 `min_games` 降至 32 以下。

容量为 8 时，正在对局引用的对手、anchor 和最新快照先得到保护，然后优先保留最近 2 个非 anchor、最多 3 种不同标签的专家、一个旧参照及一个较新候选；不足的空位使用时间上分散的快照补齐。前面的硬保护占满容量时，后面的软配额让位。硬保护本身超过容量会在修改池之前报错。

证据默认有效至参考 learner version 后 20 个版本，之后不再提供专家保护。证据和报告来源随 `pool.save()` 保存，恢复不重复读取大型 checkpoint。未启用时保持原有剪枝和存档格式。

剪枝仅返回待淘汰条目。删除实际 checkpoint 仍由训练器执行原有的活跃对局、回滚和评估引用检查。
