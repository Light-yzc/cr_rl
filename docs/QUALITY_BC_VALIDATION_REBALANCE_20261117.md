# 高质量 BC：全量筛选结束后重分配 Validation 到 Train

日期：2026-11-17。项目：G:/Code/cr_rl。

## 决策

不重新扫描 52 个 Parquet 源分片，不重建原生轨迹，也不修改正在运行的筛选 worker。
维持原先 v1 分片级约 20% Validation 的生产方式，在最终 catalog 合并完成时
直接把一部分 v1 Validation 改到 Train。**固定 Heldout 和历史已分配对局不动。**

新规则：先保留固定评估牌组 Heldout，再从其余新对局中按 canonical_game_signature
哈希取约 10% 为 Validation，约 90% 为 Train。整池比例约 81/9/10，
它是结果的近似比例，不是强制配额。

## 防泄漏约束

1. 历史 configs/quality_bc_prior_split.json 中的 Train、Validation、Heldout
   全部保留原分区，不把旧模型用过的验证局冒充全新独立测试局。
2. configs/decks_eval.yaml 所含评估牌组，任意座位命中则整场 Heldout，
   对手座位也不得混进 Train。历史分区若与固定评估牌组冲突则报错。
3. 对其他新对局，v1 用 SHA256 前 8 位取整数后对 5 求余，余数 0 为 Validation；
   v2 改为对 10 求余，余数 0 才为 Validation。这样旧 v1 Validation 的约一半
   会变成 Train，但原 v1 Train 不会变成 Validation，且同一局两个座位不跨分区。
4. 原分片 split_registry、原始 npz 与每个 game 的 SHA256 一律不变。
   顶层保存新的版本化分区，另存 v1 完整分区快照；旧版本实验仍使用旧 registry。
5. 只有 producer state 是 full_scan_completed 且所有分片都已提交时，
   自动切换最终 catalog 到 v2。未完成时只保留旧 v1 增量 catalog。

## 25/52 分片的实际预测

2026-11-17 通过 Colab CLI 只读读取已完成的 25 个分片：
共 1,586 个跨片唯一候选，当前没有跨片重复，历史 registry 重合 4 局
（Train 3，Validation 1），它们不改变。

| 划分 | 当前 v1 | 最终规则应用于同一批数据的 v2 | 变化 |
|---|---:|---:|---:|
| Train | 1,115 | 1,279 | +164 |
| Validation | 310 | 146 | -164 |
| Heldout | 161 | 161 | 0 |
| 合计 | 1,586 | 1,586 | 0 |

这是部分分片上的确定性重分配估算，不代表 52 个分片已完成，也不代表
1,586 局已经通过玩家技术水平或原始战场一致性验收。
新的 Validation 分布与旧 v1 不同，因此不能直接沿用旧版验证损失
来比较 checkpoint；必须以新 split 独立跑对照。

## 实现

- src/cr_rl/data/quality_bc_split.py：新分区的确定性纯函数。
- scripts/repartition_quality_bc_v2.py：读取已提交分片的元数据，
  检查旧分区、源文件版本和一致性，生成 v2 清单及转换报告，不读大数组。
- scripts/catalog_quality_bc.py：只有全量完成后才自动调用 v2 重分配，
  将 v1 归档到 split_versions 下，再更新最终顶层 split_registry.json。
- scripts/colab_finalize_full_bc.py：保留原调度逻辑，源扫描完成后仍会再次调用 catalog。

预期在 Google Drive 的 MyDrive/cr_rl_ppo/preparation/full_quality_bc 中：

- dataset.manifest.json：同一批数据、原数组路径与 hash。
- split_registry.json：最终 v2 分区，供新的 BC 训练使用。
- catalog_summary.json：新旧数量、Validation 转 Train 计数和版本目录。
- split_versions/quality_bc_v2_final_52of52_哈希/
  保存 v2 split_registry.json、旧 split_registry_v1.json 和 report.json。

如果现有 CPU session 未同步新版 catalog，扫描完成后手动在云端工程目录执行：

    python scripts/catalog_quality_bc.py --root /content/drive/MyDrive/cr_rl_ppo/preparation/full_quality_bc

未完成时可仅预览，不带 --publish：

    python scripts/repartition_quality_bc_v2.py --root /content/crrl_training/public_bc_full --prior-split configs/quality_bc_prior_split.json --eval-decks configs/decks_eval.yaml --allow-partial

## 验收与训练边界

本地测试覆盖转换哈希、历史 Validation 和固定 Heldout、
原分片 registry 不动、全量未完成不切换、
finalizer 重复运行仍保持相同不可变分区、原 catalog 去重和校验。
相关 tests/test_quality_bc_split.py、tests/test_quality_bc_catalog.py、
tests/test_bc_quality.py 合计 14 项通过。

## 现有 Colab CPU 会话部署

使用 Colab CLI 0.7.5，对正在运行的 crrl-cpu-prep-1117 会话完成了以下上传，
CLI 均返回 Uploaded 成功：

- src/cr_rl/data/quality_bc_split.py
- scripts/repartition_quality_bc_v2.py
- scripts/catalog_quality_bc.py

未修改/重启源扫描 worker，未申请 GPU，未启动 BC 训练。现有 finalizer
在收到 full_scan_completed 后会重新执行磁盘上的 catalog，因此届时会使用
已同步的 v2 最终合并逻辑。注意这里描述的是**已部署的自动触发路径**，
不是声称扫描已完成。

通过 Colab CLI 发起了只读分区预览，但 CLI 在 kernel preflight
阶段报 Timeout waiting for reply，预览代码尚未执行，故云端 v2 报告目前
没有完成实际验证。上方 25/52 分片的转换计数来自之前成功读取的已提交
云端元数据；新脚本的正确性目前由本地 14 项回归测试支持。
正式使用新数据前，还须复核最终 catalog_summary.json 的转换统计和版本路径。

最终 catalog 仍保持 training_ready=false 和 ppo_rollouts_allowed=false。
这只是数据切分方案，尚不能证明候选对局是专家、可以用于真实 PPO rollout，
或已经接入 PPO 的周期性 BC stage。仍须单独完成专家质量核验与 Actor/KL 验收。
