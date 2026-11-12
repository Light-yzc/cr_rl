# BC 本地诊断阶段结论（2026-11-11）

完成范围：8/8/8局先导容量对照15支、普通选牌诊断6支；所有权重源为本地冻结step2274。当前全175/53/65落点结果另见[全量报告](G:/Code/cr_rl/docs/BC_FULL_TRAIN_RESULT_20261111.md)，不能用本表的小样本负结果否定全量结果。

主要判断：

1. 小集上，空间CNN把训练落点CE降到3.98、2格命中56.7%，说明结构/空间参数共享会显著改变可拟合程度；同8局验证没有稳定改善，因此单扩容不保证泛化。
2. 最后MLP增宽有小幅验证收益，完整落点头高LR/长曝光会出现验证反弹；不能把训练更快拟合当成战力更强。
3. 训练牌种位置初始化＋可学习残差在本小集快速得到验证CE4.845，是初始化/训练路线候选，不是公平同函数容量胜利，也不是全量赢家的既定结论。
4. 整模型savedAdam联合训练优于freshAdam联合；tile-only fresh未明显胜过fresh联合。已观察梯度冲突不能单独定罪，需要和匹配A/B一起解释。
5. 只拟合普通出牌的选牌头，原/宽/实体attention都能改善train，未改善该小集validation/heldout。这个诊断不含WAIT/技能正标签，不能直接部署或外推所有数据。

## 普通选牌6支（座位等权）

| 变体/LR | train acc | val acc | held acc | val ordinary4 NLL |
|---|---:|---:|---:|---:|
| original/3e-05 | 51.97% | 44.55% | 37.55% | 1.1498 |
| wide_residual/3e-05 | 52.68% | 41.37% | 35.92% | 1.1835 |
| original/0.0003 | 59.36% | 39.83% | 32.79% | 1.2633 |
| wide_residual/0.0003 | 59.35% | 39.62% | 32.06% | 1.2639 |
| entity_attention/3e-05 | 53.91% | 44.60% | 37.76% | 1.1584 |
| entity_attention/0.0003 | 59.37% | 39.39% | 32.71% | 1.2623 |

原始val ordinary4 NLL=1.12385、slot accuracy=49.16%。ability参数全程冻结，但6类CE仍会通过slot相对分数改变ability概率质量，因此报告普通4类NLL，不能只看6类CE小降。

实体attention3e-5相对同LR原头验证准确率差+0.047个百分点，整局95%CI约[-2.03,+2.57]个百分点；未发现明确优势。后续full175普通选牌头对照已完成：同1epoch实体读出相对原头的val4NLL差约−0.000041，95%CI跨零；续3epoch可小幅降低heldout4NLL，准确率提升仅约0.18个百分点、区间跨零。见[全量选牌诊断](G:/Code/cr_rl/docs/BC_CARD_FULL_RESULT_20261111.md)，避免将8game结论直接外推。

## 工程与保留

- 原readout_version=2已包含FiLM、空间投影/norm/门；本次没有将其冒称新增功能。
- 宽落点读出是256→1024→1024→1深两层残差，新增1,314,305参数，非26万浅层结构。
- 所有临时输入复制已回收；没有保存任意先导拟合权重。GPU微诊断峰值PyTorch allocated0.114GiB，不含CUDA context。
- 8game paired bootstrap保留双座位关联；筛多模型的探索性区间不能当独立确认试验。
- 保持最新安全checkpoint，已本地修复aggregate nonfinite norm时拒绝Adam更新的边界；77项相关检查通过，云端进程未热改。

证据：[容量结果](G:/Code/cr_rl/artifacts/bc_capacity_20261111/summary.json)、[普通选牌配对结果](G:/Code/cr_rl/artifacts/bc_capacity_20261111/card_capacity/paired_summary.json)。
