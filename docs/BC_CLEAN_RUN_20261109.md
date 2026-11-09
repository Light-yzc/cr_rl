# 单 GPU 清洁 BC 路线

这条路线用于算力有限的情况。它不启动 PPO，也不扩 300M 模型；先从严格、可审计的
BC 前缀中做一个 A-tier-only snapshot，再用一个 GPU 训练一个候选。

## 数据构建

使用新的输出目录，避免污染旧 broad/omission2 数据：

```bash
python scripts/build_bc_shards.py \
  --il-parquet /content/data/IL_Replay/replays.parquet \
  --cochon-parquet /content/data/cochon.parquet \
  --out /content/bc_strict_20261109 \
  --drive /content/drive/MyDrive/cr_rl/datasets/bc_strict_20261109 \
  --target 0 \
  --shard-replays 16 \
  --max-omissions 0 \
  --strict-subset \
  --infer-opening-cycle
```

这会启用：固定来源 revision、共享 replay registry、至少 20 个成功出牌、至少 90% 事件
覆盖、无碰撞、首次 omission 即截断和开局牌序推断。旧数据只保留作对照，不继续追加。

## A-tier 快照

先审计并生成只包含 A-tier replay 的新 manifest；脚本只准备文件和训练命令，不启动训练：

```bash
python scripts/prepare_clean_bc_run.py \
  /content/bc_strict_20261109 \
  --out /content/bc_clean_a_20261109 \
  --min-a-games 128 \
  --init-from-bc /content/drive/MyDrive/cr_rl/runs/121m_bc_step5096/bc.pt \
  --epochs 2
```

A-tier 默认要求双座完整时间网格、至少 60 秒保留时长、至少 8 个合法出牌/技能标签、无
碰撞和 prefix-safe provenance。B/C replay 保留在原始快照中，但通过
`excluded_replay_ids` 排除出训练。

如果 A-tier 少于 128 局，不降低阈值；增加严格数据扫描量或先停止。脚本输出的
`clean_run.json` 是唯一训练命令来源，`quality_report.json` 保存每个 replay 的质量判定。

## 单 GPU BC 配置

默认使用已有 121M/122M checkpoint 作为表示初始化，但使用新 optimizer，避免继承旧数据的
动量。训练只跑 2 个 epoch：

- conditional action objective；
- natural timing prior；
- 不使用旧 card-prior correction；
- ability fraction 0.1；
- placement loss coefficient 1.0；
- backbone learning rate `1e-5`；
- tile head learning rate `3e-5`；
- batch 16、gradient accumulation 6；
- shard mixing window 64。

这套配置的目标是稳定修正动作时机和落点，不追求一次训练把所有容量都榨干。训练后只检查：

1. natural validation 的误出牌率和实际动作概率；
2. held-out replay 的 slot accuracy、tile-within-2 和平均 L1；
3. 16 个固定卡组换边对局；
4. 与旧 BC 的成功出牌数、满费等待和原生执行审计。

BC 候选没有同时通过自然时机、落点和换边门禁时，不启动 PPO。通过后再用分支中的
choice-only Actor、card/tile 分离 KL 和完整对局预算做一个小规模 PPO；不做四臂矩阵，也不
重新训练 300M。

## 轨迹边界

这条路线只能保证“模拟器内部一致的高质量前缀”。它不能从 IL/Cochon 事件恢复人类原局的
真实后半局。真正完整的人类轨迹仍需安卓模拟器内存状态、动作请求、执行确认和 250ms 状态
快照；在此之前，A-tier 前缀是低算力条件下最稳妥的 BC 冷启动来源。

## Salvage 数据路线（技能跳过）

如果目标是让 episode 尽量完整，可以使用已经实现的 salvage 模式：

```bash
python scripts/import_il_replay.py \
  /content/replay_repair_benchmark_20261109_skipability300/cohort.parquet \
  --out /content/bc_repaired_skipability300 \
  --limit 300 --max-source-rows 300 \
  --source-revision 059d43a02138a34b1b3009cc2acc7630fb99a638 \
  --seed 20261109 --min-plays 1 --min-coverage 0 --max-omissions 0 \
  --infer-opening-cycle --repair-hand --max-hand-repairs -1 \
  --repair-tile --max-tile-shift 50 --repair-time --max-delay-ticks 200 \
  --skip-unavailable-abilities --require-source-or-terminal
```

300 局实测得到 292 局可加载 episode、465,498 行状态。联合完成率为 97.33%；严格所有源事件（包含技能）完成率仍为 47.33%。技能跳过只改变 episode 的连续性，不把技能伪造为成功。

训练时必须使用监督 mask：

- 正常 native 成功的出牌和技能：可以进入 BC action loss；
- 技能跳过事件：`supervised=false`，只保留状态和时间上下文；
- 正常提前终局后的 absorbing state：只用于 value/terminal/sequence 目标，不用于 action loss；
- fault、损坏或无法恢复的 episode：丢弃。

salvage 数据必须单独标记为 `dataset_kind=repaired`，不能和 exact 人类轨迹合并成同一个 A-tier 数据集。

## 训练期间的固定对照

每 500～1000 个 optimizer steps 保存一次候选，并在同一验证集、同一随机种子和同一模拟器版本上比较旧 BC：

1. natural validation 的实际出牌率、WAIT 率和误出牌率；
2. play slot accuracy；
3. placement tile-within-2 和平均 L1；
4. 技能跳过率、native refusal 率和非法动作率；
5. 16 局固定卡组换边的胜负、成功出牌数、满费等待比例；
6. value/terminal 指标单独报告，不能用提前终局后的 action loss 代替。

候选只有在自然时机、落点、原生合法率和固定对照都没有退化时才进入 PPO。训练日志和 checkpoint 写入 Drive；每个评估点同时保存 `metrics.json`、验证集版本、数据 manifest SHA 和 simulator backend SHA。
