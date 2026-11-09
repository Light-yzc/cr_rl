# BC 数据集重建与模拟器对齐（2026-11-09）

本轮整理 BC 重建方法，暂不启动训练。目标是统一监督样本与模型的 `crforge` 输入格式、
合法动作 mask 和 2307 维动作编码契约。格式与 backend 内部合法性一致，不等于重建状态
与人类原局一致。人类回放主要提供动作事件和坐标，缺少可验证的完整战场状态。

本轮代码只增加跨来源/跨 shard 的共享 replay registry，并补充文档；没有修复模拟器
动力学、恢复被截断的后半局或重训 BC。初始手牌推断与 omission-prefix 机制此前已存在。
视觉数据候选、限制及待实现的接入方案见
[BC_DATASET_ALTERNATIVES_20261109.md](BC_DATASET_ALTERNATIVES_20261109.md)。

## 数据来源

保留来源、revision 和 SHA-256，不把网页当前内容当作可复现实验输入：

| 来源 | 固定版本 | 用途 | 已知限制 |
|---|---|---|---|
| [VanguardX101/IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) | `059d43a02138a34b1b3009cc2acc7630fb99a638` | 有卡组、等级、事件和坐标的人类回放 | 没有原始 arena state；不同客户端/版本会造成重放失步 |
| [Cochon123/clash-royale-replays](https://huggingface.co/datasets/Cochon123/clash-royale-replays) | `10cca44e9d863d06cf078f142d3e744f78698b0a` | 清洗后的补充回放和技能事件 | 历史数据的卡牌等级未知；只保留明确的 Hero 事件，其他形式按 base 处理 |

`scripts/collect_training_sources.py` 下载 IL 分片并按上游 manifest 校验；
`scripts/prepare_cochon_replays.py` 将 Cochon 的 `matches.jsonl/events.jsonl` 转成同一
个 `royaleapi-battle-actions.v1` 事件格式。源 parquet、转换报告和 revision 必须进入
Drive/Colab 的数据目录，不能只保存一个压缩后的训练数组。

## 为什么原始 BC 会和模型轨迹不一致

1. 原始数据没有完整初始手牌、随机发牌和每个 tick 的实体状态。直接按固定 seed 重放时，
   第一张牌就可能不同，之后手牌、费用、实体和合法落点全部分叉。
2. 人类事件可能在模拟器中不合法：牌不在手牌、费用不足、坐标落在 mask 外、技能无法
   激活，或者同一 tick 两个事件发生冲突。把这些事件后的帧继续标成 WAIT 会制造错误
   标签。
3. source level、tower level、进化/英雄形式与固定 backend 表不一定相同。当前 BC/PPO
   模型的契约是 pinned `crforge`、`decision_ticks=5`、`mask_version=2`、统一 level 11、
   2307 action codec；等级差异只能作为 provenance，不能混进没有记录的观测。

在当前完整原生 observation 的重建管线中，状态和 legal mask 来自模拟器；人类事件
用于产生动作标签和检查重放前缀。这只保证 backend 内部一致，不能恢复缺失的原局状态。
若接入视频观测，需要另外定义视频与 simulator 共有的观测契约。

## 重建流程

`scripts/build_bc_shards.py` 是 Colab/Drive 的生产入口，流程如下：

1. 读取固定 revision 的 parquet，检查 1v1、双方各 8 张不重复卡和形式标记；卡名通过
   `import_il_replay.py` 的 pinned vocabulary/alias 表归一化。
2. 以 replay tag 派生稳定 seed，在 `crforge` 中用 level/tower level 11、统一 action
   codec reset。若启用 `--infer-opening-cycle`，先用源事件序列反推可行的初始发牌顺序，
   只有该顺序能在 backend 中重现才采用；否则保留 seeded reset 并在 provenance 中标明。
   推断结果只是与事件序列相容的候选，不是真实初始手牌的核验结果。
3. 按 replay tick 回放 card/ability 事件。每个成功事件必须同时满足：卡在手牌、动作在
   当前 legal mask 中、坐标能映射到 18×32 网格、技能槽唯一可用。
4. 使用有界 omission policy。`--max-omissions 2` 是放宽覆盖率的实验设置：前两次
   遗漏后继续模拟，只对遗漏附近默认 ±5 tick 的双方监督行加 `SOURCE_COMMAND_OMITTED`；
   第三次遗漏所在决策窗口及以后才整体截断。前两次遗漏可能持续改变费用、轮转和战场，
   局部屏蔽不能修复后续状态。保守对照用 `--max-omissions 0 --strict-subset`；这也只是在
   首次检测到的遗漏前截断，`strict-subset` 另限制来源/覆盖率，都不是状态真值校验。
5. 调用 pinned backend 的 `label_replays`，得到模型真正使用的 observations、masks、
   actions、ticks、seats 和 native flags。保留所有历史帧以维持 5 tick 间隔，训练只接收
   `flags == 0` 且 `masks[row, action] == 1` 的行；这些条件不证明人类标签与重建状态匹配。
   BC 的 128 帧上下文由
   `build_bc_batch()` 从同一 replay/seat 的时间轴构造。
6. 每个 shard 写入 `human_bc_reconstructed.npz`、metadata、import stats、原生重放和
   SHA-256。跨 IL/Cochon 和跨 shard 使用同一个 replay registry 按
   `canonical_game_signature` 去重，避免同一场换边视图或重复来源同时进入训练。

本次代码已让 `build_bc_shards.py` 默认使用 `<out>/replay_registry.json`，并把 registry
传给每个 importer。旧的未去重 build 不能原地续用，必须换新的 output 目录；否则容易把
重复回放误认为数据补全。

## Colab 生产命令

源文件放在 Colab 临时磁盘或 Drive，训练 shard 放 Drive；不要把完整 parquet 和展开的
`.npy` 缓存在本机 HDD。下面的命令会按 16 场一个 shard、每次成功后发布 manifest，
可中断后从 `build_status.json` 继续：

```bash
python scripts/prepare_cochon_replays.py \
  --cleaned /content/data/cochon123-cleaned \
  --out /content/data/cochon.parquet

python scripts/build_bc_shards.py \
  --il-parquet /content/data/IL_Replay/replays.parquet \
  --cochon-parquet /content/data/cochon.parquet \
  --out /content/bc_dataset_build \
  --drive /content/drive/MyDrive/cr_rl/datasets/bc_20261109 \
  --target 0 --shard-replays 16 --max-omissions 0 \
  --strict-subset --infer-opening-cycle
```

`--target 0` 扫描已提供的候选全集；若先做小验证，改成 `--target 4` 或 `--target 8`，
仍使用同样的 registry 和 prefix policy。小验证只检查重建与 manifest，不启动 BC/PPO：

```bash
python scripts/audit_bc_coverage.py \
  /content/bc_dataset_build --out /content/bc_dataset_build/audit.json
python - <<'PY'
import json
p = json.load(open('/content/bc_dataset_build/audit.json'))
assert p['shard_hashes_verified']
assert p['active_games_with_complete_paired_history_grid'] == p['active_metadata_games']
assert p['unflagged_illegal_target_rows'] == 0
print({k: p[k] for k in ('active_metadata_games','valid_supervision_rows',
                          'valid_card_play_labels','valid_ability_labels')})
PY
```

上述审计只检查文件完整性、时间网格和内部合法性，不能证明后半局已恢复或状态已对齐。
需补充小样本状态核验和截断分布后，再决定是否用 `train_bc.py` 读取 `dataset.manifest.json`。
训练必须按完整 replay 做
train/validation split，并排除整个 held-out deck，而不是随机切行；训练缓存使用
`ShardedBCDataset` 的有界 memmap，避免把全量 observation 展开到 RAM。

## 质量门槛和验收顺序

每个 immutable manifest 必须满足：

- metadata、每个 shard、源 parquet 都有 SHA-256，revision/backend commit/content SHA
  写入 provenance；
- 每个 active replay 有两席成对的 5 tick 时间网格，或者被明确排除；
- `unflagged_illegal_target_rows == 0`，且有效 card/ability 标签都通过 native mask；
- 首个超过所选 omission budget 的事件窗口及以后不参与监督；预算内遗漏仅局部屏蔽，
  须单列其后样本占比与持续失步风险，不能宣称后缀均已排除；
- registry 去重数、每个来源计数、level/form 分布和 held-out deck 列表写入报告；
- 先做少量 BC `--evaluate-only`/quick validation，再决定 Colab 正式训练。模仿 loss、
  play recall 或单次对战不能单独证明 BC 强度；必须保留自然 WAIT 误出牌率、实际成功出牌数、
  落点距离和固定换边对局。

如果源数据没有足够可重建事件，不用随机状态补齐缺失后缀。保留截断和未知状态记录，
通过小样本状态核验、held-out 验证及固定对战判断是否适合 BC 初始化。BC 是离线模仿，
这些检查不能保证 PPO 的 on-policy 条件或模型强度。即使没有检测到遗漏，双方交战、
路径和伤害仍可能偏离原局；现有事件流无法单独验证这些误差。

## 与当前 PPO 结果的关系

update 216 的 BC 对战合计为 33 胜 31 负（64 局，Wilson 95% 区间约
39.6%–63.4%，双侧二项检验约 0.90），这不能证明 BC 冷启动太弱，也不能证明 PPO 已
学会超越 BC。检查点日志显示近期 Actor 的 choice 行很少，而且 factorized tile-KL
超限会提前结束本轮 Actor epoch；这属于 PPO 样本利用率问题。重新训练 BC 的首要理由是
验证并减少状态/标签错位、提高动作质量，不能把去重或 BC 重训本身当成 PPO 变强的证据。PPO 恢复时
应固定 BC SHA、数据 manifest SHA、backend commit 和评估种子，再单独比较训练改动。
