# BC 数据源替代方案核查（2026-11-09）

目的：解决动作事件在 crforge 开环重放失步、导致后半局轨迹大量截断的问题。本次只查公开说明、生产/读取代码、文件清单及 Parquet 文件尾元数据；没有下载全量数据、没有训练，也没有证明候选已经改善 BC。现有事件流重建方法及命令见 [BC_DATASET_REBUILD_20261109.md](BC_DATASET_REBUILD_20261109.md)。

## 结论

找到两个比纯动作事件表更适合研究此问题的候选。它们保留实际画面或从画面提取的状态，可避免“从开局模拟到结尾”造成的战场状态累积漂移。但是，视觉估计误差、动作时间标签误差、版本差异仍然存在，二者都不是当前 7024-int crforge observation 与合法 mask 的直接替代。

## 1. KataCR / Clash-Royale-Replay-Dataset

- 项目：https://github.com/wty-yy/KataCR
- 数据：https://github.com/wty-yy/Clash-Royale-Replay-Dataset
- 核查的数据仓提交：`ce86e10bedcf97762c3d207633f035cd12be8936`
- 生产代码：https://github.com/wty-yy/KataCR/blob/master/katacr/policy/replay_data/offline_data_builder.py
- 读取代码：https://github.com/wty-yy/KataCR/blob/master/katacr/policy/offline/dataset.py

公开 tree 中有 347 个 `.npy.xz` 文件：fast_hog_2.6 221 个、golem_ai 126 个。压缩文件总计 2,016,380,664 bytes，约 1.88 GiB。这是文件数，不代表 347 场互不重复、完整正常终局的专家对局。文件名含 OYASSU/lan77，不能只按命名认定质量。

生产代码记录 state/action/reward。state 包含 time、unit_infos、cards（next + 四手牌）、elixir；单位包含类别、位置、阵营和血条小图 bar1/bar2；action 包含 card_id/xy，部分动作时间按识别的圣水变化 offset 回填。缺少当前 backend 的精确单位 HP、目标/冷却等原生状态及合法 mask。

局限：KataCR 卡牌分类说明注明针对 2.6 速猪；golem_ai 目录名不能证明己方石头人/高费卡组覆盖。作者 loader 会裁到最后一个动作 frame 并设 terminal，不能照搬为真实终局标签。代码 MIT 不代表数据仓有相同许可；数据仓未发现 README/LICENSE，许可待确认。未下载数组做逐局审计。

适用：小规模验证“共有可观测状态 -> BC”转换、视觉表征预训练。不适宜直接作为全卡组主训练集，尤其不能据此宣称解决高费牌不足。

## 2. chrisrca/clash-royale-tv-replays

- 数据：https://huggingface.co/datasets/chrisrca/clash-royale-tv-replays
- 固定 revision：`f3a750881e008e0daf237afd45736231b51a955a`
- 生产项目：https://github.com/chrisrca/CS541-Deep-Learning-Clash-Royale-Project/tree/08aafd87cb3707b779804918cc65f4ae430d37ab
- 标签关联示例：https://github.com/HumaidAlhuthali/clash-royale-dataset-browser/blob/8976803f2cbbaf5bd631a8616a9686fcc7bd12d1/patch_labels.py

实际文件清单有 2,525 个按 replay 分目录的 frames.parquet，覆盖 arena_01 至 arena_31。README 标称约 10fps、540x960、PNG，且移除重复帧。因此不能仅用行号乘 0.1 秒假定精确物理时间；需核对原 frame_id、采集代码与游戏计时。

图像与动作/手牌标签分别保存。通过 HTTP Range 只读取尾部元数据，核实：

| 文件 | 行数 | 文件大小 | 字段 |
|---|---:|---:|---|
| training.parquet | 3,060 | 1,389,524,075 bytes | card, png_bytes, x, y, arena, replay, frame |
| initial_training_hand_elixir.parquet | 3,060 | 1,388,750,868 bytes | 上述字段 + hand, elixir |
| training_offset_1.parquet | 21,848 | 10,016,106,287 bytes | 上述字段 + offset |
| training_offset_1_all_arenas.parquet | 21,848 | 10,031,799,806 bytes | 上述字段 + offset |
| new_arena_placement.parquet | 4,624 | 2,063,054,974 bytes | 上述字段 + offset |

这些是标签行数，不是完整对局数、互不重复的出牌数或已验证可用样本数。两个 offset 表可能重叠，不能相加。许多标签文件嵌入 PNG，不能因其叫“标签表”就下载 GB 文件到本机。浏览器关联键为 arena/replay/frame/card；坐标是 540x960 画面坐标，不能直接当作 18x32 backend 网格。

优势：能直接检查后半局真实画面，不需要相信失步后的模拟状态。局限：仍要检测实体、OCR 血条/圣水、确认出牌前后时间、处理缺失手牌和 WAIT 标注；高 arena 不自动等于高手。HF dataset card 声明 MIT，本次仅核查许可声明；视觉识别标签并非人工验真的原生状态。未读取实际标签行计算错误率。

适用：比单卡组数据更值得验证多卡组视觉 BC，但要先抽查卡组和标签覆盖，不能把 2,525 段画面当作现成专家监督全集。

## 3. 现有来源为何仍有局限

- IL_Replay：https://huggingface.co/datasets/VanguardX101/IL_Replay ，revision `059d43a02138a34b1b3009cc2acc7630fb99a638`。README 为 252,238 replays / 17,836,160 actions；体量更大不能修复缺少真实状态的问题。
- Cochon123：https://huggingface.co/datasets/Cochon123/clash-royale-replays ，revision `10cca44e9d863d06cf078f142d3e744f78698b0a`。README 明说不含实时单位 HP、连续位置、projectile state。2026-11-01 snapshot 有 tower-HP 弱标签/画面与 board annotation queue，但 queue 不是已验证标注，metadata 也未必对应历史版本。它与此前已用来源相同，不能当作新发现的完整状态数据。
- 其他比赛胜负/卡组统计、卡牌属性表、YOLO 检测图片不能直接提供完整 state-action 历史，不能替代此任务的 BC 轨迹。

## 建议的小样本验收

先从两个候选各选 2–3 段独立对局，在 Colab 配套抽查帧和动作标签，不下载全集。按真实时间分别抽查开局、中局、后局，报告：卡组/费用覆盖、可用状态比例、手牌匹配、动作时间偏移、坐标转换误差、缺失实体字段与持续历史长度。保留未知字段标记，禁止将未知 HP/冷却直接编造成精确 0。

如果采用视觉来源，需要定义“视频与 simulator 都能提供”的共有观测契约，并在 BC 与 PPO 两端一致使用，或验证独立的表征预训练/迁移方案。若只抽出动作又从头重模拟，会回到原来的 mismatch。若要求保持当前完整 simulator observation 不变，目前本次核查没有找到可直接替换的公开全状态专家数据集。


## 转换路线与当前接入状态

本次只完成数据源核查和接入方案，尚未实现 KataCR/TV Royale 转换器，也没有重训 BC。
当前 `import_il_replay.py` / `build_bc_shards.py` 读取的是动作事件流，不能直接接入这两个视觉数据源。

保持当前完整原生 observation 的路线，需要真实状态与 crforge 状态之间可验证的对齐。
公开数据缺少精确 HP、目标和冷却等字段，不能靠数组补零就达到这个条件。
若尝试视觉路线，下一步应先实现一个小样本 adapter，按以下顺序验收后再决定是否改模型：

1. 固定数据 revision、文件路径和 SHA-256；按完整对局留出验证集，去重后再切分，记录卡组与版本。
2. 按原始时间轴读取观测，定位出牌前的状态和随后的动作；核对视角/换边、棋盘投影、手牌槽和卡牌 ID。
   TV Royale 的屏幕坐标需先校准 arena 投影；KataCR 的 `xy` 也需核查其坐标定义，不能默认共用单位。
3. 只采用来源实际可观测的实体类别、位置、手牌、圣水和时间，逐项记录有效性与缺失标记；
   血条小图需经校准才能变成 HP 估计，不能伪装成原生精确 HP。
4. BC 与 PPO 共享同一观测定义、动作 codec 和历史采样时序。WAIT 需要连续时间证据；
   不能将未标注帧一律当作 WAIT，也不能从错误/缺失手牌强造 legal mask。
5. 分开报告开局、中局、后局的标注可用率与人工抽查误差；检查真实终局，不照搬最后一次出牌为 terminal。
   若共有观测方案通过，再验证离线动作质量和固定换边对战；若采用表征预训练，则另行验证迁移效果。

这些步骤是待实现、待验收的方案，不是已恢复后半局轨迹或已提高胜率的结果。
