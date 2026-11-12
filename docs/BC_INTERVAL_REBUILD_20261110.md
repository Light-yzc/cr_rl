# 0.4秒 BC 因果重建入口（2026-11-10）

新脚本 `scripts/rebuild_bc_interval.py` 从原始命令时间在模拟器重新执行，默认间隔8 ticks（400ms）。它不读取旧5tick观测作下采样，也不把旧动作移动到更早时间。源码依赖显式由 `--source-root` 指定，本地使用 `.tmp_latest`，Colab源码overlay完成后使用 `.`。

## 已完成与尚未验证

- 六项CPU测试通过：原命令时间保持、同座位碰撞FIFO、双座位同时操作、缺原命令拒绝、形态身份映射、原生20tick命令延迟与网格量化分开报告。
- 当前真实metadata中152/293局保留了所有原始命令的 `source_event`；离线8tick计划覆盖这些152局/9874个命令，发现36个同座位量化格碰撞。FIFO不丢命令，0个提前命令，纯网格等待最大7 ticks，额外FIFO等待最大8 ticks。这只是计划，不是模拟器回放保留率。
- 其余141局metadata存在未记录的原命令。默认拒绝；不能把修复后的native replay JSON冒称原始命令源。
- 本地现有conda环境缺JPype，未安装新环境，因此本轮尚未在真实JVM上完成8tick重建。live保留率、非法率、执行时间偏差和训练性能仍需Colab预检实测。不会把离线计划算成已完成数据重建。

## 原始来源与完整性

293局原始源为 `VanguardX101/IL_Replay`，revision `059d43a02138a34b1b3009cc2acc7630fb99a638`。原训练流水线读 `/content/data/IL_Replay_replays.parquet`，SHA256为 `8f99ddd7b6f460c3efaf6873b4727e746a9e9edd8c796d120084aaaa68b4e261`。目前已下载的234715500字节NPZ是修复后的数组，不是该parquet。

正式重建优先传 `--source-parquet`。脚本流式读取必要列，并核验原parquet SHA、`source_row_index`、replay tag和原始命令数。保持原metadata的卡组、形态槽和初始seed/推断初始发牌，命令时间来自原始parquet，而非旧模拟器执行时间。

无parquet时，只能显式选择 `--skip-incomplete-ledgers` 后重建命令完整的子集；这改变了数据覆盖，需独立记录选择偏差。`--allow-recorded-subset` 是有洞命令的调试模式，任何输出均标 `smoke_only=true`、`training_ready=false`，不可用于正式训练。

## 时间映射与碰撞

同座位每个8tick格最多提交一个动作。命令最早提交时刻为 `ceil(source_tick/8)*8`；一格内多条命令按原先顺序延到后续格，双方座位可同时提交。法术使用同一因果规则，不提前，不扩成宽时间标签。

离线计划忽略费用、pending和战场变化。实际重建使用 `execute_repaired_events` 的live legal mask/手牌修复/落点修复/原生命令确认。旧策略同座位等待上一命令native确认后才提交下一命令，在8tick网上实际上至少间隔24ticks；这会额外推迟不同牌的合法操作。

更新后的v2-strict重建显式设置 `serial_native_confirmation=False`，允许不同牌同时处于native pending，但每座位每格仍只提交一条原命令，按source FIFO处理，继续服从费用、同牌pending和legal mask。旧executor默认仍为True。并发模式发生任何native失败立即拒绝整局，不把旧命令重试到已经提交的后续动作之后。技能不再跳过。

实际等待仍可能因费用/mask大于7ticks。audit分别记录网格映射、额外等待与native执行延迟；不将7ticks纯网格上界冒称总延迟上界。

网格内snap容忍度随8tick设为7ticks，额外重试总预算沿用原200ticks；不注入圣水。成功执行和原人类战术状态一致是两件事，8tick重建也不能修复缺少原始完整战场真值的问题。

## Colab运行命令

先在CPU上测试8局，验收后再将 `--limit` 设0。必须使用已核验当前补丁receipt的模拟器；旧源与新后端的hash分别记录，不强行宣称二者一致：

```bash
PYTHONPATH=src:.cache/vendor/crforge/python python scripts/rebuild_bc_interval.py \
  --metadata /content/drive/MyDrive/cr_rl/runs/bc_salvage_fp16_20261109/ready2/dataset_metadata.json \
  --source-parquet /content/data/IL_Replay_replays.parquet \
  --source-root . --decision-ticks 8 --limit 8 \
  --simulator-install .cache/vendor/crforge/env/build/install/env \
  --tables /root/.crforge/tables/16.635.3 \
  --out /content/bc_interval8_smoke
```

完整构建换新output目录并 `--limit 0`，不能覆盖旧5tick数据。默认output目录非空时拒绝运行，避免混合不同时间网格。CPU重建不需要GPU；正式训练前确保没有另一个trainer占用GPU。

## 输出与验收

- `interval_protocol.json`：源hash、决策间隔、所选原replay IDs、命令完整性和理论碰撞统计。
- `shards/replay_XXXXXX.npz`：每局独立压缩数组；字段为 observations、masks、actions、ticks、seats、flags、replay_ids（局内为0）。不会把整个dataset展开入RAM。
- `dataset.manifest.json`：兼容 `crrl-bc-shards-v1`，每个shard有hash、rows、replays=1、compact `replay_offset`。
- `dataset_metadata.json`：新schema/decision_ticks=8、compact replay列表、原 `canonical_game_signature`/source/replay_tag、original_replay_id，以原split registry保持对局级train/validation/heldout分配。
- `audit/replay_原ID.json`：外置完整修复ledger，区分原命令时间、实际提交时间、native执行时间，记录拒绝/遗漏/非法/fault/终局情况。
- `interval_summary.json`：每局命令保留率和实际延迟分位数、接受局数、训练可用性。必须读取实际结果后决定是否训练；输出格式可用不等于数据战术监督已核验。

严格准入要求每个原始命令一对一映射到唯一的native-confirmed、flags0、合法非WAIT监督行，且座位/原source tick/FIFO顺序均一致。任何技能跳过、遗漏、native失败、重复/多余动作或提前终局缺尾都会拒绝整局。无法在终局后伪造缺失动作；拒局的完整source命令和执行ledger仍写入audit。`accepted_source_fraction`说明全源25004条中有多少进入严格完整对局，不能把“入选局内100%保留”说成“全源100%保留”。

`training_ready`仅在本次所有selected局的每条原命令都保留且不是smoke子集时为True；任一selected局转换失败就为False。`strict_subset_ready`仅表示已保存的逐局完整池可分析，不授权自动过滤后启动正式训练。完整293局要求用 `--limit 0`；8局smoke即使全部通过也仅验证这8局，不等于全源通过。

旧v1实际结果已经确认不满足新要求：293局/25004命令中nativeconfirmed20633（82.52%）；288个旧accepted局确认20132/24497（82.18%）。158局提前正常终局共缺4365条，5局不完整共缺6条。仅按旧结果过滤完整局将剩130局8257条（全源33.02%），其中train78局4420、validation24局1792、heldout28局2045。去掉全座位确认锁后能保留多少，需要新的真实JVM重建测量，不能凭离线plan保证。

纯时序反事实计划显示25004条中2046条受旧24tick确认间隔额外推迟，涉及269局，额外延迟p95为8ticks、p99为16、最大56。这只评估调度条件，不含战场/费用/终局，不能当成新数据保留率。

本地复查：`tests/test_interval_rebuild.py`、`artifacts/bc_interval8_20261110/plan_complete/interval_protocol.json`。全部新文件位于G盘，无云实例或GPU任务由本分工启动。
