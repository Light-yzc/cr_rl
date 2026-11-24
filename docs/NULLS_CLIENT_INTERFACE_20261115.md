> 2026-11-16 状态更新：此文包含历史接口/缺口描述；当前接入状态与本轮验收见 [MODEL_INTERFACE_INTEGRATION_20261116.md](MODEL_INTERFACE_INTEGRATION_20261116.md)。

# Nulls Royale 实机接口与模型接入文档（2026-11-15）

> 本文档整理**当前仓库已落地且可调用**的真实游戏客户端接口（[src/cr_rl/client/](file:///g:/Code/cr_rl/src/cr_rl/client/__init__.py) 与 [run_nulls_dryrun.py](file:///g:/Code/cr_rl/scripts/run_nulls_dryrun.py)），供后续开发者将 `architecture-v3` 或其他自定义模型接入 Nulls Royale (`16.635.1`) 实机对战使用。  
> 历史上早期调研与停机导出说明见 [WINDOWS_REAL_GAME_INTERFACE_20261115.md](file:///g:/Code/cr_rl/docs/WINDOWS_REAL_GAME_INTERFACE_20261115.md) 与 [NULLS_READER_REPAIR_20261115.md](file:///g:/Code/cr_rl/docs/NULLS_READER_REPAIR_20261115.md)；**API 契约、字段覆盖率及当前限制一切以本文与当前源码为准**。

---

## 1. 当前交付边界：已打通能力 vs 尚未完成项

在开始接入模型前，请先确认当前代码处于什么阶段：

### 1.1 已完成并验证的部分
1. **只读内存战场状态采集（[nulls_reader.py](file:///g:/Code/cr_rl/src/cr_rl/client/nulls_reader.py)）**：
   - 通过 ADB 只读访问 `/proc/<pid>/mem`，定位 Nulls Royale `16.635.1`（`nullsroyale.rel.free` ARM64 `libg.so`）实时战场状态。
   - 已解码：对局时钟（`tick`, `time_ms`, `sec_elapsed_ms`, `sec_len_ms`, `phase`, `clock_running`, `clock_ended`）、双端阵营自动匹配 `own_seat`、双方圣水（`1 费 = 10000`）、出牌计数 `played_count`、手牌槽位与队列、8 张卡组（名/SCID/费/等级）、已见敌方卡牌 SCID、场上 4 类实体（部队/建筑、国王/公主塔、投射物、法术区域）及单位 HP/护盾/目标/Buff。
   - **强制只读与流量预算**：硬上限 `HARD_CAP_BYTES = 4 MiB`，默认单帧预算 `DEFAULT_MAX_POLL_BYTES = 1 MiB`；绝不写内存、不注入进程。
2. **Schema v4 (`int32[7024]`) 适配与合法动作掩码（[schema_v4.py](file:///g:/Code/cr_rl/src/cr_rl/client/schema_v4.py)、[deploy_mask.py](file:///g:/Code/cr_rl/src/cr_rl/client/deploy_mask.py)、[schema_adapter.py](file:///g:/Code/cr_rl/src/cr_rl/client/schema_adapter.py)）**：
   - `NullsToSchemaV4Adapter.adapt()` 将 `NullsBattleSnapshot` 转为与 `crforge` 一致的 `obs: int32[7024]`（含自动翻转到己方屏幕视角 `seat frame`）和 `legal_mask: bool[2307]`。
   - `build_legal_mask()` 精确移植 `crforge` `16.635.3` `DeployMask` / `PlacementSearch` 逻辑（经 `test_client_deploy_mask.py` 对齐校验）。
3. **ADB 触控执行桥（[adb_actions.py](file:///g:/Code/cr_rl/src/cr_rl/client/adb_actions.py)）**：
   - `TouchCalibration` 将动作 `1..2304` 解码为 `(hand_xy, board_xy)`，内置 MuMu `900x1600` 预设参数 `MUMU_900X1600` / `mumu_900x1600_calibration()`。
   - `AdbActionBridge` 实现安全前置校验：新鲜度检查（`max_age=0.4s`）、`match_id` 一致性、手牌 `hand_card_ids` 核验、屏幕分辨率核验、合法掩码二次校验、单发 `pending` 防抖锁与 `resolve()` 解锁。
4. **只读推理参考脚本（[run_nulls_dryrun.py](file:///g:/Code/cr_rl/scripts/run_nulls_dryrun.py)）**：
   - 已跑通 `NullsMemoryReader -> NullsToSchemaV4Adapter -> StateInference (actor_export.pt) -> AdbActionBridge(execute=False)` 完整链路（当前仅干跑打印坐标，不点击屏幕）。
5. **单元测试**：`tests/test_client_deploy_mask.py`、`tests/test_nulls_adapter.py`、`tests/test_adb_actions.py` 共 **27 项测试**当前全部通过（详见第 7 节）。

### 1.2 当前尚未完成 / 接入方必须自行处理的限制（重点）
1. **尚未提供实机自动闭环出牌脚本**：
   - [run_nulls_dryrun.py](file:///g:/Code/cr_rl/scripts/run_nulls_dryrun.py) 硬编码 `execute=False` 且使用 `validated=False`。如要让模型真实出牌，接入方需自写控制循环（参考第 5 节），并自行确认设备分辨率后传 `validated=True`。
2. **内存读取耗时高于 `max_age=0.4s` 决策节拍**：
   - 当前 `BoundedAdbMemReader` 每次抓取会对页批次（`_BATCH_PAGES = 32`）和未预取读取分别派生 `adb exec-out` 子进程；实测即使改用常驻 `adb shell -T sh` 在 shell 内 `for` 循环逐页 `dd | base64 -w 0` 读取 100 页（约 `400 KiB`），仍耗时 **`1378.4 ~ 1584.2 ms`**。
   - **直接后果**：若保持 `AdbActionBridge(..., max_age=0.4)` 且将 `ActionObservation.observed_monotonic` 设在抓取快照时刻（`snap.monotonic_ts`），实机触发 `submit(..., execute=True)` 时会因超时触发 `ValueError('Observation source is unverified or stale')`。
   - **解决方向**：接入方如需高速闭环，建议先优化底层只读 `MemoryBackend` 传输（如单连接常驻/二进制聚合读取，严守只读与 ≤4 MiB），或在明确评估状态滞后风险后调整轮询与 `max_age` 策略（注意 `run_nulls_dryrun.py` 把 `observed_monotonic` 设在了 `adapt()` 之后，实机建议改用 `snap.monotonic_ts` 保证真实新鲜度）。
3. **缺自动确认/超时回收（`reconcile`）机制**：
   - `AdbActionBridge` 一旦发送点击，`self.pending` 会一直锁定直至调用方显式调用 `bridge.resolve('confirmed' | 'rejected' | 'timed_out')`。目前需要接入方自己在后续帧对比 `played_count` / `hand_card_ids` 或超时后释放锁，并将未确认槽位传给 `adapter.adapt(..., pending_slots=...)`。
4. **开局禁用期与英雄技能暂未接入**：
   - `NullsToSchemaV4Adapter.adapt()` 当前传给 `build_legal_mask` 的 `play_open=True`、`ability_ready=(False, False)`，且 `MUMU_900X1600.ability_points=None`。
   - 开局约 2.5~3 秒动画期间游戏尚不接收下牌，需接入方依赖“超时未生效自动 `resolve('timed_out')`”或自己结合 `tick` 门控处理；英雄技能 (`2305/2306`) 当前既不可选也未标定像素点。
5. **`load_policy_inference()` 精度提示**：
   - [run_nulls_dryrun.py](file:///g:/Code/cr_rl/scripts/run_nulls_dryrun.py) 第 94 行构造 `StateInference` 时未传 `precision` 参数，且 `--device` 默认为 `"cpu"`。接入方在 CUDA 上运行 `StateInference` 时推荐显式传入 `precision="fp16"`，CPU 调试必须为 `precision="fp32"`。

---

## 2. 整体数据流与文件索引

```text
Nulls Royale 进程 (/proc/<pid>/mem, 只读)
   │  adb exec-out dd  (<= 4 MiB/poll)
   ▼
NullsMemoryReader.read_snapshot()        ──▶ NullsBattleSnapshot
   │
   ▼  (全局坐标 -> 己方视角 + 合法动作计算)
NullsToSchemaV4Adapter.adapt(snap, pending_slots=...)
   │                                     ──▶ SchemaObservation
   │                                           ├─ obs: int32[7024]
   │                                           └─ legal_mask: bool[2307]
   ▼  (固定 400ms 节拍推理，不可跳过 WAIT)
你的模型 / StateInference (actor_export.pt) ──▶ action: int (0..2306)
   │
   ▼  (新鲜度 / match_id / 手牌ID / mask / 单发防抖校验)
AdbActionBridge.submit(action, ..., execute=True/False)
   │  adb shell "input tap x1 y1 && input tap x2 y2"
   ▼
Android 客户端 ──(下帧检查 played_count/hand 变化)──▶ bridge.resolve(...)
```

| 层级 | 文件路径 | 核心导出符号 |
|---|---|---|
| 包入口 | [__init__.py](file:///g:/Code/cr_rl/src/cr_rl/client/__init__.py) | 仅模块说明，无 re-export（从子模块直接导入） |
| 内存采集 | [nulls_reader.py](file:///g:/Code/cr_rl/src/cr_rl/client/nulls_reader.py) | `NullsMemoryReader`, `BoundedAdbMemReader`, `MemoryBackend`, `NullsBattleSnapshot`, `NotInBattleError` |
| 协议定义 | [schema_v4.py](file:///g:/Code/cr_rl/src/cr_rl/client/schema_v4.py) | `SchemaV4`, `Vocabulary`, `VOCAB_TABLES`, `SPELL_TABLES`, `load_table_rows` |
| 合法掩码 | [deploy_mask.py](file:///g:/Code/cr_rl/src/cr_rl/client/deploy_mask.py) | `DeployCatalog`, `TileMap`, `MaskEntity`, `landing_tiles`, `build_legal_mask` |
| 帧适配器 | [schema_adapter.py](file:///g:/Code/cr_rl/src/cr_rl/client/schema_adapter.py) | `NullsToSchemaV4Adapter`, `SchemaObservation`, `DEFAULT_NAME_ALIASES` |
| 触控执行 | [adb_actions.py](file:///g:/Code/cr_rl/src/cr_rl/client/adb_actions.py) | `TouchCalibration`, `MUMU_900X1600`, `mumu_900x1600_calibration`, `ActionObservation`, `AdbActionBridge` |
| 干跑脚本 | [run_nulls_dryrun.py](file:///g:/Code/cr_rl/scripts/run_nulls_dryrun.py) | `schema_from_model_config`, `load_policy_inference`, `format_action`, `main` |
| 模型特征转换 | [observations.py](file:///g:/Code/cr_rl/src/cr_rl/data/observations.py) | `CrForgeObservationAdapter`（将 `int32[7024]` 拆为归一化 `(header, entities, entity_mask)`） |
| 权重与推理 | [state_policy.py](file:///g:/Code/cr_rl/src/cr_rl/models/state_policy.py), [inference.py](file:///g:/Code/cr_rl/src/cr_rl/training/inference.py) | `state_model_from_checkpoint`, `StateInference` |

---

## 3. 逐层 API 详细契约

### 3.1 Layer 1：战场状态读取（[nulls_reader.py](file:///g:/Code/cr_rl/src/cr_rl/client/nulls_reader.py)）

- **连接与初始化**：
  ```python
  from cr_rl.client.nulls_reader import (
      DEFAULT_MAX_POLL_BYTES,  # 1 MiB (1048576)
      HARD_CAP_BYTES,          # 4 MiB (4194304)
      DEFAULT_PACKAGE,         # "nullsroyale.rel.free"
      NotInBattleError,
      NullsMemoryReader,
      ReadBudgetExceededError,
  )

  reader = NullsMemoryReader.connect(
      serial="127.0.0.1:16416",
      package=DEFAULT_PACKAGE,
      adb="adb",
      max_poll_bytes=DEFAULT_MAX_POLL_BYTES,
      require_running_clock=True,
  )
  ```
- **抓取单帧**：`snap: NullsBattleSnapshot = reader.read_snapshot()`
  - 当进程未运行、不在战斗模式（`mode != 4`）、关键指针为空、或 `require_running_clock=True` 且 `clock_ended=True`（已弹出结算界面）时，抛出 `NotInBattleError`。
  - 当单轮读取字节数超过 `max_poll_bytes` 时抛出 `ReadBudgetExceededError`。
  - `NullsBattleSnapshot` 核心字段：
    - 时序：`mode: int`、`battle_substate: int`、`tick: int`（50 ms/tick）、`time_ms: int`、`sec_elapsed_ms: int`、`sec_len_ms: int`、`phase: int`（`>0` 视为加时）、`clock_running: bool`、`clock_ended: bool`。
    - 阵营：`own_seat: int | None`（自动比对本地账号 `local_acc` 与双方玩家 ID 得到 `0` 或 `1`，无法匹配时为 `None`）。
    - 双方摘要 `summoners: tuple[NullsSummoner, NullsSummoner]`（索引 `0` 和 `1`）：
      - `seat: int`、`acc_id: tuple[int, int]`
      - `elixir: int`（**`1 费 = 10000`**，满 10 费 = `100000`）
      - `played_count: int`（该侧累计出牌次数，**用于判断出牌是否被客户端受理**）
      - `hand: tuple[int, int, int, int]`（对应 `deck[0..7]` 的下标）
      - `queue: tuple[int, ...]`（后续摸牌下标队列）
      - `seen_scids: tuple[int, ...]`（客户端已记录的敌方已出卡牌 SCID）
      - `deck: tuple[NullsCard, ...]`（含 `name: str, scid: int, cost: int, level_raw: int`）
    - 实体列表 `entities: tuple[NullsEntity, ...]`：
      - `eid: int`、`vt: int`（相对 `libg.so` 的 vtable 偏移）、`name: str`、`scid: int`、`side: int`
      - **`x: int, y: int, z: int`**：**注意这里是游戏全局逻辑坐标（`0..18000` × `0..32000`）**，尚未按 `own_seat` 翻转；进入模型前务必由 `NullsToSchemaV4Adapter` 或 `SchemaV4.to_seat_x/y` 转换。
      - `level_raw: int | None`、`hp: int | None`、`max_hp: int | None`、`shield: int | None`、`max_shield: int | None`、`target_eid: int | None`、`buffs: tuple[NullsBuff, ...] | None`
    - 审计字段：`bytes_read: int`、`monotonic_ts: float`。

### 3.2 Layer 2：Schema v4 与动作编码（[schema_v4.py](file:///g:/Code/cr_rl/src/cr_rl/client/schema_v4.py)）

- **动作空间（共 `SchemaV4.ACTIONS = 2307` 维）**：
  - `0`（`SchemaV4.WAIT`）：等待，不操作屏幕。
  - `1 .. 2304`：`PLAY(slot, col, row)`
    - 编码：`action = 1 + slot * 576 + row * 18 + col`（`SchemaV4.play(slot, col, row)`）
    - 解码：`SchemaV4.decode_action(action)` 返回 `{"kind": "play", "slot": slot, "col": col, "row": row, "tile": tile}`
    - 坐标系（`seat frame`，以己方屏幕为基准）：
      - `slot` ∈ `0..3`（从左到右 4 张手牌）
      - `col` ∈ `0..17`（**左 `0` → 右 `17`**）
      - `row` ∈ `0..31`（**己方底线 `0` → 敌方底线 `31`**）
  - `2305, 2306`（`SchemaV4.FIRST_ABILITY + slot`）：英雄主动技能 `0, 1`（当前未标定，掩码固定为 `False`）。
- **观测平面向量**：长度 `SchemaV4.SEAT_INTS = 7024`（`Header 224 + 200 × Entity 34`）。
- **语义缺失约定**：未知/不可读字段固定填 `SchemaV4.UNKNOWN = -1`，切勿用 `0` 混充“已知”。

### 3.3 Layer 3：Schema v4 适配器（[schema_adapter.py](file:///g:/Code/cr_rl/src/cr_rl/client/schema_adapter.py)）

- **初始化与调用**：
  ```python
  from cr_rl.client.schema_adapter import NullsToSchemaV4Adapter, SchemaObservation

  adapter = NullsToSchemaV4Adapter(
      tables_dir=r"C:\Users\Administrator\.crforge\tables\16.635.3",
      decision_ticks=8,      # 默认 8 (= 400ms)
      default_own_seat=None, # 当 snap.own_seat 为 None 时回退到该座位
  )

  obs_out: SchemaObservation = adapter.adapt(
      snap,
      own_seat=None,
      pending_slots=(False, False, False, False),  # 有正在等待确认的槽位时置 True
  )
  ```
- **返回体 `SchemaObservation`**：
  - `obs: np.ndarray`（`np.int32`, shape `(7024,)`）
  - `legal_mask: np.ndarray`（`np.bool_`, shape `(2307,)`）
  - `own_seat: int`（`0` 或 `1`；若既无法识别也无默认值则抛 `ValueError`）
  - `tick: int`、`time_ms: int`
  - `hand_cards: tuple[str | None, ...]`（4 张手牌归一化内部名）
  - `hand_card_ids: tuple[int, ...]`（4 张手牌词表 ID，空槽 `-1`）
  - `unmatched_names: tuple[str, ...]`（未能命中词表/部署表的名称，健康接入时应为空 `()`）
- **状态重置规则**：当检测到 `snap.tick < _last_tick` 或卡组签名/座次变化时，`adapt()` 会自动调用 `reset_match()`；但**外部模型（如 `StateInference`）的隐状态需要调用方自己监测并 `reset()`**。

<details>
<summary>点击展开：schema_adapter.py 当前字段填充状态清单（实机可读 vs 固定 UNKNOWN/-1）</summary>

- **已从实机内存解析并对齐填入 `obs[7024]`**：
  - **Header**：`tick(0)`, `result(1, 进行中=0)`, `elixir(2)`, `crowns(4)`, `crowns_enemy(5)`, `hand_card(7..10)`, `hand_cost(11..14)`, `hand_pending(15..18, 来自 pending_slots)`, `next_card(19)`, `deck(20..27)`, `enemy_seen(28..35)`, 6 塔 `hp/max_hp(36..47, 已按屏幕左右翻转)`, `entity_count(48)`, `entity_overflow(49)`, `decision_ticks(50, 默认8)`, `play_delay_ticks(51, 20)`, `seat(52)`, `episode_step(54)`, `time_ms(55)`, `section(56)`, `section_type(57)`, `section_left_ms(58)`, `overtime(60)`, `enemy_play_count(114)`, `enemy_elixir(115)`, `queue(188..191)`, `deck_level(192..199)`。
  - **Entity（每行 34 int）**：`id`（Adapter 分配局内连续 ID）, `kind(1..4)`, `side(0=己/1=敌/2=中)`, `row(词表ID)`, `x/y`（已转为 seat frame）, `hp/max_hp/shield`（部队/塔读取；投射物/区域为 `0`）, `flags`（`F_ALIVE/AIR/BUILDING/KING`）, `level`, `dx/dy`（相邻采样帧位移差）, `target/target_x/target_y`, `buff_count` 与前 3 个 `buff/buff_ms`。
- **当前固定输出 `-1` (`SchemaV4.UNKNOWN`) 或静态默认的值**：
  - **Header**：`elixir_reserved(3)`, `hand_refill_ms(6)`, `pending_count(53)`, `full_bar_ms(59)`, `play_open_in_ticks(61)`, `last_refusal(62)`, `last_run_refusal(63)`, `illegal(64)`, `pending_*(65..80)`, `deck_slot(81..88)`, `deck_evolution(89..96)`, `hand_form(97..100)`, `ability_*(101..113)`, `enemy_play_*(116..187)`, `enemy_seen_evolution*(200..215)`。
  - **Entity**：`state` / `deploy_ms` 现在有时序推算兜底，直接验证值优先；`attack_ms` 在缺少验证值时仍未知。其余 `load_ms`, `speed_percent`, `hit_speed_percent`, `aim_x`, `aim_y`, `time_left_ms`, `radius`, `ability_ticks` 仍可能未知。详见 [实机对齐改造](REAL_ALIGNMENT_PPO_REWORK_20261115.md)。
  - **Mask 默认值**：`play_open=True`、`ability_ready=(False, False)`。

</details>

### 3.4 Layer 4：触控标定与 ADB 动作桥（[adb_actions.py](file:///g:/Code/cr_rl/src/cr_rl/client/adb_actions.py)）

- **预设标定（MuMu `900x1600`）**：
  - `MUMU_900X1600` 参数：
    - `serial='127.0.0.1:16416'`，`width=900`，`height=1600`
    - `hand_points=((280, 1448), (450, 1448), (619, 1448), (788, 1448))`
    - `board_quad=((68, 129), (832, 129), (832, 1229), (68, 1229))`（顺时针：左上、右上、右下、左下）
    - `validated=False`、`ability_points=None`
  - 创建实例：
    ```python
    from cr_rl.client.adb_actions import mumu_900x1600_calibration

    # 干跑安全默认 validated=False；实机确认屏幕为 900x1600 后再设 validated=True
    cal = mumu_900x1600_calibration(serial="127.0.0.1:16416", validated=True)
    ```
  - **坐标插值规则（`TouchCalibration.taps(action)`）**：
    - `u = (col + 0.5) / 18.0`（从左 `0` 到右 `1`）
    - `v = 1.0 - (row + 0.5) / 32.0`（使 `row=0` 落到屏幕棋盘底部，`row=31` 落到屏幕棋盘顶部）
    - 用 `board_quad` 做双线性插值并四舍五入为像素坐标，先返回手牌坐标，再返回棋盘坐标。
- **动作提交与状态保护（`AdbActionBridge`）**：
  ```python
  from cr_rl.client.adb_actions import ActionObservation, AdbActionBridge

  bridge = AdbActionBridge(
      calibration=cal,
      adb="adb",
      state_source=lambda: latest_action_obs,
      max_age=0.4,  # 超过该秒数视为 stale 并拒绝执行
  )
  receipt = bridge.submit(
      action,
      match_id=match_id,
      expected_card_id=expected_card_id,  # 1..2304 动作必填
      execute=True,                        # False 时仅返回 dry_run 坐标
  )
  ```
  `execute=True` 且 `action != 0` 时，依次检查 6 项条件（任一不满足即拒绝且不触碰设备）：
  1. `self.pending is None`（上一笔动作必须已 `resolve()`，否则抛 `RuntimeError`）；
  2. `self.calibration.validated is True`；
  3. `state_source()` 的 `verified is True` 且 `0 <= clock() - observed_monotonic <= max_age`；
  4. `state.match_id == match_id` 且 `(state.width, state.height) == (cal.width, cal.height)`；
  5. 若为 `1..2304`，`expected_card_id` 非空且等于 `state.hand_card_ids[slot]`；
  6. `len(state.legal_actions) == 2307` 且 `state.legal_actions[action] is True`。

  实际下发 ADB 命令为：`adb -s <serial> shell "input tap x1 y1 && input tap x2 y2"`（单次调用，超时 5s）。若命令抛出异常，`self.pending['status']` 置为 `'transport_failed_unconfirmed'` 并**保持锁定**；正常返回时 `self.pending['status']` 为 `'sent_unconfirmed'`。
  - **ADB 返回成功不代表游戏已扣费落牌**（例如网络抖动、开局动画中、第 1 下或第 2 下被丢）。调用方必须在后续帧自行确认并调用 `bridge.resolve('confirmed' | 'rejected' | 'timed_out')` 解锁。

---

## 4. 其它模型 / 本仓库模型如何接入

无论接入哪种架构，模型侧都只需要围绕两个张量交互：**输入 `(obs, legal_mask)`，输出 `action: int`**。

### 4.1 方案 A：接入第三方 / 自定义 PyTorch 模型（最小接口契约）

```python
# 你只需要实现一个 Policy Wrapper：
# 1. 新对局开始时重置内部状态（如有 RNN/Transformer KV 记忆）
# 2. 每步接收:
#    obs_int:    np.ndarray[np.int32] shape (7024,)   # 来自 SchemaObservation.obs
#    legal_mask: np.ndarray[np.bool_] shape (2307,)   # 来自 SchemaObservation.legal_mask
# 3. 可选: 借助 CrForgeObservationAdapter(schema, vocabulary_size=1466) 直接转为归一化浮点:
#    header_f32:  (1, 224)
#    entity_f32:  (1, 200, 34)
#    entity_mask: (1, 200)
# 4. 输出合法整数 action in [0, 2306] 且必须满足 legal_mask[action] == True
```

> 若使用自定义特征缩放，请务必遵循 [observations.py](file:///g:/Code/cr_rl/src/cr_rl/data/observations.py) 中的除数与 `-1` 保留约定，避免和缺失哨兵值 `-1` 混淆。

### 4.2 方案 B：接入本仓库 `architecture-v3`（`actor_export.pt`）

[run_nulls_dryrun.py](file:///g:/Code/cr_rl/scripts/run_nulls_dryrun.py) 中的 `load_policy_inference()` 已经示范了标准加载方式。**实机 GPU 部署时注意补充两点差异**：

1. **显式指定推理精度**：
   ```python
   infer = StateInference(
       model,
       obs_adapter,
       device=device,
       streams=1,
       compute_value=False,  # 实机只推理可关闭 critic，省算力
       precision="fp16" if str(device).startswith("cuda") else "fp32",
   )
   ```
   `model` 内部会在 `fp16` autocast 下自动保持 `frame_encoder` 与 `tile_head` 为 FP32；**CPU 上严禁开 `fp16`**。
2. **严格保持 `400 ms` 节拍连续推进记忆**：
   - 本仓库模型训练于 `decision_ticks = 8`（`8 × 50 ms = 400 ms`）。
   - **即使当前只有 `WAIT` 合法（圣水不够或刚下完牌处于等待），也必须在每拍调用 `infer.sample(...)` 推进 `infer.state`**，切勿跳帧；
   - 对局重启（`snap.tick < prev_tick`）或结束（`NotInBattleError`）后开始下一局前，必须调用 `infer.reset()` 与 `state_adapter.reset_match()`。

---

## 5. 若开启实机点击（`execute=True`）的推荐主循环骨架

目前 `scripts/run_nulls_dryrun.py` 仅实现到干跑。接入方实现自动对战循环时，请按下面结构处理 **`pending_slots` 反馈与 `resolve()` 确认**：

```python
def _slot_vocab_id(adapter: NullsToSchemaV4Adapter, own_sum, slot: int) -> int:
    idx = own_sum.hand[slot] if slot < len(own_sum.hand) else -1
    if 0 <= idx < len(own_sum.deck):
        cname = adapter.normalize_name(own_sum.deck[idx].name)
        return adapter.vocab.row_id(cname)
    return -1


pending_slot: int | None = None
baseline_played: int | None = None
baseline_card_id: int | None = None
sent_ts: float = 0.0
last_tick: int = -1

while True:
    snap = reader.read_snapshot()  # Note: 若当前 read_snapshot 耗时 > max_age，需先优化后端或调整 max_age
    if last_tick >= 0 and snap.tick < last_tick:
        infer.reset()
        state_adapter.reset_match()
        if bridge.pending is not None:
            bridge.resolve("rejected")
        pending_slot = None
    elif last_tick >= 0 and snap.tick == last_tick:
        time.sleep(0.05)
        continue
    last_tick = snap.tick

    own_seat = snap.own_seat if snap.own_seat is not None else default_seat
    own_sum = snap.summoners[own_seat]

    # 1) 在 adapt() 之前先直接通过 hand/deck 对账上一笔在途动作（避免同一帧重复调用 adapt() 污染 dx/dy）
    if bridge.pending is not None and pending_slot is not None:
        cur_card_id = _slot_vocab_id(state_adapter, own_sum, pending_slot)
        slot_changed = cur_card_id != baseline_card_id
        played_advanced = baseline_played is not None and own_sum.played_count > baseline_played
        if played_advanced or slot_changed:
            bridge.resolve("confirmed")
            pending_slot = None
        elif time.monotonic() - sent_ts > 1.8:  # 开局动画或丢触控超时解锁
            bridge.resolve("timed_out")
            pending_slot = None

    # 2) 用最新的 pending 状态生成 obs 与 legal_mask（每帧只调用一次 adapt）
    pending_slots = [False, False, False, False]
    if pending_slot is not None:
        pending_slots[pending_slot] = True
    adapted = state_adapter.adapt(snap, own_seat=own_seat, pending_slots=tuple(pending_slots))

    # 3) 用模型采样动作（400ms 每拍必须调用，即使只有 WAIT 合法）
    action = your_policy_step(adapted.obs, adapted.legal_mask)

    # 4) 如选中了非 WAIT 且当前无在途动作，校验并触控下发
    if action != 0 and bridge.pending is None:
        match_id = f"nulls-{reader.mem.pid}-s{adapted.own_seat}"
        # 注：若真实读耗较大，用 snap.monotonic_ts 会比 time.monotonic() 更真实反映观测年龄
        latest_obs = ActionObservation(
            match_id=match_id,
            observed_monotonic=snap.monotonic_ts,
            hand_card_ids=adapted.hand_card_ids,
            legal_actions=tuple(bool(x) for x in adapted.legal_mask),
            width=calibration.width,
            height=calibration.height,
            verified=True,
        )
        slot = (action - 1) // 576 if 1 <= action <= 2304 else None
        expected_card_id = int(adapted.hand_card_ids[slot]) if slot is not None else None
        try:
            receipt = bridge.submit(
                action,
                match_id=match_id,
                expected_card_id=expected_card_id,
                execute=True,
            )
            if receipt["status"] == "sent_unconfirmed" and slot is not None:
                pending_slot = slot
                baseline_played = own_sum.played_count
                baseline_card_id = expected_card_id
                sent_ts = time.monotonic()
        except (ValueError, RuntimeError) as exc:
            print(f"[skip] {exc}")
```

---

## 6. 本机关键资源路径（Windows）

以下路径已在当前工作区核对存在：
- **纯推理 Actor 权重**：`G:\Code\cr_rl\artifacts\ppo_v3_v100\actor_export.pt`（`743,166,134` bytes）
- **CrForge 数据表目录（词表与 DeployMask 用）**：`C:\Users\Administrator\.crforge\tables\16.635.3`
- **默认测试设备与分辨率**：MuMu Android 12，ADB 端口 `127.0.0.1:16416`，预设屏幕分辨率 `900x1600`（实机接入前请先执行 `adb -s 127.0.0.1:16416 shell wm size` 核对一致）
- **Python 环境分工（本机实测）**：
  - 单元测试（含 `pytest`）：`F:\yzc\Miniconda\envs\cs285\python.exe`
  - 实机推理 / CUDA 环境：`F:\python\python.exe`

---

## 7. 复现实验与测试命令

在 PowerShell（工作目录 `G:\Code\cr_rl`）中执行：

```powershell
# 1. 运行客户端 27 项离线单元测试（已实测通过：27 passed in 6.40s）
$env:PYTHONPATH="src"
& 'F:\yzc\Miniconda\envs\cs285\python.exe' -m pytest -v tests/test_client_deploy_mask.py tests/test_nulls_adapter.py tests/test_adb_actions.py

# 2. 进入 Nulls Royale 对局/训练营后，运行当前只读 Dry-Run 链路（不会点击屏幕）
$env:PYTHONPATH="src"
& 'F:\python\python.exe' scripts/run_nulls_dryrun.py `
  --serial 127.0.0.1:16416 `
  --checkpoint artifacts/ppo_v3_v100/actor_export.pt `
  --tables C:\Users\Administrator\.crforge\tables\16.635.3 `
  --device cuda `
  --ticks 15 `
  --interval-ms 400
```

> **安全红线**：所有接入改动必须保持 `/proc/<pid>/mem` **严格只读**、单次 poll 字节不超过 `HARD_CAP_BYTES (4 MiB)`，绝不写入游戏内存或注入库文件；一切实机动作必须走合法 ADB 触控通道并在异常/结算时主动放行。
