# Windows 实机接口迁移与接入指南（2026-11-15）

本文档面向 **Windows 端真实游戏接口接入**，汇总当前 `architecture-v3`（KDA + GQA 显式状态策略）的最终停机产物、推理加载方式、400 ms 决策节拍约束、Schema v4 观测与合法动作掩码契约、词表映射方法以及屏幕触控/坐标变换细节。

- 前置调研与客户端路线参见：[CLIENT_STATE_ACTION_BRIDGE_20261104.md](file:///run/media/regen/D02497132496FC22/Code/cr_rl/docs/CLIENT_STATE_ACTION_BRIDGE_20261104.md) 与 [DIRECT_GAME_STATE_RESEARCH_20261115.md](file:///run/media/regen/D02497132496FC22/Code/cr_rl/docs/DIRECT_GAME_STATE_RESEARCH_20261115.md)。

---

## 1. 停机权重与 Windows 拷贝清单

`crrl-v100-v3.service` 已于 **2026-11-15 16:53** 在 `update_000696` 定期保存并推进完成 **`u=697`** 后干净停机（双 V100 显存已释放为 `7 MiB / 0%`，无残留 `.tmp` 临时文件）。

- **停机前改参全阶段表现（`u=425..697`，共 273 个 updates，覆盖全 51 套卡组）**：
  - 总胜率 `3006/5366 (56.0%)`，对阵 BC 锚点胜率 **`1508/2430 (62.1%)`**，对阵历史池 **`845/1629 (51.9%)`**，对阵当前池 `653/1307 (50.0%)`。
  - 停机时 `u=697` 指标：`skill_advantage_vs_bc = +0.2257`（**51 卡组均衡对阵 BC 胜率 `61.28%`**），`anchor_pool_ema_win_rate = 0.6355`，`historical_pool_mean_ema_win_rate = 0.5305`，`anchor_kl = 0.2347`，`ref_kl = 0.00243`（均值），`EV = 0.730`（末段 21 轮均值 `0.758`）。
- **已校验的文件完整性（SHA256）**：
  - 轻量推理导出 `actor_export.pt`（`updates=697`，`decision_ticks=8`）：`743,166,134 bytes`，SHA256 `0de37b17a70bbfde0f6d5291eca4aec22de0d5bcbff088d49b2f0eb5268d5fc6`
  - 最新续训全量断点 `ppo_latest.pt`（`updates=697`）：`4,375,478,073 bytes`，SHA256 `ab30918a205e0eaad67b4df5a953cb0dc921183066df5b4b35ac623e7152e4cd`
  - 定期保存全量断点 `checkpoints/update_000696.pt`：`4,375,477,945 bytes`，SHA256 `afaf5ca7fe1090f4bb4dde4437ed3bacf7acdae24a6b072ddf830c50e708d856`
  - 早期对照导出 `actor_best_val.pt`（`u=184`）：`743,166,134 bytes`，SHA256 `b101a07de7cc9312fe3b5d39667db16618489cdfd9aeeed25cd7ab1631dd37f3`

### 推荐携带到 Windows 的最小文件集合

实机推理 **不需要** 携带 `4.1 GiB` 的完整训练断点 `ppo_latest.pt`（其中包含 Adam 优化器状态与独立 Critic 参数），只需携带轻量纯推理导出文件：

| 用途 | 源文件（Linux） | 大小 | 说明 |
|---|---|---:|---|
| **主推推理权重（推荐）** | [actor_export.pt](file:///run/media/regen/D02497132496FC22/Code/cr_rl/artifacts/ppo_v3_v100/actor_export.pt) | 709 MiB (`743,166,134 B`) | 实测 `updates=697`；仅含 Actor 参数、`config`、`temporal_config`、`learned_placement_definition`、`decision_ticks=8`，已验证可直接加载完成前向推理。 |
| 训练早期对照权重（可选） | [actor_best_val.pt](file:///run/media/regen/D02497132496FC22/Code/cr_rl/artifacts/ppo_v3_v100/actor_best_val.pt) | 709 MiB (`743,166,134 B`) | `u=184` 的 `best_train_traj` 导出（扩至全 51 卡组之前旧平台权重，仅作 AB 对照）。 |
| 代码包 | `src/cr_rl/`、`configs/`、`scripts/evaluate_state_policy.py` | < 10 MiB | 包含 `StateActorCritic`、`CrForgeObservationAdapter`、`StateInference`。 |
| 词表与游戏数值表 | `~/.crforge/tables/16.635.3/*.json`（或单独导出的词表 JSON） | ~几 MB | 用于把客户端/内存读取的 SCID 或卡牌内部名映射到模型词表 ID（`0..1465` / buff `0..264`）。 |

> 如果后续还要在 Linux 继续从 `u=697` 断点续训，**不要删除** `artifacts/ppo_v3_v100/ppo_latest.pt`、`checkpoints/`、`opponent_pool.json` 和 `metrics.jsonl`。

---

## 2. Windows 推理环境与纯 Python 加载方式

### 2.1 Python 与 GPU 要求
- Python 3.12 + PyTorch（支持 CUDA 的版本；参考 `scripts/run_local_state_preview.ps1` 中已配置好的 Windows 路径如 Miniconda + CUDA）。
- **纯真实游戏接口推理时，Windows 端不需要启动 JVM / `crforge`**：只要把游戏客户端或内存读取器解析出的战场状态组装成 `int32[7024]`（或 `(B, 7024)`）观测与 `bool[2307]` 合法掩码，就能直接喂给 `CrForgeObservationAdapter` + `StateInference`。
- 如果要在 Windows 本地先跑 `crforge` 模拟器对局/离线回放做一致性自测，则需要 JDK 21 并设置 `JAVA_HOME` 与 `PYTHONPATH="$root/src;$root/.cache/vendor/crforge/python"`（可参考 [run_local_state_preview.ps1](file:///run/media/regen/D02497132496FC22/Code/cr_rl/scripts/run_local_state_preview.ps1)）。

### 2.2 最小纯 Python 推理封装示例

`CrForgeObservationAdapter`（见 [observations.py](file:///run/media/regen/D02497132496FC22/Code/cr_rl/src/cr_rl/data/observations.py)）只读取 `schema` 的极少数布局属性，因此 Windows 实机脚本可用轻量 dataclass 构造静态 `Schema v4` 契约，完全解耦 JVM：

```python
from dataclasses import dataclass
import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.inference import StateInference


@dataclass(frozen=True)
class StaticSchemaV4:
    header_ints: int
    entity_ints: int
    max_entities: int
    seat_ints: int
    header_fields: tuple[str, ...]
    entity_fields: tuple[str, ...]
    header: dict[str, tuple[int, ...]]


def build_schema_from_ckpt(ckpt: dict) -> StaticSchemaV4:
    cfg = ckpt["config"]
    header_fields = tuple(cfg["header_fields"])
    entity_fields = tuple(cfg["entity_fields"])
    header: dict[str, tuple[int, ...]] = {}
    for idx, name in enumerate(header_fields):
        if name != "reserved":
            header[name] = (*header.get(name, ()), idx)
    return StaticSchemaV4(
        header_ints=int(cfg["header_dim"]),           # 224
        entity_ints=int(cfg["entity_feature_dim"]),   # 34
        max_entities=int(cfg["max_entities"]),        # 200
        seat_ints=int(cfg["header_dim"]) + int(cfg["max_entities"]) * int(cfg["entity_feature_dim"]),  # 7024
        header_fields=header_fields,
        entity_fields=entity_fields,
        header=header,
    )


# 1) 加载纯推理权重（Windows CUDA 推荐 fp16；若 CPU 调试必须改为 precision="fp32"）
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("actor_export.pt", map_location="cpu", weights_only=False)
assert ckpt["model_architecture_version"] == 3
assert ckpt["decision_ticks"] == 8  # 8 * 50ms = 400ms / step

model = state_model_from_checkpoint(ckpt, device=device).eval()
schema = build_schema_from_ckpt(ckpt)
adapter = CrForgeObservationAdapter(schema, vocabulary_size=int(ckpt["config"]["vocab_size"]))  # 1466
engine = StateInference(
    model,
    adapter,
    device=device,
    streams=1,
    compute_value=False,
    precision="fp16" if device == "cuda" else "fp32",
)

# 2) 新开一局对战前重置显式时序记忆
engine.reset()

# 3) 每 400 ms 决策一次：
# raw_obs: np.int32 shape (1, 7024)
# legal_mask: np.bool_ shape (1, 2307)
logits, _ = engine.evaluate(raw_obs, legal_mask, compute_value=False)
# 确定性模式可 argmax，或按训练同款分布采样：
probs = logits.softmax(dim=-1)
action_id = int(torch.multinomial(probs, 1).item())  # 或 int(logits.argmax(-1).item())
```

> **精度关键点**：`model` 自带 `frame_encoder_fp32=True` 和 `tile_head_fp32=True`。CUDA 上配合 `torch.autocast("cuda", dtype=torch.float16)`（即 `StateInference(..., precision="fp16")`）会自动让关键读取/落点头保持 FP32；**CPU 上切勿开启 `fp16`**，请始终用 `precision="fp32"`。

---

## 3. 核心时序契约（必读）

1. **固定 400 ms 决策节拍（`decision_ticks = 8`）**
   - 模拟器 1 tick = `50 ms`，`decision_ticks = 8` ⇒ 模型训练于固定 **`400 ms`（2.5 Hz）** 的观测节拍。
   - **即使当前只有 `WAIT` 合法（圣水不够 / 开局冷却中），也必须按节拍喂入每一帧推进 `StateInference.state`！** `architecture-v3` 的 KDA + GQA 记忆在训练与评估时都是对所有物理 0.4 s 帧连续推进的，跳过强制 `WAIT` 帧会破坏内部位置编码与时间节奏感知。
2. **单流单局隔离**
   - 严禁多个对局复用同一份 `engine.state`；每场对局开始/重开时必须执行 `engine.reset()`。
3. **权重只读与内存管理**
   - `GQAState` 的 KV 缓存随对局步数增长（正常 3~5 分钟对局约 450~750 帧，显存极小）；跨局务必 `reset()` 释放上局 KV。

---

## 4. 动作空间与坐标变换（`action_count = 2307`）

动作空间共 `2307` 维（见 `Schema.java` 与 `schema.py`）：

| 动作 ID 范围 | 语义 | 参数与解码公式 |
|---|---|---|
| `0` | `WAIT` | 本拍不出牌、不按技能（不发送屏幕点击）。 |
| `1 .. 2304` | `PLAY(slot, col, row)` | 4 个手牌槽位 × `18 × 32` 格子（每槽占 `576` 格）：<br>`slot = (action - 1) // 576` (`0..3`)<br>`rem = (action - 1) % 576`<br>`row = rem // 18` (`0..31`，己方底线 `0` → 敌方底线 `31`)<br>`col = rem % 18` (`0..17`，屏幕左 `0` → 屏幕右 `17`)<br>**编码公式**：`action = 1 + slot * 576 + row * 18 + col` |
| `2305` | `ABILITY(0)` | 点击己方第 1 个英雄主动技能按钮 |
| `2306` | `ABILITY(1)` | 点击己方第 2 个英雄主动技能按钮 |

### 4.1 竞技场逻辑坐标与视角规则
- 竞技场逻辑尺寸：`WIDTH_UNITS = 18000`（`18` 格 × `1000` 单位/格），`HEIGHT_UNITS = 32000`（`32` 格 × `1000` 单位/格）。
- 格子 `(col, row)` 对应的 **玩家视角（seat frame）** 中心坐标：
  - `x_seat = col * 1000 + 500`（范围 `500 .. 17500`）
  - `y_seat = row * 1000 + 500`（范围 `500 .. 31500`）
- **与屏幕/客户端坐标系对齐**：
  - 模型期望的 **seat frame** 就是**当前玩家自己手机屏幕看到的视角**：己方国王塔在底部（小 `y`），敌方国王塔在顶部（大 `y`），`x` 从左向右递增。
  - 如果实机状态读取器（如内存读取）直接读出的是**对局全局坐标**而非当前客户端渲染坐标，必须按己方所在阵营 `side`（`0` 或 `1`）转换到 seat frame（`Schema.toSeatX` / `Schema.toSeatY`）：
    - `side == 0`：`x_seat = 18000 - x_world`，`y_seat = y_world`
    - `side == 1`：`x_seat = x_world`，`y_seat = 32000 - y_world`
  - **屏幕触控标定（重要）**：手游画面中手牌栏与棋盘区域的比例不是全屏等比。请单独标定：
    1. **手牌 4 槽位屏幕中心**：`hand_slot_xy[0..3]`
    2. **英雄技能按钮中心**：`ability_btn_xy[0..1]`（如有）
    3. **棋盘可点击矩形映射**：`(x_seat / 18000, y_seat / 32000)` 线性映射到设备上棋盘边界的像素 `(screen_x_left..screen_x_right, screen_y_bottom..screen_y_top)`（注意图像像素 `screen_y` 通常向下增大，而 `y_seat` 向上增大，因此 `screen_y` 方向须反向插值）。

---

## 5. 观测契约 Schema v4（`int32[7024]`）

每帧观测是一个长度为 `7024` 的 `int32` 一维向量：
- **Header**：前 `224` 个整数（索引 `0 .. 223`）
- **Entities**：后 `200 × 34 = 6800` 个整数（索引 `224 .. 7023`），按实际写入数量 `header[48]`（`entity_count`）依次填充，多余行全部填 `0`。

> **关键原则（见 `observations.py`）**：训练时对不可见/缺失语义固定使用 **`-1`**（例如敌方未显血条单位的 `hp/shield`、空槽位的 `hand_card/next_card`、钻地单位的 `row`、未识别卡组等）。接入实机读取器时，**确实未知的字段应保持 `-1`，不要随意填 `0` 伪装成“已知为 0”**。

### 5.1 Header 关键字段索引与缩放（共 224 int）

| 索引 | 字段名 | 数量 | 实机数值含义与单位（传入 raw 前的整数尺度） | `Adapter` 除数 |
|---|---|---:|---|---:|
| `0` | `tick` | 1 | 对局逻辑 tick（`50 ms / tick`） | `7200.0` |
| `1` | `result` | 1 | `0=running, 1=won, 2=lost, 3=draw` | `1.0` |
| `2` | `elixir` | 1 | 己方当前圣水，**1 费 = `10000`**（满 10 费 = `100000`） | `100000.0` |
| `3` | `elixir_reserved` | 1 | 已点击待生效出牌预占圣水（单位同上） | `100000.0` |
| `4, 5` | `crowns`, `crowns_enemy` | 2 | 己方皇冠数 / 敌方皇冠数（`0..3`） | `3.0` |
| `6` | `hand_refill_ms` | 1 | 补牌冷却毫秒数（通常 `0`） | `360000.0` |
| `7..10` | `hand_card` | 4 | 4 个手牌槽位的**词表 ID**（空为 `-1`） | `/ 1466.0`（`<0` 保留 `-1.0`） |
| `11..14` | `hand_cost` | 4 | 4 个手牌槽位牌面费用（整数费，空为 `0`） | `10.0` |
| `15..18` | `hand_pending` | 4 | 该槽位是否有已发出未落地出牌指令（`0/1`） | `1.0` |
| `19` | `next_card` | 1 | 下一张将摸到的卡牌**词表 ID**（未知 `-1`） | `/ 1466.0` |
| `20..27` | `deck` | 8 | 己方 8 张卡组**词表 ID** | `/ 1466.0` |
| `28..35` | `enemy_seen` | 8 | 己方视角已见过的敌方卡牌**词表 ID**（按出现顺序，未见槽位填 `-1`） | `/ 1466.0` |
| `36..47` | `own_{king,left,right}_{hp,max_hp}`<br>`enemy_{king,left,right}_{hp,max_hp}` | 12 | 6 座塔当前 HP 与最大 HP（按己方屏幕左右视角；已破塔 `hp=0`） | `10000.0` |
| `48` | `entity_count` | 1 | 当前写入的实体行数（`0..200`） | `1.0` |
| `49` | `entity_overflow` | 1 | 超出 200 行被截断的实体数（通常 `0`） | `1.0` |
| `50` | `decision_ticks` | 1 | 固定 `8` | `1.0` |
| `51` | `play_delay_ticks` | 1 | 下牌到生效的延迟 tick（通常填模拟器默认或 `20`） | `1.0` |
| `52` | `seat` | 1 | `0` 或 `1`（若单端始终视为 seat frame 可写 `0`） | `1.0` |
| `53` | `pending_count` | 1 | 当前处于待执行队列的己方出牌数 | `1.0` |
| `54` | `episode_step` | 1 | 本局已执行的 0.4s 决策步数 | `1440.0` |
| `55` | `time_ms` | 1 | 对局时间轴已进行毫秒数 | `360000.0` |
| `56, 57, 58` | `section`, `section_type`, `section_left_ms` | 3 | 当前时间段序号、类型、本段剩余毫秒数（battle clock） | `1, 1, 360000.0` |
| `59, 60, 61` | `full_bar_ms`, `overtime`, `play_open_in_ticks` | 3 | 涨满 1 费毫秒数、是否加时(`0/1`)、开局禁出牌剩余 ticks | `360000.0, 1, 100.0` |
| `62, 63, 64` | `last_refusal`, `last_run_refusal`, `illegal` | 3 | 上一步拒绝码与非法计数（正常填 `0`） | `1.0` |
| `65..80` | `pending_{card,x,y,ticks}` × 4 | 16 | 最多 4 条已发出待落地己方出牌；空闲行填 `(-1, 0, 0, 0)` | `vocab / 18000 / 32000 / 100` |
| `81..88` | `deck_slot` | 8 | 卡组各张牌槽位标记（`1=进化槽, 2=英雄槽`） | `1.0` |
| `89..96` | `deck_evolution` | 8 | 进化槽卡牌距离下次进化已打出次数（未用 `-1` 或 `0`） | `1.0` |
| `97..100` | `hand_form` | 4 | 当前手牌打出形态：`0=普通, 1=进化, 2=英雄` | `1.0` |
| `101..113` | `ability_*` × 2 + `last_ability_run_refusal` | 13 | 每槽 6 int：`champion_vocab(-1), unit_id(-1), state, cooldown_ms, charges(-1), command_ticks`；最后 1 int 拒绝码 | `vocab / 1 / 1 / 360000 / 1 / 100` |
| `114, 115` | `enemy_play_count`, `enemy_elixir` | 2 | 已观测敌方出牌次数、估算敌方圣水（`1 费 = 10000`，未知可追踪或 `-1`） | `1, 100000.0` |
| `116..187` | `enemy_play_*` × 8 | 72 | 最近 8 条敌方出牌记录（最新在前），每条 9 int；空槽位见 `Schema.java`（前 3 项 `-1`、点位 `-1,-1`、其余 `0`） | 见 `observations.py` |
| `188..191` | `queue` | 4 | 手牌后 4 张牌库队列（`queue[0] == next_card`），未知 `-1` | `/ 1466.0` |
| `192..199` | `deck_level` | 8 | 卡组每张牌等级（从 `1` 起，如锦标赛标准 `11` 级） | `15.0` |
| `200..215` | `enemy_seen_evolution`, `enemy_seen_evolution_need` | 16 | 敌方进化计数与所需周期；未知填 `-1` / `0` | `1.0` |
| `216..223` | `reserved` | 8 | 固定 `0` | `1.0` |

### 5.2 Entity 行布局（每行 `34 int`，最多 `200` 行）

| 索引 | 字段名 | 含义与取值说明 |
|---|---|---|
| `0` | `id` | 实体 ID（Adapter 会将该列置 `0`，但若使用 `ability_unit` 关联则需局内稳定） |
| `1` | `kind` | `1=character（部队/场上建筑）, 2=tower（国王/公主塔）, 3=projectile（投射物）, 4=area（法术区域效果）` |
| `2` | `side` | `0=己方, 1=敌方, 2=中立` |
| `3` | `row` | 实体在词表中的 ID（`0..1465`；例如掘地矿工隐身钻地轨迹未知单位时填 `-1`） |
| `4, 5` | `x, y` | **seat frame 逻辑坐标**（`x ∈ [0, 18000]`, `y ∈ [0, 32000]`；模型内部对坐标使用 Fourier 位置编码） |
| `6, 7, 8` | `hp, max_hp, shield` | 当前 HP、最大 HP、护盾值；无血量单位填 `0`；敌方未显示血条或地下单位填 `-1` |
| `9` | `state` | 单位状态枚举（`0..16`，详见下表；投射物与区域效果固定 `-1`） |
| `10` | `flags` | 位掩码（`F_ALIVE=1, F_HIDDEN=2, F_INVISIBLE=4, F_UNDERGROUND=8, F_AIR=16, F_BUILDING=32, F_KING=64, F_CLONE=128, F_AWAKE=256, F_NO_MOVE=512, F_NO_ATTACK=1024, F_CHARGED=2048`） |
| `11` | `level` | 等级（从 `1` 计；未知 `-1`） |
| `12, 13` | `dx, dy` | 相比上一决策帧在 seat frame 下的位移增量（无上一帧或未知填 `0, 0`） |
| `14, 15, 16` | `target, target_x, target_y` | 目标实体 ID（Adapter 会将 `target` 置 `0`；无目标填 `-1, 0, 0`） |
| `17, 18, 19` | `deploy_ms, attack_ms, load_ms` | 部署剩余毫秒、攻击计时毫秒、前摇加载剩余毫秒（不可读可暂写 `0`，但须标注） |
| `20, 21` | `speed_percent, hit_speed_percent` | Buff 后移速/攻速百分比（正常未受减速/狂暴影响为 `100`，建筑/投射物视情况为 `0`） |
| `22..28` | `buff_count, buff_0, buff_ms_0, buff_1, buff_ms_1, buff_2, buff_ms_2` | Buff 数量及前 3 个 Buff（Buff 词表 ID `0..264` 与剩余 ms；空槽 `-1, 0`） |
| `29, 30` | `aim_x, aim_y` | 投射物飞行终点 seat frame 坐标（非投射物写 `0, 0`） |
| `31, 32` | `time_left_ms, radius` | 区域效果剩余毫秒、半径（逻辑单位） |
| `33` | `ability_ticks` | 距离上次释放技能 tick 数（无 `-1`） |

`E_STATE`（字段 `9`）枚举对照：
- `0=standing`, `1=moving`, `2=attacking`, `3=dashing`, `4=deploying`, `5=jumping`, `6=spawn_pathfind`, `7=ingame_pathfind`, `8=clone_setup`, `9=morphing`, `10=casting`, `11=waiting_to_deploy`, `12=following_removed`, `13=following_removed_building`, `14=components_disabled`, `15=route_following_alternate`, `16=ability_follow_up`；无状态（投射物/区域效果）为 `-1`。

---

## 6. 词表（`vocab_size = 1466`）与客户端 SCID 映射方法

- 模型固定数据版本：`data_version = 16.635.3`，`content_sha = dae6c6cc7e9126338da0c606e47e48a9ef80462c`。
- 主词表（1466 个名字，全局按 ASCII 字典序排列去重）合并自 `~/.crforge/tables/16.635.3/` 下 9 张表：
  - `spells_characters.json` (`table_id = 26`, 108 行)
  - `spells_other.json` (`table_id = 28`, 27 行)
  - `spells_buildings.json` (`table_id = 27`, 18 行)
  - `spells_evolved.json` (`table_id = 13`, 109 行)
  - `spells_hero_form.json` (`table_id = 203`, 113 行)
  - `characters.json` (`table_id = 34`, 514 行)
  - `buildings.json` (`table_id = 35`, 151 行)
  - `projectiles.json` (`table_id = 10`, 312 行)
  - `area_effect_objects.json` (`table_id = 22`, 257 行)
- Buff 词表（265 个名字）来自 `character_buffs.json` (`table_id = 9`, 262 行) 加上动作内联 buff。
- **客户端 SCID → 词表 ID 映射公式**：
  - 皇室战争内部每个数据条目的 SCID 为 `scid = table_id * 1_000_000 + row["index"]`（例如 `Knight` 在 `spells_characters` 的 `index=0`，`scid = 26000000`）。
  - 可用下面一段 Python 从本地表生成 `scid -> vocab_id` 与 `name -> vocab_id` 静态 JSON 给 Windows 直接读取（不依赖 JVM）：

```python
import json, pathlib

tables_dir = pathlib.Path.home() / ".crforge/tables/16.635.3"
vocab_tables = [
    "spells_characters", "spells_other", "spells_buildings",
    "spells_evolved", "spells_hero_form",
    "characters", "buildings", "projectiles", "area_effect_objects",
]
names = set()
scid_to_name = {}
for t in vocab_tables:
    data = json.loads((tables_dir / f"{t}.json").read_text(encoding="utf-8"))
    tid = int(data["id"])
    for row_name, row in data["rows"].items():
        names.add(row_name)
        scid_to_name[tid * 1_000_000 + int(row["index"])] = row_name

vocab = sorted(names)  # len == 1466 (与 crforge Vocabulary.java 完全一致)
name_to_id = {name: i for i, name in enumerate(vocab)}
scid_to_vocab_id = {scid: name_to_id[name] for scid, name in scid_to_name.items()}
```

---

## 7. 合法动作掩码（`legal_action_mask[2307]`）构造规则

实机没有 `crforge` 的 Java 合法性判定器，需要在 Windows 适配层严格构造 `bool[2307]` 掩码，否则模型会把概率分配给非法牌或非法格子：

1. **动作 `0`（`WAIT`）**：永远为 `True`。
2. **出牌动作 `1 .. 2304`**：按 `slot = 0..3` 判定：
   - 若 `play_open_in_ticks > 0`（开局前冷却）、或 `hand_card[slot] < 0`（手牌空）、或 `hand_pending[slot] != 0`（已发出一张尚在等待生效）、或可用圣水 `elixir - elixir_reserved < hand_cost[slot] * 10000`：该槽位所有 `576` 格全部置 `False`。
   - **地面区域限制（参考 `DeployMask.java`）**：
     - **全图法术**（`CanDeployOnEnemySide=True`，如火球、箭雨、滚木起始范围等，其中如滚木/皇家速递仍有 `DeployStartY`/`DeployEndY` 行限制，需查表）：对应合法行全部置 `True`。
     - **普通部队 / 建筑**：双方公主塔均存活时，仅己方半场可放置（`row <= 14`，并按塔底/建筑占用关闭对应格子；镜像或特殊建筑同理）；若**敌方左/右公主塔已破**，则开放对应侧敌方口袋区（该侧前场约到敌塔前一行开放，国王塔及另一侧存活公主塔周围仍保留禁放矩形框）。
3. **英雄技能动作 `2305 / 2306`**：仅当对应 `ability_champion[k] >= 0`、场上该英雄存活、技能冷却为 `0` 且当前圣水足够技能费时置 `True`。

---

## 8. 实机闭环工程接入检查表（Windows 开发路线）

建议按以下 5 步逐步联调，切勿第一步就直接挂机自动乱点：

1. **Step 1：冻结版本与导出离线静态表**
   - 把本轮最终 `actor_export.pt` 与上面生成的 `vocab_mapping.json`（含卡名、SCID、费用、`CanDeployOnEnemySide`/`DeployStartY`/`DeployEndY`）拷入 Windows 工程目录。
2. **Step 2：只读数据流审计（Dry-Run 1 局真实对局）**
   - 接入实机状态源（例如先确认游戏版本号/架构，若使用 `cr-memory-reader` 或自研读取器，先跑通一局只读日志）。
   - 检查实机坐标朝向（红/蓝方是否已翻转成底部为己方）、手牌与 `hand_cost`、`elixir` 单位是否换算为 `× 10000`、未知字段是否为 `-1`。
3. **Step 3：Offline Dry-Run 推理叠加图**
   - 把录制或实时的实机帧按 `400 ms` 间隔送入 `StateInference`，打印每步耗时（目标远小于 `400 ms`）、动作概率分布与推荐落点 `(slot, col, row)`，人工核对模型建议是否符合直觉且无非法选牌。
4. **Step 4：ADB / 窗口触控标定与单发防抖**
   - 分别标定手牌 4 个槽位像素点与 `18 × 32` 棋盘矩形四边界（注意 Y 轴反转）。
   - **必须加单步 in-flight 去重锁**：发出一记“点击手牌 → 点击落点”后，标记本地 `in_flight`（对应 `hand_pending[slot]=1` 直到手牌轮换、圣水扣除或超时 `~1.5s` 解锁），防止手牌未刷新时连续重复下同一格。
5. **Step 5：受控闭环试跑与日志回溯**
   - 保存每步的 `(tick, raw_obs_summary, legal_count, action_id, screen_xy, infer_ms, click_ms, confirmed)`，统计 P95 延迟与未确认出牌率，再放开完整对局自动对战。
