# 模型输入复查与ADB实测（2026-11-16）

## 修复状态更新

下文两处映射问题现已修复，历史观察保留。

- 实体仅接纳已验证vtable和正ID。空对象、未支持布局不再默认当普通角色；原始快照
  仍保留，并通过`rejected_entities`和`entities_rejected`事件记录eid/vtable/name/原因。
  它们不参与模型实体、SeatId、目标引用和部署碰撞计算。未声称已解码未知塔弹道。
- Buff总数使用完整列表长度，前三个槽仍排序截取，永久/未知sentinel保持不变。
  四Buff测试验证总数4、三个槽及速度叠加结果。
- manifest input_contract升为3。53项回归测试通过，含假塔名字、空对象、未知布局、
  同tick幂等、SeatId和mask保护，以及已有reader/adapter/控制链路回归。
- 重建旧53帧录制，隔离tick407的空对象、tick606的未支持塔弹道。所有合法动作
  mask不变，归一化后的实际模型输入仅这两帧改变。原始obs另有后续SeatId重新编号，
  但id/target列在预处理归零。证据：`artifacts/nulls_live/mapping_repair_report_20261116.json`。
- 修复后的53帧输入通过最新PPO929 CUDA FP16前向，logits全部有限、mask检查通过。
  证据：`artifacts/nulls_live/mapping_repair_model_validation_20261116.json`。
  重建数据仅供接口验证，不包含冒充新策略输出的旧概率，不是新采集的训练轨迹。

四个无同构来源的拒绝码继续保留UNKNOWN；AOE/觉醒/技能动作仍需各自实机验证。

## 以下为修复前排查依据

范围：核对当前代码、PPO929字段定义、已有完整录制及当前MuMu只读采样。
本轮没有修改reader/adapter的字段映射，没有启动模型自动出牌。

## 当前ADB实测

连接 `127.0.0.1:16416`，root正常，Nulls PID2623。客户端最初在大厅，进入训练营
只读采样25秒，不发牌。原始证据：`artifacts/nulls_live/input_recheck_live_20261116/`。

- 56帧、411条实体记录：112条国王塔、290条普通单位、9条已识别弹道。
- state/attack_ms/load_ms/speed_percent/hit_speed_percent 在适用的单位/塔行中
  未出现UNKNOWN，读页错误0，未识别vtable为0。
- 单次读取平均29.44ms，最大46.57ms。
- 无超过3个Buff的实例，无持续区域效果、无觉醒循环实际触发；不据此宣布这些场景验证完成。
- 没有己方出牌，不能用这段0-pending样本证明服务端pending所有过渡均可靠。

## 仍然存在的实际缺口

1. `last_refusal`、`last_run_refusal`、`illegal`、`last_ability_run_refusal`：
   客户端没有已经验证的模拟器同构拒绝码来源。运行器没有传入，保持UNKNOWN。
2. 手动出牌或未被本机追踪的命令，其pending落点仍未知。本机命令已经补传请求点，
   但那是local requested point，不是服务器队列中直接读取的最终落点。
3. 复杂同时出牌、法术、多波生成和潜地等敌方历史仍由实体/计数推断，不是完整客户端
   事件流；可能漏记、晚记、聚类错误，需要事件级验证。
4. 持续区域效果、正倍率Rage、真实觉醒循环、加时/阶段变化仍缺相应场景验证。
   已有代码路径及FakeMem测试，不等于相关动态内存语义已经全部实机确认。
5. 技能状态已接入输入；双技能按钮与技能命令确认尚未接通，属于动作接口缺口。

## 本次静态/旧录制复查发现的两处映射问题

### A. 未知实体类型默认当成普通单位

`NullsToSchemaV4Adapter._entity_kind()` 对不匹配既有vtable/塔判断的实体，最终
直接返回KIND_CHARACTER。

在 `final_verify_20261116_v2` 的53帧603行录制中实际存在：

- `vt=0x1a05af0, eid=0, name=''`：空/未识别对象被归入角色行。
- `vt=0x1a4fff8, eid=4000020, name='TowerPrincessProjectile'`：未支持布局的
  塔弹道被归为普通单位，坐标为0且HP/状态/等级等字段缺失。

这不是再找一个HP偏移就能解决：应验证真实对象类别/生命周期，拒绝空对象，
对未支持布局显式诊断；只有验证过的布局才能进入模型，不应套用Character布局。
当前25秒实测未复现这两类，所以不能说问题已消失。本轮只审计，没有更改分类逻辑。

### B. Buff总数被前三槽数量截断

当前adapter在 `len(buffs)>3` 时写 `buff_count=min(len(buffs),3)`。
核对训练后端 `.tmp_crforge` 的实际HEAD：
`7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`，与checkpoint backend_commit一致。
其 `LadderEnv.java` 写 `buff_count=items.size()`，只对展示的三个Buff槽做截断。

因此四个Buff时，实机接口总数为3，训练侧为4。这是确定的语义差异，虽没有破坏
shape或产生NaN，但应修改为真实总数，仍仅保留前三槽内容，并更新多Buff回归测试。
当前实机段没有多Buff实例，不能通过这段数据验证修复效果。本轮未修改此逻辑。

## 容易被误判为缺失的合法值

- 没有Buff时 `buff_0/1/2=-1`，未使用槽剩余时间为0，是正常情况。
- `ability_ticks=-1` 在未运行技能或不适用实体上是NO_STATE。
- 弹道/区域 `state=-1` 合法；它们的HP、攻击计时等不适用值不能算成普通单位缺口。
- 未见的敌方卡牌/觉醒信息、未占用的pending槽和空技能槽可以合法为-1。
- dx/dy是两次观测的位移，负数（包括-1）是合法方向信息，不等于未知。
- 预留圣水/队列并发释放时允许短暂未知。旧53帧中有1帧此情况；不能用0掩盖读失败。

## 复现

```powershell
$env:PYTHONPATH='G:/Code/cr_rl/src'
& 'F:/python/python.exe' scripts/inspect_nulls_inputs_live.py --out artifacts/nulls_live/new_read_audit --seconds 25
```

脚本不加载模型、不执行触控，只读当前对局。若在大厅，只记录nonbattle，不能拿到
战斗字段。检查必须按实体kind和实际触发条件进行，不能单看全局“-1数量”。
