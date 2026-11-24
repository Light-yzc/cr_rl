# 模型数据接口接入验收（2026-11-16）

这份文档补充 `NULLS_MODEL_INPUT_GAPS_HANDOFF_20261116.md`。原文件是最初缺口清单，
其中大量“恒为-1”的描述已被后续 reader/adapter 实现取代，不能继续当当前状态使用。

## 本轮完成

1. 核对现有 reader → adapter：真实状态、攻击/前摇、Buff速度倍率、弹道/区域字段、
   卡牌形态、服务端pending/预留、补牌、技能状态等已由现有数据接口映射到7024维输入。
2. `run_nulls_policy.py` 现在保存每次发牌的 battle-frame请求落点。后续快照使用
   `pending_targets_for` 按原deck index匹配卡牌，将落点传给adapter。手牌换槽不会
   把旧请求坐标赋给另一张牌；没有匹配来源时保持未知。仅匹配游戏中实际pending记录。
   坐标来源是**本机请求点**，不是从服务器命令队列读出的实际位置。
3. 录制增加 `adapter_context`：local pending_slots、pending_targets_battle和来源。
   离线回放同时还原这些上下文，不再只有raw snapshot；旧录制仍兼容。
4. 每帧日志/训练诊断记录 `input_quality`：按角色/塔统计state、attack_ms、load_ms、
   speed_percent、hit_speed_percent缺失数，以及预留圣水/补牌/pending是否未知。
   projectile不适用的攻击字段不会误计为单位输入缺失。dx/dy负值不被视作缺失。
5. manifest增加版本化 `input_contract`，说明技能输入存在但技能动作关闭、pending
   坐标来源、未验证场景及不能同构的拒绝码。7024/2307布局保持不变。
6. 启动环境：优先现有 `F:/python/python.exe`，可用 `-PythonExecutable` 或
   `CRRL_PYTHON` 指定；备用已有cs285。增加 `-Checkpoint` 参数，默认仍使用最新导出。
   启动记录解释器/权重路径，连接选定loopback模拟器，检查root ADB；必要时恢复
   已配置模拟器的root adbd，10秒命令超时，失败明确停止，不使用部分/伪造快照。

## 验证证据

- 本轮 **52项**客户端/runtime/回放上下文回归测试通过，含双席位坐标、手牌换槽、
  无来源坐标保留未知、转换序列化复现、root恢复/拒绝路径、模型输入缺失计数。
- `final_verify_20261116_v2` 53帧重建：`old_model_obs_mismatches=0`。
- 新增 `scripts/validate_nulls_model_interface.py`，实际严格加载本地最新PPO **929**，
  在这53帧上做CUDA FP16前向：logits全部有限、形状(1,2307)、概率和及合法mask检查通过。
  中位前向20.2ms；含首次冷启动最大7202.6ms。它是离线接口检查，不是L4时延证明。
- 权重SHA256：`80a0dc9f4428e8caaa928a6b89f93465d599f60ae9ad20ea076640b1e0441f25`。
- 真启动器使用 `-DryRun -Seconds 2 -MaxPlays 1` 实跑到ready，确认PPO929载入且能连接
  当前MuMu PID2623。客户端处于大厅，0次动作，正常按时限退出；本轮没有另外声称打完新对局。

## 界限

- 新reader已实现的持续AOE和觉醒循环，仍需要相应实机场景印证；此前53帧无这些场景，
  不能把“不为-1”或FakeMem测试当所有偏移都完成实机验证。
- 技能状态/冷却/charges进入模型；技能点击仍禁用。缺两按钮标定及独立技能确认逻辑，
  不通过普通卡牌played_count轮换来冒充技能确认。
- last_refusal、last_run_refusal、illegal、last_ability_run_refusal无已验证的客户端
  同构来源，保持UNKNOWN，不用0冒充成功。
- action_selection仍使用原有部署阈值/法术保护，属于off-policy部署选择；本轮没有改变
  这些策略，也没有训练/替换权重或宣称最新模型是最佳模型。

## 命令

```powershell
# 默认使用新接口和最新导出，原启动命令不变
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_nulls_policy.ps1

# 仅回放录制数据验证模型，不连ADB、不点击
$env:PYTHONPATH='G:/Code/cr_rl/src'
& 'F:/python/python.exe' scripts/validate_nulls_model_interface.py
& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/replay_nulls_diagnostics.py artifacts/nulls_live/final_verify_20261116_v2
```
