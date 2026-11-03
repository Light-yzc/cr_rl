# 看真实对局回放

crforge 自带 LibGDX 桌面调试可视化器。这里使用与训练一致的上游 commit `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6` 和数据表 `16.635.3`。默认打开最近一次准备好的回放集合；显式传入 `-Replay` 可以选择历史回放。

## 当前默认：32局正式BC换边评估

根目录 `view-replay.cmd` 当前读取 `artifacts/replay_viewer/latest_replay.json`，指向
`artifacts/v2_cycle_final_eval_20261103/all_matches.jsonl`。其中前16局为
`BC CYCLE 1920 WAIT+1.5` 对 `BC EARLY 237 WAIT+0`，后16局对
`BC CONDITIONAL 896 WAIT+0`。双方都是BC，本轮没有PPO对局。

对early237为8胜8负，对conditional896为11胜5负，尚未通过PPO交接门槛。
旁边的 `evaluation.json`、`paired.json` 和 `execution_audit.json` 分别保存原始结果、
换边配对统计与实际成功出牌/支付审计。当前训练已继续到新的BC阶段，不能把该归档
当成最新训练权重的胜率证明。详见 [最新训练进度](TRAINING_PROGRESS_20261104.md)。

这些回放和本机缓存未提交到Git；新clone需要从自己的Drive备份取得，不能只靠仓库打开。

## 历史：step1280 的四局排查样本

`artifacts/v2_gui_bc_20261103_01/evaluation/all_matches.jsonl` 包含四场完整模拟器对局，使用两个固定种子、两组留出卡组并交换模型席位。面板和窗口标题明确显示：

- `BCv2 NEW 1280`：v2 BC 第 1,280 个优化步的冻结权重。
- `BCv2 EARLY 237`：同一 v2 架构第 237 步的早期权重。

双方都是 BC。第一场新模型在红方，第二场在蓝方；以面板标签为准。四场均无模拟器 fault/truncation，原始回放未筛选；新模型四场全负。这是可视化排查样本，不能用四场推断稳定胜率。

采样期间 BC 更新临时暂停，采样进程在 `finally` 中恢复训练。模型和数据集保留在 Colab/Drive，本地下载包只有 4,540 字节；`manifest.json` 记录源 checkpoint 和导出 actor 的 SHA-256。`evaluation.json` 保存卡组、席位与结果。

显式指定该历史归档即可打开这四局。空格暂停，`[` / `]` 切换对局，`R` 重播，`+` / `-` 调速。下面关于六场 PPO/BC 的描述也是历史样本。

上游：[crforge README / Watching a replay](https://github.com/voonhous/crforge/tree/7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6#watching-a-replay)。原有 Apache 2.0 许可证和源码保留在缓存目录中，未重写战斗规则。

## 打开

Windows 双击项目根目录 **`view-replay.cmd`**，或在 PowerShell 中执行：

```powershell
.\view-replay.cmd
# 单独看其中一局：
.\view-replay.cmd -Replay artifacts/bc_multideck_20261103/evaluation/match_000.json
```

依赖 Java 21，需确保 `java` 位于PATH。首次构建需要 Gradle 和 Java 桌面依赖，之后直接启动已经构建的程序。可视化在本机运行，无需继续占用 Colab GPU。

可视化器、固定版本游戏表位于 `.cache/replay-viewer/`；六份原生回放和合并的 `all_matches.jsonl` 位于 `artifacts/bc_multideck_20261103/evaluation/`。窗口启动日志和进程记录位于 `artifacts/replay_viewer/`。

## 操作

| 按键/操作 | 功能 |
|---|---|
| 空格 | 暂停 / 继续 |
| `.` / `,` | 暂停并前进 / 后退一个 50ms tick |
| `+` / `-` | 加速 / 减速 |
| `R` | 从头播放当前对局 |
| `F` | 翻转视角；默认是游戏展示 side 1 的视角 |
| `[` / `]` | 上一局 / 下一局 |
| `L` | 显示 / 隐藏对局列表 |
| 鼠标左键点单位 | 查看单位详情 |

这是调试风格的 2D 画面，可以观察单位移动、血量、出牌和战斗过程。它重演保存的真实模拟器输入，不会再次执行策略网络推理。**牌组面板标题会直接显示 PPO 或 BC**，换局和翻转视角时跟着更新；窗口标题也显示两个逻辑席位的模型身份。

仅在历史六局PPO/BC归档中，默认未翻转时，第 1、3、5 场 PPO 在红方（上方），第 2、4、6 场 PPO 在蓝方（下方）。按 F 会翻转视角和显示颜色，应以面板标签为准；当前32局归档的双方都是BC。

原始回放不改动。`scripts/prepare_replay_archive.py` 生成带显示名称的播放集及 `viewer/` 下的单场显示副本，转换格式参考固定 commit 的 `ReplayArchive.java`。模型席位和每局卡组、胜负仍以旁边的 `evaluation.json` 为依据。

## 重新取得缓存

本次从训练 VM 下载的完整输入包保存在 `artifacts/bc_multideck_20261103/replay_viewer_inputs.tar.gz`，SHA-256 为 `53037aecb5c2da6b477f144a48559ed9d13a01428fb750730cded294b12e6255`。它包含固定版本源码、游戏表和六份回放。训练日志和原始回放另已保存在 Drive 的对应运行目录。
