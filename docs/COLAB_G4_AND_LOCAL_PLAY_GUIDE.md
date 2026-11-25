# 跨电脑开发、Colab G4 训练启动与本地私服测试操作指南

本指南用于每天切换电脑时，在 **3 分钟内** 拉取代码、连接/恢复 Google Colab **G4 (`RTX PRO 6000 Blackwell 95GB`) / A100 / L4** 训练任务、保持后台保活监控，并将最新模型一键高速拉回本地（如 RTX 4090 + MuMu 模拟器）进行私服实战验证。

---

## 一、新电脑首次准备（仅需一次）

### 1. 克隆/更新代码仓库
```powershell
git clone <你的仓库URL> cr_rl
cd cr_rl
git checkout codex/training-game-budget
git pull
```

### 2. 配置 `google-colab-cli`（用于本地终端远程查看/保活/高速下载）
```powershell
uv tool install google-colab-cli
colab auth login
```
> **注意**：`scripts/colab_g4_ops.py` 严格只连接你账号下**已经打开的** Colab 实例（`list_assignments()`），绝不会私自新开计费实例。

### 3. 本地模拟器与推理环境（私服实战用）
- **Python 环境**：安装支持 CUDA 的 PyTorch、`numpy`、`pyyaml`（或直接设置环境变量 `$env:CRRL_PYTHON = "你的python.exe路径"`）。
- **游戏数据表**：确保 `%USERPROFILE%\.crforge\tables\16.635.3` 存在（若新电脑未初始化，可在 WSL/Linux 运行 `bash scripts/bootstrap_linux.sh` 或从旧机复制 `~/.crforge/tables/16.635.3`）。
- **MuMu 模拟器**：分辨率设为 **`900x1600`（竖屏）**，开启 ADB 调试（默认端口 `127.0.0.1:16416`），确保 `adb` 在 `PATH` 中或设置 `$env:CRRL_ADB = "路径/adb.exe"`。

---

## 二、Colab G4 训练：启动、断线恢复、保活监控与安全停机

训练的所有持久化状态（最新一代完整 `ppo_latest.pt`、历史对手池、`best_validation/actor_best_val.pt`、代码覆盖包与配置）均自动保存在 Google Drive 的 `/content/drive/MyDrive/cr_rl_ppo/` 下。

### 场景 A：Colab 实例仍在运行（换了一台电脑继续接管）
在本地 PowerShell 用 `colab-cli` 自带的 Python 直接运行 [colab_g4_ops.py](../scripts/colab_g4_ops.py)：

```powershell
# 定义 colab-cli Python 快捷变量（Windows 默认路径）
$colabPy = "$env:APPDATA\uv\tools\google-colab-cli\Scripts\python.exe"

# 1) 单次心跳保活 + 查看 GPU、最近 8 轮吞吐/胜率/EV/KL 与验证集战绩
& $colabPy scripts/colab_g4_ops.py status

# 2) 持续后台保活监控（每 15 分钟自动发 WebSocket 心跳并刷新代理 Token；实例关闭自动退出）
& $colabPy scripts/colab_g4_ops.py watch --interval 900

# 3) 安全重启或安全停机（停机会等待当前轮完整保存并同步到 Google Drive）
& $colabPy scripts/colab_g4_ops.py restart
& $colabPy scripts/colab_g4_ops.py stop
```

### 场景 B：Colab 实例刚重开（已挂载 Google Drive），一键恢复到 SSD 并续训
如果已在浏览器中打开了任意 GPU 实例（G4 / A100 / L4 / T4）并完成了 `drive.mount('/content/drive')`，可直接在本地一键恢复并启动（自动根据当前显存大小适配 G4 95GB / L4 22GB / T4 15GB 参数，绝不 OOM）：
```powershell
$colabPy = "$env:APPDATA\uv\tools\google-colab-cli\Scripts\python.exe"
& $colabPy scripts/colab_g4_ops.py restore-and-start
```
或者直接在浏览器打开仓库中的 [notebooks/colab_g4_heavy_mcts.ipynb](../notebooks/colab_g4_heavy_mcts.ipynb) 运行前 **2 个单元格**：
   - **第 1 格**：挂载 Google Drive，将模拟器、最新 `g4_heavy_mcts_overlay.zip` 代码包、`sparky_plan_b` 数据集以及最新一代 `ppo_latest.pt` + 对手池高速解压到 `/content` 本地 SSD，并按显存自动生成 `cr_train_config.json`。
   - **第 2 格**：启动后台守护进程 `cr_train_pipeline.py start`（即使关掉浏览器标签页，只要内核保活，后台子进程也会持续训练并每轮异步备份到 Drive）。
3. 随后即可在本地终端运行 `& $colabPy scripts/colab_g4_ops.py watch --interval 900` 接管保活。

---

## 三、一键高速下载模型并在本地 MuMu 私服测试

### 1. 从正在运行的 Colab G4 高速拉取最新/最佳权重（16 线程压缩分块 + SHA256 校验）
```powershell
$colabPy = "$env:APPDATA\uv\tools\google-colab-cli\Scripts\python.exe"
& $colabPy scripts/colab_g4_ops.py download-actors --out artifacts/g4_heavy_mcts
```
下载完成后，`artifacts/g4_heavy_mcts/` 下会自动生成：
- `actor_best_val.pt`：最近正式 100 局验证集胜率最高的稳健权重（**实战首选推荐**）
- `actor_latest.pt`：当前最新一轮导出的实时权重
- `manifest.json`：记录对应的 `update` 轮数与完整 SHA256 指纹

> 如果 Colab 已关机，也可直接从 Google Drive 的 `MyDrive/cr_rl_ppo/training/longrun_sparky_value_bc_v2/best_validation/actor_best_val.pt` 或 `generations/update_XXXXXX/actor_export.pt` 下载。

### 2. 启动本地私服常驻推理控制器（加载一次显存，无限连打）
确保 MuMu 已启动并进入私服大厅，然后运行 [start_nulls_policy.ps1](../scripts/start_nulls_policy.ps1)：

```powershell
# 默认自动寻找 Python、ADB 与 artifacts/g4_heavy_mcts/actor_best_val.pt（或 actor_latest.pt）
powershell -ExecutionPolicy Bypass -File scripts/start_nulls_policy.ps1

# 也可显式指定权重、模拟器端口或 Python 路径：
powershell -ExecutionPolicy Bypass -File scripts/start_nulls_policy.ps1 `
  -Checkpoint artifacts/g4_heavy_mcts/actor_latest.pt `
  -Serial 127.0.0.1:16416 `
  -SelectionMode argmax `
  -TouchTransport shell
```
看到控制台输出 `"event": "ready"` 与 `"waiting_for_battle"` 后，直接在模拟器中点击进入对战即可自动接管出牌；每局录像与决策遥测会自动保存在 `artifacts/nulls_live/play_<时间戳>/`。

---

## 四、当前核心训练与客户端防变形架构说明

1. **G4 Heavy-MCTS 加速管线**（`u=1651+`）：
   - 原生 JVM `fork_from_lane` 状态克隆（[crforge.py](../src/cr_rl/envs/crforge.py)）+ 144 路并行双阶段 `MacroCardSearch`（[macro_card_search.py](../src/cr_rl/training/macro_card_search.py)）+ 504 路并行 `LookaheadDistiller`（[lookahead_distiller.py](../src/cr_rl/training/lookahead_distiller.py)），G4 采样吞吐达 `1,150–1,429 sps`，PPO 单轮优化仅 `~40s`。
2. **逆风局防变形与防秒甩保护**：
   - **训练侧**（[lookahead_distiller.py](../src/cr_rl/training/lookahead_distiller.py)）：对称 `mean_p_wait` 信任域门控（防止连续负向漂移压垮 4~5 费憋费能力）、平衡背景帧权重，并专门纳入**逆风翻盘帧（`1.6x` 加权）与小负局中经 Critic 验证的纯 MCTS 防守修正帧（`override_only`）**，消除顺风 3 冠幸存者偏差。
   - **客户端侧**（[run_nulls_policy.py](../scripts/run_nulls_policy.py) & [action_selection.py](../src/cr_rl/client/action_selection.py)）：自动将私服专属超模/英雄单位（如 `ChefTower`、`Ronin`、`MiniPekkaHero`、`ValkyrieHero`）映射到已充分训练的词表嵌入；空场手握 6 费核心（电车/采集器）时阻止 4 费桥头秒甩；满费防漏优先沉底投资；单路重兵压境时锁定同路/中场拉扯防守格，杜绝逆风向空路桥头乱拍变形。
