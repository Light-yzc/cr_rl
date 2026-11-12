# Colab Drive 挂载与恢复

## 用户提供的取数入口

2026-11-10 用户提供的项目文件夹：

[Google Drive 文件夹](https://drive.google.com/drive/folders/13zk63XLqt_n_J1tGXf0SNb0TedCafQGG?usp=drive_link)

此链接作为目录定位入口；是否可免登录下载以实际权限为准，不改变文件分享设置。当前BC训练目录为MyDrive/cr_rl/runs/bc_salvage_fp16_20261109。需要本地试验时，数据、模型、缓存和结果写入G:\Code\cr_rl，TMP/TEMP、PYTHONPYCACHEPREFIX、CUDA_CACHE_PATH、TORCH_HOME和TORCHINDUCTOR_CACHE_DIR也显式指向G盘；不在C盘存放实验文件或安装新环境。

## 推荐流程：Colab CLI + Drive ephemeral 授权

2026-11-10补充：本轮`state_global_l4_20261110`已实测CLI挂载，授权后出现`Credentials propagated`并最终`Mounted at /content/drive`。若CLI `exec`自动kernel连接失效，不要重开GPU；可用标准Jupyter JSON websocket连接同一个已存在kernel，脚本见`.tmp_latest/state_runtime_ws.py`。必须复用当前session的代理token、显式kernel ID，不能把token打印或写进源码；本机代理仅作网络传输。`colab upload/download`仍可独立使用，不需要重新授权Drive。

先创建或选择 L4 runtime，然后执行：

```powershell
colab drivemount -s <session> /content/drive
```

CLI 输出授权地址后，必须保留 URL 中完整的 `state` JSON，包括：

- `token`
- `endpoint`
- `scopes`

部分 CLI 版本会输出 `response_type=none+gsession`，Google 会把它当成非法类型。只将这一段改成：

```text
response_type=none%20gsession
```

其余参数保持原样，尤其不要删掉 `state.scopes`。浏览器显示 `Please close this tab / window` 后，回到 CLI 按 Enter。成功标志是：

```text
[colab] Credentials propagated. Resuming mount...
Mounted at /content/drive
```

如果凭据传播成功但 DriveFS 报 `mount failed`，不要重新授权，直接在 runtime 中运行：

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

如果仍失败，先清理残留挂载点，再重试：

```python
!fusermount -uz /content/drive 2>/dev/null || true
!rm -rf /content/drive
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

## 授权后检查

```python
from pathlib import Path
root = Path("/content/drive/MyDrive/cr_rl/runs")
print(root.exists(), root)
```

确认能看到目标 run 后，再复制 checkpoint。训练过程只从本地缓存读取，结束时再同步到 Drive，避免直接在 Drive 上高频写大文件。

## 网页端 fallback

如果 CLI 的 kernel 连接中断，打开对应 runtime 的网页 URL，在第一个 cell 执行：

```python
from google.colab import drive
drive.mount("/content/drive")
```

网页端完成挂载后，CLI 和网页共享同一个 runtime，CLI 可继续执行剩余脚本。

## rclone 备用方案

长期运行不要把 OAuth token 写入 notebook。将本机完成 OAuth 后的 `rclone.conf` 放入 Colab Secret，启动时写入临时路径：

```bash
mkdir -p /root/.config/rclone
printf '%s' "$RCLONE_CONF" > /root/.config/rclone/rclone.conf
rclone copy gdrive:cr_rl/runs/<run> /content/drive_cache/<run>
```

训练结束只同步候选 checkpoint、最新状态和日志：

```bash
rclone copy /content/drive_cache/<run> gdrive:cr_rl/runs/<run> \
  --include 'candidate/**' --include '*.json' --include '*.log'
```

不要把 `rclone.conf`、refresh token 或临时授权 URL 提交到 Git 或公开 notebook。
