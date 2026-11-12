# Colab自动恢复：Secret-backed rclone

原生`drive.mount()`属于当前运行时的交互授权，换新L4可能再次要求授权。恢复流程可以使用rclone的Google Drive remote；refresh token只放在Colab Secret，不写入仓库、日志或checkpoint。

一次性在本机完成`rclone config`并创建Google Drive remote，然后把配置base64放入Colab Secret：

```powershell
$config = Get-Content "$env:APPDATA\rclone\rclone.conf" -Raw
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($config)) | Set-Clipboard
```

Secret名称固定为`RCLONE_CONFIG_B64`。不要把配置、refresh token或base64值发到聊天或提交Git。

新L4启动后安装rclone并拉取：

```bash
apt-get update -qq && apt-get install -y -qq rclone
python scripts/colab_rclone_sync.py pull \
  gdrive:cr_rl/runs/122m_long1000_sample2x_20261106 \
  /content/crrl-drive-cache
```

训练使用cache目录。每轮checkpoint、rollout和receipt用不可变文件名；上传成功且SHA核对后，最后才更新`latest.json`：

```bash
python scripts/colab_rclone_sync.py push \
  gdrive:cr_rl/runs/122m_long1000_sample2x_20261106 \
  /content/crrl-drive-cache
```

脚本使用`--immutable`，拒绝覆盖同名不同内容的文件；配置临时文件为600权限，输出不包含Secret。它解决新运行时的Drive交互授权和文件恢复，不能突破Colab寿命或配额。

当前L4已重新创建但Drive尚未挂载；109号完整checkpoint仍在Drive。下一步应先pull到cache，再启动训练，不能在空cache初始化新模型。
