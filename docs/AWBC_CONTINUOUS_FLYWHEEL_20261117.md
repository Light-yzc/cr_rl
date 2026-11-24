# 持续实战采集 → Drive → Colab AWBC

## 一键启动与停止

在项目目录 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_awbc_flywheel.ps1
```

默认两套现有预设 `2:10,1:10`，每批 20 局；循环运行直到收到 STOP。
启动器在后台运行，关闭启动命令窗口不会停止采集。停止命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_awbc_flywheel.ps1
```

停止会等待当前对局结束，随后尝试上传不足 10 局的最后一批。恢复：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_awbc_flywheel.ps1 -Resume
```

`-Resume` 会归档 STOP 标记，不删除对局。未完成批次继续使用原先冻结的 Actor。
不要同时启动另一个策略控制器操作同一台模拟器。

## 模型下行

- 默认启动时请求最新已提交 Actor，此后按 60 分钟刷新；已有未使用候选时，等当前批次结束再请求另一版，避免白下模型。
- Drive 恢复指针：`MyDrive/cr_rl_ppo/training/longrun/latest.json`。
- 指针中的 `actor` 指向该恢复点的 `actor_export.pt`；本地不下载 Critic、优化器或完整 PPO checkpoint。
- 本次实测 U985 Actor 为 **743,169,262 bytes**，architecture v3、8 tick。
- Colab 将 Actor 固定到 SSD 临时传输目录，分为 8 MiB 二进制块；客户端直接下载二进制，避免 base64 膨胀。
- 本地单个 `.partial` 支持按完成块续传；长度和 ZIP 结构检查通过后原子替换成 `.pt`，再更新 `latest_actor.json`。
- 原始 Actor 权重格式保留，未量化或修改架构。
- 本地缓存：`artifacts/awbc_flywheel/models`。在用 Actor 与待切换 Actor 最多各一份；完成批次后清理不用的旧 Actor，失败续传仅保留最新一份。
- Colab 临时传输缓存最多保留两个版本。Drive 继续保留完整训练恢复点供训练重启使用。

整批采集冻结一个 Actor。新模型下载在后台进行，只有批次边界才会切换，绝不中途替换一局的权重。每局保留 checkpoint SHA、Actor 指纹、架构、采样概率及版本；AWBC 继续按行为策略分组。

## 数据上行

1. 采集记录仍来自现有只读内存接口和动作桥，保留观察、合法掩码、概率、动作提交/确认以及真实终局。
2. 写入器关闭、结果校验完成后生成 `collection_sealed.json`，封存 manifest/result。封口后元数据变化会被拒绝。
3. 每满 10 局打一个 ZIP，只含训练必要的 JSON、事件日志和压缩帧，不传重复 NPZ 导出、源代码归档或模型。
4. 本地 outbox 保存待发送批次，失败后重试同一批；传输块为 8 MiB，不增加大文件哈希扫描。
5. Drive：`preparation/awbc_inbox/<batch>.zip`。ZIP 完成后才提交 `<batch>.ready.json`。
6. Colab 的独立 CPU 接收器每 60 秒检查 ready 批次，复制 ZIP 到 SSD，在 `sessions` 目录之外解压、检查，再将完整对局目录原子移入 `sessions`。
7. PPO 不等待上传。已有 PlayBuffer 在每个 AWBC 周期刷新，按现有训练导入规则检查观察/动作/结果。无效数据隔离并记录原因。

旧自动采集数据允许以最后的 `menu_lobby_verified` 事件证明写入结束。新流程使用显式封口；结果尚未稳定的文件不会进入上传队列。已有对局同名且元数据冲突时拒绝覆盖，重复批次不重复导入。

当前 AWBC 每 **50 次 PPO 更新**触发，默认批次 24 局、最少 12 局新的训练分区数据；行为组也必须满足 Value 拟合和留出门槛。刚上传少数几局会先等待，不保证下一周期立刻做 Actor 更新。AWBC 的 Value/留出/KL 门控保留。

如需立即上传不足 10 局的已完成部分，可以创建 `live/outbox/FLUSH` 文件；上传器处理完本次短批后自动恢复 10 局分批，不停止采集。

## 导航与异常恢复

支持这台 900×1600 私服客户端的主界面、牌组、结算、奖杯之路、游戏模式弹层、个人资料和退出确认框。

- 每次操作要求连续两次页面证据；特殊弹层使用多个局部标记。
- 奖杯之路点击底部“确定”；游戏模式点击顶部收起箭头；退出确认点击“取消”。本客户端 Android BACK 会弹出退出框，不能作为通用返回。
- 返回操作后必须再次确认主界面的对战按钮，才会开始下一局。
- 战斗状态由现有内存读取接口辅助判断。活动对局不会被菜单恢复逻辑重启。
- 未知页面不盲点；保存截图并重试。只有持续未知、反复确认没有活动对局且前台确为 `nullsroyale.rel.free` 时，才允许重启该私服应用。
- ADB 只连接配置的 `127.0.0.1:16416`，不杀共享 ADB server。
- 完整、结果已核验的对局不会因为随后菜单返回失败而被丢弃。结果不明或未完成的录制仍不计入配额。

菜单识别仍依赖当前版本的校准图像，不能保证任意 UI 改版都能自动处理。未知情况会保留证据并停止冒险点击。战场观察来自内存，不来自这些菜单模板。

Windows 默认要求至少 6 GiB 可用提交内存再启动一批推理；启用 CUDA 懒加载。显存空闲但系统分页/提交内存不足也会导致 CUDA OOM，此时开局前失败会等待资源恢复，不上传假对局。磁盘不足 5 GiB 时停止新批次并保留已有数据。

运行期间持有临时的 Windows 系统防休眠请求，退出后释放；不修改电源配置，也不取消已设置的定时关机。

本地采集监督器也检查已有事件日志；300 秒没有进展时终止该采集进程树，再通过内存/菜单状态恢复。未完成的录制不会充当完整训练对局。

## Colab 与重启

- 独立对战 valid 已设为 `validation_interval=0`，最佳模型沿用训练 traj 判断。
- traj 最佳链接在异步 checkpoint 写盘完成后更新，避免串错更新编号或阻塞每次保存。
- 本地 SSD checkpoint 和 Drive publication 均异步，保存间隔 5 次更新。
- PPO 控制器监测现有进度文件；600 秒无进展先请求退出，再按完整的 checkpoint/pool 配对恢复，最多自动重试 5 次。没有宣称修复原生模拟器的内部卡死。
- 数据接收器是独立 CPU 子进程，可单独重启，不需中断 GPU 训练。
- 代码和配置通过既有 `preparation/crrl_training_update.zip` 与 `cr_train_config.json` 持久化。新实例仍从 notebook 的共享准备脚本恢复 SSD 和接收器。
- 新数据 ZIP 持久化在 Drive inbox。SSD 消失后重新导入，消费次数继续以 PPO checkpoint 的 PlayBuffer 状态为准。

## 已完成的验证

- 完整相关回归 **42 passed**：传输中断、重复批次、路径检查、元数据封口、Actor 续传、缓存清理、冻结版本恢复、菜单恢复、异步保存/发布与监督器。新增短批次上传后，其所在的 **23 项定向回归也全部通过**。
- MuMu 实测 6 次导航恢复：奖杯之路两次、游戏模式、个人资料、退出确认和牌组页，均回到主界面。
- U985 完整采集两套卡组各一局：475 帧胜局、445 帧负局；均记录真实终局并上传到 Drive。
- 在此之前，一局已有真实录制经新链路传输后，云端 AWBC 导入通过（277 帧，67 个可用 Actor 标签，终局 reward=-1）。
- 上述两局新录制均通过云端 AWBC 导入，各有 154 个可用 Actor 标签，reward 分别为 +1/-1。
- 正式飞轮随后下载 U1000 Actor 并开始采集；U1000 的首份显式封口录制经云端导入通过：440 帧、148 个可选决策中 147 个可用标签，1 个执行歧义被排除，reward=-1。此小样本不能代表长期利用率。
- 验收时 Colab SSD 已从原 80 局增长到 **85 局**，接收无报错。读取 U1005 checkpoint 的 PlayBuffer 状态副本后刷新，识别新增 3 局、总 85 局、无导入排除；没有修改正式消费次数。两个新行为组当时分别仅 2/3 局，仍等待足够训练与留出样本。
- 验收时正式后台采集继续使用冻结的 U1000，最新候选 U1005 已下载，批次结束后切换；模型缓存仅这两份 Actor。已实测正常结束当前对局、上传剩余短批、恢复未满配额批次，未混用模型。
- 这次云端代码同步只重启 CPU 接收器，GPU PPO 进程保持运行；验收读取到 U1008，Drive 完整恢复点为 U1005。

## 看日志

正式目录：`artifacts/awbc_flywheel/live`。

```powershell
Get-Content .\artifacts\awbc_flywheel\live\console.log -Tail 20
Get-Content .\artifacts\awbc_flywheel\live\uploader.log -Tail 10
Get-Content .\artifacts\awbc_flywheel\live\model_sync.log -Tail 10
Get-Content .\artifacts\awbc_flywheel\live\flywheel.json
```

每批的 `collector.log` 和每局的 `events.jsonl` 在 `live/sessions` 下。
Colab 接收日志：`runs/longrun/awbc_inbox.log`；接收状态：`prepared/awbc_inbox_state.status.json`。
