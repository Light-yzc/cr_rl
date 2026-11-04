# Colab CLI 连接与运行时存活复核

最新指令：用户要求停止CPU测试，改用一台L4直接验证。CPU长测在约405秒时停止，未通过75分钟/跨一小时验收。新会话为`bc-live-l4`（`gpu-l4-s-usw4c0-qctdzvvyw84q`）；当前已挂载Drive，并在真正前台kernel cell中完成安装/模拟器编译和141项远端测试（13.13秒），进入基线恢复流程。PPO仍未启动。

本机实际运行句柄见`artifacts/l4_live_20261104/worker.json`：前台exec客户端、独立连接守护，以及仅在守护存在期间阻止本机自动休眠的`caffeinate -i -w PID`。没有永久修改电源设置；手动关机/合盖/断网仍不受保证。当前已核验连接为busy，但GPU跨一小时/长训练存活仍需后续实际记录，不能根据启动认定通过。

已使用的完整前台准备入口保存在`scripts/colab_bootstrap_bc_repair.py`，以及Drive运行目录`tools/live_bootstrap_20261104.py`（后者为实际发送到kernel的原文）。该入口在已批准的ephemeral凭据服务上重试挂载，不重复索要OAuth；之后运行真实安装、构建、测试和BC监督。

## 社区与上游证据

- [维护者对SSH/tmux的答复 #160](https://github.com/googlecolab/google-colab-cli/issues/160)：建议在长任务期间保留实际连接；断开SSH后后台进程存在不等于平台还认为实例被使用。
- [社区跟进 #161](https://github.com/googlecolab/google-colab-cli/discussions/161)：报告过约一小时代理令牌过期导致CLI误判会话死亡，升级后解决。它说明“连接失败”和“VM被回收”必须分别核实，不证明本项目两次中断的唯一原因。
- [现行CLI会话文档](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)：旧keep-alive daemon已取消；依赖kernel活动。代理令牌约一小时过期，新命令可刷新，已有长连接不会自动重新认证。
- [超时问题 #82](https://github.com/googlecolab/google-colab-cli/issues/82)、[上游修复 #41](https://github.com/datalayer/jupyter-kernel-client/pull/41)、[默认timeout问题 #183](https://github.com/googlecolab/google-colab-cli/issues/183)：旧客户端超时后可能Event.wait(0)忙循环。timeout是总执行预算，不应保留默认30秒来跑长训练。
- [旧TFE ping修复 #61](https://github.com/googlecolab/google-colab-cli/pull/61)是历史版本机制。没有直接安装第三方fork或把旧ping说明当作当前CLI已启用保活的证据。

本机CLI为0.7.5。update查询发生SSL超时，不能依据其“up to date”后续输出认定已验证最新版本。本机`jupyter-kernel-client`实际为0.9.0，已从可信包源升级至1.0.2，并检查安装源码包含deadline/丢唤醒修复。CLI其他依赖随解析更新，未修改项目训练依赖锁文件。

## 落地方式

1. `scripts/colab_resume_bc_repair.py`在真正Colab内核cell里启动/接管后台controller并持续监督真实任务。它不是SSH后台sleep脚本。任务完成、失败、心跳或进度停滞时退出，不生成虚假GPU工作。
2. `scripts/watch_colab_session.py`在常开且联网的客户端机器保持现有kernel的实际WebSocket连接。定期从assignments刷新代理凭据，默认45分钟轮换连接，先建立新连接再关闭旧连接。断线时仅接回同一endpoint/kernel。
3. 连接守护器不执行用户代码、不创建VM、不创建kernel、不重启训练。单次网络错误或空列表不被当作重启许可；重复缺失明确报告assignment gone。HTTP头、proxy token、OAuth响应和kernel输出均不写入它的日志。
4. 主机断网/休眠/关机后客户端守护器也会中止，不能保证Colab继续运行。支持的后台执行套餐、实例寿命、额度和平台回收均不能被这套代码保证或绕过。

先在已创建的会话中执行真实kernel监督入口，再用CLI虚拟环境的Python启动连接守护：

```sh
colab exec --session SESSION --file scripts/colab_resume_bc_repair.py --timeout 15000
# 另一常开终端；PYTHON是安装colab CLI的虚拟环境解释器
PYTHON scripts/watch_colab_session.py --session SESSION --minutes 240 \
  --out artifacts/colab_watch.jsonl
```

`PYTHON`在这里是解释器路径占位符，不是要原样执行的命令。必须先确认会话存在，不能用可能auto-create的SSH命令代替存在性检查。当前机器解释器为`/Users/Regenin/.local/share/uv/tools/google-colab-cli/bin/python`。

## 验收证据与边界

- 本地141项测试通过，覆盖连接轮换先接后断、同kernel重连、单次缺失容忍、重复缺失终止、身份变化拒绝、网络失败不重启，以及真实任务完成/失败/心跳停滞/无进度检测。适配器回归还验证SDK发现kernel不存在时不会调用其隐式创建路径、关闭连接不关闭kernel、同名新endpoint不被冒充为原实例。
- 已在CPU会话实测旧问题触发方式：远端sleep3秒、CLI timeout1秒，客户端在13.59秒内返回TimeoutError并退出，没有无限空转。证据：`artifacts/colab_liveness_20261104/timeout_probe.json`。
- 原75分钟CPU连接诊断已按用户要求停止并释放CPU实例。实际覆盖90秒后关闭exec客户端、同kernel重连、持续至约405秒；未完成跨一小时验证。当前在真实L4任务上继续采集连接和kernel监督证据。
- 已实际覆盖初始断线：原始exec客户端已关闭，之后远端health继续推进到120秒，nonce与kernel PID均未改变；连接守护器出现过一次同kernel重连并再次读到busy。它是短测证据，不是75分钟结论。CPU测试当前计费显示约0.08计算单位/小时，不能称完全免费。
- 本地driver PID等实时句柄见`artifacts/colab_liveness_20261104/worker.json`；进度见`acceptance_75m/status.json`，最终结果必须以`acceptance_75m/acceptance.json`为准。未完成前不能说已经通过一小时或75分钟验收。CPU验收不等于GPU长训练寿命保证，也不验证客户端主机关机后的存活。

后续恢复训练前，先读最终验收、确认会话清理及实际连接行为，再恢复基线4480剩余616步。BC完整验证/对战仍未完成，不能因连接测试通过就启动PPO。
