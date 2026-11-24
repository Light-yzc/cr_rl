# Nulls Royale 下载与读取器兼容性核查

本轮仅请求网页、源码文本和 APK 的 HTTP HEAD；没有下载、安装或执行 APK，也未启动训练。

## 核实结果

- Nulls 团队频道 https://t.me/nulls_ru/1033 的 2026-11-02 更新公告明确提及 Nulls Royale、新卡与 11 月平衡调整，给出 Android 下载入口 `http://dnull.xyz/latest_cr`。这是近期维护证据，不是当前对局服务实时健康证明。
- 请求 `https://dnull.xyz/latest_cr` 的 HEAD 返回 HTTP 200，重定向到 `https://tempweb.nullsusercontent.com/fpapk/nr_16.635.1_release_951fe421.apk`，Content-Type 为 `application/vnd.android.package-archive`，Content-Length 为 841223777 字节（约 802.3 MiB）。16.635.1 来自文件名，尚未读取 APK manifest 核实版本或包名。
- `https://nulls.gg/` 及 Royale 页面在本次请求返回 403。这不能证明游戏停服，也不能证明下载不可用。
- 用户截图显示 Nulls Clash，而不是 Nulls Royale；Clash 的 TimeoutExceeded 无法证明 Royale 停服。

## Windows 读取链路

源码：https://github.com/Jason-XII/cr-memory-reader

- 查询 GitHub 元数据得到 pushed_at 为 2026-08-25T07:09:02Z。
- README Windows 步骤指定 MuMu、Android 12、root 和 ADB。
- `x86_64/tools/capture_native_entity_stream.py` 默认 package 为 `nullsroyale.rel.free`，serial 为 `localhost:5557`，会在 adbd 非 root 时退出。
- `x86_64/My Own Findings.md` 出现历史分析包 content version 13.300.6，并记载过旧包被要求升级。不能据此认定整个读取器只能支持该版本，但没有当前 16.635.1 的适配证明。
- `deploy/proc_memory.py` 和 native helper 含固定结构/偏移；技术指南要求按 libg 指纹建立 profile。因此当前 APK 能下载，不代表现有读取器可直接使用。
- MuMu 官网 https://www.mumuplayer.com/ 当前描述 Android 15/12 实例支持；匹配读取器优先使用 Android 12。尚无本项目模拟器性能对照实验。

## 尚未验证

安装后的 manifest/ABI/签名；模拟器实际 root adbd；成功登录和完整对局；libg 指纹与读取 profile；实体/血量/手牌/圣水等连续采集正确性；到模型 Schema v4 的语义一致性。

结论：有近期维护公告和可访问的发布下载入口，不应称为“已停服的废包”；当前网络的游戏服务可用性和当前版状态读取仍需最小实测，不能承诺下载后即能接模型。

## 本机启动失败实测

通过 MuMuManager info 与 ADB 只读诊断，实际实例为 MuMu Android 15.0，ADB `127.0.0.1:16384`，Royale 包 `nullsroyale.rel.free`，versionName `16.635.1`、versionCode `160635001`、ABI `arm64-v8a`，经 Houdini 转译。

进程 5583 的日志显示启动顺序：

```text
17:52:30.342 I WILDCAT: Hello
17:52:30.927 F Frida: Unable to locate bsd_signal(); please file a bug
17:52:30.927 I houdini: [SIGABRT] abort
17:52:30.931 I houdini: arm backtrace ... libwildcat.so
17:52:35.912 I WildcatLoader: Loaded with status: TimeoutExceeded
17:52:35.912 I WildcatLoader: Script result: null
```

因此本次 TimeoutExceeded 前有明确的客户端加载组件/转译兼容性失败，不是服务器停服证据，也不应先以网络超时解释。尚未验证换 Android 12 后是否恢复；优先用读取器文档指定的 Android 12 实例保留同一 APK 对照测试，不必重新下载游戏或修改 APK。本轮未创建新实例、未修改系统/游戏。

权限备注：用户已在 UI 开启 root/ADB；本次普通 `adb shell id` 返回 uid=2000，`su -c id` 返回 su 不存在。仅证明当前 shell 未取得 root，尚未测试 `adb root`，不能据此断言整个实例无 root 能力；它与上述启动组件错误应分开排查。
