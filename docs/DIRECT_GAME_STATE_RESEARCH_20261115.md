# 非 CV 游戏状态读取调研（2026-11-15）

目标是直接读取游戏结构化状态，再交给现有 v3 模型。此次只下载、检查源码，没有安装或执行第三方注入脚本，没有修改训练流程。尚未验证当前版本客户端的实时读取。

## 项目与实际能力

下表日期为 GitHub `pushed_at`，不使用可能由其他活动改变的仓库更新时间判断代码是否适配新版本。

| 项目 | 最后推送 | 检查结果 |
| --- | --- | --- |
| [iGio90/frida_cr_proxy](https://github.com/iGio90/frida_cr_proxy) | 2017-11-17 | Frida 挂钩 `libg.so` 加解密路径，硬编码偏移，仓库描述限定 x86。输出消息编号和明文载荷；定义文件只有登录消息 10101，没有完整战场实体解码。不能直接给模型输入。 |
| [weeco/supercell-packet-listener](https://github.com/weeco/supercell-packet-listener) | 2018-03-24 | README 明确仅提供未加密的包元数据，不解密载荷。无法获得单位、血量、坐标。 |
| [Galaxy1036/TwistedProxy](https://github.com/Galaxy1036/TwistedProxy) | 2018-11-22 | README 指定游戏 v2.4，依赖 Frida 和旧版加密绕行方式。没有证据表明适配当前版本。 |
| [royale-proxy/node-cr-proxy](https://github.com/royale-proxy/node-cr-proxy) | 2018-07-14 | README 标识游戏 1.9.2，要求修改 APK，并明确项目已放弃维护。 |
| [royale-proxy/cr-proxy](https://github.com/royale-proxy/cr-proxy) | 2017-11-26 | 旧协议代理，要求修改 APK。没有当前版本完整战场状态读取的验证。 |
| [doio/ClashRoyaleProxy](https://github.com/doio/ClashRoyaleProxy) | 2016-04-16 | 旧 C# 代理，安装步骤要求当年的修改版 APK。 |

前两个项目源码已保存到 `.cache/direct_state_research/`，总源码约 71 KB。`sources.json` 保存固定 commit 和逐文件 SHA256；未执行这些源码。Frida 项目固定 commit 为 `912065bd9f73b344ade02ad509c77acafe2732d7`；packet-listener 为 `dff096ba639e03e109c772885eab2a6ce7cdf2c2`。其余项目仅查仓库元数据和 README。

## 最简单的接入路线

`游戏客户端内存状态 → 字段/单位 ID 映射 → CrForgeObservationAdapter → StateActorCritic.step`

不增加 CV，也不另建一套模型。先用目标设备上的准确游戏版本验证读取，再决定是否有可复用的读取器；目前没有证据可以把以上老项目直接用于当前客户端。

现有模型入口在 `src/cr_rl/data/observations.py` 和 `src/cr_rl/models/state_policy.py`。除了手牌、圣水和时间，还需要实体的类型、阵营、坐标、HP、等级、状态、目标、计时器等字段，以及塔状态、待部署信息、历史信息和合法动作掩码。单位 ID 必须映射到训练时的 vocabulary，坐标和数值单位必须与 crforge schema 一致。部分历史字段可以在接入端维护，但隐藏计时器等字段需要验证是否可读、能否可靠推导；缺失字段不能任意填零就声称兼容。

仅代理出牌/回放事件再用模拟器还原，仍然存在客户端与模拟器的状态误差；这不能解决此前 BC 数据的 mismatch。官方战绩类 API 也不能代替实时战场状态。

## 尚需验证

需要确定 Android/iOS、手机或模拟器、游戏版本及进程架构。本机此次未发现可用的 adb 或正在运行的 Android 模拟器。获取这些信息后，先验证一帧真实战场数据能否覆盖模型输入，再进行最小接口转换。当前进度是源码调研和下载完成，实时读取与模型接入尚未完成。
