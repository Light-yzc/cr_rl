# Nulls Royale 16.635.1 读取适配进度

**交接状态：用户已要求交给其它 agent，本轮到此停止。新版读取尚未修复成功。** 以下命令是复现入口，不代表当前设备和进程仍在线；接手先查询状态。

本轮用户授权修正读取器，只读采集/静态分析；不写游戏内存、不注入、不自动出牌，不恢复 PPO 训练。其它 agent 在同一工作树的训练改动未修改或提交。

## 当前环境与证据

- MuMu Android 12 实例 1，ADB `127.0.0.1:16416`，root adbd 实测成功。
- 进程 `nullsroyale.rel.free`，本轮 PID 2661（以后必须动态查询），游戏 versionName 16.635.1，ARM64，经 MuMu 转译。
- `scripts/probe_nulls_memory_readonly.py` 成功读取 ELF magic `7f454c46`；可以访问进程内存，不代表字段解析成功。
- 旧 `ARM_MANAGER_GLOBAL=0x19BBDD8` 解引用到 `0x85028e4` 后，下层 context 得到 `0xfd4013e8a9474ff4`，不在可读映射内。旧指针链失效，不应继续使用。
- `scripts/diagnose_nulls_locator.py` 测试现有 x86_64 堆定位算法，所需区域超过 32 MiB 预算，未读取堆快照；没有声称算法匹配 ARM64 客户端。上限参数最大 64 MiB，禁止无界整进程 dump。
- 截图/探测 JSON：`artifacts/nulls_capture_20261115/`。截图用于人工核对，不作模型输入。已保存的一帧中计时 1:43，双方公主塔显示 3123/4858、4858/1559；这些都是截图值，不是内存已验证结果。

## 已取得的新版核心库

`artifacts/nulls_profile_166351/libg.so`：29,408,752 bytes（只导出该库，没有下载整 APK）。

- SHA256：`0b580222e5a3502cf153bb5e6297ec32d2b154b605b777c74873dd75722e4eaa`
- ELF Build ID：`179ac4bd38c205bc5041d372a79f1c2f929b1c30`
- ELF64 little-endian AArch64，符号不完整，须用 program header 映射文件偏移与虚拟地址。

## 静态分析工具与结果

本机已有 `F:/BaiduNetdiskDownload/IDA Professional 9.0/idat.exe`。初始 IDAPython 未配置；使用自带 `idapyswitch` 指定同目录 `python311/python3.dll` 后脚本可运行（修改了 IDA 的用户 Python 选择）。没有安装额外依赖或运行目录中的其它安装/修改程序。第三方插件缺可选模块会报错，但核心 IDAPython 脚本已输出结果。

`scripts/ida_nulls_survey.py` 配合 `-A -a` 运行，仅 loader 与少量指定地址反汇编，不做整库自动分析；进程已退出，没有常驻 GPU/RAM 实验。输出 `ida_survey.json`、`candidate_disassembly.txt`、`ida.log` 在上述 artifact 目录。8 个 segment、482 个 import；没有直接可用的战场字段导出符号。

`scripts/find_nulls_string_xrefs.py` 用 ELF PT_LOAD 和 AArch64 ADRP/ADD 提取候选引用，不将候选当作函数边界/真实状态字段：

- 六座塔 current_hitpoints 字符串引用集中于 `0xf127f0..0xf12984`。
- 可见 `LDP W24,W25,[X19,#0x40/0x48/0x50]`，后续构造 current/max 字段字符串并调用校验函数，附近还检查卡牌参数；**更像对局配置参数校验，不能直接认定是实时塔血结构**。
- `SpendElixir` 引用候选 `0xeb9bd4`、`0xeb9bf8`、`0x10434d8`、`0x10434fc`，语义尚未验证。

## 实现状态与下一步

未定位新版实时 battle manager、玩家对象、实体列表，尚无可信圣水/手牌/塔血输出，未向模型输入任何推测字段。

下一步应从具体函数和调用关系继续定位新版对象入口，再用多帧只读样本验证：圣水自然恢复与出牌扣除、手牌轮换、塔血变化、计时单位、阵营坐标、跨局指针生命周期。只在当前库指纹匹配且结构/时序校验通过时启用 profile。不能用范围内的随机值或截图值代替真正解析。

出牌桥接已独立实现 `src/cr_rl/client/adb_actions.py`，6 项单元测试通过；当前未校准，未对设备发起出牌。执行路径要求 verified observation、fresh timestamp、match ID、手牌 ID、合法 mask 和 validated calibration，ADB 回执仅标为 sent_unconfirmed。

## 接手复现命令

工作目录 `G:/Code/cr_rl`，PowerShell。Python 使用现有 conda，避免重新下载 Torch；本任务的探测/静态脚本不依赖 Torch。不启动训练或额外模拟器。用户本地 RAM 16GB，实验放 G 盘。

```powershell
& 'E:/Program Files/Netease/MuMu/nx_main/MuMuManager.exe' info --vmindex all
& G:/sdk/platform-tools/adb.exe devices -l
# 如果实例仍是 1 且端口未变：
& G:/sdk/platform-tools/adb.exe connect 127.0.0.1:16416
& G:/sdk/platform-tools/adb.exe -s 127.0.0.1:16416 shell id
& G:/sdk/platform-tools/adb.exe -s 127.0.0.1:16416 shell pidof nullsroyale.rel.free

& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/probe_nulls_memory_readonly.py --serial 127.0.0.1:16416 --out artifacts/nulls_capture_20261115
& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/diagnose_nulls_locator.py --serial 127.0.0.1:16416 --out artifacts/nulls_capture_20261115/locator_diagnostic.json
& 'F:/yzc/Miniconda/envs/cs285/python.exe' scripts/find_nulls_string_xrefs.py --library artifacts/nulls_profile_166351/libg.so --strings artifacts/nulls_profile_166351/string_candidates.json --out artifacts/nulls_profile_166351/string_xrefs.json

$env:PYTHONPATH="$PWD/src"
& 'F:/yzc/Miniconda/envs/cs285/python.exe' -m unittest discover -s tests -p test_adb_actions.py -v
```

IDA 现有数据库：`artifacts/nulls_profile_166351/libg.so.i64`（约 141 MiB）；可直接复用。批处理复现：

```powershell
& 'F:/BaiduNetdiskDownload/IDA Professional 9.0/idat.exe' -A -a '-LG:/Code/cr_rl/artifacts/nulls_profile_166351/ida.log' '-SG:/Code/cr_rl/scripts/ida_nulls_survey.py' 'G:/Code/cr_rl/artifacts/nulls_profile_166351/libg.so'
```

读取器源码已下载到 `.cache/direct_state_research/cr-memory-reader/`，`source.json` 记录 revision `9af94ca2f273ce526dda6256bdbd5d5555fb82d4` 及 SHA256；最新提交日期 2026-08-25，Windows 明确相关更新 2026-08-09。它早于当前客户端，不能照抄旧偏移。

## 文件与工作树注意

- 本交接涉及的脚本：`probe_nulls_memory_readonly.py`、`diagnose_nulls_locator.py`、`find_nulls_string_xrefs.py`、`ida_nulls_survey.py`。
- 本轮新增 `src/cr_rl/client/__init__.py`、`adb_actions.py` 与 `tests/test_adb_actions.py`。6 项测试验证的是触控命令计划和拒绝逻辑，不是实机出牌成功。
- 上述新文件及本文当前尚未 commit。仓库还有其它 agent 的大量 PPO/V100 修改，不要整体 add、reset、stash 或覆盖；交接仅以这些明确文件和 artifacts 为范围。
- `artifacts/` 和 `.cache/` 通常被 Git 忽略，迁移给另一台机器时需另带核心库、数据库、源码和 JSON 证据；同机 agent 可直接读取。
- 前轮 “读取器可挂载” 仅指 root 和 `/proc/PID/mem` 访问成功。实际状态流、模型接入、触控标定和闭环均未验收。不得将截图数值当内存解析输出。
