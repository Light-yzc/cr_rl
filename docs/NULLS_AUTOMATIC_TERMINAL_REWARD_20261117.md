# 自动记录实战胜负与终局 reward

此前采集器自动打完和返回大厅，但只有可选人工prompt_result能写结果，导致自动采集没有
result.json。这是代码缺口；已修复，未修改Actor权重、温度、出牌策略或游戏内存。

## 结果来源

`client/outcome.py`读取同一对局的terminal.json，校验clock_ended、own_seat、tick与观测
一致，双方国王塔HP尺度已知。双方终局皇冠数不等时，按本方视角判win/loss。
明确标记source=terminal_status_inference_v1，不把推断伪称为直接读取到原生winner字段。
不单凭一座塔缺失判胜；不把未知/相等比分直接标成draw。

若有900x1600结算截图，再用独立的固定区域金色皇冠检测校对。动画造成的部分皇冠记作
区间而不是强行四舍五入；可确认的截图胜负与status冲突时拒绝标注。
截图明确、status暂时同分时允许用截图给出胜负，并在证据记录decisive_source。
这是结算证据核对，游戏中的模型输入仍来自内存接口。

`result.json`写outcome、reward（win +1/loss -1）、来源、完整证据、terminal SHA与checkpoint
SHA；manifest同时更新outcome_verified。终局原始观测不改写，避免破坏原始证据或泄漏结果。

## 自动采集修复

- run_nulls_policy.py在写入完整终局后，先保存结算截图（只读，不点击），自动记结果，再
  运行训练导入/导出检查，最后才点确认返回大厅。
- 结果无法确认时保留截图/终局并停止，不再无声进入下一局。
- collect_nulls_presets.py验收完成局时也调用结果校验，不把无reward局当作正常完成。
- 当前已经运行的旧child不会热更新，轻量backfill watcher每30秒处理遗留；它不抢写新
  控制器的自动结果，不占GPU。后续child自然加载新代码，无需中断正在打的游戏。

## 已补救数据

第一次回补45个完整局，15胜30负。随后一批训练检查覆盖47局，15胜32负；全部有结果，
其中24局通过load_episode，23局因10例>0.8秒观测间断、13例执行未确认/歧义被排除。
这是该次快照，不是停止采集后的最终局数。失败局的真实reward仍保存，不伪造缺失轨迹。

结果恢复报告：`preset_60_20261117/result_recovery_report.json`。
训练资格报告：`preset_60_20261117/result_training_readiness.json`。
只做检查而不复制大数组：

```powershell
$env:PYTHONPATH='src'
& 'F:/python/python.exe' scripts/backfill_nulls_results.py `
  --sessions artifacts/nulls_live/preset_60_20261117 --validate
```

`--export`才额外导出NPZ；playbuffer直接读取日志，不要求复制一套完整数据。
含同分/冲突/未知证据的局不自动猜结果；现有注释与reward矛盾也会拒绝。
