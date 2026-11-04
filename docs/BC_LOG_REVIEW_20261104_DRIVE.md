# 2026-11-04 Drive 日志复核

通过已连接的 Google Drive 插件读取本轮运行目录、status.json、release_guard.json、repaired/train.log、bc.quick.jsonl、bc.json、bc/latest.json，并复查 Colab CLI 无活动实例。原始小日志留在 artifacts/bc_repair_20261104/drive_*。

## 执行状态

- 最后训练日志 step4944，即从4096新增848/1000步；status文件最后更新时间北京时间10:04:08，仍为 training/repaired。
- 目标5096未达到，目录没有 baseline 组、comparison.json 或完成标记。原计划两组对比未完成。
- release_guard.json仍为 armed，未记录完成备份后释放。结合运行时已消失，符合外部中断/回收特征，但没有平台终止原因证据，不能确定为额度、空闲回收或其他原因。
- Drive latest.json记录step4864，checkpoint文件大小与记录相符（1,494,224,317字节），SHA记录为 aa59613954d1a4ffba06c3fe07ed237e68c180f5cd23c2a965c082f9e44fdd03。本次只读取元数据和备份记录，未重新下载1.49GB并重算SHA或加载权重。恢复前须验证。
- 如该备份验证通过，修复组还需232步达到5096；日志中4864之后80步没有新备份记录。PPO未启动。

## 同一固定快速验证集

比较修复组step4128与4928，均256行，其中110个部署标签。4128已接受32次更新，并非原始4096的零步基线；不能替代未运行的原版组对照。

| 指标 | step4128 | step4928 |
|---|---:|---:|
| 条件选牌准确率 | 39.09% | 36.36% |
| 条件落点两格内 | 19.09% | 18.18% |
| 条件落点平均L1距离 | 12.69 | 11.15 |
| 落点NLL | 5.021 | 4.768 |

首5次与末5次验证均值：L1从12.51降到11.41，两格内从16.0%到17.82%，选牌从38.73%到36.0%。存在小幅距离/NLL改善，尚无落点命中率或选牌突破。各次验证使用相同小集合，不是独立重复实验。

自然验证集512行，只有20个部署标签：step4928平均非WAIT概率3.19%，真实动作行5.22%，WAIT行3.11%。硬决策play_recall为0；代码比较全部部署概率质量与WAIT概率，不是拿WAIT与单个格子比较。自然先验训练下概率低于0.5不自动证明随机采样策略永不出牌；但当前动作与WAIT行区分弱，不能用96.09%的总体准确率（全WAIT同样可得）宣称时机学好。尚无新的随机采样对战/动作窗口验证。

## 数值检查与下一步

- 可见日志没有Traceback、OOM或非有限loss。首8/末8条训练batch loss均值6.442/6.382，改善有限；不同batch均值仅作描述。
- step4432/4448裁剪前全局梯度范数4198/4553，之后恢复；代码对全模型clip到1.0并拒绝非有限梯度。这是需要定位的尖峰，不足以认定已发散；全局裁剪可能在尖峰步压缩其他分支梯度。
- 空间残差gate末值约0.004，新增分支贡献应通过激活/梯度和禁用分支消融测量；仅凭gate小不能判定失效。
- 不直接进入PPO。先核验4864备份并恢复剩余232步，完成同父模型的原版1000步对照，再做更大自然验证、动作窗口校准和换边对战。对尖峰样本/分支梯度增加诊断，而不是直接扩模型或认定结构修复成功。

来源：[运行目录](https://drive.google.com/drive/folders/1_qou2M-klzhc5H-auNI0dswlE5nmgyXN)、[训练日志](https://drive.google.com/file/d/178FLw9HM9PqL3HTO1TlOXa7DD48zgSkX/view?usp=drivesdk)、[快速验证](https://drive.google.com/file/d/1mgwEI-QiJyoj63Cu0gTwSIX-T5UKoBDN/view?usp=drivesdk)、[备份记录](https://drive.google.com/file/d/1XKKZAxhXwaNIWuW36JKi1E2DvvJE_dj5/view?usp=drivesdk)。
