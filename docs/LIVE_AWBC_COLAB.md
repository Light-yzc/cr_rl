# 实战AWBC：第一版训练入口

入口：`scripts/train_live_awbc.py`；Colab：`notebooks/colab_live_awbc.ipynb`。模型架构不变，冻结现有独立Critic的实体/时序骨干，只训练其MLP价值头；Actor仅更新WAIT、卡槽、技能与wait_hand_gate读出，落点、实体和时间骨干均冻结。此版本专门用于少量实战数据的保守适应，不宣称已经解决长期规划。

## 数据

使用完整已结束、有可信result.json的会话目录，而不是只上传NPZ。至少12个通过`load_episode`严格检查的对局。现已取消采集/初始化Actor指纹必须一致的限制；来源指纹仍用于审计和Value分组。仍不支持旧贪心日志。独立入口默认最多驻留64局；PPO playbuffer默认每次最多24局。排除清单写入`excluded_sessions.json`，不合格局不会因换算法自动合格。

保留全部状态的时序上下文。Actor仅训练有选择的决策，排除已证实未执行的开场动作、明确stale跳过的出牌以及被法术守卫拦下的提议；这些帧仍进入Critic和记忆序列，不把未执行提议改标为WAIT。无解释的未提交出牌、孤立/冲突/重复执行事件拒收。奖励只有真实终局+1/-1/0，按实际tick间隔计算MC回报，暂不混入模拟器shaping。

## Value与加权

独立入口按采集策略版本和己方卡组分层，整局划分80%开发训练/20%留出诊断。
同一个采集策略版本内做3折交叉拟合，不混用不同版本的胜负来拟合一个价值头。
每个版本不足9条训练轨迹、只有一种终局结果，或没有同版本外层验证局时，该版本Actor权重为0并跳过；其余版本独立做泛化门槛检查。
预测对象不进入该折拟合或早停；编码器特征按32帧从开局推进记忆，缓存为本地npy/mmap。
这是用于离线赋权的V_mu，不是当前PPO的V_pi；真实MC回报不含模拟器shaping。PPO Critic不被覆盖。

比较out-of-fold误差与按卡组、仅由该折训练局计算的常数回报基线。如果MSE没有至少改善2%，该版本跳过Actor更新，不再用普通BC强化自己的旧行为。2%为工程门槛，并非显著性检验。Value通过时：`w=1+strength*(clip(exp((G-V)/beta),0.25,4)-1)`，默认strength=0.5、beta=0.5，实际权重范围0.625—2.5。权重固定后训练Actor，不在每epoch重算制造正反馈。没有逐帧全局标准化去掉胜负或卡组差异。

外层留出局由训练折模型集成预测价值并生成固定权重；其回报不参与价值头拟合、早停或OOF门槛。最终必须至少有2条可信留出局。没有可信训练版本或验证不足时，仅输出诊断，不进行Actor优化，也不新建best.pt/latest.pt。PPO playbuffer另外要求过滤后仍有至少12条新训练局；这与独立入口的至少12条总输入局不是同一计数。

`value_head.pt`现在只保留各版本交叉拟合报告（head=None、diagnostic_only=True），不再输出一个混合版本的“可部署Value”。权重计算后丢弃影子头；不参与在线选牌或替换PPO Critic。20%仍是开发验证集，不是最终独立测试集。

## 默认设置

- Actor最多8 epochs，lr=1e-6；Critic头最多50 epochs、lr=1e-4、早停patience=5。价值头较小，学习率不同于此前完整Critic训练的建议。
- Actor每整局提交一步，按局平均loss；微批为32帧，冻结骨干下跨chunk只传递状态，避免整局梯度内存开销。
- CUDA投影使用FP16 autocast，缩放器保护Actor梯度；回报、价值头拟合与loss累积使用FP32，不声称纯FP16数值算术。
- Actor保持eval模式避免dropout导致参考分布漂移；只有readout参数需要梯度。
- 每epoch评估固定优势加权的留出NLL及相对本次AWBC开始前Actor的前向KL，不以采集策略概率作KL守卫。未加权NLL仅作诊断。独立入口KL超过0.02或加权NLL连续3次未改善就停；epoch0也是best候选。NLL/KL是离线守卫，不是胜率。
- `best.pt`为可部署Actor；`latest.pt`为AWBC恢复状态（只含Actor与小优化器，不含完整Critic）。不覆盖原PPO checkpoint，不能把它冒充可直接恢复独立Critic的PPO全量checkpoint。
- 输出含权重、划分、排除记录、trainable参数名、原始计数、Value折报告和JSONL日志；`--wandb`记录同一日志并保存run id。

## 本地命令（先停止实机采集，避免16GB RAM争用）

```powershell
$env:PYTHONPATH='src'
& 'F:/python/python.exe' scripts/train_live_awbc.py --sessions artifacts/nulls_live/preset_60_20261117 --checkpoint artifacts/ppo_v3_v100/ppo_latest.pt --out artifacts/live_awbc_v1 --device cuda --epochs 8 --wandb
```

本次没有启动这条真实训练命令，也没有修改当前采集策略。Windows入口持有与实战采集相同的进程锁，拒绝二者并行。

## Colab

保留原4.38GB `crrl_live_ppo_colab.zip`（完整U929 checkpoint）和采集好的会话目录，额外上传小文件`live_awbc_update.zip`及新notebook即可，无需重传权重。固定Drive目录`MyDrive/cr_rl_live_ppo/`，会话放`sessions/`。

Notebook逐文件校验并复制到`/content/crrl_live_ppo/`，沿用已有CUDA Torch。W&B通过Colab Secrets中的WANDB_API_KEY读取。训练期间每30秒显示阶段/epoch日志，Drive只同步已完整写入且SHA256校验通过的文件；仅保留latest、best与价值头等小文件，特征缓存留在临时本地盘。

相同数据、设置与RUN_NAME重跑恢复latest；改变数据/seed/beta等需新RUN_NAME。已early-stop的运行不会因重复点击cell继续无限复用。若想扩展epoch，仅在未触发早停的情况下提高`--epochs`。
旧版使用未加权NLL选择模型的latest不能直接续训，新入口会明确拒绝；请使用新输出目录，
保留旧实验作对照。旧输出目录已有的best也不能当作本次数据门槛跳过所产生的新模型。

## 验证范围

测试包括实际时间间隔MC、权重上限、三折与外层整局隔离、只更新Actor读出、12个合成完整会话的端到端训练与断点恢复。测试不是实战效果评估；实际U929在L4上的显存、吞吐和训练后胜率仍需小规模验收。
2026-11-17修复后的完整验收记录见`PPO_AUX_REVIEW_AND_FIXES_20261117.md`；可信/不可信价值
两条CLI路径分别测试，可信门槛分支在小型fixture中受控触发，不代表真实数据已经过关。
