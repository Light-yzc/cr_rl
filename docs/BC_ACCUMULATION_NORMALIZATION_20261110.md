# BC 有效batch归一化数值检验（2026-11-10）

## 结论

现有conditional BC按各microbatch内部的非WAIT/普通出牌数取均值，随后把microbatch梯度等权平均。它仅在各microbatch的样本数、非WAIT数和普通出牌数都相同时等价于整有效batch目标。

人工logits的16项CPU数值检验显示，使用整个有效batch的分母可以恢复整batch的loss与梯度；最大梯度绝对误差2.38e-7，属于float32归约顺序误差。当前实现最大绝对误差0.619、最大相对L2误差49.66%。这是目标定义/梯度算术验证，不是实战提升证据。

源代码位置：`.tmp_latest/src/cr_rl/training/bc.py:105,141`定义条件选牌/落点局部均值，`.tmp_latest/scripts/train_bc.py:1123`附近再按实际累积microbatch数平均梯度。

## 应实现的目标

令一个有效batch共有N行、W个WAIT、A个非WAIT（普通出牌+技能）、P个普通出牌。a_i是非WAIT标志，p_i是普通出牌标志。当前natural timing使用固定epoch抽样先验q和目标先验p（本轮q约0.5、p=0.2），定义权重：

\[
w_i=\begin{cases}p/q,&a_i=1\\(1-p)/(1-q),&a_i=0\end{cases}
\]

整batch条件目标应是：

\[
L=\frac{\sum_{i=1}^N w_i\ell_i^{timing}}{N}
 +\frac{\sum_{i:a_i=1}\ell_i^{conditional\ card}}{\max(A,1)}
 +c_{position}\frac{\sum_{i:p_i=1}\ell_i^{tile}}{\max(P,1)}.
\]

每microbatch只计算自己行的三个分子，分母始终使用同一有效batch的N/A/P。各microbatch贡献直接相加反向传播，即与整batch一致。q仍使用原epoch抽样先验，不要为了这项修正而按microbatch重新估计q。

若沿用每microbatch已有的三个局部均值，可写成：

\[
L=\sum_m\left[\frac{n_m}{N}T_m+\frac{a_m}{A}C_m
 +c_{position}\frac{p_m}{P}S_m\right].
\]

空组贡献取0。**不能仅把整个micro loss乘以n_m/N**，因为选牌与落点项分别需要a_m/A和p_m/P。

balanced timing（不传target prior）的整batch定义为WAIT条件均值与动作条件均值按全batch中实际存在的类别数平均；其micro贡献也应分别用全batch W/A分母，而不是复用局部balanced均值。

## 最小落地方式

1. 在有效batch开始前，从CPU labels计算N/W/A/P，开销仅数十个整数索引，无需一次构建全部history。
2. loss函数增加可选归一化上下文，或暴露三个可微分子；当前返回的loss_parts大多已经detach，不能用它们重新组合反向目标。
3. 每个micro使用全组分母后执行scaler.scale(contribution).backward()。
4. 删除原来optimizer step前对梯度再除accumulated的操作，否则会重复缩小梯度。
5. 整个有效batch仍只做一次unscale、finite检查、clip和optimizer step；尾部短batch用实际计数。
6. 若为兼容旧训练而暂时保留原梯度除K，则每micro贡献先乘K，但更清晰的实现是移除这一二次归一化。
7. 将归一化版本记录到训练设置中；它改变旧实现的逐样本权重，不能声称是旧阶段逐步完全相同的续训。

双卡DDP补充：若每卡使用全局N/A/P计算本卡分子贡献，而DDP在梯度同步时取各rank平均，则本卡贡献还需乘world_size；也可采用等价的梯度求和协议。不要把单卡公式直接搬入平均梯度的DDP后漏掉这一因子。全局计数只需all-reduce少量标量，不需汇总history。

## CPU数值结果

脚本`.tmp_latest/scripts/probe_bc_accumulation.py`只构造小型人工features/共享logit参数，直接复用真实hierarchical_bc_loss；不加载模型、checkpoint或GPU。覆盖prior0.2/无prior、硬落点/sigma1软落点、不均动作组成、全WAIT微批、尾部短microbatch。

下面为prior0.2、sigma1的代表案例：

| 情形 | 现行梯度相对L2误差 | 修正梯度最大绝对误差 |
|---|---:|---:|
| 两micro各2WAIT+2普通出牌 | 0 | 5.96e-8 |
| 两micro非WAIT数1/3、普通出牌数1/2 | 34.35% | 7.45e-9 |
| 一个micro全部WAIT | 45.67% | 1.19e-7 |
| micro长度4/4/2，动作组成不同 | 31.66% | 5.96e-8 |

固定小集五组GPU对照采用每micro固定2WAIT+2普通出牌，属于第一行，所以这些对照本身不受上述组成差异污染。归一化修正主要用于正式随机组成训练、尾部batch和未来双卡/不同microbatch设置。

建议保留此修正作为后续训练规范，并独立验证训练稳定性/泛化。它使“有效batch96”有明确、与microbatch切分无关的目标，但不保证单独解决数据质量、表示或控制集退化。

结果文件：`artifacts/bc_diagnostic_20261110/accumulation/probe.json`。所有脚本、临时文件、CPU测试缓存均位于G盘。
