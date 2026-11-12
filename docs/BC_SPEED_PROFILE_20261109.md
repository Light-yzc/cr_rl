# BC L4 性能分析与加速（2026-11-09）

基线：122,621,959 参数 V2/readout2 BC，FP16 autocast + FP32 帧编码器/落点头，microbatch4 × accumulation24，有效 batch96，context128。checkpoint128 作为固定基准起点，同一批96行、同一模型权重；benchmark Adam lr=0，不保存更新后的模型/optimizer。

| 阶段（每96样本） | 1 GiB缓存 | 16 GiB缓存，暖缓存 |
|---|---:|---:|
| CPU组历史、解压和H2D | 5.888s | 0.613s |
| 前向＋loss | 1.143s | 1.156s |
| 反向 | 2.383s | 2.381s |
| optimizer和梯度检查 | 0.043s | 0.041s |
| 总墙钟 | 9.458s | 4.192s |
| NPZ成员重复解压次数 | 76 | 0 |

计时在阶段前后CUDA synchronize；这些同步仅用于基准，正式训练不加同步。原配置约62%的时间在数据准备，约37%在GPU前后向。16 GiB是SSD解压缓存上限，不是把16 GiB数组装入Python RAM；数组使用mmap，OS文件页缓存可回收。该Colab主机53GiB RAM、228GiB可用磁盘。迁移到16GiB RAM机器仍须独立核对RSS。

microbatch8/12/16/24的暖缓存时间约4.125/4.047/4.015/3.987s，额外收益只有约5%，显存峰值从4.21GiB升到14.65GiB。不同microbatch的条件动作损失会因各批次动作数量归一化而改变梯度，因此这次保留microbatch4和既定有效batch96。

新增一批CPU预取（固定顺序、单worker、队列只提前一批，CPU tensors pin_memory，主线程non_blocking搬入CUDA）。真实同一批GPU基准：暖缓存3.687s，梯度范数2.8405025005340576，与无预取4.192s配置一致。CPU顺序/恢复偏移/尾部batch/空输入测试通过。

落地：先将正式run解压缓存从1GiB提升到16GiB，从step128完整恢复，首16步70.69秒（含启动开销），比原正式run每16步约160秒明显改善。预取将在下一份完整checkpoint（至少step256）原子保存后交接，恢复相同optimizer、scaler、RNG、epoch selected数组与next_sample，避免重新随机采样或重置优化器。交接状态以runtime的speed_handoff.log/status.json为准。

边界：9.458→3.687s是固定真实批次的暖缓存基准，约2.56倍吞吐；整次训练还包含冷缓存、验证和checkpoint发布，不能把这个数字当作整次训练已测得的加速倍数。第一次冷缓存预取批次为7.00s。

证据：artifacts/bc_20261109/ready2/speed_benchmark.json、prefetch_benchmark.json、speed_benchmark.log、prefetch_benchmark.log、train_fast.log；本机镜像位于.tmp_l4/。这些优化不改变训练卡组、验证划分、LR、FP16边界和有效batch。

## 进一步的TF32实测

在保持FP32指数范围的帧编码器/落点头上，将float32_matmul_precision从highest改为high，时序Transformer仍使用FP16。固定四行的概率差检查：最大概率绝对差5.05447e-5，已选动作log-prob最大差0.00348282，全输出有限；该检查覆盖有限样本，不等于完整验证集强度验收。相同96行反向基准全部有限，梯度范数2.82798（原2.84050，约0.44%差），暖缓存墙钟3.277s，约为原9.458s的2.89倍吞吐。

已将TF32高吞吐配置安排在全局step384的完整checkpoint后交接，使用原selected数组、optimizer和scaler；不回退正在进行的训练来强行立即切换。已启用的缓存+预取阶段正在完成第二遍；最终运行状态以status.json为准。

12次GPU采样（缓存优化阶段）利用率59–82%，平均约69%，功耗约56–76W，显存约7GiB。prefetch/TF32后的持续利用率与墙钟另验。

## 正式运行验收

TF32+缓存+预取在step384完整checkpoint交接后已实际运行至step400。16个真实optimizer steps耗时56.3985s（含启动），约3.525s/step；原正式训练每16步约160s，因此该段实际吞吐提升约2.84倍。这里保守报告约2.9倍，不按固定批次暖缓存3.277s夸大整次训练收益。

step400：epoch2，optimizer_batch_loss6.00155，未裁剪梯度范数20.3451（随后clip到1.0），GradScaler256，恢复后的growth_tracker由35递增到51，无新的溢出跳步。当前训练PID47127，RAM约5.7GiB。更早两次FP16梯度溢出已被GradScaler跳过并降低scale，未继续使用非有限梯度。

用户确认：16GiB RAM限制仅用于本地V100，Colab可使用其约53GiB主机内存。Colab配置没有16GiB RAM进程上限；16GiB是SSD解压缓存上限，OS页缓存自动利用可用RAM。预取只提前一批，避免为未测得收益而无界预加载整批history。

旧递归备份会扫到benchmark缓存，现已替换为排除.bc_arrays/.validation_cache/prefetch_cache/临时checkpoint的版本；只回收已结束基准的专用缓存，不删除活动训练数据、恢复checkpoint或当前解压缓存。
