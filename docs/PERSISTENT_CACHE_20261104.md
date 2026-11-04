# Drive缓存与离线训练交接（2026-11-04）

## 已实现和实测

优先备份原始状态的紧凑验证历史，不是模型embedding，也不复制整份128 GiB训练LRU。
按冻结数据SHA、验证/历史行索引和context生成签名；换模型权重可复用，换数据、窗口或验证样本则使用不同缓存。

当前签名：`12ad9e6be7d91cf3e6d88c2c48252a56e4d597572bec00a6247a44d331c9c079`。

| 实测项目 | 结果 |
|---|---:|
| 原始缓存合计 | 12,019,731,564字节，约11.2 GiB |
| Zstandard level 1归档 | 50,448,245字节，约48.1 MiB |
| 归档占原体积 | 0.420%，约缩小238倍 |
| 一次性压缩及源SHA验证 | 86.73秒，低优先级单线程 |
| Drive路径回读、解压到SSD、逐文件SHA校验 | 76.10秒 |

这是训练同时运行时的**同一VM回读**；DriveFS和页缓存可能加速读取，不是新VM网络带宽测试。不能把约0.32秒的回读单独外推成云盘带宽，也不能声称整个训练重启仅需76秒。新VM仍需恢复原始压缩分片、数据元信息、模型和optimizer。这里减少的是重复解压验证历史的成本。

压缩收益足够大：重启时只搬约48 MiB归档，随后在SSD重建11.2 GiB数组。不会从Drive直接mmap训练数据。空间不足、归档缺失或校验失败会记录事件并回退到原始分片；完成标记最后发布，半成品不被当成缓存。

原始压缩分片本来已在Drive；暂不为每个stage再复制全量训练SSD缓存。训练LRU规模大、工作集随采样变化，整份恢复可能比按需解压更慢。新增Drive占用主要是本次约48 MiB验证归档，而不是11 GiB或128 GiB的解压文件。

## 存放与恢复

全局缓存目录：`MyDrive/cr_rl/caches/validation-v1/`。
归档：`<signature>/98be6a7e7d62fd1da50120eb88d2c6103304bf5bf48755c6e7eff483aeef7037.tar.zst`。
清单：`<signature>/archive.json`；实测报告：`benchmarks/<signature>.json`。
独立缓存工具源码另存`tools/source.tar.gz`；当前训练模型、数据分区、采样和优化器没有因缓存实验而更改。

后续BC或阶段控制器启动时加：

```bash
--persistent-validation-cache /content/drive/MyDrive/cr_rl/caches/validation-v1
```

`train_bc.py`先恢复到本地SSD；同签名才复用。新`run_bc_repair_trial.py`已传递这个路径，`run_all_bc.py`也支持透传。当前正在运行的旧控制器不热换代码；它的两组对照继续复用已经建好的本地缓存。新的持久化路径供下次启动使用。

实现：`src/cr_rl/data/persistent_cache.py`、`validation_cache.py`、`scripts/cache_validation.py`。已覆盖归档精确往返、重复发布、坏包回退、缺失成员、路径越界和数据绑定校验。

## 本地关机后的行为

训练控制器及子训练器均在Colab内运行，脱离本地连接后继续当前两组各1000步的BC对照，不自动进入PPO。
`finish_and_release.py`守护进程等待控制器/训练器退出，成功时要求两组都到step5096、最终Drive checkpoint SHA/大小匹配且比较报告已备份；失败时先保存最后的本地checkpoint和日志。备份核验后才请求释放本L4。

状态：`MyDrive/cr_rl/runs/121m_bc_repair_20261104/release_guard.json`。
备份核验失败时保留VM并记录错误，不冒险释放。此守护不能绕过Colab回收或额度限制；若平台提前回收，从最近已校验的Drive备份恢复。
