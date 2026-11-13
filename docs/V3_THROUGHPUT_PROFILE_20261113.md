# v3 双 V100 吞吐 profile（2026-11-13）

当前长训命令使用 `CUDA_VISIBLE_DEVICES=1,2`、`envs=1`、`steps=8`、FP16 和 full-deck。profile 期间没有停止训练。

## 观测

- 两个 rank 的 CPU 占用约 90–98%。
- V100 SM 利用率在 60–85% 间波动，功耗约 90–140 W（300 W 上限）。
- 显存约 5.2–5.6 GiB/卡；4090 仅有桌面占用，没有训练显存。
- `nvidia-smi topo -m` 显示两张 V100 之间为 NV5，NCCL all-reduce 使用 NVLink 拓扑。
- 训练进度约 2 update/s；每个 update 当前每卡 8 个 decision samples，工作量很小，因此不会持续拉满 V100 功耗。

## 瓶颈判断

主要瓶颈是 Java simulator step、Python/NumPy observation adapter 和小批量 recurrent forward 之间的同步，而不是 PCIe 或 NVLink 带宽。提高 `steps` 到 16 的短测为 3 updates/19.93 s，按样本吞吐反而低于 `steps=8`，所以当前默认保持 `steps=8`。

经过 `frame_microbatch=8` 的 lane microbatch 调整，`envs=2` 已通过 20-update benchmark：无 OOM/NCCL 错误，单 update 约 0.10 s rollout + 0.17 s PPO，32 samples/update，约 115–123 samples/s。正式长训已切换到该配置；它比 envs=1 的样本吞吐更高。

profile 命令示例：

```bash
nvidia-smi dmon -s pucm -d 1
nvidia-smi topo -m
```
