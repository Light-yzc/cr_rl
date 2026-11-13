# v3 双卡 full-deck PPO 启动说明

v3 PPO 现在支持用 `torchrun` 启动两个同步 rank。每张 GPU 独立运行 crforge 环境和 recurrent state，采集不同 seed 的 rollout；PPO 反向后对非空梯度做 all-reduce 平均，再由两张卡同步 optimizer step。只有 rank 0 保存 `ppo_latest.pt`、写 progress JSONL、发布备份和写 TensorBoard，避免两个进程覆盖同一个文件。

默认牌组池是 `configs/decks_ppo46.yaml`，共 46 个训练牌组。首个权重固定为 `artifacts/ppo_prep_20261113/best.pt`，它是 architecture-v3、8-tick 的 BC weights-only checkpoint。

v3 自博弈使用完整 role league：每条 lane 训练一个 learner seat，另一席按 30% 当前策略、50% 历史冻结策略 PFSP 桶、20% 固定 v3 BC anchor 抽样。历史快照保存在输出目录的 `opponents/` 并有界保留；对手席轨迹不会混入 PPO，避免把冻结策略动作当成 learner on-policy 样本。

Linux/V100 启动：

```bash
CHECKPOINT=artifacts/ppo_prep_20261113/best.pt \
DECKS=configs/decks_ppo46.yaml \
OUT=artifacts/ppo_v3_full_deck_dual \
NPROC=2 \
TENSORBOARD_DIR=artifacts/ppo_v3_full_deck_dual/tensorboard \
./loc_transfer/run_v100_v3.sh
```

TensorBoard：

```bash
tensorboard --logdir artifacts/ppo_v3_full_deck_dual/tensorboard
```

事件包括 `ppo_update/*`、`rollout/sequences` 和 `rollout/samples`。正式运行环境需要安装 TensorBoard；V100 使用 FP16，不使用 BF16。

双 V100 已实测通过 full-deck 长训：`envs=1, steps=8, precision=fp16` 在 16 GiB V100 上稳定同步；将 `steps` 提高到 16 会显著降低吞吐，因此启动脚本默认保持 `steps=8`。启动前仍应核对 `nvidia-smi` 显示两张卡。

## recurrent PPO memory 约定

每条 rollout segment 保存采样前的 `h0`（detached、clone 的 `PolicyMemory`）。PPO 重算 old/new log-prob 时都从同一个 `h0` 开始 unroll；更新完成后 live environment state 会重置，旧 memory 不会驱动下一轮环境。这是截断 BPTT 下的 recurrent PPO 条件输入，避免用零 state 造成 old-log-prob mismatch。

采样概率会先做有限值清洗，并在异常时退回合法动作均匀分布；FP16 梯度出现非有限值时丢弃该 update 并重置 GradScaler，避免单个坏 batch 触发 CUDA/NCCL 级联退出。长训 checkpoint 间隔设为 100 update，以减少 SSD 写放大。
