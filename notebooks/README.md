# Colab BC notebooks

当前全量版本：[colab_bc040_full.ipynb](colab_bc040_full.ipynb)。它保留云端最新的
10,654局 packed 数据与8路并行，使用最新完整父checkpoint，进入独立的8轮新落点后端阶段。
保留W&B、tqdm、自动恢复及逐局数据回收。选L4后顺序运行6个代码单元格，代码5启动训练。
实现、固定划分和冻结范围见 [交接说明](../docs/BC_FULL_PLACEMENT_STAGE_20261111.md)。

以下是较早的 `colab_bc040_wandb.ipynb` 100轮 notebook 说明，
其目录、调度和采样规模不适用于当前 full 版本。

## 旧版 BC 100轮续训

打开`colab_bc040_wandb.ipynb`，上传到Colab，选择GPU（推荐L4），顺序运行6个代码单元格。

1. 挂载Drive并设置总epoch目标。
2. 检查CUDA，仅补W&B/tqdm/PyYAML；沿用当前Torch。
3. 使用Colab Secret `WANDB_API_KEY`或隐藏密码框登录W&B，打印run链接。
4. 缓存manifest所需数据，自动选择最新完整checkpoint，恢复原训练配置并预检。
5. 开始/连接已有训练进程，显示tqdm进度并转发W&B指标。暂停显示后重跑此单元格。
6. 查看最近验证和进程状态。

默认来源`MyDrive/cr_rl/runs/state_global_event040_20261110`；本notebook续训备份在其`bc100_cosine/training`。
最新恢复点从原目录、notebook备份和本地缓存中按有效训练进度选择，保留Adam、scaler、RNG及固定数据划分。
`TARGET_EPOCHS=100`是当前数据阶段的总目标，包含已经完成的epoch。350次成功更新预热至3e-5，之后余弦衰减到3e-6；恢复保留调度位置，不重新预热。

L4实测选择frame microbatch32、TF32 high、fused AdamW与集中指标回读。同权重/同两条完整轨迹吞吐约为frame8基线的1.54倍；这不包含Drive保存或全epoch验证。每128座位及epoch结束保存latest，验证每350座位/epoch结束，固定16座位。

只缓存现有0.4秒数据manifest引用的文件和一个checkpoint；逐局解压缓存由现有训练器回收。
当前训练器要求校验完整manifest，所以不能遗漏其中引用的shard。已有且SHA校验通过的缓存不会重新从Drive读数据。
初始不加载原始重建资料、历史checkpoint、对战评测模型或模拟器。

Notebook自带经过SHA校验的工具模块，直接使用本阶段已有源码包，无需手动复制多个辅助文件。
训练器、模型、优化目标不因notebook展示层而改变。W&B只记录标量，不上传模型或数据集。

本地验证：notebook工具、调度、速度路径与协调器测试通过；4090真实FP16 tiny模型完成保存、自动选最新、独立进程启动检查、续训、备份和进度显示。
测试中的W&B服务为模拟接口；Google Drive挂载和真实W&B在线上传未在新Colab会话中端到端验证。
测试回执：`artifacts/notebook_bc_gpu_smoke/report.json`。

100轮实现与真实L4性能证据见`docs/BC100_SPEED_COSINE_20261110.md`。
