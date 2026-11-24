# Colab PPO 上传包

上传 `crrl_ppo_upload_20261116.zip` 到 `MyDrive/cr_rl_ppo/`。上传完成后再运行配套 `notebooks/colab_ppo_prepare.ipynb`。文件较大，推荐Drive网页或桌面客户端上传；不经过Colab临时文件上传入口。

本包包含当前完整PPO恢复权重、历史验证Actor、两个旧PPO Actor、BC锚点、当前工作区代码、配置与测试、固定版本模拟器Python接口与预编译JAR、16.635.3卡表及小量实战诊断记录。不包含BC训练大数据、虚拟环境、账号凭据、完整旧训练目录和全部联赛池。

ZIP使用ZIP64且不压缩：Torch权重已是ZIP容器，大文件直接流式打包，避免长时间压缩与额外副本。文件级SHA256在包内 `BUNDLE_MANIFEST.json`，ZIP SHA256在旁边 `.zip.sha256` 文件。两者都保留以检查上传完整性。

准备notebook只负责Drive挂载、本地复制/解压、逐文件校验、卡表安装及环境检查，不会创建额外实例或自动训练。新课程、单GPU训练启动器和自动备份仍待单独完成和smoke；旧双V100脚本不可直接照搬。完整恢复权重保留在原路径，但本包刻意没有携带全部历史池；新分支需要用包中Actor重新建立小型对手池并显式重设路径，不能原样复用旧Linux绝对路径。

训练前需要：Python 3.12/3.13、Java21、CUDA Torch、NumPy、JPype1、PyYAML、gymnasium、pettingzoo、msgpack、pyarrow、tqdm、tensorboard与zstandard。沿用Colab现有CUDA Torch，不为准备上传包安装或下载本地Torch。W&B通过Colab Secrets读取，凭据不写ZIP。

恢复训练后使用独立的新输出目录；先小预算测速和数值一致性验证，再决定CU预算。训练读Colab本地盘；Drive只承担验证后的checkpoint备份。当前best标签来自历史小规模验证，不等于已证明优于最新PPO。
