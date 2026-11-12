# 本地迁移下载清单

来源：迁移包 checkpoints/latest.json，恢复点 update 142。

云端运行目录：`MyDrive/cr_rl/runs/122m_long1000_sample2x_20261106/candidate/`。

## 必需：完整续训状态（已下载并通过 SHA256）

- 云端相对路径：`checkpoints/update_000142.pt`
- 下载到：`artifacts/v100_handoff/weights/update_000142.pt`
- 完整大小：2,919,979,125 字节（2.719 GiB）
- SHA256：`b945e63ae21fb7d8987076a2e12fed8ed09ed10e3fb032ed487e7ec520846f6c`
- 完整 checkpoint 归档在 HDD，训练运行目录使用 Intel SSD；已删除旧的未完成 `.partial`。

## 可选：历史审计 rollout（不阻碍恢复训练）

- `rollouts/rollout_000110.npz` → `artifacts/v100_handoff/rollouts/rollout_000110.npz`，状态 `missing`，SHA256 `a530bad8dfc68788ca887e1d274c4f936c33b71f06ab15bfd5f19c2bbe13bf60`
- `rollouts/rollout_000110.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000110.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000111.npz` → `artifacts/v100_handoff/rollouts/rollout_000111.npz`，状态 `missing`，SHA256 `bcede34693dd2d43fcd0105071c37222afebe6c7999edd753ccb470de001138e`
- `rollouts/rollout_000111.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000111.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000112.npz` → `artifacts/v100_handoff/rollouts/rollout_000112.npz`，状态 `missing`，SHA256 `99c1c7f9790387a0c3615884b14c19dc532f22a69ce18110e89ff731cc9d7876`
- `rollouts/rollout_000112.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000112.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000113.npz` → `artifacts/v100_handoff/rollouts/rollout_000113.npz`，状态 `missing`，SHA256 `ff42de0fbcb5c7160e70d03dff3b50ddd681174f8a2e19ff8c575b7bd0a7d6b7`
- `rollouts/rollout_000113.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000113.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000114.npz` → `artifacts/v100_handoff/rollouts/rollout_000114.npz`，状态 `missing`，SHA256 `9c796553b1e0f8e45e3db566dc8b4c6902c04b55030eaf8ab6ef9abbf5494326`
- `rollouts/rollout_000114.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000114.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000115.npz` → `artifacts/v100_handoff/rollouts/rollout_000115.npz`，状态 `missing`，SHA256 `fbab12349728e9b18d3eeb730cdbecb85802a7503789c29d5af96e0c558661f5`
- `rollouts/rollout_000115.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000115.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000116.npz` → `artifacts/v100_handoff/rollouts/rollout_000116.npz`，状态 `missing`，SHA256 `c928928f905dfcbd93eba708fb2abdcdb46dec1cd72e7f56b1faef1c6f0b96f9`
- `rollouts/rollout_000116.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000116.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000117.npz` → `artifacts/v100_handoff/rollouts/rollout_000117.npz`，状态 `missing`，SHA256 `23c448ffe5c74f7199867ab879e6713f28f3b0e4685298a32e4b51ac81ce2ac2`
- `rollouts/rollout_000117.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000117.npz.json`，状态 `missing`，SHA256 `4536b3367d115d69e7dd2bb6671d8c60c71f12615b8d92e02e34b877f85a8f20`
- `rollouts/rollout_000118.npz` → `artifacts/v100_handoff/rollouts/rollout_000118.npz`，状态 `missing`，SHA256 `813935c6b00741f74800173226133d0ee9a8c28d13784f6d2b232ab23ebbcfa9`
- `rollouts/rollout_000118.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000118.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000119.npz` → `artifacts/v100_handoff/rollouts/rollout_000119.npz`，状态 `missing`，SHA256 `a0eb7c906402bad8516a7801d38d1e2fc31ef2635a9336adaf2dee3b0d1be277`
- `rollouts/rollout_000119.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000119.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000120.npz` → `artifacts/v100_handoff/rollouts/rollout_000120.npz`，状态 `missing`，SHA256 `3d7c20ea12df3b753190502c04cc3cb9f5428263d83f14b4f70548c6408a3924`
- `rollouts/rollout_000120.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000120.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000121.npz` → `artifacts/v100_handoff/rollouts/rollout_000121.npz`，状态 `missing`，SHA256 `f45da0013fc90757c6b178be270d8e3b129046767200bf9c198723de4fc9bf4e`
- `rollouts/rollout_000121.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000121.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000122.npz` → `artifacts/v100_handoff/rollouts/rollout_000122.npz`，状态 `missing`，SHA256 `203c843fddf3e675458c095b28a887fd5112063bd0fed02926b40ebf63c1ebcc`
- `rollouts/rollout_000122.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000122.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000123.npz` → `artifacts/v100_handoff/rollouts/rollout_000123.npz`，状态 `missing`，SHA256 `6d03936ab7a6828159156d9458eaa05606092edee57107eb57ace63a22bc3dfa`
- `rollouts/rollout_000123.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000123.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000124.npz` → `artifacts/v100_handoff/rollouts/rollout_000124.npz`，状态 `missing`，SHA256 `bf686c53b8909f218adc6cba106b3e502a75c4b194769bc21a520c07735287b3`
- `rollouts/rollout_000124.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000124.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000125.npz` → `artifacts/v100_handoff/rollouts/rollout_000125.npz`，状态 `missing`，SHA256 `25f788faa00a858b464cb42f51ebcae6cce26386e6385e9a811cb96efca3a8e8`
- `rollouts/rollout_000125.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000125.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000126.npz` → `artifacts/v100_handoff/rollouts/rollout_000126.npz`，状态 `missing`，SHA256 `9e30275e49199e0123f80ce7768f4de9a90d8266e10202026f8b692719314494`
- `rollouts/rollout_000126.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000126.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000127.npz` → `artifacts/v100_handoff/rollouts/rollout_000127.npz`，状态 `missing`，SHA256 `edb4214c439e7477798f1136bc68b04b5195b9120a04a1241e9c7c4fc924ea3b`
- `rollouts/rollout_000127.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000127.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000128.npz` → `artifacts/v100_handoff/rollouts/rollout_000128.npz`，状态 `missing`，SHA256 `c79c38f416b4210fd6128dff1e9420d53513b1665f0abe2bcd0ec366ab1dd31d`
- `rollouts/rollout_000128.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000128.npz.json`，状态 `missing`，SHA256 `4536b3367d115d69e7dd2bb6671d8c60c71f12615b8d92e02e34b877f85a8f20`
- `rollouts/rollout_000129.npz` → `artifacts/v100_handoff/rollouts/rollout_000129.npz`，状态 `missing`，SHA256 `4f3c51fb2bf7b38c9af314f21041089db8fcf268f820486860a736a6fba0623d`
- `rollouts/rollout_000129.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000129.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000130.npz` → `artifacts/v100_handoff/rollouts/rollout_000130.npz`，状态 `missing`，SHA256 `09fd179565f5d61eb78d6bb79b7bf2cc465c44b12f9c80bfe000f86e5dc94fee`
- `rollouts/rollout_000130.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000130.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000131.npz` → `artifacts/v100_handoff/rollouts/rollout_000131.npz`，状态 `missing`，SHA256 `9e665a83b5941a851275625b87c24e11ed43f3c78324cdc644c52e207b760d39`
- `rollouts/rollout_000131.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000131.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`
- `rollouts/rollout_000132.npz` → `artifacts/v100_handoff/rollouts/rollout_000132.npz`，状态 `missing`，SHA256 `d0cb879ef9172c89b21c9c76e7d476dd78329b12d0b3aee25e9a59c67d525737`
- `rollouts/rollout_000132.npz.json` → `artifacts/v100_handoff/rollouts/rollout_000132.npz.json`，状态 `missing`，SHA256 `eaf72b9f918623e318645a7f64359b66828792ebc07a81bc328e2b3dc2ab2da1`

已校验 31/77 个文件。完整机器清单见 download_audit.json。

下载完整 checkpoint 后重新执行：

```bash
python3 loc_transfer/audit_downloads.py
```

原始 rollout 用于审计；请保留迁移源目录，训练运行目录的保留策略独立执行。
