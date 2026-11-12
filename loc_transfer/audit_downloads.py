"""Audit the immutable handoff manifest without loading or changing checkpoints."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "artifacts/v100_handoff"
manifest = json.loads((HANDOFF / "checkpoints/latest.json").read_text())
records = [*manifest["files"], {"path": manifest["pool"], "sha256": manifest["pool_sha256"]}]
report = []
for record in records:
    relative = record["path"]
    local = HANDOFF / relative
    if relative == manifest["checkpoint"]:
        local = HANDOFF / "weights" / Path(relative).name
    status = "missing"
    if local.exists():
        if record.get("bytes", local.stat().st_size) != local.stat().st_size:
            status = "size_mismatch"
        else:
            with local.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            status = "verified" if actual == record["sha256"] else "sha_mismatch"
    report.append({**record, "local": str(local.relative_to(ROOT)), "status": status})
result = {"update": manifest["update"], "files": report}
(ROOT / "loc_transfer/download_audit.json").write_text(json.dumps(result, indent=2) + "\n")
missing = [r for r in report if r["status"] != "verified"]
lines = [
    "# 本地迁移下载清单",
    "",
    "来源：迁移包 checkpoints/latest.json，恢复点 update 142。",
    "",
    "云端运行目录：`MyDrive/cr_rl/runs/122m_long1000_sample2x_20261106/candidate/`。",
    "",
    "## 必需：完整续训状态",
    "",
]
for r in missing:
    if r["path"] == manifest["checkpoint"]:
        lines += [
            f"- 云端相对路径：`{r['path']}`",
            f"- 下载到：`{r['local']}`",
            f"- 完整大小：{r['bytes']:,} 字节（{r['bytes'] / 2**30:.3f} GiB）",
            f"- SHA256：`{r['sha256']}`",
            "- 现有 `.partial` 未完成，不能用 Actor 导出替代完整模型、Critic 和优化器状态。",
            "",
        ]
others = [r for r in missing if r["path"] != manifest["checkpoint"]]
lines += ["## 可选：历史审计 rollout（不阻碍恢复训练）", ""]
lines += [
    f"- `{r['path']}` → `{r['local']}`，状态 `{r['status']}`，SHA256 `{r['sha256']}`"
    for r in others
] or ["无。"]
lines += [
    "",
    f"已校验 {len(report) - len(missing)}/{len(report)} 个文件。"
    "完整机器清单见 download_audit.json。",
    "",
    "下载完整 checkpoint 后重新执行：",
    "",
    "```bash",
    "python3 loc_transfer/audit_downloads.py",
    "```",
    "",
    "原始 rollout 用于审计；请保留迁移源目录，训练运行目录的保留策略独立执行。",
    "",
]
(ROOT / "loc_transfer/DOWNLOADS.md").write_text("\n".join(lines))
print(
    json.dumps(
        {
            "verified": len(report) - len(missing),
            "total": len(report),
            "missing": [r["path"] for r in missing],
        },
        indent=2,
    )
)
