"""Read back one immutable local restore point and check all optimizer/model tensors."""

import hashlib
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(4)
run = Path(sys.argv[1]).resolve()
latest = torch.load(run / "ppo_latest.pt", map_location="cpu", mmap=True, weights_only=False)
update = latest["updates"]
checkpoint = run / "checkpoints" / f"update_{update:06d}.pt"
with checkpoint.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
count = 0


def check(value):
    global count
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        assert torch.isfinite(value).all(), "Non-finite saved tensor"
        count += 1
    elif isinstance(value, dict):
        for child in value.values():
            check(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            check(child)


for key in ("model", "critic", "optimizer", "critic_optimizer"):
    check(latest[key])
for key in ("actor_amp_scaler", "critic_amp_scaler"):
    assert latest[key]["scale"] > 0
pool = json.loads((run / "checkpoints" / f"update_{update:06d}.pool.json").read_text())
assert all(Path(entry["checkpoint"]).exists() for entry in pool["entries"])
result = {
    "update": update,
    "sha256": digest,
    "finite_tensors": count,
    "completed_games": latest["training_game_counter"]["completed_games"],
    "actor_optimizer_steps": max(int(v["step"]) for v in latest["optimizer"]["state"].values()),
    "critic_optimizer_steps": max(
        int(v["step"]) for v in latest["critic_optimizer"]["state"].values()
    ),
    "actor_scaler": latest["actor_amp_scaler"],
    "critic_scaler": latest["critic_amp_scaler"],
    "pool_entries_resolve": len(pool["entries"]),
    "passed": True,
}
out = Path(__file__).resolve().parents[1] / "artifacts/local_dual_v100_checks"
(out / f"checkpoint_{update:06d}_audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
