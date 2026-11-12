"""Measure actual P2P copy bandwidth on the visible V100 pair."""

import json
from pathlib import Path

import torch

assert torch.cuda.device_count() == 2
assert torch.cuda.can_device_access_peer(0, 1)
size = 128 * 2**20
x = torch.ones(size // 4, device="cuda:0")
y = torch.empty_like(x, device="cuda:1")
records = []
for source, target in ((x, y), (y, x)):
    with torch.cuda.device(target.device):
        for _ in range(4):
            target.copy_(source, non_blocking=True)
        torch.cuda.synchronize(target.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            target.copy_(source, non_blocking=True)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 1000
        records.append(
            {
                "source": str(source.device),
                "target": str(target.device),
                "GB_per_second": size * 20 / elapsed / 1e9,
            }
        )
result = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "arch": torch.cuda.get_arch_list(),
    "copies": records,
    "devices": [torch.cuda.get_device_name(i) for i in range(2)],
}
path = Path("artifacts/local_dual_v100_checks/nvlink.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
