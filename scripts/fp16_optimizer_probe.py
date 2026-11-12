"""Reproduce the native-FP16 optimizer safety check on a CUDA GPU.

This is a model-path probe, not a full PPO update. It distinguishes direct
half-parameter AdamW (which is unsafe here) from FP32-master AdamW.
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "src")
from cr_rl.models import ActorCritic, ModelConfig  # noqa: E402


class FakeSchema:
    entity_fields = ("row", "kind", "state", "buff_0", "buff_1", "buff_2", "flags") + tuple(
        f"field_{i}" for i in range(27)
    )
    header_fields = tuple(f"header_{i}" for i in range(224))


def batch(device, batch_size=2, context_len=8, entities=32):
    return (
        torch.randn(batch_size, context_len, 224, device=device, dtype=torch.float16),
        torch.randn(batch_size, context_len, entities, 34, device=device, dtype=torch.float16),
        torch.ones(batch_size, context_len, entities, device=device, dtype=torch.bool),
        torch.ones(batch_size, 2307, device=device, dtype=torch.bool),
        torch.ones(batch_size, context_len, device=device, dtype=torch.bool),
        torch.randint(0, 100, (batch_size, 4), device=device),
        torch.randint(0, 100, (batch_size, 8), device=device),
        torch.randint(0, 100, (batch_size, 8), device=device),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    cfg = ModelConfig(
        entity_layers=1, entity_dim=64, entity_heads=4,
        temporal_layers=2, temporal_dim=128, temporal_heads=4,
        context_len=8, max_entities=32, header_dim=224, entity_feature_dim=34,
    ).for_v2(FakeSchema())
    model = ActorCritic(cfg).to(device).half().train()
    inputs = batch(device)

    # Native half AdamW is intentionally tested and expected to fail.
    native = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    native_result = "finite"
    for _ in range(2):
        native.zero_grad(set_to_none=True)
        logp, value = model(*inputs[:5], hand_card_ids=inputs[5], deck_card_ids=inputs[6],
                             enemy_seen_card_ids=inputs[7])
        loss = -logp[:, :10].mean() + value.float().square().mean()
        if not torch.isfinite(loss):
            native_result = "nonfinite"
            break
        loss.backward(); native.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            native_result = "nonfinite_parameters"
            break

    # FP32 master parameters with an FP16 model copy.
    model = ActorCritic(cfg).to(device).half().train()
    masters = [p.detach().float().clone().requires_grad_() for p in model.parameters()]
    optimizer = torch.optim.AdamW(masters, lr=1e-5, foreach=False)
    master_losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
        logp, value = model(*inputs[:5], hand_card_ids=inputs[5], deck_card_ids=inputs[6],
                             enemy_seen_card_ids=inputs[7])
        loss = -logp[:, :10].mean() + value.float().square().mean()
        loss.backward()
        for parameter, master in zip(model.parameters(), masters, strict=True):
            if parameter.grad is not None:
                master.grad = parameter.grad.float().clone()
        torch.nn.utils.clip_grad_norm_(masters, 0.5)
        optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(model.parameters(), masters, strict=True):
                parameter.copy_(master)
        master_losses.append(float(loss.detach()))
    passed = all(torch.isfinite(torch.tensor(master_losses))) and all(
        torch.isfinite(p).all() for p in model.parameters()
    )
    print({"device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
           "direct_half_adam": native_result, "master_losses": master_losses,
           "master_optimizer_pass": bool(passed)})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
