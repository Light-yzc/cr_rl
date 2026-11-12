"""Small CUDA probe for native-FP16 PPO boundary arithmetic.

This deliberately does not construct the full model. It catches the failure
modes that can invalidate a full run before spending a long update on it.
"""
from __future__ import annotations

import argparse
import math

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(7)
    logits = torch.randn(64, 2307, device=device, dtype=dtype, generator=g) * 3
    legal = torch.zeros_like(logits, dtype=torch.bool)
    legal[:, :20] = True

    # The safe action boundary used by policy.py.
    masked = logits.float().masked_fill(~legal, -1e9)
    logp = torch.log_softmax(masked, dim=-1)
    probs = logp.exp()
    assert torch.isfinite(logp).all() and torch.isfinite(probs).all()
    assert (probs.sum(-1) - 1).abs().max() < 1e-5

    delta = torch.linspace(-11, 11, 4096, device=device, dtype=torch.float32)
    ratio = delta.exp()
    assert torch.isfinite(ratio).all()

    rewards = torch.randn(3072, device=device, dtype=torch.float32) * 0.1
    values = torch.randn(3073, device=device, dtype=torch.float32) * 0.5
    gae = torch.zeros_like(rewards)
    running = torch.zeros((), device=device, dtype=torch.float32)
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + 0.9995 * (values[t + 1] - values[t]) + 0.9995 * 0.993 * running
        gae[t] = running
    assert torch.isfinite(gae).all()

    print({
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "fp16_logits": str(logits.dtype),
        "masked_logp_finite": bool(torch.isfinite(logp).all()),
        "prob_sum_max_error": float((probs.sum(-1) - 1).abs().max()),
        "ratio_max": float(ratio.max()),
        "gae_std": float(gae.std()),
        "status": "PASS",
    })


if __name__ == "__main__":
    main()
