"""Native-FP16 model support with FP32 Adam master parameters."""

from __future__ import annotations

import math
import os
from contextlib import nullcontext

import torch


def native_fp16_enabled() -> bool:
    return os.environ.get("CRRL_NATIVE_FP16", "0") == "1"


class FP32MasterAdamW:
    """AdamW over FP32 master copies while the live model remains FP16.

    The wrapper deliberately exposes the optimizer interface used by the
    trainer and stores the same state-dict shape as AdamW. Model gradients are
    copied to the masters at step time and updated weights are copied back.
    """

    def __init__(self, model, *, lr, foreach=False):
        self.model = model
        self.master_parameters = [
            parameter.detach().float().clone().requires_grad_(True)
            for parameter in model.parameters()
        ]
        self.optimizer = torch.optim.AdamW(self.master_parameters, lr=lr, foreach=foreach)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    def copy_grads_to_master(self, samples=1):
        for parameter, master in zip(self.model.parameters(), self.master_parameters, strict=True):
            if parameter.grad is None:
                master.grad = None
            else:
                master.grad = (parameter.grad.detach().float() / samples).clone()

    def step(self):
        self.copy_grads_to_master()
        self.optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(
                self.model.parameters(), self.master_parameters, strict=True
            ):
                parameter.copy_(master)

    def step_with_samples(self, samples):
        self.copy_grads_to_master(samples)
        self.optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(
                self.model.parameters(), self.master_parameters, strict=True
            ):
                parameter.copy_(master)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def autocast_context(device):
    if native_fp16_enabled():
        return nullcontext()
    if str(device).startswith("cuda") and os.environ.get("CRRL_AMP_DTYPE") == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if os.environ.get("CRRL_AMP_DTYPE") == "fp32":
        return nullcontext()
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def sanitize_grad_scaler_state(
    state: dict,
    *,
    min_scale: float = 1.0,
    reset_scale: float = 128.0,
    growth_interval: int = 100,
) -> dict:
    """Repair subnormal/zero or stale GradScaler state before restoring on resume."""
    clean_state = dict(state)
    raw_scale = clean_state.get("scale", reset_scale)
    try:
        saved_scale = float(reset_scale if raw_scale is None else raw_scale)
    except (TypeError, ValueError):
        saved_scale = 0.0
    if not math.isfinite(saved_scale) or saved_scale < min_scale:
        clean_state["scale"] = float(reset_scale)
        clean_state["_growth_tracker"] = 0
    clean_state["growth_interval"] = int(growth_interval)
    return clean_state


def build_grad_scaler(
    enabled: bool,
    state: dict | None = None,
    *,
    init_scale: float = 128.0,
    growth_interval: int = 100,
) -> torch.amp.GradScaler:
    """Create an AMP GradScaler with fast recovery and safe resume state."""
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=enabled,
        init_scale=init_scale,
        growth_interval=growth_interval,
    )
    if enabled and state:
        scaler.load_state_dict(
            sanitize_grad_scaler_state(
                state,
                reset_scale=init_scale,
                growth_interval=growth_interval,
            )
        )
    return scaler


def configure_optimizer_amp(optimizer, device, state=None):
    enabled = str(device).startswith("cuda") and os.environ.get("CRRL_AMP_DTYPE") == "fp16"
    optimizer.amp_scaler = build_grad_scaler(enabled, state)


def backward_loss(loss, optimizer):
    scaler = getattr(optimizer, "amp_scaler", None)
    (scaler.scale(loss) if scaler is not None else loss).backward()
