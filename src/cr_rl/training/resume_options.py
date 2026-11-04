"""Explicit optimizer LR transitions while preserving resumed Adam state."""

import math


def override_resume_rates(actor_optimizer, critic_optimizer, actor_lr=None, critic_lr=None):
    rates = {"actor": actor_lr, "critic": critic_lr}
    for rate in rates.values():
        if rate is not None and (not math.isfinite(rate) or rate <= 0):
            raise ValueError("Resume learning rates must be finite and positive")
    report = {}
    for name, optimizer in (("actor", actor_optimizer), ("critic", critic_optimizer)):
        before = [float(group["lr"]) for group in optimizer.param_groups]
        if rates[name] is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(rates[name])
        report[name] = dict(
            before=before,
            after=[float(group["lr"]) for group in optimizer.param_groups],
            explicit_override=rates[name] is not None,
        )
    return report
