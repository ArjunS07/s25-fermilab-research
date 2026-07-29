"""Optimizer-step learning-rate schedules.

Epoch-level stepping caused abrupt LR jumps every ~500 optimizer updates in the
g30 pipeline. These schedules preserve the existing epoch-valued configuration
while applying it smoothly at optimizer-step resolution.
"""

from __future__ import annotations

import math

import torch


def learning_rate_multiplier(step: int, *, total_steps: int, warmup_steps: int,
                             eta_min_factor: float, schedule: str,
                             restart_steps: int | None = None) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        # Retain the prior nonzero start without a first-boundary discontinuity.
        return 1e-6 + (1.0 - 1e-6) * step / warmup_steps

    if schedule == "monotonic_cosine":
        cosine_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / cosine_steps))
    elif schedule == "legacy_restarts":
        if restart_steps is None or restart_steps < 1:
            raise ValueError("legacy_restarts requires positive restart_steps")
        progress = ((step - warmup_steps) % restart_steps) / restart_steps
    else:
        raise ValueError(f"unknown learning-rate schedule {schedule!r}")
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return eta_min_factor + (1.0 - eta_min_factor) * cosine


def build_step_scheduler(optimizer, *, total_steps: int, steps_per_epoch: int,
                         warmup_epochs: int, eta_min_factor: float,
                         schedule: str, restart_epoch: int,
                         warmup_steps: int | None = None):
    if warmup_steps is None:
        warmup_steps = warmup_epochs * steps_per_epoch
    default_restart_epochs = max(1, total_steps // steps_per_epoch // 4)
    restart_steps = (
        (restart_epoch if restart_epoch > 0 else default_restart_epochs)
        * steps_per_epoch
    )
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            eta_min_factor=eta_min_factor,
            schedule=schedule,
            restart_steps=restart_steps,
        ),
    )


def build_epoch_scheduler(optimizer, *, total_epochs: int, warmup_epochs: int,
                          eta_min_factor: float, schedule: str,
                          restart_epoch: int):
    """Reproduce the historical scheduler, advanced once after every epoch."""
    base_lr = optimizer.param_groups[0]["lr"]
    if schedule == "monotonic_cosine":
        tail = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=base_lr * eta_min_factor,
        )
    elif schedule == "legacy_restarts":
        default_restart = total_epochs // 4 if total_epochs >= 20 else max(1, total_epochs // 2)
        tail = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_epoch if restart_epoch > 0 else default_restart,
            T_mult=1,
            eta_min=base_lr * eta_min_factor,
        )
    else:
        raise ValueError(f"unknown learning-rate schedule {schedule!r}")
    if warmup_epochs <= 0:
        return tail
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, tail], milestones=[warmup_epochs]
    )
