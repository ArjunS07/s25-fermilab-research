"""Cosine learning-rate schedule advanced after each optimizer update."""

from __future__ import annotations

import math

import torch


def learning_rate_multiplier(step: int, *, total_steps: int, warmup_steps: int,
                             eta_min_factor: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        # Retain the prior nonzero start without a first-boundary discontinuity.
        return 1e-6 + (1.0 - 1e-6) * step / warmup_steps

    cosine_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / cosine_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return eta_min_factor + (1.0 - eta_min_factor) * cosine


def build_step_scheduler(optimizer, *, total_steps: int, warmup_steps: int,
                         eta_min_factor: float):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            eta_min_factor=eta_min_factor,
        ),
    )
