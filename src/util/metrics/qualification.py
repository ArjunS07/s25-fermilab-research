"""Small, deterministic helpers for bounded training qualifications."""

from __future__ import annotations

import numpy as np


def optimizer_limit_reached(global_step: int, max_steps: int | None) -> bool:
    return max_steps is not None and global_step >= max_steps


def loss_improvement_summary(losses, max_window: int = 200) -> dict:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("qualification loss history must be a non-empty vector")
    window = min(max_window, max(1, values.size // 2))
    first = float(np.median(values[:window]))
    final = float(np.median(values[-window:]))
    improvement = (first - final) / max(abs(first), 1e-12)
    return {
        "loss_window": window,
        "first_loss_median": first,
        "final_loss_median": final,
        "loss_improvement_fraction": improvement,
        "losses_finite": bool(np.isfinite(values).all()),
    }
