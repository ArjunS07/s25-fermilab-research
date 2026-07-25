"""Joint-structure auxiliary losses for mass-shell flow matching."""

from __future__ import annotations

import torch

from util.minkowski_utils import dotsq4


def _masked_mean_per_jet(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    support64 = support.to(torch.float64)
    count = support64.sum(dim=tuple(range(1, support64.ndim))).clamp(min=1.0)
    total = (values * support64).sum(dim=tuple(range(1, values.ndim)))
    return total / count


def gram_transport_derivative(
    state: torch.Tensor, velocity: torch.Tensor,
) -> torch.Tensor:
    """Return d/dt <x_i,x_j> for every ordered constituent pair."""
    x = state.to(torch.float64)
    v = velocity.to(torch.float64)
    return (
        dotsq4(v.unsqueeze(2), x.unsqueeze(1))
        + dotsq4(x.unsqueeze(2), v.unsqueeze(1))
    )


def gram_transport_loss(
    state: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalized pairwise Gram-derivative error, averaged per jet."""
    particles = state.shape[1]
    real = mask > 0
    eye = torch.eye(particles, dtype=torch.bool, device=state.device).unsqueeze(0)
    support = real.unsqueeze(2) & real.unsqueeze(1) & ~eye
    pred_dot = gram_transport_derivative(state, prediction)
    target_dot = gram_transport_derivative(state, target)
    numerator = _masked_mean_per_jet((pred_dot - target_dot).square(), support)
    denominator = _masked_mean_per_jet(target_dot.square(), support).detach()
    valid_pairs = support.sum(dim=(1, 2)) > 0
    if not valid_pairs.any():
        return prediction.to(torch.float64).sum() * 0.0
    return (numerator[valid_pairs] / denominator[valid_pairs].clamp(min=eps)).mean()


def lab_time_positive_norm(vector: torch.Tensor, lab_time: torch.Tensor) -> torch.Tensor:
    """Positive norm 2<a,e_t>²-<a,a> for a unit timelike lab reference."""
    a = vector.to(torch.float64)
    et = lab_time.to(torch.float64)
    return 2.0 * dotsq4(a, et).square() - dotsq4(a, a)


def total_momentum_transport_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lab_time: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalized error in the collective ambient four-momentum derivative."""
    mask64 = mask.to(torch.float64).unsqueeze(-1)
    pred_total = (prediction.to(torch.float64) * mask64).sum(dim=1)
    target_total = (target.to(torch.float64) * mask64).sum(dim=1)
    et = lab_time.to(torch.float64)
    if et.ndim == 3:
        et = et.squeeze(1)
    numerator = lab_time_positive_norm(pred_total - target_total, et)
    denominator = lab_time_positive_norm(target_total, et).detach()
    return (numerator / denominator.clamp(min=eps)).mean()


def auxiliary_warmup(step: int, start_step: int, warmup_steps: int) -> float:
    """Linear continuation-local warmup in [0, 1]."""
    if warmup_steps <= 0:
        return 1.0
    return max(0.0, min(1.0, (step - start_step) / warmup_steps))
