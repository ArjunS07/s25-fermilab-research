"""Vanilla ambient-space conditional flow matching.

The component-wise loss is intentionally frame dependent.  This is the conventional
Euclidean FM control; the LorentzNet neural field itself remains equivariant.
"""

import torch


def euclidean_interpolant_and_target(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (-1,) + (1,) * (x0.ndim - 1)
    time = t.view(shape).to(x0.dtype)
    return (1 - time) * x0 + time * x1, x1 - x0


def euclidean_flow_loss(*, model, x0, x1, t, mask, conditions, references):
    state, target = euclidean_interpolant_and_target(x0, x1, t)
    prediction = model(
        x=state,
        t=t.to(next(model.parameters()).dtype),
        jet_conditions=conditions.to(next(model.parameters()).dtype),
        mask=mask,
        ref_vectors=references,
    )
    mask4 = mask.unsqueeze(-1).to(prediction.dtype)
    squared_error = (prediction - target.to(prediction.dtype)).square() * mask4
    return squared_error.sum() / (mask.sum().clamp_min(1).to(prediction.dtype) * x0.shape[-1])


__all__ = ["euclidean_flow_loss", "euclidean_interpolant_and_target"]
