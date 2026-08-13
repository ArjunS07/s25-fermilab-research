"""Low-overhead training-gradient diagnostics."""

import torch


@torch.no_grad()
def collect_gradient_stats(model):
    """Collect per-parameter diagnostics with one device-to-host synchronization."""
    names = []
    values = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        names.append(name)
        values.append(torch.stack((
            param.data.norm(2),
            param.grad.norm(2),
            param.grad.abs().mean(),
        )))
    if not values:
        return {}

    rows = torch.stack(values).detach().cpu().tolist()
    return {
        name: {
            "norm": grad_norm,
            "mean": grad_mean,
            "weight_norm": weight_norm,
            "update_ratio": grad_norm / (weight_norm + 1e-8),
        }
        for name, (weight_norm, grad_norm, grad_mean) in zip(names, rows)
    }
