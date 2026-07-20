"""Numerically safe endpoint-tail summaries independent of optional JetNet metrics."""

import torch


def endpoint_tail_diagnostics(samples: torch.Tensor) -> dict:
    """Describe endpoint failures without treating finite explosions as valid bulk."""
    x = samples.detach().to(torch.float64).cpu()
    finite = torch.isfinite(x).all(dim=-1).all(dim=-1)
    finite_x = x[finite]
    report = {
        "n_total": int(x.shape[0]),
        "n_nonfinite": int((~finite).sum()),
        "fraction_nonfinite": float((~finite).to(torch.float64).mean()),
    }
    if not finite_x.numel():
        return report

    max_abs = finite_x.abs().amax(dim=(1, 2))
    jet = finite_x.sum(dim=1)
    jet_pt = torch.linalg.vector_norm(jet[:, 1:3], dim=-1)
    jet_m2 = jet[:, 0].square() - jet[:, 1:4].square().sum(dim=-1)
    jet_mass = jet_m2.clamp(min=0).sqrt()

    def quantiles(values):
        q = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64)
        return {name: float(value) for name, value in zip(
            ("p50", "p90", "p99", "p999", "max"), torch.quantile(values, q))}

    report.update({
        "finite_max_abs_quantiles": quantiles(max_abs),
        "finite_jet_pt_quantiles": quantiles(jet_pt),
        "finite_jet_mass_quantiles": quantiles(jet_mass),
        "n_finite_max_abs_gt_1e3": int((max_abs > 1e3).sum()),
        "n_finite_max_abs_gt_1e6": int((max_abs > 1e6).sum()),
        "fraction_finite_max_abs_gt_1e6": float((max_abs > 1e6).to(torch.float64).mean()),
    })
    return report
