"""Write compact shell-vs-massless sample diagnostics for one or more saved tensors."""

import argparse
import json
import os

import torch

from util.geometry.mass_shell import massless_energy_view
from util.geometry.minkowski_utils import normsq4


def _quantiles(values):
    values = values[torch.isfinite(values)].to(torch.float64)
    if not values.numel():
        return [None] * 4
    levels = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64)
    return [float(v) for v in torch.quantile(values, levels)]


def summarize(samples):
    samples = samples.detach().cpu()
    valid_jet = torch.isfinite(samples).all(dim=-1).all(dim=-1)
    finite = samples[valid_jet]
    mask = finite.abs().sum(dim=-1) > 0
    massless = massless_energy_view(finite)
    spatial_equal = torch.equal(finite[..., 1:4], massless[..., 1:4])
    pt = torch.linalg.vector_norm(finite[..., 1:3], dim=-1)
    pnorm = torch.linalg.vector_norm(finite[..., 1:4], dim=-1)
    delta_e = finite[..., 0] - massless[..., 0]
    total = finite.sum(dim=1)
    total_mass = torch.sqrt(normsq4(total).clamp(min=0.0))
    total_massless = torch.sqrt(normsq4(massless.sum(dim=1)).clamp(min=0.0))
    return {
        "n_total": int(samples.shape[0]),
        "n_invalid": int((~valid_jet).sum()),
        "spatial_momentum_identical_in_massless_view": spatial_equal,
        "particle_energy_quantiles": _quantiles(finite[..., 0][mask]),
        "particle_pt_quantiles": _quantiles(pt[mask]),
        "particle_pnorm_quantiles": _quantiles(pnorm[mask]),
        "regulator_delta_energy_quantiles": _quantiles(delta_e[mask]),
        "summed_fourvector_mass_quantiles": _quantiles(total_mass),
        "massless_view_summed_mass_quantiles": _quantiles(total_massless),
        "shell_msq_quantiles": _quantiles(normsq4(finite)[mask]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {}
    for item in args.runs:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"expected LABEL=PATH, got {item!r}")
        report[label] = summarize(torch.load(path, map_location="cpu", weights_only=False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote sample diagnostics to {args.out}")


if __name__ == "__main__":
    main()
