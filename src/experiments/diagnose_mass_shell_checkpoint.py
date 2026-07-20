"""Checkpoint-level mass-shell diagnostics on real held-out jets.

This is intentionally separate from headline metrics: it measures the learned tangent field
on known conditional paths and a short sampler trajectory, exposing numerical failure modes
that a final W1 number cannot localize.
"""

import argparse
import json
import os
import pickle
import random
import sys

# Support both `python -m experiments...` and direct script execution from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data import get_data_path
from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.distributions import gen_initial_distribution
from util.mass_shell import (
    conditional_vector_field,
    geodesic_interpolant,
    project_to_shell,
    pushforward_to_tangent,
    tangent_error_diagnostics,
    massless_energy_view,
)
from jetnet.utils import cartesian_to_EtaPhiPtE, EtaPhiPtE_to_relEtaPhiPt


TIMES = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def _quantiles(x):
    x = x[torch.isfinite(x)]
    if not x.numel():
        return [None] * 4
    q = torch.tensor([0.5, 0.9, 0.99, 0.999], device=x.device, dtype=x.dtype)
    return [float(v) for v in torch.quantile(x, q).cpu()]


def paired_endpoint_diagnostics(prior, generated, mask):
    """Exact paired prior→endpoint transport summary (no reconstructed prior allowed)."""
    real = mask > 0
    pp = EtaPhiPtE_to_relEtaPhiPt(cartesian_to_EtaPhiPtE(prior))
    gp = EtaPhiPtE_to_relEtaPhiPt(cartesian_to_EtaPhiPtE(generated))
    dphi = torch.remainder(gp[..., 1] - pp[..., 1] + torch.pi, 2 * torch.pi) - torch.pi
    deta = gp[..., 0] - pp[..., 0]
    angular = torch.sqrt(deta.square() + dphi.square())
    spatial = torch.linalg.vector_norm(generated[..., 1:4] - prior[..., 1:4], dim=-1)
    def girth(x):
        dr = torch.sqrt(x[..., 0].square() + x[..., 1].square())
        return (x[..., 2].clamp(min=0) * dr).sum(dim=1)
    return {
        "angular_displacement_quantiles": _quantiles(angular[real]),
        "delta_eta_change_quantiles": _quantiles(deta[real].abs()),
        "delta_phi_change_quantiles": _quantiles(dphi[real].abs()),
        "delta_r_change_quantiles": _quantiles((torch.sqrt(gp[..., 0].square() + gp[..., 1].square()) -
                                                 torch.sqrt(pp[..., 0].square() + pp[..., 1].square())).abs()[real]),
        "spatial_momentum_displacement_quantiles": _quantiles(spatial[real]),
        "prior_girth_quantiles": _quantiles(girth(pp)),
        "generated_girth_quantiles": _quantiles(girth(gp)),
    }


def _load_model(checkpoint, model_cfg, num_particles, device):
    model = LEFTJeN(
        max_num_jet_types=5,
        max_particles=num_particles,
        num_layers=model_cfg["n_layers"],
        hidden_dim=model_cfg["n_hidden"],
        use_residual_update=model_cfg["use_residual"],
        include_pt=True,
        use_reference_vectors=model_cfg["use_reference_vectors"],
        use_node_scalars=model_cfg["use_node_scalars"],
        node_scalar_seed=model_cfg.get("node_scalar_seed", "physics"),
        use_adaln=model_cfg["use_adaln"],
        use_attention=model_cfg["use_attention"],
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-jets", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max-step-rapidity", type=float, default=None)
    parser.add_argument("--max-substeps", type=int, default=64)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(args.run_dir, "train", "models", "final_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    full_cfg = checkpoint["full_config"]
    model_cfg = full_cfg["model"]
    num_particles = full_cfg["data"]["num_particles"]
    mass = float(model_cfg["regulator_mass"])
    model = _load_model(checkpoint, model_cfg, num_particles, device)

    with open(os.path.join(get_data_path(args.run_dir), "x_test.pkl"), "rb") as handle:
        test_data = pickle.load(handle)
    transformed = transform_rel_particle_coordinates_to_cartesian(test_data)[:args.n_jets, :num_particles]
    with open(os.path.join(args.run_dir, "train", "scale.txt")) as handle:
        scale = float(handle.read().strip())
    transformed[..., :4] /= scale
    x1 = transformed[..., :4].to(device)
    mask = transformed[..., 4].to(device)
    jet_info = test_data[:][1][:args.n_jets].clone().to(device)
    jet_info[:, 3] = jet_info[:, 3].clamp(max=num_particles)
    cond = torch.cat([
        jet_attributes.one_hot_enc_jet_type(jet_info[:, 4].long()).to(device),
        jet_info[:, 3:4],
        jet_info[:, 1:2],
    ], dim=-1)
    x0 = gen_initial_distribution(
        x_1=x1,
        prior_dist=full_cfg["training"]["prior_dist"],
        jet_features=jet_info,
        device=device,
    ) * mask.unsqueeze(-1)
    p0 = project_to_shell(x0, mass)
    p1 = project_to_shell(x1 * mask.unsqueeze(-1), mass)

    report = {
        "run_dir": args.run_dir,
        "seed": args.seed,
        "n_jets": int(x1.shape[0]),
        "regulator_mass": mass,
        "prior_dist": full_cfg["training"]["prior_dist"],
        "path_field": {},
        "sampler": [],
        "trajectory_failures": {
            "first_nonfinite_step_by_jet": {},
            "first_max_abs_gt_1e6_step_by_jet": {},
        },
    }

    with torch.no_grad():
        for value in TIMES:
            t = torch.full((x1.shape[0],), value, device=device)
            point = geodesic_interpolant(p0, p1, t, mass)
            target = conditional_vector_field(point, p1, t, mass) * mask.unsqueeze(-1)
            model_dtype = next(model.parameters()).dtype
            pred = model(point.to(model_dtype), t.to(model_dtype), cond.to(model_dtype), mask)
            pred_tan = pushforward_to_tangent(point, pred, mass) * mask.unsqueeze(-1)
            report["path_field"][str(value)] = tangent_error_diagnostics(
                point, pred_tan, target, mask, mass, dt=1 / args.steps
            )

        state = p0
        times = torch.linspace(0, 1 - 1e-5, args.steps + 1, device=device)
        seen_nonfinite = torch.zeros(x1.shape[0], dtype=torch.bool, device=device)
        seen_explosive = torch.zeros_like(seen_nonfinite)
        for step in range(args.steps):
            t = times[step].expand(x1.shape[0])
            pred = model(state.to(model_dtype), t.to(model_dtype), cond.to(model_dtype), mask)
            pred_tan = pushforward_to_tangent(state, pred, mass) * mask.unsqueeze(-1)
            if step % 8 == 0 or step == args.steps - 1:
                pt = torch.linalg.vector_norm(state[..., 1:3], dim=-1)[mask > 0]
                stats = tangent_error_diagnostics(
                    state, pred_tan, torch.zeros_like(pred_tan), mask, mass,
                    dt=float(times[step + 1] - times[step]),
                )
                jet_max_abs = state.abs().amax(dim=(1, 2))
                finite_max = jet_max_abs[torch.isfinite(jet_max_abs)]
                stats.update({
                    "step": step,
                    "t": float(times[step]),
                    "pt_quantiles": _quantiles(pt),
                    "state_max_abs_quantiles": _quantiles(finite_max),
                    "state_max_abs_max": (float(finite_max.max()) if finite_max.numel() else None),
                })
                report["sampler"].append(stats)
            state = model.step_hyperbolic(
                state, cond, mask, times[step], times[step + 1],
                hyperbolic_model="mass_shell", regulator_mass=mass,
                use_cfg=False,
                max_step_rapidity=args.max_step_rapidity,
                max_substeps=args.max_substeps,
            )
            finite_jet = torch.isfinite(state).all(dim=(1, 2))
            max_abs = state.abs().amax(dim=(1, 2))
            newly_nonfinite = (~finite_jet) & (~seen_nonfinite)
            newly_explosive = finite_jet & (max_abs > 1e6) & (~seen_explosive)
            for idx in newly_nonfinite.nonzero(as_tuple=False).flatten().tolist():
                report["trajectory_failures"]["first_nonfinite_step_by_jet"][str(idx)] = step
            for idx in newly_explosive.nonzero(as_tuple=False).flatten().tolist():
                report["trajectory_failures"]["first_max_abs_gt_1e6_step_by_jet"][str(idx)] = step
            seen_nonfinite |= ~finite_jet
            seen_explosive |= finite_jet & (max_abs > 1e6)

        physical_prior = massless_energy_view(p0, mask)
        physical_generated = massless_energy_view(state, mask)
        report["paired_endpoint"] = paired_endpoint_diagnostics(
            physical_prior, physical_generated, mask)
        report["trajectory_failures"].update({
            "n_nonfinite": int(seen_nonfinite.sum()),
            "n_finite_max_abs_gt_1e6": int((~seen_nonfinite & seen_explosive).sum()),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote diagnostics to {args.out}")


if __name__ == "__main__":
    main()
