#!/usr/bin/env python3
"""Build a replay bundle from a selected Stage-1 model without running Stage 2."""

from __future__ import annotations

import argparse
import os

import torch

from generate_samples import gen_initial_distribution
from util import jet_attributes
from util.mass_shell import project_to_shell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run", required=True)
    parser.add_argument("--stage2-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regulator-mass", type=float, default=0.1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = jet_attributes.load_model(
        os.path.join(args.stage1_run, "jet_attr_model.pth"), device=device
    ).eval()
    with open(os.path.join(args.stage2_run, "train", "scale.txt")) as handle:
        scale = float(handle.read())

    attrs_parts, phi_parts, mask_parts, prior_parts = [], [], [], []
    with torch.no_grad():
        for start in range(0, args.n_samples, args.batch_size):
            size = min(args.batch_size, args.n_samples - start)
            attrs, _ = jet_attributes.generate_jets(
                model, device, n_jet_types=1, num_jets=size
            )
            phi = 2 * torch.pi * torch.rand(size, device=device)
            mask = jet_attributes.generate_masks(
                attrs[:, -1].long().clamp(max=30),
                max_particles_per_jet=30, device=device,
            )
            prior = gen_initial_distribution(
                batch_size=size, num_particles=30,
                prior_dist="axis_aligned_per_jet",
                jet_features=torch.stack([attrs[:, 5], attrs[:, 6]], dim=-1),
                jet_phi=phi, device=device, model_scale=scale,
            )
            prior = project_to_shell(
                prior * mask.unsqueeze(-1), args.regulator_mass
            )
            attrs_parts.append(attrs.cpu())
            phi_parts.append(phi.cpu())
            mask_parts.append(mask.cpu())
            prior_parts.append(prior.cpu())
    torch.save({
        "format_version": 1,
        "metadata": {
            "n_samples": args.n_samples, "num_particles": 30,
            "prior_dist": "axis_aligned_per_jet", "final_scale": scale,
            "use_hyperbolic": True, "hyperbolic_model": "mass_shell",
            "regulator_mass": args.regulator_mass,
            "provenance": "stage1_v3_generated_conditions",
        },
        "generated_jet_attrs": torch.cat(attrs_parts),
        "jet_phi": torch.cat(phi_parts),
        "masks": torch.cat(mask_parts),
        "internal_prior": torch.cat(prior_parts),
    }, args.output)
    print(f"Saved shared Stage-1 replay bundle to {args.output}")


if __name__ == "__main__":
    main()
