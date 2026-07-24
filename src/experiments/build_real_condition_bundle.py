#!/usr/bin/env python3
"""Build an inference replay bundle conditioned on held-out real jet attributes."""

from __future__ import annotations

import argparse
import os
import pickle

import torch

from generate_samples import gen_initial_distribution
from util import jet_attributes
from util.mass_shell import project_to_shell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regulator-mass", type=float, default=0.1)
    args = parser.parse_args()
    with open(os.path.join(args.source_run, "data", "x_test.pkl"), "rb") as handle:
        dataset = pickle.load(handle)
    with open(os.path.join(args.source_run, "train", "scale.txt")) as handle:
        scale = float(handle.read())
    jets = dataset[:][1][:args.n_samples].cpu()
    onehot = jet_attributes.one_hot_enc_jet_type(jets[:, 4].long()).float()
    attrs = torch.cat([onehot, jets[:, :4]], dim=-1)
    torch.manual_seed(args.seed)
    phi_parts, mask_parts, prior_parts = [], [], []
    for start in range(0, len(attrs), args.batch_size):
        batch = attrs[start:start + args.batch_size]
        phi = 2 * torch.pi * torch.rand(len(batch))
        mask = jet_attributes.generate_masks(
            batch[:, -1].long(), max_particles_per_jet=30, device="cpu"
        )
        prior = gen_initial_distribution(
            batch_size=len(batch), num_particles=30,
            prior_dist="axis_aligned_per_jet",
            jet_features=torch.stack([batch[:, 5], batch[:, 6]], dim=-1),
            jet_phi=phi, device="cpu", model_scale=scale,
        )
        prior = project_to_shell(prior * mask.unsqueeze(-1), args.regulator_mass)
        phi_parts.append(phi)
        mask_parts.append(mask)
        prior_parts.append(prior)
    torch.save({
        "format_version": 1,
        "metadata": {
            "n_samples": len(attrs), "num_particles": 30,
            "prior_dist": "axis_aligned_per_jet", "final_scale": scale,
            "use_hyperbolic": True, "hyperbolic_model": "mass_shell",
            "regulator_mass": args.regulator_mass,
            "provenance": "held_out_real_jet_attributes",
        },
        "generated_jet_attrs": attrs,
        "jet_phi": torch.cat(phi_parts),
        "masks": torch.cat(mask_parts),
        "internal_prior": torch.cat(prior_parts),
    }, args.output)
    print(f"Saved {len(attrs)} held-out condition records to {args.output}")


if __name__ == "__main__":
    main()
