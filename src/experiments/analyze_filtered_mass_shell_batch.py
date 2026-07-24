#!/usr/bin/env python3
"""Replay Stage-1 attributes and audit outlier-filter sensitivity for one saved batch."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
from jetnet.utils import EtaPhiPtE_to_relEtaPhiPt, cartesian_to_EtaPhiPtE
import jetnet.evaluation as jetnet_eval

from generate_samples import gen_initial_distribution
from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.eval_report import _jet_girth, _jet_invariant_mass, _polar_abs_to_cartesian
from util.mass_shell import massless_energy_view, project_to_shell
from util.metrics import run_save_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regulator-mass", type=float, default=0.1)
    return parser.parse_args()


def abs_test_particles(dataset):
    torch.manual_seed(42)
    particles, jets = dataset[:]
    jet_eta = jets[:, 0].unsqueeze(1)
    jet_phi = (2 * torch.pi * torch.rand(len(dataset))).unsqueeze(1)
    eta_rel, phi_rel, pt_rel = particles[:, :, :3].unbind(-1)
    pt = pt_rel * jets[:, 1].unsqueeze(1)
    eta = eta_rel + jet_eta
    phi = phi_rel + jet_phi
    energy = pt * torch.cosh(eta)
    polar = torch.stack([eta, phi, pt, energy], dim=-1)
    mask = particles[:, :, 3] > 0
    return polar, mask


def endpoint_features(cartesian):
    polar = cartesian_to_EtaPhiPtE(cartesian)
    relative = EtaPhiPtE_to_relEtaPhiPt(polar)
    mask = cartesian.abs().sum(-1) > 0
    relative = relative * mask.unsqueeze(-1)
    total = cartesian[..., 1:4].sum(1)
    jet_pt = torch.sqrt(total[:, 0].square() + total[:, 1].square())
    return torch.stack([
        jet_pt,
        _jet_invariant_mass(cartesian),
        _jet_girth(relative),
        polar[..., 2].masked_fill(~mask, 0).amax(1),
        cartesian.abs().amax(dim=(1, 2)),
    ], dim=-1)


def envelope(values, lower_q=0.0, upper_q=1.0):
    lower = torch.quantile(values.to(torch.float64), lower_q, dim=0)
    upper = torch.quantile(values.to(torch.float64), upper_q, dim=0)
    return lower, upper


def inside(values, bounds):
    lower, upper = bounds
    return ((values >= lower) & (values <= upper)).all(dim=1)


def metric_summary(test_rel, gen_rel):
    n = min(len(test_rel), len(gen_rel))
    test = test_rel[:n, :, :3]
    gen = gen_rel[:n, :, :3]
    result = {"n": n}
    result["w1m"] = float(jetnet_eval.w1m(jets1=gen, jets2=test)[0])
    result["w1p"] = jetnet_eval.w1p(jets1=gen, jets2=test)[0].tolist()
    result["w1efp"] = jetnet_eval.w1efp(jets1=gen, jets2=test)[0].tolist()
    result["fpd"] = float(jetnet_eval.fpd(
        real_features=test.reshape(n, -1),
        gen_features=gen.reshape(n, -1),
        seed=42,
    )[0])
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.source_run, "data", "x_train.pkl"), "rb") as handle:
        x_train = pickle.load(handle)
    with open(os.path.join(args.source_run, "data", "x_test.pkl"), "rb") as handle:
        x_test = pickle.load(handle)
    with open(os.path.join(args.source_run, "train", "scale.txt")) as handle:
        scale = float(handle.read())
    samples = torch.load(os.path.join(args.eval_dir, "samples.pt"), map_location="cpu")
    saved_prior = torch.load(os.path.join(args.eval_dir, "prior_samples.pt"), map_location="cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    stage1 = jet_attributes.load_model(
        model_path=os.path.join(args.source_run, "jet_attr_model.pth")
    ).to(device).eval()
    # infer.py constructs the Stage-2 model after seeding and before drawing Stage-1
    # attributes. Reproduce that CPU-RNG consumption exactly; otherwise the replayed
    # jet-type stream diverges even though Stage-2 integration itself is deterministic.
    rng_consumer = LEFTJeN(
        max_num_jet_types=5,
        max_particles=30,
        num_layers=6,
        hidden_dim=256,
        use_residual_update=True,
        include_pt=True,
        use_reference_vectors=True,
        backbone="tangent_attention",
        include_mass_condition=True,
        num_attention_heads=8,
        vector_channels=16,
        regulator_mass=args.regulator_mass,
        velocity_readout_init="zero",
    ).to(device)
    del rng_consumer

    attrs_parts = []
    replay_prior_parts = []
    for start in range(0, args.n_samples, args.batch_size):
        size = min(args.batch_size, args.n_samples - start)
        attrs, _ = jet_attributes.generate_jets(stage1, device, n_jet_types=1, num_jets=size)
        attrs_parts.append(attrs.cpu())
        eta = attrs[:, 5]
        pt = attrs[:, 6]
        n_particles = attrs[:, -1].long().clamp(max=samples.shape[1])
        jet_phi = 2 * torch.pi * torch.rand(size, device=device)
        prior = gen_initial_distribution(
            batch_size=size,
            num_particles=samples.shape[1],
            prior_dist="axis_aligned_per_jet",
            jet_features=torch.stack([eta, pt], dim=-1),
            jet_phi=jet_phi,
            device=device,
            model_scale=scale,
        )
        mask = jet_attributes.generate_masks(
            n_particles, max_particles_per_jet=samples.shape[1], device=device
        )
        prior = project_to_shell(prior * mask.unsqueeze(-1), args.regulator_mass)
        prior = massless_energy_view(prior, mask)
        replay_prior_parts.append((scale * prior).cpu())
    attrs = torch.cat(attrs_parts)
    replay_prior = torch.cat(replay_prior_parts)
    finite_prior = torch.isfinite(saved_prior) & torch.isfinite(replay_prior)
    prior_max_abs_diff = float((saved_prior[finite_prior] - replay_prior[finite_prior]).abs().max())
    prior_exact = bool(torch.equal(saved_prior, replay_prior))
    if prior_max_abs_diff > 1e-8:
        raise RuntimeError(f"Stage-1 replay failed prior verification: {prior_max_abs_diff}")
    torch.save(attrs, os.path.join(args.out_dir, "replayed_stage1_attributes.pt"))

    train_jets = x_train[:][1].cpu()
    train_stage1 = torch.stack([
        train_jets[:, 0], train_jets[:, 1], train_jets[:, 2],
        train_jets[:, 2] / train_jets[:, 1].clamp(min=1e-8),
    ], dim=-1).to(torch.float64)
    gen_stage1 = torch.stack([
        attrs[:, 5], attrs[:, 6], attrs[:, 7],
        attrs[:, 7] / attrs[:, 6].clamp(min=1e-8),
    ], dim=-1).to(torch.float64)

    test_polar, test_mask = abs_test_particles(x_test)
    test_cart = _polar_abs_to_cartesian(test_polar) * test_mask.unsqueeze(-1)
    test_features = endpoint_features(test_cart)
    gen_features = endpoint_features(samples)
    finite = torch.isfinite(samples).all(dim=(1, 2))

    train_range = envelope(train_stage1)
    train_q999 = envelope(train_stage1, 0.0005, 0.9995)
    test_range = envelope(test_features[torch.isfinite(test_features).all(1)])
    test_q999 = envelope(test_features[torch.isfinite(test_features).all(1)], 0.0, 0.999)

    masks = {
        "finite_only": finite,
        "stage1_train_range": finite & inside(gen_stage1, train_range),
        "stage2_test_range": finite & inside(gen_features, test_range),
        "combined_train_test_range": (
            finite & inside(gen_stage1, train_range) & inside(gen_features, test_range)
        ),
        "combined_q999_sensitivity": (
            finite & inside(gen_stage1, train_q999) & inside(gen_features, test_q999)
        ),
    }

    test_rel = EtaPhiPtE_to_relEtaPhiPt(test_polar) * test_mask.unsqueeze(-1)
    gen_polar = cartesian_to_EtaPhiPtE(samples)
    gen_mask = samples.abs().sum(-1) > 0
    gen_rel = EtaPhiPtE_to_relEtaPhiPt(gen_polar) * gen_mask.unsqueeze(-1)
    sensitivity = {}
    for name, mask in masks.items():
        selected = mask.nonzero(as_tuple=False).flatten()
        sensitivity[name] = {
            "retained": int(mask.sum()),
            "removed": int((~mask).sum()),
            "metrics": metric_summary(test_rel, gen_rel[selected]),
        }

    combined = masks["combined_train_test_range"]
    selected = combined.nonzero(as_tuple=False).flatten()
    filtered_dir = os.path.join(args.out_dir, "combined_filtered_eval")
    filtered_metrics = run_save_metrics(
        X_test=x_test,
        gen_samples=samples[selected],
        jet_types=["g"],
        output_path=filtered_dir,
        device=device,
        gen_jet_types=torch.zeros(len(selected), dtype=torch.long),
        gen_pt_cond=attrs[selected, 6],
        prior_samples=saved_prior[selected],
    )

    report = {
        "prior_replay_exact": prior_exact,
        "prior_replay_max_abs_diff": prior_max_abs_diff,
        "stage1_feature_order": ["eta", "pt", "mass", "mass_over_pt"],
        "stage1_train_range": [train_range[0].tolist(), train_range[1].tolist()],
        "stage1_train_q0005_q9995": [train_q999[0].tolist(), train_q999[1].tolist()],
        "stage2_feature_order": ["jet_pt", "jet_mass", "girth", "max_particle_pt", "max_abs"],
        "stage2_test_range": [test_range[0].tolist(), test_range[1].tolist()],
        "stage2_test_q999": test_q999[1].tolist(),
        "sensitivity": sensitivity,
        "combined_filtered_full_metrics": filtered_metrics,
    }
    with open(os.path.join(args.out_dir, "filtered_analysis.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["sensitivity"], indent=2))


if __name__ == "__main__":
    main()
