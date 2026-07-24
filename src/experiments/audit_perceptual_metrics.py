#!/usr/bin/env python3
"""Audit FPD/FPND contracts and condition-stratified perceptual distances."""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch
import jetnet.evaluation as jetnet_eval
from jetnet.utils import EtaPhiPtE_to_relEtaPhiPt, cartesian_to_EtaPhiPtE

from experiments.analyze_filtered_mass_shell_batch import abs_test_particles
from util.perceptual import conditional_fpnd, particlenet_activations


FEATURES = ("eta", "pt", "mass", "mass_over_pt", "n_particles")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--replay-bundle", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def condition_features(values):
    return torch.stack([
        values[:, 0], values[:, 1], values[:, 2],
        values[:, 2] / values[:, 1].clamp(min=1e-8), values[:, 3],
    ], dim=-1).to(torch.float64)


def fpd(real_features, generated_features):
    return float(jetnet_eval.fpd(
        real_features=real_features, gen_features=generated_features, seed=42,
    )[0])


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with open(os.path.join(args.source_run, "data", "x_test.pkl"), "rb") as handle:
        x_test = pickle.load(handle)
    samples = torch.load(os.path.join(args.eval_dir, "samples.pt"), map_location="cpu")
    bundle = torch.load(args.replay_bundle, map_location="cpu")
    attrs = bundle["generated_jet_attrs"]
    finite = torch.isfinite(samples).all(dim=(1, 2))
    samples, attrs = samples[finite], attrs[finite]

    test_polar, test_mask = abs_test_particles(x_test)
    test_rel = EtaPhiPtE_to_relEtaPhiPt(test_polar)[..., :3]
    test_rel = test_rel * test_mask.unsqueeze(-1)
    gen_polar = cartesian_to_EtaPhiPtE(samples)
    gen_mask = samples.abs().sum(-1) > 0
    gen_rel = EtaPhiPtE_to_relEtaPhiPt(gen_polar)[..., :3]
    gen_rel = gen_rel * gen_mask.unsqueeze(-1)
    n = min(len(test_rel), len(gen_rel))
    test_rel, gen_rel = test_rel[:n].float(), gen_rel[:n].float()

    test_conditions = condition_features(x_test[:][1].cpu())[:n]
    gen_conditions = condition_features(torch.stack([
        attrs[:, 5], attrs[:, 6], attrs[:, 7], attrs[:, -1],
    ], dim=-1))[:n]
    test_efp = jetnet_eval.get_fpd_kpd_jet_features(test_rel)
    gen_efp = jetnet_eval.get_fpd_kpd_jet_features(gen_rel)
    test_act = particlenet_activations(test_rel, device="cuda")
    gen_act = particlenet_activations(gen_rel, device="cuda")

    permutation = rng.permutation(n)
    half = n // 2
    split_a, split_b = permutation[:half], permutation[half:2 * half]
    report = {
        "n_finite_generated": int(n),
        "input_contract": {
            "feature_order": ["eta_rel", "phi_rel", "pt_rel"],
            "dtype": "float32",
            "padding": "exact_zero",
            "particle_count": 30,
            "jet_type": "g",
        },
        "global": {
            "fpd_efp": fpd(test_efp, gen_efp),
            "fpnd_jetnet": float(jetnet_eval.fpnd(
                jets=gen_rel, jet_type="g", device="cuda", use_tqdm=False
            )),
            "cfpnd_test_vs_generated": conditional_fpnd(test_act, gen_act),
            "real_split_fpd": fpd(test_efp[split_a], test_efp[split_b]),
            "real_split_cfpnd": conditional_fpnd(test_act[split_a], test_act[split_b]),
        },
        "sample_size_fpd": {},
        "strata": {},
    }
    for size in (10_000, 25_000, 50_000):
        if size <= n:
            report["sample_size_fpd"][str(size)] = {
                "test_generated": fpd(test_efp[:size], gen_efp[:size]),
                "real_split": fpd(test_efp[permutation[:size // 2]],
                                  test_efp[permutation[size // 2:size]]),
            }

    for feature_index, feature_name in enumerate(FEATURES):
        edges = torch.quantile(
            test_conditions[:, feature_index],
            torch.linspace(0, 1, 5, dtype=torch.float64),
        )
        bins = []
        for bin_index in range(4):
            lower, upper = edges[bin_index], edges[bin_index + 1]
            test_bin = ((test_conditions[:, feature_index] >= lower)
                        & (test_conditions[:, feature_index]
                           < upper if bin_index < 3 else
                           test_conditions[:, feature_index] <= upper))
            gen_bin = ((gen_conditions[:, feature_index] >= lower)
                       & (gen_conditions[:, feature_index]
                          < upper if bin_index < 3 else
                          gen_conditions[:, feature_index] <= upper))
            test_idx = test_bin.nonzero(as_tuple=False).flatten().numpy()
            gen_idx = gen_bin.nonzero(as_tuple=False).flatten().numpy()
            matched = min(len(test_idx), len(gen_idx))
            record = {
                "lower": float(lower), "upper": float(upper),
                "n_test": len(test_idx), "n_generated": len(gen_idx),
                "n_matched": matched,
            }
            if matched >= 500:
                test_idx = test_idx[:matched]
                gen_idx = gen_idx[:matched]
                record["fpd_efp"] = fpd(test_efp[test_idx], gen_efp[gen_idx])
                record["cfpnd"] = conditional_fpnd(test_act[test_idx], gen_act[gen_idx])
            bins.append(record)
        report["strata"][feature_name] = bins

    with open(os.path.join(args.out_dir, "perceptual_audit.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["global"], indent=2))


if __name__ == "__main__":
    main()
