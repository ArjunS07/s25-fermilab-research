#!/usr/bin/env python3
"""Large-sample Stage-1 marginal and joint-tail audit."""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch

from util import jet_attributes


FEATURES = ("eta", "pt", "mass", "mass_over_pt", "n_particles")
QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def feature_tensor(jets):
    return torch.stack([
        jets[:, 0], jets[:, 1], jets[:, 2],
        jets[:, 2] / jets[:, 1].clamp(min=1e-8), jets[:, 3],
    ], dim=-1).to(torch.float64)


def summaries(values):
    q = torch.tensor(QUANTILES, dtype=torch.float64)
    return {
        name: {
            "min": float(values[:, index].min()),
            "max": float(values[:, index].max()),
            "quantiles": {
                str(level): float(value)
                for level, value in zip(q.tolist(), torch.quantile(values[:, index], q))
            },
        }
        for index, name in enumerate(FEATURES)
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.source_run, "data", "x_train.pkl"), "rb") as handle:
        train_dataset = pickle.load(handle)
    train_jets = train_dataset[:][1].cpu()
    train = feature_tensor(train_jets)
    model = jet_attributes.load_model(
        model_path=os.path.join(args.source_run, "jet_attr_model.pth")
    ).to(device).eval()
    generated_parts = []
    with torch.no_grad():
        for start in range(0, args.n_samples, args.batch_size):
            size = min(args.batch_size, args.n_samples - start)
            attrs, _ = jet_attributes.generate_jets(
                model, device, n_jet_types=1, num_jets=size
            )
            generated_parts.append(torch.stack([
                attrs[:, 5], attrs[:, 6], attrs[:, 7],
                attrs[:, 7] / attrs[:, 6].clamp(min=1e-8), attrs[:, -1],
            ], dim=-1).cpu())
    generated = torch.cat(generated_parts).to(torch.float64)
    lower, upper = train.min(0).values, train.max(0).values
    outside_by_feature = (generated < lower) | (generated > upper)
    q = torch.tensor(QUANTILES, dtype=torch.float64)
    train_q = torch.quantile(train, q, dim=0)
    gen_q = torch.quantile(generated, q, dim=0)
    report = {
        "n_train": len(train),
        "n_generated": len(generated),
        "feature_order": FEATURES,
        "train": summaries(train),
        "generated": summaries(generated),
        "outside_train_support_fraction": float(outside_by_feature.any(1).double().mean()),
        "outside_train_support_by_feature": {
            name: float(outside_by_feature[:, index].double().mean())
            for index, name in enumerate(FEATURES)
        },
        "quantile_ratios_generated_over_train": {
            str(level): {
                name: float(gen_q[q_index, feature_index]
                            / train_q[q_index, feature_index].clamp(min=1e-12))
                for feature_index, name in enumerate(FEATURES)
            }
            for q_index, level in enumerate(QUANTILES)
        },
        "train_correlation": np.corrcoef(train.numpy(), rowvar=False).tolist(),
        "generated_correlation": np.corrcoef(generated.numpy(), rowvar=False).tolist(),
    }
    with open(os.path.join(args.out_dir, "stage1_audit.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    torch.save(generated.float(), os.path.join(args.out_dir, "stage1_generated_features.pt"))
    print(json.dumps({
        "outside_train_support_fraction": report["outside_train_support_fraction"],
        "outside_train_support_by_feature": report["outside_train_support_by_feature"],
    }, indent=2))


if __name__ == "__main__":
    main()
