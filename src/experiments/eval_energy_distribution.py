#!/usr/bin/env python3
"""
Experiment: Generated Sample Energy Distribution

Compares energy and momentum distributions between real JetNet test data and
generated samples, to detect whether the λ² loss omission suppresses high-energy
tails in generation.

Expects generated samples saved by infer.py as samples.pt:
  shape (N, max_particles, 4), Cartesian (E, px, py, pz), physical units.
  Padding slots are exactly zero (infer.py multiplies by mask before saving).

Usage:
    python experiments/eval_energy_distribution.py \\
        --gen_samples_path /mnt/data/output/INFER_RUN/euclidean/samples.pt \\
        --output_path      /mnt/data/output/TRAIN_RUN \\
        --label            euclidean \\
        --num_particles    30
"""

import os
import sys
import pickle
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from data import get_data_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare energy/momentum distributions: real vs generated"
    )
    parser.add_argument("--gen_samples_path", type=str, required=True,
                        help="Path to samples.pt written by infer.py")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Training output dir (contains data/x_test.pkl)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Defaults to <gen_samples_dir>/energy_distribution/")
    parser.add_argument("--label", type=str, default="generated",
                        help="Label for generated samples in plots (e.g. 'euclidean', 'hyperbolic')")
    parser.add_argument("--num_particles", type=int, default=30)
    return parser.parse_args()


def _extract(samples, mask):
    """Extract per-particle quantities from (N, P, 4) tensor filtered by mask."""
    flat = mask.flatten().bool()
    E  = samples[:, :, 0].flatten()[flat]
    px = samples[:, :, 1].flatten()[flat]
    py = samples[:, :, 2].flatten()[flat]
    pz = samples[:, :, 3].flatten()[flat]
    return {
        'E':      E.cpu().numpy(),
        'p_norm': torch.sqrt(px**2 + py**2 + pz**2).cpu().numpy(),
        'pt':     torch.sqrt(px**2 + py**2).cpu().numpy(),
    }


def compare_and_plot(real, gen, out_dir, label):
    sns.set_style("whitegrid")
    plt.rc("mathtext", fontset="cm")
    colors = sns.color_palette("deep")

    key_labels = {
        'E':      r"Energy $E/c$",
        'p_norm': r"$|\mathbf{p}|$",
        'pt':     r"$p_T$",
    }
    percentiles = [50, 75, 90, 95, 99]

    stat_lines = [f"=== Energy Distribution Comparison: real vs {label} ==="]
    stat_lines.append(f"Real particles: {len(real['E'])}  |  Generated: {len(gen['E'])}\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col, key in enumerate(['E', 'p_norm', 'pt']):
        r_arr, g_arr = real[key], gen[key]
        p99 = np.percentile(r_arr, 99)

        # Percentile comparison
        r_pct = np.percentile(r_arr, percentiles)
        g_pct = np.percentile(g_arr, percentiles)
        ratio = g_pct / np.where(r_pct != 0, r_pct, 1.0)
        ks = stats.ks_2samp(r_arr, g_arr)

        stat_lines.append(f"--- {key} ---")
        stat_lines.append(f"KS stat: {ks.statistic:.4f}  (p = {ks.pvalue:.2e})")
        stat_lines.append(f"{'Pct':>5}  {'Real':>10}  {'Gen':>10}  {'Ratio G/R':>10}")
        for p, rp, gp, rat in zip(percentiles, r_pct, g_pct, ratio):
            stat_lines.append(f"{p:5d}  {rp:10.4f}  {gp:10.4f}  {rat:10.4f}")
        stat_lines.append("")

        # Row 0: histogram (log-y scale)
        ax = axes[0, col]
        bins = np.linspace(0, p99, 60)
        ax.hist(r_arr, bins=bins, alpha=0.6, label="Real", density=True, color=colors[2])
        ax.hist(g_arr, bins=bins, alpha=0.6, label=label.capitalize(), density=True, color=colors[3])
        ax.set_xlabel(key_labels[key])
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.set_title(f"{key_labels[key]} distribution")

        # Row 1: Q-Q plot
        ax = axes[1, col]
        pct_grid = np.linspace(1, 99, 100)
        rq = np.percentile(r_arr, pct_grid)
        gq = np.percentile(g_arr, pct_grid)
        ax.scatter(rq, gq, s=12, alpha=0.7, color=colors[0])
        lim = max(rq.max(), gq.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
        ax.set_xlabel(f"Real {key_labels[key]} quantiles")
        ax.set_ylabel(f"{label.capitalize()} quantiles")
        ax.set_title(f"{key_labels[key]} Q-Q plot")
        ax.legend(fontsize=9)

    plt.suptitle(f"Real vs {label}: energy & momentum distributions", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"energy_distributions_{label}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    # Percentile ratio summary plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for col, key in enumerate(['E', 'p_norm', 'pt']):
        r_arr, g_arr = real[key], gen[key]
        pct_grid = np.linspace(1, 99, 99)
        rq = np.percentile(r_arr, pct_grid)
        gq = np.percentile(g_arr, pct_grid)
        ratio_curve = gq / np.where(rq > 0, rq, 1.0)
        ax = axes[col]
        ax.plot(pct_grid, ratio_curve, color=colors[col], linewidth=1.5)
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Gen / Real quantile ratio")
        ax.set_title(f"{key_labels[key]}: quantile ratio")
        ax.set_ylim(0, 2)
    plt.suptitle(f"Quantile ratio: {label} / real", fontsize=12)
    plt.tight_layout()
    ratio_path = os.path.join(out_dir, f"quantile_ratio_{label}.png")
    plt.savefig(ratio_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {ratio_path}")

    # Stats file
    stats_path = os.path.join(out_dir, f"energy_stats_{label}.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stat_lines) + "\n")
    print(f"Saved {stats_path}")
    print("\n".join(stat_lines))


def main():
    args = parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.gen_samples_path), "energy_distribution"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Outputs → {out_dir}")

    # Load generated samples: (N, P, 4), physical units, zero-padded
    gen_samples = torch.load(args.gen_samples_path, map_location="cpu")
    print(f"Generated samples: {gen_samples.shape}")
    # Reconstruct mask from zero-padding (padded slots are all-zero after infer.py)
    gen_mask = (gen_samples.abs().sum(dim=-1) > 1e-6).float()
    print(f"Mean real particles/jet (generated): {gen_mask.sum(dim=1).mean():.1f}")

    # Load real test data
    data_path = get_data_path(args.output_path)
    with open(os.path.join(data_path, "x_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    print(f"Test set: {len(X_test)} jets")

    # Transform to Cartesian (physical units, pre-scale — same units as gen_samples)
    real_particles = transform_rel_particle_coordinates_to_cartesian(X_test)  # (N, P, 5)
    if args.num_particles < real_particles.shape[1]:
        real_particles = real_particles[:, :args.num_particles, :]

    # Match counts
    n = min(len(gen_samples), len(real_particles))
    real_particles = real_particles[:n]
    gen_samples    = gen_samples[:n]
    gen_mask       = gen_mask[:n]

    real_4mom = real_particles[:, :, :4]   # (N, P, 4)
    real_mask = real_particles[:, :, 4]    # (N, P)
    print(f"Mean real particles/jet (test data): {real_mask.sum(dim=1).mean():.1f}")

    real_quantities = _extract(real_4mom, real_mask)
    gen_quantities  = _extract(gen_samples, gen_mask)

    compare_and_plot(real_quantities, gen_quantities, out_dir, args.label)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
