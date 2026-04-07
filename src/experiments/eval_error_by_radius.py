#!/usr/bin/env python3
"""
Experiment: Error Stratification by Ball Radius

Investigates whether the λ² omission in hyperbolic_loss causes the model to fit
low-radius (center) points better than high-radius (boundary) points.

For each test particle at a given (x_0, x_1, t):
  - Ball radius    = ||y_t||             where y_t = phi(x_t) in the Poincaré ball
  - Euclidean err  = ||pred_ball - u_t||²   (the metric the model was trained on)
  - Riemannian err = λ_x² × Euclidean err  (what it should have been trained on)

If Riemannian error increases sharply with radius, the λ² omission is significant.

Usage:
    python experiments/eval_error_by_radius.py \\
        --checkpoint_path /mnt/data/output/TRAIN_RUN/train/models/final_model.pth \\
        --output_path     /mnt/data/output/TRAIN_RUN \\
        --num_particles   30 \\
        --n_layers        5 \\
        --use_residual
"""

import os
import sys
import pickle
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.jet_attributes import NUM_CLASSES
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.distributions import gen_initial_distribution, hyperbolic_interpolant
from util.hyperbolic import pushforward
from data import get_data_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Error stratification by Poincaré ball radius"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True,
                        help="Training output dir (contains data/ and train/scale.txt)")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_particles", type=int, default=30)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_radius_bins", type=int, default=25)
    parser.add_argument("--jet_types", type=str, nargs="+", default=["g", "q", "t"])
    return parser.parse_args()


def _load_model(checkpoint_path, n_hidden, n_layers, use_residual, num_particles, device):
    model = LEFTJeN(
        max_num_jet_types=NUM_CLASSES,
        max_particles=num_particles,
        num_layers=n_layers,
        hidden_dim=n_hidden,
        use_residual_update=use_residual,
        include_pt=True,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}) from {checkpoint_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded raw state dict from {checkpoint_path}")
    model.eval()
    return model


def evaluate_error_by_radius(model, loader, device, n_radius_bins):
    all_radius = []
    all_error_E = []
    all_lambda_sq = []

    with torch.no_grad():
        for batch_idx, (batch_jet_info, batch_particle_info) in enumerate(loader):
            batch_jet_info = batch_jet_info.to(device)
            batch_particle_info = batch_particle_info.to(device)

            x_1 = batch_particle_info[:, :, :4]
            true_masks = batch_particle_info[:, :, 4]
            mask_exp = true_masks.unsqueeze(-1).expand_as(x_1)

            # Conditioning: [one_hot_type(5), n_particles(1), pT(1)]
            # jet_info layout: [eta, pt, mass, num_particles, type]
            n_particles = batch_jet_info[:, 3]
            pt          = batch_jet_info[:, 1]
            jet_type    = batch_jet_info[:, 4].long()
            one_hot     = jet_attributes.one_hot_enc_jet_type(jet_type).to(device)
            cond = torch.cat([one_hot, n_particles.unsqueeze(-1), pt.unsqueeze(-1)], dim=-1)

            x_0 = gen_initial_distribution(x_1=x_1).to(device)
            x_0 = mask_exp * x_0
            x_1 = mask_exp * x_1

            t = torch.rand(x_1.shape[0], device=device)
            x_t, y_t, u_t_ball = hyperbolic_interpolant(x_0, x_1, t)
            u_t_ball = u_t_ball * mask_exp

            pred = model.forward(x=x_t, t=t, jet_conditions=cond, mask=true_masks)
            pred = pred * mask_exp
            pred_ball = pushforward(x_t, pred) * mask_exp

            # Per-particle Euclidean error in ball tangent space
            diff = pred_ball - u_t_ball
            error_E = (diff * diff).sum(dim=-1)   # (batch, particles)

            # Per-particle ball radius
            radius = y_t.norm(dim=-1)             # (batch, particles)

            # Conformal factor squared: λ_x² = (2 / (1 - ||y||²))²
            y_norm_sq  = (y_t * y_t).sum(dim=-1)
            lambda_sq  = (2.0 / (1.0 - y_norm_sq).clamp(min=1e-6)) ** 2

            mask_flat = true_masks.bool().flatten()
            all_radius.append(radius.flatten()[mask_flat].cpu())
            all_error_E.append(error_E.flatten()[mask_flat].cpu())
            all_lambda_sq.append(lambda_sq.flatten()[mask_flat].cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches")

    all_radius    = torch.cat(all_radius).numpy()
    all_error_E   = torch.cat(all_error_E).numpy()
    all_lambda_sq = torch.cat(all_lambda_sq).numpy()

    print(f"Total real particles: {len(all_radius)}")
    print(f"Radius range: [{all_radius.min():.4f}, {all_radius.max():.4f}]")
    print(f"Mean Euclidean error: {all_error_E.mean():.6f}")
    print(f"Mean λ²: {all_lambda_sq.mean():.4f}")

    bin_edges = np.linspace(0, all_radius.max() * 1.001, n_radius_bins + 1)
    bin_idx   = np.clip(np.digitize(all_radius, bin_edges) - 1, 0, n_radius_bins - 1)

    euclidean_error  = np.zeros(n_radius_bins)
    riemannian_error = np.zeros(n_radius_bins)
    lambda_sq_mean   = np.zeros(n_radius_bins)
    counts           = np.zeros(n_radius_bins)

    for i in range(n_radius_bins):
        sel = (bin_idx == i)
        if sel.sum() > 0:
            euclidean_error[i]  = all_error_E[sel].mean()
            riemannian_error[i] = (all_lambda_sq[sel] * all_error_E[sel]).mean()
            lambda_sq_mean[i]   = all_lambda_sq[sel].mean()
            counts[i]           = sel.sum()

    return {
        'bin_centers':      (bin_edges[:-1] + bin_edges[1:]) / 2,
        'bin_edges':        bin_edges,
        'euclidean_error':  euclidean_error,
        'riemannian_error': riemannian_error,
        'lambda_sq_mean':   lambda_sq_mean,
        'counts':           counts,
    }


def plot_results(results, out_dir):
    sns.set_style("whitegrid")
    plt.rc("mathtext", fontset="cm")
    colors = sns.color_palette("deep")

    r = results['bin_centers']
    w = (r[1] - r[0]) * 0.85 if len(r) > 1 else 0.04
    valid = results['counts'] > 0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].bar(r[valid], results['counts'][valid], width=w, alpha=0.8, color=colors[2])
    axes[0].set_xlabel(r"Ball radius $\|y_t\|$")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("Particle density per radius bin")

    axes[1].bar(r[valid], results['lambda_sq_mean'][valid], width=w, alpha=0.8, color=colors[4])
    axes[1].axhline(y=4.0, color='k', linestyle='--', linewidth=1, label=r"$\lambda^2$ at origin")
    axes[1].set_xlabel(r"Ball radius $\|y_t\|$")
    axes[1].set_ylabel(r"Mean $\lambda_x^2$")
    axes[1].set_title(r"Conformal factor $\lambda_x^2$ vs radius")
    axes[1].legend()

    axes[2].bar(r[valid], results['euclidean_error'][valid], width=w, alpha=0.8, color=colors[0])
    axes[2].set_xlabel(r"Ball radius $\|y_t\|$")
    axes[2].set_ylabel(r"Mean $\|\hat{v} - u\|^2_E$")
    axes[2].set_title(r"Euclidean error (trained metric)")

    axes[3].bar(r[valid], results['riemannian_error'][valid], width=w, alpha=0.8, color=colors[3])
    axes[3].set_xlabel(r"Ball radius $\|y_t\|$")
    axes[3].set_ylabel(r"Mean $\lambda^2 \|\hat{v} - u\|^2_E$")
    axes[3].set_title(r"Riemannian error (correct metric)")

    plt.suptitle(r"Error stratification: does $\lambda^2$ omission matter?", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_by_radius.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved error_by_radius.png")

    # Ratio plot: Riemannian / Euclidean per bin
    ratio = np.where(
        valid & (results['euclidean_error'] > 0),
        results['riemannian_error'] / results['euclidean_error'],
        0.0
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(r[valid], ratio[valid], width=w, alpha=0.8, color=colors[1])
    ax.axhline(y=4.0, color='k', linestyle='--', linewidth=1, label=r"$\lambda^2$ at origin ($= 4$)")
    ax.set_xlabel(r"Ball radius $\|y_t\|$")
    ax.set_ylabel(r"Riemannian / Euclidean error")
    ax.set_title(r"Ratio: how much does $\lambda^2$ weighting matter?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_ratio_by_radius.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved error_ratio_by_radius.png")


def save_stats(results, out_dir):
    path = os.path.join(out_dir, "error_by_radius_stats.txt")
    with open(path, "w") as f:
        f.write("=== Error Stratification by Ball Radius ===\n\n")
        f.write(
            f"{'Bin center':>12}  {'Count':>8}  "
            f"{'Euclidean err':>14}  {'Riemannian err':>15}  "
            f"{'λ² mean':>10}  {'Ratio R/E':>10}\n"
        )
        f.write("-" * 80 + "\n")
        for i, rc in enumerate(results['bin_centers']):
            if results['counts'][i] > 0:
                ratio = results['riemannian_error'][i] / max(results['euclidean_error'][i], 1e-12)
                f.write(
                    f"{rc:12.4f}  {int(results['counts'][i]):8d}  "
                    f"{results['euclidean_error'][i]:14.6f}  {results['riemannian_error'][i]:15.6f}  "
                    f"{results['lambda_sq_mean'][i]:10.4f}  {ratio:10.3f}\n"
                )
    print(f"Saved stats → {path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = args.out_dir or os.path.join(args.output_path, "experiments", "error_by_radius")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Outputs → {out_dir}")

    # Load test data
    data_path = get_data_path(args.output_path)
    with open(os.path.join(data_path, "x_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    print(f"Test set: {len(X_test)} jets")

    particle_data = transform_rel_particle_coordinates_to_cartesian(X_test)  # (N, P, 5)
    if args.num_particles < particle_data.shape[1]:
        particle_data = particle_data[:, :args.num_particles, :]

    scale_path = os.path.join(args.output_path, "train", "scale.txt")
    with open(scale_path) as f:
        final_scale = float(f.read().strip())
    print(f"Scale: {final_scale:.6f}")
    particle_data[:, :, :4] = particle_data[:, :, :4] / final_scale

    jet_info = X_test[:][1].clone()  # (N, 5): eta, pt, mass, num_particles, type
    if args.num_particles < 150:
        jet_info[:, 3] = jet_info[:, 3].clamp(max=args.num_particles)

    loader = DataLoader(
        TensorDataset(jet_info, particle_data),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )
    print(f"DataLoader: {len(loader)} batches")

    model = _load_model(
        checkpoint_path=args.checkpoint_path,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        use_residual=args.use_residual,
        num_particles=args.num_particles,
        device=device,
    )

    print("\n=== Evaluating error by radius ===")
    results = evaluate_error_by_radius(model, loader, device, args.n_radius_bins)

    plot_results(results, out_dir)
    save_stats(results, out_dir)
    torch.save(
        {k: v for k, v in results.items() if k != 'bin_edges'},
        os.path.join(out_dir, "results.pt")
    )
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
