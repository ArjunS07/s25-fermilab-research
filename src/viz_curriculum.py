"""
viz_curriculum.py — Visualise the curriculum sampling schedule over training.

Loads the real training set and shows:

  Left   Particle-count distribution of the training data, coloured by bucket.
  Centre Heatmap of P(bucket k, epoch) — how the sampling weight of each
         bucket evolves as alpha decays from alpha_start → 0.
  Right  Effective per-jet sampling probability over epochs (P(bucket) /
         count(bucket)), which reveals which individual jets are over- or
         under-sampled at each phase of training.

All defaults match train.py.

Usage:
    python viz_curriculum.py --output_path /mnt/data/output
    python viz_curriculum.py --output_path /mnt/data/output --num_epochs 100 --alpha_start 3.0
"""

import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch

from data import data_args, get_data_path

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")


def main():
    parser = argparse.ArgumentParser(description="Visualise curriculum learning schedule")
    parser.add_argument("--output_path", type=str, default="/mnt/data/output")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_particles", type=int, default=data_args["num_particles"])
    parser.add_argument("--n_curriculum_buckets", type=int, default=10)
    parser.add_argument("--curriculum_alpha_start", type=float, default=2.0)
    parser.add_argument("--n_train_samples", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="curriculum.png")
    args = parser.parse_args()

    MAX_N_PARTICLES = 150

    # ── Load training jet info ─────────────────────────────────────────────
    data_path = get_data_path(args.output_path)
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    train_jet_info = X_train[:][1]                       # (N, jet_feature_dim)
    if args.num_particles < MAX_N_PARTICLES:
        train_jet_info[:, 3] = train_jet_info[:, 3].clamp(max=args.num_particles)
    train_jet_info = train_jet_info[:args.n_train_samples]

    n_particles_per_jet = train_jet_info[:, 3].cpu().float()   # (N,)
    N_train = len(n_particles_per_jet)

    # ── Bucket assignment (mirrors train.py exactly) ───────────────────────
    N_BUCKETS = args.n_curriculum_buckets
    p_min = n_particles_per_jet.min().item()
    p_max = n_particles_per_jet.max().item()
    bucket_width = (p_max - p_min + 1e-6) / N_BUCKETS
    bucket_assignments = ((n_particles_per_jet - p_min) / bucket_width).long()
    bucket_assignments = bucket_assignments.clamp(0, N_BUCKETS - 1)
    bucket_counts = torch.bincount(bucket_assignments, minlength=N_BUCKETS).float()

    # Particle count range per bucket (for axis labels)
    bucket_ranges = []
    for k in range(N_BUCKETS):
        lo = p_min + k * bucket_width
        hi = lo + bucket_width
        bucket_ranges.append(f"{int(lo)}–{int(hi)}")

    # ── Compute P(bucket, epoch) for each epoch ────────────────────────────
    num_epochs = args.num_epochs
    alpha_start = args.curriculum_alpha_start
    alphas = [alpha_start * (1.0 - e / max(num_epochs - 1, 1)) for e in range(num_epochs)]

    bucket_probs_matrix = np.zeros((num_epochs, N_BUCKETS))   # rows=epochs, cols=buckets
    per_jet_prob_matrix = np.zeros((num_epochs, N_BUCKETS))   # normalised per-jet prob

    k_vals = torch.arange(1, N_BUCKETS + 1, dtype=torch.float)
    counts_safe = bucket_counts.clamp(min=1).numpy()

    for e, alpha in enumerate(alphas):
        raw = torch.pow(k_vals, alpha)
        probs = (raw / raw.sum()).numpy()
        bucket_probs_matrix[e] = probs
        per_jet_prob_matrix[e] = probs / counts_safe   # un-normalised; relative scale

    # Normalise per-jet matrix row-wise so each epoch sums to 1
    per_jet_prob_matrix /= per_jet_prob_matrix.sum(axis=1, keepdims=True).clip(min=1e-12)

    # ── Colourmap shared across buckets ───────────────────────────────────
    bucket_cmap = plt.cm.get_cmap("plasma", N_BUCKETS)
    bucket_colors = [bucket_cmap(k / (N_BUCKETS - 1)) for k in range(N_BUCKETS)]

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.35)
    ax_hist = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_eff  = fig.add_subplot(gs[2])

    # --- Left: particle-count histogram coloured by bucket ----------------
    bin_edges = [p_min + k * bucket_width for k in range(N_BUCKETS + 1)]
    n_particles_np = n_particles_per_jet.numpy()
    for k in range(N_BUCKETS):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        mask = (n_particles_np >= lo) & (n_particles_np < hi)
        if mask.sum() == 0:
            continue
        ax_hist.bar(
            (lo + hi) / 2, mask.sum(), width=bucket_width * 0.85,
            color=bucket_colors[k], edgecolor="white", linewidth=0.4,
            label=f"k={k}" if k in (0, N_BUCKETS - 1) else None,
        )
    ax_hist.set_xlabel("Particles per jet", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.set_title("Training data distribution", fontsize=12)
    ax_hist.set_xlim(p_min - 1, p_max + 1)

    # Annotate sparsest / densest
    ax_hist.text(0.03, 0.95, "k=0\n(sparsest)", transform=ax_hist.transAxes,
                 va="top", fontsize=9, color=bucket_colors[0])
    ax_hist.text(0.97, 0.95, f"k={N_BUCKETS-1}\n(densest)", transform=ax_hist.transAxes,
                 va="top", ha="right", fontsize=9, color=bucket_colors[-1])

    # --- Centre: heatmap of P(bucket k, epoch) ----------------------------
    im = ax_heat.imshow(
        bucket_probs_matrix.T,           # (buckets, epochs)
        aspect="auto", origin="lower",
        extent=[0, num_epochs, -0.5, N_BUCKETS - 0.5],
        cmap="plasma", interpolation="nearest",
    )
    fig.colorbar(im, ax=ax_heat, label=r"$P(\text{bucket } k)$", shrink=0.85)
    ax_heat.set_xlabel("Epoch", fontsize=12)
    ax_heat.set_ylabel("Bucket k  (0 = sparsest)", fontsize=12)
    ax_heat.set_title(
        rf"Bucket sampling weight  ($\alpha$: {alpha_start:.1f} → 0)", fontsize=12
    )
    ax_heat.set_yticks(range(N_BUCKETS))
    ax_heat.set_yticklabels([f"k={k}" for k in range(N_BUCKETS)], fontsize=8)

    # Overlay alpha value on top axis
    ax_alpha = ax_heat.twiny()
    alpha_ticks = np.linspace(alpha_start, 0, 5)
    alpha_tick_epochs = [alpha_start - a for a in alpha_ticks] / np.array(
        alpha_start) * num_epochs if alpha_start > 0 else [0] * 5
    ax_alpha.set_xlim(0, num_epochs)
    ax_alpha.set_xticks(np.linspace(0, num_epochs, 5))
    ax_alpha.set_xticklabels([f"α={a:.1f}" for a in alpha_ticks], fontsize=8)

    # --- Right: effective per-jet sampling probability --------------------
    im2 = ax_eff.imshow(
        per_jet_prob_matrix.T,
        aspect="auto", origin="lower",
        extent=[0, num_epochs, -0.5, N_BUCKETS - 0.5],
        cmap="RdYlGn", interpolation="nearest",
    )
    fig.colorbar(im2, ax=ax_eff, label="Effective per-jet prob (normalised)", shrink=0.85)
    ax_eff.set_xlabel("Epoch", fontsize=12)
    ax_eff.set_ylabel("Bucket k  (0 = sparsest)", fontsize=12)
    ax_eff.set_title("Effective per-jet sampling probability", fontsize=12)
    ax_eff.set_yticks(range(N_BUCKETS))
    ax_eff.set_yticklabels([f"k={k}  ({bucket_ranges[k]})" for k in range(N_BUCKETS)],
                            fontsize=7)

    # Annotate bucket counts
    for k in range(N_BUCKETS):
        ax_eff.text(num_epochs + 0.5, k, f"n={int(bucket_counts[k]):,}",
                    va="center", fontsize=6.5, color="gray")

    fig.suptitle(
        f"Curriculum learning — {N_train:,} jets, {N_BUCKETS} buckets, "
        f"{num_epochs} epochs",
        fontsize=13, y=1.01,
    )

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {args.output}")
    print(f"\nBucket summary:")
    for k in range(N_BUCKETS):
        print(f"  k={k}  particles={bucket_ranges[k]}  count={int(bucket_counts[k]):,}  "
              f"P(k,epoch=0)={bucket_probs_matrix[0, k]:.3f}  "
              f"P(k,last epoch)={bucket_probs_matrix[-1, k]:.3f}")


if __name__ == "__main__":
    main()
