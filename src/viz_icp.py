"""
viz_icp.py — Animate ICP point-cloud alignment in 2-D, 3-D, or all-pairs panels.

Modes
-----
2d      Two chosen dimensions as a scatter plot (default dims 0,1 = E/c vs px).
3d      3-momenta (px, py, pz) as a 3-D scatter; E/c encoded as point colour.
        The view slowly rotates so the 3-D structure is visible.
panels  All 6 pairwise projections of (E/c, px, py, pz) in a 2×3 grid,
        animated simultaneously — gives a complete 4-D picture.

Usage
-----
    python viz_icp.py --output_path /mnt/data/output --mode 3d
    python viz_icp.py --output_path /mnt/data/output --mode panels
    python viz_icp.py --output_path /mnt/data/output --mode 2d --dims 1 2 --jet_idx 7
"""

import argparse
import itertools
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import torch

from data import data_args, get_data_path
from util.coordinates import transform_rel_particle_coordinates_to_cartesian
from util.distributions import gen_initial_distribution

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")
_PALETTE = sns.color_palette("deep")
_DIM_LABELS = [r"$E/c$", r"$p_x$", r"$p_y$", r"$p_z$"]


# ---------------------------------------------------------------------------
# Step-by-step ICP recorder
# ---------------------------------------------------------------------------

def _collect_icp_frames(x_0_init: np.ndarray, x_1: np.ndarray, max_iter: int = 40):
    """
    Run ICP step-by-step and return a list of frame dicts:
        x0       : (n, 4) float64  current prior-cloud positions
        col_ind  : (n,)   int      current Hungarian matching (x0[i] → x1[col_ind[i]])
        iteration: int
        dist     : float           Σ‖x0 – x1‖ after this step
    """
    x_0 = x_0_init.copy()

    def _match(a, b):
        cost = cdist(a, b, metric="euclidean")
        cost = cost * (1_000.0 / max(cost.max(), 1e-8))
        _, col = linear_sum_assignment(cost)
        return col

    dist = float(np.linalg.norm(x_0 - x_1, axis=1).sum())
    col_ind = _match(x_0, x_1)

    best_dist = dist
    best_x0 = x_0.copy()
    best_iter = 0

    frames = [dict(x0=x_0.copy(), col_ind=col_ind.copy(), iteration=0,
                   dist=dist, is_best=True, best_dist=dist)]

    dist_delta = np.inf
    for i in range(1, max_iter + 1):
        if dist_delta <= 1e-8:
            break
        x_0 = x_0[col_ind]
        try:
            rot, _, _ = R.align_vectors(x_1[:, 1:4], x_0[:, 1:4], return_sensitivity=True)
            x_0 = x_0.copy()
            x_0[:, 1:4] = x_0[:, 1:4] @ rot.as_matrix().T
        except Exception:
            pass
        dist_new = float(np.linalg.norm(x_0 - x_1, axis=1).sum())
        dist_delta = abs(dist_new - dist)
        dist = dist_new

        if dist_new < best_dist:
            best_dist = dist_new
            best_x0 = x_0.copy()
            best_iter = i

        col_ind = _match(x_0, x_1)
        frames.append(dict(x0=x_0.copy(), col_ind=col_ind.copy(), iteration=i,
                           dist=dist, is_best=(dist_new == best_dist),
                           best_dist=best_dist))

    # Append one final frame showing the restored-to-best state
    best_col = _match(best_x0, x_1)
    frames.append(dict(x0=best_x0, col_ind=best_col, iteration=best_iter,
                       dist=best_dist, is_best=True, best_dist=best_dist,
                       restored=True))

    return frames


# ---------------------------------------------------------------------------
# Shared title helper
# ---------------------------------------------------------------------------

def _title(f):
    tag = ""
    if f.get("restored"):
        tag = "  ★ RESTORED TO BEST"
    elif f.get("is_best"):
        tag = "  ★ best so far"
    return (rf"ICP — iteration {f['iteration']}   "
            rf"$\Sigma\|x_0-x_1\|={f['dist']:.4f}$" + tag)


def _highlight_best(ax_or_axes, f):
    """Draw a gold spine on every axis when this is a best / restored frame."""
    if not f.get("is_best") and not f.get("restored"):
        return
    color = "gold" if f.get("restored") else "#a8d08d"
    lw = 3.0 if f.get("restored") else 1.8
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(lw)


def _build_dist_norm(frames, x_1):
    """Global distance normalisation across all frames (0 = perfect match, 1 = worst)."""
    max_dist = max(
        np.linalg.norm(f["x0"] - x_1[f["col_ind"]], axis=1).max()
        for f in frames
    )
    line_cmap = cm.RdYlGn_r           # green = close, red = far
    line_norm = plt.Normalize(vmin=0, vmax=max(max_dist, 1e-8))
    return line_cmap, line_norm


def _pair_colors(x0, col, x_1, line_cmap, line_norm):
    """Return (n,4) RGBA array: one colour per correspondence line."""
    dists = np.linalg.norm(x0 - x_1[col], axis=1)
    return line_cmap(line_norm(dists))


# ---------------------------------------------------------------------------
# Mode: 2-D
# ---------------------------------------------------------------------------

def make_icp_animation_2d(frames, x_1, output_path, dim0=0, dim1=1, fps=4):
    line_cmap, line_norm = _build_dist_norm(frames, x_1)

    fig, ax = plt.subplots(figsize=(8, 7))
    x1_a, x1_b = x_1[:, dim0], x_1[:, dim1]

    all_a = np.concatenate([f["x0"][:, dim0] for f in frames] + [x1_a])
    all_b = np.concatenate([f["x0"][:, dim1] for f in frames] + [x1_b])
    pad_a = 0.10 * (all_a.ptp() or 1.0)
    pad_b = 0.10 * (all_b.ptp() or 1.0)
    xlim = (all_a.min() - pad_a, all_a.max() + pad_a)
    ylim = (all_b.min() - pad_b, all_b.max() + pad_b)

    # Fixed colorbar for line distances
    sm = cm.ScalarMappable(cmap=line_cmap, norm=line_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Pair distance", shrink=0.75, pad=0.02)

    def _draw(fi):
        ax.clear()
        ax.set_facecolor("#f8f8f8")
        f = frames[fi]
        x0, col = f["x0"], f["col_ind"]
        colors = _pair_colors(x0, col, x_1, line_cmap, line_norm)
        for i, j in enumerate(col):
            ax.plot([x0[i, dim0], x1_a[j]], [x0[i, dim1], x1_b[j]],
                    color=colors[i], alpha=0.7, linewidth=1.0, zorder=1)
        ax.scatter(x1_a, x1_b, color=_PALETTE[0], s=60, label=r"Target $x_1$",
                   zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter(x0[:, dim0], x0[:, dim1], color=_PALETTE[3], s=60,
                   label=r"Prior $x_0$", zorder=3, edgecolors="white", linewidths=0.4)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(_DIM_LABELS[dim0], fontsize=13)
        ax.set_ylabel(_DIM_LABELS[dim1], fontsize=13)
        ax.set_title(_title(f), fontsize=12)
        ax.legend(loc="upper right", fontsize=11)
        _highlight_best(ax, f)

    anim = animation.FuncAnimation(fig, _draw, frames=len(frames), interval=1000 // fps)
    anim.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    print(f"[2D]     → {output_path}")


# ---------------------------------------------------------------------------
# Mode: 3-D  (3-momenta + colour for E/c, lines coloured by distance)
# ---------------------------------------------------------------------------

def make_icp_animation_3d(frames, x_1, output_path, fps=4):
    """
    3-D scatter of (px, py, pz).  E/c encoded as point colour; correspondence
    lines coloured by pair distance (green = close, red = far).
    """
    line_cmap, line_norm = _build_dist_norm(frames, x_1)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Point colour range: E/c over both clouds, fixed
    e_all = np.concatenate([f["x0"][:, 0] for f in frames] + [x_1[:, 0]])
    pt_cmap = cm.plasma
    pt_norm = plt.Normalize(vmin=e_all.min(), vmax=e_all.max())

    def _range(dim):
        vals = np.concatenate([f["x0"][:, dim] for f in frames] + [x_1[:, dim]])
        pad = 0.10 * (vals.ptp() or 1.0)
        return vals.min() - pad, vals.max() + pad

    xlim, ylim, zlim = _range(1), _range(2), _range(3)

    def _draw(fi):
        ax.cla()
        f = frames[fi]
        x0, col = f["x0"], f["col_ind"]
        colors = _pair_colors(x0, col, x_1, line_cmap, line_norm)

        for i, j in enumerate(col):
            ax.plot(
                [x0[i, 1], x_1[j, 1]],
                [x0[i, 2], x_1[j, 2]],
                [x0[i, 3], x_1[j, 3]],
                color=colors[i], alpha=0.6, linewidth=0.9,
            )

        sc1 = ax.scatter(x_1[:, 1], x_1[:, 2], x_1[:, 3],
                         c=x_1[:, 0], cmap=pt_cmap, norm=pt_norm,
                         s=55, marker="^", edgecolors=_PALETTE[0],
                         linewidths=0.8, label=r"Target $x_1$", zorder=3)
        ax.scatter(x0[:, 1], x0[:, 2], x0[:, 3],
                   c=x0[:, 0], cmap=pt_cmap, norm=pt_norm,
                   s=55, marker="o", edgecolors=_PALETTE[3],
                   linewidths=0.8, label=r"Prior $x_0$", zorder=4)

        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel(r"$p_x$", fontsize=11, labelpad=6)
        ax.set_ylabel(r"$p_y$", fontsize=11, labelpad=6)
        ax.set_zlabel(r"$p_z$", fontsize=11, labelpad=6)
        ax.set_title(_title(f), fontsize=11, pad=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.view_init(elev=20, azim=-60)
        # 3-D axes don't have .spines; annotate instead
        if f.get("restored") or f.get("is_best"):
            color = "gold" if f.get("restored") else "#a8d08d"
            ax.set_title(_title(f), fontsize=11, pad=10, color=color, fontweight="bold")
        return sc1,

    sm_pt = cm.ScalarMappable(cmap=pt_cmap, norm=pt_norm)
    sm_pt.set_array([])
    fig.colorbar(sm_pt, ax=ax, label=r"$E/c$ (normalised)", shrink=0.55, pad=0.08)

    sm_ln = cm.ScalarMappable(cmap=line_cmap, norm=line_norm)
    sm_ln.set_array([])
    fig.colorbar(sm_ln, ax=ax, label="Pair distance", shrink=0.55, pad=0.14)

    anim = animation.FuncAnimation(fig, _draw, frames=len(frames), interval=1000 // fps)
    anim.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    print(f"[3D]     → {output_path}")


# ---------------------------------------------------------------------------
# Mode: panels — all 6 pairwise 2-D projections
# ---------------------------------------------------------------------------

def make_icp_animation_panels(frames, x_1, output_path, fps=4):
    """
    2×3 grid showing all C(4,2)=6 pairwise projections of (E/c, px, py, pz)
    animated simultaneously.  Lines coloured by pair distance (green→red).
    """
    line_cmap, line_norm = _build_dist_norm(frames, x_1)
    pairs = list(itertools.combinations(range(4), 2))   # 6 pairs

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # Pre-compute fixed axis limits for each panel
    limits = []
    for da, db in pairs:
        all_a = np.concatenate([f["x0"][:, da] for f in frames] + [x_1[:, da]])
        all_b = np.concatenate([f["x0"][:, db] for f in frames] + [x_1[:, db]])
        pa = 0.10 * (all_a.ptp() or 1.0)
        pb = 0.10 * (all_b.ptp() or 1.0)
        limits.append(((all_a.min()-pa, all_a.max()+pa),
                        (all_b.min()-pb, all_b.max()+pb)))

    # Shared colorbar for line distances
    sm = cm.ScalarMappable(cmap=line_cmap, norm=line_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), label="Pair distance",
                 shrink=0.6, pad=0.02, aspect=30)

    def _draw(fi):
        f = frames[fi]
        x0, col = f["x0"], f["col_ind"]
        colors = _pair_colors(x0, col, x_1, line_cmap, line_norm)
        for ax, (da, db), (xlim, ylim) in zip(axes, pairs, limits):
            ax.clear()
            ax.set_facecolor("#f8f8f8")
            for i, j in enumerate(col):
                ax.plot([x0[i, da], x_1[j, da]], [x0[i, db], x_1[j, db]],
                        color=colors[i], alpha=0.7, linewidth=0.9, zorder=1)
            ax.scatter(x_1[:, da], x_1[:, db], color=_PALETTE[0], s=35,
                       label=r"$x_1$", zorder=3, edgecolors="white", linewidths=0.3)
            ax.scatter(x0[:, da], x0[:, db], color=_PALETTE[3], s=35,
                       label=r"$x_0$", zorder=3, edgecolors="white", linewidths=0.3)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(_DIM_LABELS[da], fontsize=10)
            ax.set_ylabel(_DIM_LABELS[db], fontsize=10)
        axes[0].legend(loc="upper right", fontsize=9, markerscale=0.8)
        color = "gold" if f.get("restored") else ("#a8d08d" if f.get("is_best") else "black")
        fig.suptitle(_title(f), fontsize=12, color=color,
                     fontweight="bold" if f.get("is_best") or f.get("restored") else "normal")
        _highlight_best(axes, f)
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, _draw, frames=len(frames), interval=1000 // fps)
    anim.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    print(f"[panels] → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ICP alignment")
    parser.add_argument("--output_path", type=str, default="/mnt/data/output")
    parser.add_argument("--jet_idx", type=int, default=0)
    parser.add_argument("--num_particles", type=int, default=data_args["num_particles"])
    parser.add_argument("--max_iter", type=int, default=40)
    parser.add_argument("--mode", type=str, default="2d",
                        choices=["2d", "3d", "panels"],
                        help="Visualization mode (default: 2d)")
    parser.add_argument("--dims", type=int, nargs=2, default=[0, 1],
                        metavar=("DIM0", "DIM1"),
                        help="[2d mode only] Which two dims to plot (0=E/c,1=px,2=py,3=pz)")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    MAX_N_PARTICLES = 150

    data_path = get_data_path(args.output_path)
    with open(f"{data_path}/x_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    X_transformed = transform_rel_particle_coordinates_to_cartesian(X_train)
    if args.num_particles < MAX_N_PARTICLES:
        X_transformed = X_transformed[:, :args.num_particles, :]

    jet = X_transformed[args.jet_idx]
    mask = jet[:, 4].numpy().astype(bool)
    n_real = int(mask.sum())
    print(f"Jet {args.jet_idx}: {n_real} real particles")

    mask_flat = X_transformed[:, :, 4].flatten().numpy().astype(bool)
    final_scale = float(np.mean([
        np.std(np.array(X_transformed[:, :, c].flatten())[mask_flat])
        for c in range(4)
    ]))

    x_1_real = (jet[:n_real, :4] / final_scale).numpy().astype(np.float64)

    x_0_full = gen_initial_distribution(
        batch_size=1, num_particles=args.num_particles
    ).squeeze(0).numpy().astype(np.float64)
    x_0_real = x_0_full[:n_real]

    print(f"Running ICP for up to {args.max_iter} iterations …")
    frames = _collect_icp_frames(x_0_real, x_1_real, max_iter=args.max_iter)
    print(f"Converged in {frames[-1]['iteration']} iterations  "
          f"(Σdist {frames[0]['dist']:.4f} → {frames[-1]['dist']:.4f})")

    out_dir = os.path.join(args.output_path, "icp_viz")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"icp_jet{args.jet_idx}"

    if args.mode == "2d":
        out = os.path.join(out_dir, f"{stem}_d{args.dims[0]}d{args.dims[1]}.mp4")
        make_icp_animation_2d(frames, x_1_real, out,
                              dim0=args.dims[0], dim1=args.dims[1], fps=args.fps)
    elif args.mode == "3d":
        out = os.path.join(out_dir, f"{stem}_3d.mp4")
        make_icp_animation_3d(frames, x_1_real, out, fps=args.fps)
    elif args.mode == "panels":
        out = os.path.join(out_dir, f"{stem}_panels.mp4")
        make_icp_animation_panels(frames, x_1_real, out, fps=args.fps)
