#!/usr/bin/env python3
"""Generate compact paper ablation figures from the local results snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


CLASS_COLORS = {"g": "#0072B2", "q": "#D55E00", "t": "#009E73"}
SEMANTIC_COLORS = {
    "canonical": "#222222",
    "no_icp": "#CC79A7",
    "latent": "#8C8C8C",
    "evolving": "#6A51A3",
    "fixed": "#BDBDBD",
    "small_normal": "#E6AB02",
    "zero": "#6A51A3",
    "euclidean": "#8C8C8C",
    "mass_shell": "#6A51A3",
}


def configure_style() -> None:
    sns.set_theme(context="paper", style="ticks", font="serif")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.grid": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def quiet_axes(ax: plt.Axes) -> None:
    sns.despine(ax=ax)
    ax.grid(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=240)
    plt.close(fig)


def smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            try:
                rows.append((float(row[0]), float(row[1])))
            except (ValueError, IndexError):
                continue
    data = np.asarray(rows)
    y = smooth(data[:, 1])
    x = data[24:, 0] if len(data) >= 25 else data[:, 0]
    return x, y


def plot_training_curves(data_dir: Path, out_dir: Path) -> None:
    files = {
        "canonical": ("ICP, physical", data_dir / "training_loss_icp.csv"),
        "no_icp": ("no ICP, physical", data_dir / "training_loss_noicp.csv"),
        "latent": ("no ICP, latent", data_dir / "training_loss_noicp_latent.csv"),
    }
    fig, ax = plt.subplots(figsize=(5.7, 3.5), constrained_layout=True)
    for key, (label, path) in files.items():
        x, y = load_curve(path)
        sns.lineplot(x=x, y=y, ax=ax, lw=1.5, color=SEMANTIC_COLORS[key], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Flow-matching loss")
    quiet_axes(ax)
    save(fig, out_dir, "training_curves_icp")


def plot_icp(rows: list[dict], out_dir: Path) -> None:
    metrics = [("fpnd", r"FPND$_g$"), ("fpd_x1e4", r"FPD $\times 10^4$"),
               ("w1m_x1e3", r"W1M $\times 10^3$")]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.35), sharey=True, constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for ax, (metric, ylabel) in zip(axes, metrics):
        values = [row[metric] for row in rows]
        for ypos, row, value in zip(y, rows, values):
            ax.scatter(value, ypos, s=38, color=SEMANTIC_COLORS[row["key"]], zorder=3)
            ax.text(value, ypos + 0.17, f"{value:.3g}", ha="center", va="bottom", fontsize=7)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("")
        quiet_axes(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [row["label"] for row in rows])
    save(fig, out_dir, "icp_ablation")


def plot_euler(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    for cls in ("g", "q", "t"):
        sns.lineplot(x=[row["steps"] for row in rows], y=[row[cls] for row in rows],
                     marker="o", ms=5, lw=1.7, color=CLASS_COLORS[cls], label=cls, ax=ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128], ["32", "64", "128"])
    ax.set_xlabel("Geodesic Euler steps")
    ax.set_ylabel("FPND")
    quiet_axes(ax)
    save(fig, out_dir, "euler_step_scaling")


def plot_cfg(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    for cls in ("g", "q", "t"):
        sns.lineplot(x=[row["weight"] for row in rows], y=[row[cls] for row in rows],
                     marker="o", ms=4.5, lw=1.6, color=CLASS_COLORS[cls], label=cls, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("CFG guidance weight")
    ax.set_ylabel("FPND")
    quiet_axes(ax)
    save(fig, out_dir, "cfg_guidance_sweep")


def plot_architecture(rows: list[dict], out_dir: Path) -> None:
    labels = [f'{row["run"]}\n{row["reference"]}' for row in rows]
    colors = [SEMANTIC_COLORS[row["geometry"]] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), constrained_layout=True)
    for ax, key, ylabel in [(axes[0], "fpnd", r"FPND$_g$"),
                            (axes[1], "w1m_x1e3", r"W1M $\times 10^3$")]:
        values = [row[key] for row in rows]
        ax.plot([0, 1], values[:2], color=SEMANTIC_COLORS["evolving"], lw=1.2)
        ax.plot([2, 3], values[2:], color=SEMANTIC_COLORS["fixed"], lw=1.2)
        for index, (row, value, color) in enumerate(zip(rows, values, colors)):
            marker = "o" if row["reference"] == "raw" else "s"
            ax.scatter(index, value, s=42, marker=marker, color=color, edgecolor="#333333",
                       linewidth=0.45, zorder=3)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(4), labels)
        quiet_axes(ax)
    save(fig, out_dir, "architecture_g_to_j")


def plot_geometry(rows: list[dict], out_dir: Path) -> None:
    labels = [f'{row["arm"]}\n{row["references"]}' for row in rows]
    colors = [SEMANTIC_COLORS["mass_shell" if row["geometry"] == "mass shell" else "euclidean"]
              for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), constrained_layout=True)
    for ax, key, ylabel in [(axes[0], "w1m", "W1M"),
                            (axes[1], "spacelike_pct", r"Spacelike (\%)"),
                            (axes[2], "invalid_pct", r"Invalid (\%)")]:
        values = [row[key] for row in rows]
        ax.plot([0, 1], values[:2], color="#C7C7C7", lw=1.0, zorder=1)
        ax.plot([2, 3], values[2:], color="#C7C7C7", lw=1.0, zorder=1)
        ax.scatter(range(4), values, c=colors, s=40, zorder=3)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(4), labels)
        quiet_axes(ax)
    save(fig, out_dir, "geometry_ablation")


def plot_initialization(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    for key, label in [("small_normal", "small-normal"), ("zero", "zero")]:
        sns.lineplot(x=[row["step"] for row in rows], y=[row[key] for row in rows],
                     marker="o", ms=4.5, lw=1.6, color=SEMANTIC_COLORS[key], label=label, ax=ax)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Unstable trajectories (of 64)")
    quiet_axes(ax)
    save(fig, out_dir, "readout_initialization_stability")


def plot_distributions(data_dir: Path, out_dir: Path) -> None:
    histogram_path = data_dir / "distribution_histograms.json"
    if not histogram_path.exists():
        return
    payload = json.loads(histogram_path.read_text())
    features = [
        ("eta_rel", r"Particle $\eta^{\mathrm{rel}}$"),
        ("phi_rel", r"Particle $\phi^{\mathrm{rel}}$"),
        ("pt_rel", r"Particle $p_T^{\mathrm{rel}}$"),
        ("relative_mass", r"Relative jet mass"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 6.6), constrained_layout=True)
    for row, cls in enumerate(("g", "q", "t")):
        class_data = payload["classes"][cls]
        for col, (feature, xlabel) in enumerate(features):
            ax = axes[row, col]
            hist = class_data["histograms"][feature]
            edges = np.asarray(hist["edges"])
            centers = 0.5 * (edges[:-1] + edges[1:])
            sns.lineplot(x=centers, y=hist["test_density"], color="#222222", lw=1.35,
                         drawstyle="steps-mid", ax=ax)
            sns.lineplot(x=centers, y=hist["generated_density"], color=CLASS_COLORS[cls],
                         lw=1.45, drawstyle="steps-mid", linestyle="--", ax=ax)
            ax.set_xlabel(xlabel if row == 2 else "")
            ax.set_ylabel(f"{cls} jets\nDensity" if col == 0 else "")
            quiet_axes(ax)
    handles = [
        Line2D([0], [0], color="#222222", lw=1.35, label="JetNet"),
        Line2D([0], [0], color="#555555", lw=1.45, ls="--", label="JetFUEL"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.015))
    save(fig, out_dir, "selected_cfg_distributions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    snapshot = json.loads((args.data_dir / "results_snapshot.json").read_text())
    plot_training_curves(args.data_dir, args.output_dir)
    plot_icp(snapshot["icp_ablation"], args.output_dir)
    plot_euler(snapshot["euler_scaling"], args.output_dir)
    plot_cfg(snapshot["cfg_sweep"], args.output_dir)
    plot_architecture(snapshot["architecture_gj"], args.output_dir)
    plot_geometry(snapshot["geometry_ablation"], args.output_dir)
    plot_initialization(snapshot["readout_initialization"], args.output_dir)
    plot_distributions(args.data_dir, args.output_dir)
    print(f"Local paper plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
