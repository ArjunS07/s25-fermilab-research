#!/usr/bin/env python3
"""Build paper figures from completed, immutable LEFTJeN experiment outputs.

This script intentionally does not sample models or recompute physics metrics.  It
reads the validated summaries/curves already stored on the results PVC, copies the
most useful diagnostic plots, and creates compact cross-run comparison figures.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PVC = Path("/mnt/data/output")
OUT = Path("/mnt/data/paper_figures/leftjen_2026-09-11")

RUNS = {
    "gqt_train": PVC / "2026-08-21_10-04-25--cf87d826-9bab-45fb-b52e-02c9a60af5b8-gqt30-lnet-h-cfg-994k/train",
    "gqt_w000": PVC / "2026-08-26_05-10-21--ef9da51c-6ce0-4bb2-9d68-d88e8f0af41a-paper30-gqt994k-cfg-w000-repaired/eval",
    "gqt_w025": PVC / "2026-08-25_22-18-31--a31c196a-5009-42a1-82e9-a9696e6146ee-paper30-gqt994k-cfg-w025-eval/eval",
    "gqt_w050": PVC / "2026-08-25_20-59-00--3a48a5f7-5493-4335-be28-c301daee932c-paper30-gqt994k-cfg-w050-eval/eval",
    "h_icp": PVC / "2026-08-18_20-22-22--2c831a59-7454-4778-905c-697235fd4aa2-g30-lnet-h-994k-r2/train",
    "h_noicp": PVC / "2026-09-03_17-17-34--410ae05d-160e-49fc-b8a7-6b8711039446-g30-lnet-h-noicp-994k/train",
    "h_noicp_latent": PVC / "2026-09-04_23-03-18--431bf3f5-8c53-4ed3-ab4f-3f4337f7a1b6-g30-lnet-h-noicp-latent-994k/train",
}

COLORS = {"g": "#386cb0", "q": "#ef7f1a", "t": "#2a9d3f"}


def require_inputs() -> None:
    missing = [str(path) for path in RUNS.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing required completed run(s):\n" + "\n".join(missing))


def read_summary(key: str) -> dict:
    with (RUNS[key] / "summary.json").open() as handle:
        return json.load(handle)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    labels = {
        "h_icp": "ICP + physical directions",
        "h_noicp": "no ICP + physical directions",
        "h_noicp_latent": "no ICP + latent directions",
    }
    colors = {"h_icp": "#225ea8", "h_noicp": "#ef6548", "h_noicp_latent": "#31a354"}
    for key in labels:
        rows = []
        with (RUNS[key] / "training_loss.csv").open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) >= 2:
                    try:
                        rows.append((float(row[0]), float(row[1])))
                    except ValueError:
                        continue
        data = np.asarray(rows)
        y = smooth(data[:, 1], 25)
        x = data[24:, 0] if len(data) >= 25 else data[:, 0]
        ax.plot(x, y, lw=1.8, color=colors[key], label=labels[key])
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Flow-matching loss (25-epoch mean)")
    ax.set_title("Matched gluon-30 training curves")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT / "training_curves_icp.png", dpi=220)
    fig.savefig(OUT / "training_curves_icp.pdf")
    plt.close(fig)


def plot_icp_ablation() -> None:
    rows = []
    for key, label in [
        ("h_icp", "ICP\nphysical"),
        ("h_noicp", "no ICP\nphysical"),
        ("h_noicp_latent", "no ICP\nlatent"),
    ]:
        metrics = read_summary(key)["metrics"]
        rows.append((label, metrics["fpnd_g"], metrics["fpd"][0] * 1e4, metrics["w1m"][0] * 1e3))
    values = np.asarray([row[1:] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.3), constrained_layout=True)
    titles = [r"FPND$_g$", r"FPD $\times 10^4$", r"W1M $\times 10^3$"]
    colors = ["#225ea8", "#ef6548", "#31a354"]
    for index, ax in enumerate(axes):
        bars = ax.bar(range(3), values[:, index], color=colors, width=0.7)
        ax.set_xticks(range(3), [row[0] for row in rows], fontsize=8)
        ax.set_title(titles[index])
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.3g}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Online geodesic ICP is the dominant matched-training ablation", fontsize=12)
    fig.savefig(OUT / "icp_ablation.png", dpi=220)
    fig.savefig(OUT / "icp_ablation.pdf")
    plt.close(fig)


def plot_euler_scaling() -> None:
    steps = np.asarray([32, 64, 128])
    fpnd = {
        "g": [0.4059334, 0.1419973, 0.0797202],
        "q": [0.6131796, 0.2679004, 0.1546227],
        "t": [0.9646051, 0.5243825, 0.3692616],
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for cls, values in fpnd.items():
        ax.plot(steps, values, marker="o", ms=6, lw=2, color=COLORS[cls], label=cls)
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps, [str(step) for step in steps])
    ax.set_xlabel("Geodesic Euler steps")
    ax.set_ylabel("FPND (lower is better)")
    ax.set_title(r"Inference-time integration scaling ($w=0$)")
    ax.grid(alpha=0.25)
    ax.legend(title="Jet class", frameon=False)
    fig.savefig(OUT / "euler_step_scaling.png", dpi=220)
    fig.savefig(OUT / "euler_step_scaling.pdf")
    plt.close(fig)


def plot_cfg_sweep() -> None:
    weights = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0])
    fpnd = {
        "g": [0.141994, 0.318242, 0.525544, 0.706259, 0.916041, 4.538778, 25.0270],
        "q": [0.267901, 0.213679, 0.313784, 0.579637, 1.633959, 5.721714, 15.1774],
        "t": [0.524383, 0.280137, 0.201928, 0.258170, 0.426059, 2.100775, 8.48116],
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for cls, values in fpnd.items():
        ax.plot(weights, values, marker="o", lw=2, color=COLORS[cls], label=cls)
    ax.set_yscale("log")
    ax.set_xlabel("CFG guidance weight")
    ax.set_ylabel("FPND (log scale)")
    ax.set_title("Class-conditional guidance sweep at 64 steps")
    ax.grid(alpha=0.25, which="both")
    ax.legend(title="Jet class", frameon=False)
    fig.savefig(OUT / "cfg_guidance_sweep.png", dpi=220)
    fig.savefig(OUT / "cfg_guidance_sweep.pdf")
    plt.close(fig)


def plot_architecture_ablation() -> None:
    names = ["G\nraw/evolving", "H\nnorm./evolving", "I\nraw/fixed", "J\nnorm./fixed"]
    fpnd = [0.3935, 0.3745, 0.4264, 0.4563]
    w1m = np.asarray([0.001884, 0.001808, 0.001981, 0.002191]) * 1e3
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), constrained_layout=True)
    colors = ["#74a9cf", "#0570b0", "#fdae6b", "#e6550d"]
    for ax, values, title in zip(axes, [fpnd, w1m], [r"FPND$_g$", r"W1M $\times 10^3$"]):
        bars = ax.bar(range(4), values, color=colors)
        ax.set_xticks(range(4), names, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.3g}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Reference normalization and evolving auxiliary geometry (200k)", fontsize=11)
    fig.savefig(OUT / "architecture_g_to_j.png", dpi=220)
    fig.savefig(OUT / "architecture_g_to_j.pdf")
    plt.close(fig)


def crop_class_panel(path: Path, cls_index: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    # Keep only the top-row class panel.  The source canvas also has a pooled
    # panel in its second row, so the lower crop boundary must stay above 0.51.
    left = int(width * (0.040 + cls_index * 0.328))
    right = int(width * (0.342 + cls_index * 0.328))
    top = int(height * 0.035)
    bottom = int(height * 0.505)
    return image.crop((left, top, min(right, width), bottom))


def plot_selected_distributions() -> None:
    selected = [("g", "gqt_w000", 0.0), ("q", "gqt_w025", 0.25), ("t", "gqt_w050", 0.5)]
    panels = []
    for cls_index, (cls, key, weight) in enumerate(selected):
        mass = crop_class_panel(RUNS[key] / "jet_mass.png", cls_index)
        fractions = crop_class_panel(RUNS[key] / "jet_z1_z2.png", cls_index)
        panels.append((cls, weight, mass, fractions))

    left_margin = 150
    cell_w, cell_h = 1000, 500
    header = 90
    canvas = Image.new("RGB", (left_margin + cell_w * 2, header + cell_h * 3), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)
    draw.text((left_margin + cell_w // 2, 22), "Invariant jet mass", fill="black", anchor="ma", font=font)
    draw.text((left_margin + cell_w + cell_w // 2, 22), "Leading and subleading transverse-momentum fractions",
              fill="black", anchor="ma", font=font)
    for row, (cls, weight, mass, fractions) in enumerate(panels):
        y = header + row * cell_h
        for col, panel in enumerate((mass, fractions)):
            panel.thumbnail((cell_w - 20, cell_h - 20), Image.Resampling.LANCZOS)
            x = left_margin + col * cell_w + (cell_w - panel.width) // 2
            canvas.paste(panel, (x, y + (cell_h - panel.height) // 2))
        draw.text((left_margin // 2, y + cell_h // 2), f"{cls} jets\nw={weight:g}",
                  fill="black", anchor="mm", align="center", font=font)
    canvas.save(OUT / "selected_cfg_distributions.png", dpi=(220, 220))


def copy_diagnostics() -> None:
    diagnostics = [
        "particle_kinematics_rel.png", "jet_mass.png", "jet_z1_z2.png",
        "jet_image_by_class.png", "radial_profile.png", "jet_girth.png",
        "jet_pt_total.png", "physicality.png", "isotropy.png", "energy_tails.png",
    ]
    for key in ("gqt_w000", "gqt_w025", "gqt_w050"):
        target = OUT / "source_diagnostics" / key
        target.mkdir(parents=True, exist_ok=True)
        for name in diagnostics:
            source = RUNS[key] / name
            if source.exists():
                shutil.copy2(source, target / name)


def write_manifest() -> None:
    summaries = {}
    for key, path in RUNS.items():
        summary_path = path / "summary.json"
        summaries[key] = {
            "path": str(path),
            "summary": str(summary_path) if summary_path.exists() else None,
        }
    manifest = {
        "generated_from_completed_outputs_only": True,
        "output_directory": str(OUT),
        "runs": summaries,
        "figures": {
            "selected_cfg_distributions": "Class-selected mass and z1/z2 overlays; g:w=0, q:w=0.25, t:w=0.5.",
            "training_curves_icp": "Matched gluon-30 loss curves for ICP and no-ICP variants.",
            "icp_ablation": "Matched 994k gluon-30 endpoint metrics.",
            "euler_step_scaling": "GQT-994k w=0 FPND at 32, 64, and 128 steps.",
            "cfg_guidance_sweep": "GQT-994k 64-step classwise guidance sweep.",
            "architecture_g_to_j": "Validated 200k gluon-30 G--J construction ablation.",
        },
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "README.txt").write_text(
        "LEFTJeN paper-figure bundle\n\n"
        "The root figures are derived comparisons. source_diagnostics/ contains the\n"
        "original evaluation plots for the three selected CFG settings. No model was\n"
        "sampled and no metric was recomputed by this CPU-only job. See\n"
        "figure_manifest.json for immutable PVC provenance.\n"
    )


def main() -> None:
    require_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_training_curves()
    plot_icp_ablation()
    plot_euler_scaling()
    plot_cfg_sweep()
    plot_architecture_ablation()
    plot_selected_distributions()
    copy_diagnostics()
    write_manifest()
    print(f"Paper figures written to {OUT}")


if __name__ == "__main__":
    main()
