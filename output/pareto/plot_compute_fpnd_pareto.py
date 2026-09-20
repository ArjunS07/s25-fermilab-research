#!/usr/bin/env python3
"""Build the JetNet-30 GPU-model-FLOP--FPND 3x3 Pareto figure and data table.

The accounting is conditional-generation only: jet-level conditioning variables
are supplied externally, so auxiliary jet/attribute generators are excluded.
One multiply-add is counted as two FLOPs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUT = Path(__file__).resolve().parent
DEPLOYMENT_JETS = 100_000_000
LIFECYCLE_JETS = (1_000_000, 10_000_000, 100_000_000)

# Training costs are portfolio acquisition costs for all applicable displayed
# classes: separate class-specific fits are summed, while a joint conditional
# model is counted once.  This is the primary fair accounting for a study whose
# deliverable is generation across g/q/t.
METHODS = {
    "MPGAN": {
        "train": 2_259.142030080,
        "train_class_models": {"g": 695.12062464, "q": 868.90078080, "t": 695.12062464},
        "infer": 0.192821760,
        "fpnd": {"g": 0.1200, "q": 0.3500, "t": 0.3600},
    },
    "FPCD": {
        "train": 24.097037328,
        "infer": 30.719410176,
        "fpnd": {"g": 0.0700, "q": 0.0800, "t": 0.1700},
    },
    "FPCD-1": {
        "train": 385.552597248,
        "infer": 0.059998848,
        "fpnd": {"g": 0.1100, "q": 0.0900, "t": 0.5600},
    },
    "EPiC-GAN": {
        "train": 311.121285120,
        "train_class_models": {"g": 103.707095040, "q": 103.707095040, "t": 103.707095040},
        "infer": 0.012732416,
        "fpnd": {"g": 1.0100, "q": 0.4300, "t": 0.3100},
    },
    "PET": {
        "train": 81.543163968,
        # The released 300-step sampler evaluates the particle body twice per
        # step (predictor plus second-order correction), and evaluate_models
        # invokes the conditional and zero-weight unconditional head each time.
        "infer": 111.726489600,
        "fpnd": {"g": 0.0400, "q": 0.0300, "t": 0.0700},
    },
    "OmniLearn": {
        # Full upstream pretraining plus the JetNet fine-tune.  The much smaller
        # downstream-only estimate is retained as a separate CSV column.
        "train": 109_536.215651174,
        "train_downstream_only": 65.234531174,
        "infer": 111.726489600,
        "fpnd": {"g": 0.0200, "q": 0.0200, "t": 0.0400},
    },
    "PC-JeDi DDIM": {
        "train": 119.184307200,
        "train_class_models": {"g": 59.592153600, "t": 59.592153600},
        "infer": 8.655360000,
        "fpnd": {"g": 0.1200, "t": 0.2800},
    },
    "PC-JeDi EM": {
        "train": 119.184307200,
        "train_class_models": {"g": 59.592153600, "t": 59.592153600},
        "infer": 8.655360000,
        "fpnd": {"g": 0.1000, "t": 0.1500},
    },
    "JetFUEL 64 / unguided ($w=0$)": {
        "train": 339.965605728,
        "infer": 29.185511424,
        "fpnd": {
            "g": 0.14199432330690342,
            "q": 0.26790133649231507,
            "t": 0.5243830205395739,
        },
    },
    "JetFUEL 128 / unguided ($w=0$)": {
        "train": 339.965605728,
        "infer": 58.371022848,
        "fpnd": {
            "g": 0.0797201939558363,
            "q": 0.15462267011244535,
            "t": 0.36926163373072995,
        },
    },
    "JetFUEL 64 / optimal guidance": {
        "train": 339.965605728,
        # Per-class optimum in the completed 64-step guidance sweep.
        # w=0 is an allowed deployment choice and requires only one evaluation.
        "infer": {"g": 29.185511424, "q": 58.371022848, "t": 58.371022848},
        "guidance_weight": {"g": 0.0, "q": 0.25, "t": 0.5},
        "fpnd": {
            "g": 0.14199432330690342,
            "q": 0.21367892392322574,
            "t": 0.20192831658280852,
        },
    },
    "JetFUEL 128 / optimal measured guidance": {
        "train": 339.965605728,
        # The 128-step w=0.25 arm is still pending. These are the optima among
        # the completed w=0 and w=0.5 arms; update q when w=0.25 completes.
        "infer": {"g": 58.371022848, "q": 58.371022848, "t": 116.742045696},
        "guidance_weight": {"g": 0.0, "q": 0.0, "t": 0.5},
        "fpnd": {
            "g": 0.0797201939558363,
            "q": 0.15462267011244535,
            "t": 0.17438752767836263,
        },
    },
}

# Source-level sampler accounting.  In PET/OmniLearn, the released w=0 path
# still calls the conditional head and a zero-weight unconditional head.  We
# count both calls; an aggressively optimized TensorFlow graph could prune the
# latter, in which case 101.516544 GFLOP/jet is the implementation-dependent
# lower bound rather than the portable source-level total used here.
SAMPLER_WORK = {
    "MPGAN": "1 generator evaluation",
    "FPCD": "512 DDIM steps = 512 particle evaluations",
    "FPCD-1": "1 distilled DDIM step = 1 particle evaluation",
    "EPiC-GAN": "1 generator evaluation",
    "PET": "300 steps x (2 body + 4 head) calls",
    "OmniLearn": "300 steps x (2 body + 4 head) calls",
    "PC-JeDi DDIM": "200 DDIM steps = 200 network evaluations",
    "PC-JeDi EM": "200 Euler-Maruyama steps = 200 network evaluations",
    "JetFUEL 64 / unguided ($w=0$)": "CFG-trained checkpoint, unguided inference: 64 evaluations",
    "JetFUEL 128 / unguided ($w=0$)": "CFG-trained checkpoint, unguided inference: 128 evaluations",
    "JetFUEL 64 / optimal guidance": "class-optimal w=(0,0.25,0.5): 64 or 128 evaluations",
    "JetFUEL 128 / optimal measured guidance": "best completed w=(0,0,0.5): 128 or 256 evaluations",
}

JETFUEL = {name for name in METHODS if name.startswith("JetFUEL")}
TRAIN_GROUP = {
    "PC-JeDi DDIM": "PC-JeDi",
    "PC-JeDi EM": "PC-JeDi",
    **{name: "JetFUEL" for name in JETFUEL},
}
CLASSES = ("g", "q", "t")
CLASS_TITLES = {"g": "Gluon ($g$)", "q": "Light quark ($q$)", "t": "Top ($t$)"}
ROWS = (
    ("train", "Portfolio GPU training compute [PFLOP]"),
    ("infer", "Marginal GPU inference [GFLOP / jet]"),
    ("lifecycle", "GPU lifecycle [PFLOP]"),
)

STYLE = {
    "MPGAN": ("#4C78A8", "o"),
    "FPCD": ("#59A14F", "s"),
    "FPCD-1": ("#8CD17D", "D"),
    "EPiC-GAN": ("#B6992D", "P"),
    "PET": ("#F28E2B", "v"),
    "OmniLearn": ("#E15759", "*"),
    "PC-JeDi DDIM": ("#76B7B2", "<"),
    "PC-JeDi EM": ("#2A9D8F", ">"),
    "JetFUEL 64 / unguided ($w=0$)": ("#6F4E7C", "o"),
    "JetFUEL 128 / unguided ($w=0$)": ("#6F4E7C", "s"),
    "JetFUEL 64 / optimal guidance": ("#D45087", "o"),
    "JetFUEL 128 / optimal measured guidance": ("#D45087", "s"),
}


def train_cost(record: dict, jet_class: str) -> float:
    return record["train"]


def infer_cost(record: dict, jet_class: str) -> float:
    value = record["infer"]
    return value[jet_class] if isinstance(value, dict) else value


def x_cost(
    record: dict, jet_class: str, row: str,
    deployment_jets: int = DEPLOYMENT_JETS,
) -> float:
    if row == "train":
        return train_cost(record, jet_class)
    if row == "infer":
        return infer_cost(record, jet_class)
    if row == "lifecycle":
        # There are 1e6 GFLOP in one PFLOP.
        return (
            train_cost(record, jet_class)
            + infer_cost(record, jet_class) * deployment_jets / 1_000_000
        )
    raise ValueError(row)


def pareto_names(points: list[tuple[str, float, float]]) -> set[str]:
    """Return names not weakly dominated in both lower-is-better dimensions."""
    front = set()
    for name, x, y in points:
        dominated = any(
            (x2 <= x and y2 <= y and (x2 < x or y2 < y))
            for name2, x2, y2 in points
            if name2 != name
        )
        if not dominated:
            front.add(name)
    return front


def row_points(
    row: str, jet_class: str, deployment_jets: int = DEPLOYMENT_JETS,
    excluded: frozenset[str] = frozenset(),
) -> list[tuple[str, float, float]]:
    points = [
        (
            name,
            x_cost(record, jet_class, row, deployment_jets),
            record["fpnd"][jet_class],
        )
        for name, record in METHODS.items()
        if jet_class in record["fpnd"] and name not in excluded
    ]
    if row != "train":
        return points

    # Steps, CFG, and sampler choice do not create new trained weights. Show one
    # point per checkpoint in the training row, using its best measured FPND.
    selected: dict[str, tuple[str, float, float]] = {}
    for point in points:
        name, _, y = point
        group = TRAIN_GROUP.get(name, name)
        if group not in selected or y < selected[group][2]:
            selected[group] = point
    return list(selected.values())


def write_table() -> None:
    path = OUT / "compute_fpnd_ledger.csv"
    fields = [
        "method", "jet_class", "fpnd", "train_pflop", "infer_gflop_per_jet",
        "lifecycle_pflop_at_1e6", "lifecycle_pflop_at_1e7",
        "lifecycle_pflop_at_1e8", "pareto_train", "pareto_infer",
        "pareto_lifecycle_at_1e6", "pareto_lifecycle_at_1e7",
        "pareto_lifecycle_at_1e8", "pareto_lifecycle",
        "shown_in_training_panel", "class_specific_fit_pflop",
        "train_downstream_only_pflop", "guidance_weight",
    ]
    fronts = {}
    for jet_class in CLASSES:
        fronts[("train", jet_class)] = pareto_names(row_points("train", jet_class))
        fronts[("infer", jet_class)] = pareto_names(row_points("infer", jet_class))
        for jets in LIFECYCLE_JETS:
            fronts[("lifecycle", jet_class, jets)] = pareto_names(
                row_points("lifecycle", jet_class, jets)
            )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, record in METHODS.items():
            for jet_class in CLASSES:
                if jet_class not in record["fpnd"]:
                    continue
                infer = infer_cost(record, jet_class)
                writer.writerow({
                    "method": name.replace("$", ""),
                    "jet_class": jet_class,
                    "fpnd": f'{record["fpnd"][jet_class]:.9g}',
                    "train_pflop": f"{train_cost(record, jet_class):.9f}",
                    "infer_gflop_per_jet": f'{infer:.9f}',
                    "lifecycle_pflop_at_1e6": f'{train_cost(record, jet_class) + infer:.9f}',
                    "lifecycle_pflop_at_1e7": f'{train_cost(record, jet_class) + 10 * infer:.9f}',
                    "lifecycle_pflop_at_1e8": f'{x_cost(record, jet_class, "lifecycle"):.9f}',
                    "pareto_train": name in fronts[("train", jet_class)],
                    "pareto_infer": name in fronts[("infer", jet_class)],
                    "pareto_lifecycle_at_1e6": name in fronts[("lifecycle", jet_class, 1_000_000)],
                    "pareto_lifecycle_at_1e7": name in fronts[("lifecycle", jet_class, 10_000_000)],
                    "pareto_lifecycle_at_1e8": name in fronts[("lifecycle", jet_class, 100_000_000)],
                    # Backward-compatible alias for the original M=1e8 panel.
                    "pareto_lifecycle": name in fronts[("lifecycle", jet_class, 100_000_000)],
                    "shown_in_training_panel": name in {
                        point[0] for point in row_points("train", jet_class)
                    },
                    "class_specific_fit_pflop": (
                        f'{record["train_class_models"][jet_class]:.9f}'
                        if "train_class_models" in record else ""
                    ),
                    "train_downstream_only_pflop": (
                        f'{record["train_downstream_only"]:.9f}'
                        if "train_downstream_only" in record else ""
                    ),
                    "guidance_weight": record.get("guidance_weight", {}).get(jet_class, ""),
                })


def write_lifecycle_table() -> None:
    """Write one row per deployable sampler configuration.

    Training is the one-time portfolio acquisition cost; the inference columns
    are the marginal deployment totals, and lifecycle is their sum.
    """
    path = OUT / "gpu_lifecycle_costs.csv"
    fields = [
        "method", "jet_class", "guidance_weight", "training_accounting", "train_pflop",
        "sampler_work_per_jet", "infer_gflop_per_jet",
    ]
    for jets in LIFECYCLE_JETS:
        exponent = len(str(jets)) - 1
        fields.extend([
            f"inference_pflop_at_1e{exponent}",
            f"lifecycle_pflop_at_1e{exponent}",
        ])

    rows = []
    for name, record in METHODS.items():
        for jet_class in CLASSES:
            if jet_class not in record["fpnd"]:
                continue
            rows.append((
                name.replace("$", ""), jet_class,
                record.get("guidance_weight", {}).get(jet_class, ""),
                "portfolio acquisition cost", record["train"],
                SAMPLER_WORK[name],
                infer_cost(record, jet_class),
            ))
            if name == "OmniLearn":
                rows.append((
                    "OmniLearn (downstream-only alternative)", jet_class, "",
                    "JetNet fine-tune only; upstream pretraining treated as sunk",
                    record["train_downstream_only"], SAMPLER_WORK[name],
                    infer_cost(record, jet_class),
                ))

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, jet_class, guidance_weight, accounting, train, work, infer in rows:
            row = {
                "method": name,
                "jet_class": jet_class,
                "guidance_weight": guidance_weight,
                "training_accounting": accounting,
                "train_pflop": f"{train:.9f}",
                "sampler_work_per_jet": work,
                "infer_gflop_per_jet": f"{infer:.9f}",
            }
            for jets in LIFECYCLE_JETS:
                exponent = len(str(jets)) - 1
                deployment = jets * infer / 1_000_000
                row[f"inference_pflop_at_1e{exponent}"] = f"{deployment:.9f}"
                row[f"lifecycle_pflop_at_1e{exponent}"] = f"{train + deployment:.9f}"
            writer.writerow(row)


def make_plot(
    deployment_jets: int,
    *,
    excluded: frozenset[str] = frozenset(),
    filename_suffix: str = "",
    log_x: bool = False,
) -> None:
    exponent = len(str(deployment_jets)) - 1
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 11.8), sharey=True)

    for col, jet_class in enumerate(CLASSES):
        axes[0, col].set_title(CLASS_TITLES[jet_class], fontweight="bold")
        for row_idx, (row, xlabel) in enumerate(ROWS):
            ax = axes[row_idx, col]
            points = row_points(row, jet_class, deployment_jets, excluded)
            front = pareto_names(points)
            front_points = sorted(
                ((x, y) for name, x, y in points if name in front), key=lambda p: p[0]
            )
            if len(front_points) > 1:
                ax.plot(
                    [p[0] for p in front_points], [p[1] for p in front_points],
                    color="0.25", linewidth=1.1, linestyle="--", zorder=1,
                )

            for name, x, y in points:
                color, marker = STYLE[name]
                is_jetfuel = name in JETFUEL
                ax.scatter(
                    x, y, s=72 if is_jetfuel else 48, marker=marker,
                    facecolor=color, edgecolor="black" if is_jetfuel else "white",
                    linewidth=1.0 if is_jetfuel else 0.55,
                    alpha=1.0 if name in front or is_jetfuel else 0.72,
                    zorder=4 if is_jetfuel else 3,
                )

            if log_x:
                ax.set_xscale("log")
            ax.set_ylim(0.0, 1.08)
            ax.grid(True, which="major", color="0.87", linewidth=0.7)
            ax.grid(True, which="minor", color="0.93", linewidth=0.45)
            if row == "lifecycle":
                xlabel = rf"GPU lifecycle at $M=10^{{{exponent}}}$ total jets [PFLOP]"
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(r"FPND $\downarrow$")

    handles = []
    for name, (color, marker) in STYLE.items():
        if name in excluded:
            continue
        handles.append(Line2D(
            [], [], linestyle="none", marker=marker,
            markersize=7.5 if name in JETFUEL else 6.5,
            markerfacecolor=color,
            markeredgecolor="black" if name in JETFUEL else "white",
            label=name,
        ))
    handles.append(Line2D([], [], color="0.25", linestyle="--", label="Pareto frontier"))

    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, 0.048), columnspacing=1.5, handletextpad=0.45,
    )
    fig.suptitle(
        "JetNet-30 GPU-compute--quality Pareto comparison"
        + (" (OmniLearn and EPiC-GAN omitted)" if excluded else ""),
        fontsize=15, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.012,
        "Lower-left is better. GPU model FLOPs only; CPU preprocessing, metrics, I/O, and wall-clock time are excluded. "
        "Conditional generation only; attribute/conditioning generators are excluded. "
        "Training is the portfolio cost for all applicable g/q/t models: joint training is counted once and separate fits are summed. "
        "The training row shows one point per checkpoint, using its best measured sampler for each class. "
        rf"Lifecycle adds inference for $10^{{{exponent}}}$ total generated jets. "
        "One GPU MAC = 2 FLOPs; FP32 and FP64 operations are not throughput-weighted. "
        "Training totals use published/released maximum budgets where stopping epochs are unavailable. "
        "JetFUEL guidance points use the lowest measured FPND separately by class; the 128-step $w=0.25$ arm is pending.",
        ha="center", va="bottom", fontsize=8.2,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.945, bottom=0.145, hspace=0.32, wspace=0.16)
    stem = OUT / f"jetnet30_compute_fpnd_pareto_3x3_m1e{exponent}{filename_suffix}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    if deployment_jets == DEPLOYMENT_JETS and not filename_suffix:
        # Preserve the original unsuffixed M=1e8 artifact names.
        fig.savefig(OUT / "jetnet30_compute_fpnd_pareto_3x3.pdf", bbox_inches="tight")
        fig.savefig(OUT / "jetnet30_compute_fpnd_pareto_3x3.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write_table()
    write_lifecycle_table()
    for deployment_jets in LIFECYCLE_JETS:
        make_plot(deployment_jets)
        make_plot(
            deployment_jets,
            excluded=frozenset({"OmniLearn", "EPiC-GAN"}),
            filename_suffix="_no_omnilearn_epicgan",
            log_x=True,
        )
