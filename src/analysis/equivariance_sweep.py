#!/usr/bin/env python3
"""Sweep numerical Lorentz-equivariance error for a trained H flow field.

For each transform ``Lambda``, this measures

    F(Lambda X, t, c; Lambda r0, Lambda r1) - Lambda F(X, t, c; r0, r1)

on held-out JetNet constituents or saved replay states. Boosts and rotations
are swept separately. The reference vectors are always transformed jointly
with the particles.

Example:
    PYTHONPATH=src python src/analysis/equivariance_sweep.py \
        --checkpoint /path/to/final_checkpoint.pth \
        --jetnet-data-dir src/datasets/jetnet \
        --output-dir /path/to/equivariance
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from models.lorentznet_flow import build_lorentznet
from util.data.jet_attributes import one_hot_enc_jet_type
from util.geometry.conditioning import scale_condition_pt
from util.geometry.coordinates import (
    build_reference_vectors,
    deterministic_jet_phi,
    transform_rel_particle_coordinates_to_cartesian,
)
from util.geometry.mass_shell import project_to_shell


DTYPE_STYLES = {
    "float32": {"linestyle": "-", "marker": "o"},
    "float64": {"linestyle": (0, (5, 2.4)), "marker": "s"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jetnet-data-dir")
    source.add_argument("--replay-bundle")
    parser.add_argument(
        "--final-scale", type=float,
        help="Model momentum scale; defaults to the value embedded in the checkpoint.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=41)
    parser.add_argument("--max-beta", type=float, default=0.99)
    parser.add_argument("--max-angle", type=float, default=2 * math.pi)
    parser.add_argument("--boost-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--rotation-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--time", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtypes", default="float32,float64",
        help="Comma-separated model arithmetic dtypes (float32 and/or float64).",
    )
    parser.add_argument(
        "--raw-weights", action="store_true",
        help="Use model_state_dict instead of EMA weights when both are present.",
    )
    return parser.parse_args()


def checkpoint_architecture(checkpoint: dict) -> dict:
    saved = checkpoint.get("full_config", {}).get("model", {})
    particle_mode = saved.get("particle_direction_mode")
    if particle_mode is None:
        particle_mode = (
            "physical_logmap"
            if saved.get("particle_readout_mode", "normalized_logmap") == "normalized_logmap"
            else "latent_displacement"
        )
    reference_mode = saved.get("reference_direction_mode")
    if reference_mode is None:
        reference_mode = (
            "normalized_tangent"
            if saved.get("reference_mode", "normalized_tangent_readout")
            == "normalized_tangent_readout"
            else "raw_tangent"
        )
    return {
        "n_hidden": int(saved.get("n_hidden", 96)),
        "n_layers": int(saved.get("n_layers", 6)),
        "regulator_mass": float(saved.get("regulator_mass", 0.1)),
        "particle_direction_mode": particle_mode,
        "reference_direction_mode": reference_mode,
    }


def load_model(checkpoint: dict, dtype: torch.dtype, use_ema: bool) -> torch.nn.Module:
    architecture = checkpoint_architecture(checkpoint)
    state = (
        checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict", checkpoint))
        if use_ema else checkpoint.get("model_state_dict", checkpoint)
    )
    condition_dim = int(state["null_cond"].numel())
    model = build_lorentznet(
        condition_dim - 3,
        num_layers=architecture["n_layers"],
        hidden_dim=architecture["n_hidden"],
        regulator_mass=architecture["regulator_mass"],
        particle_direction_mode=architecture["particle_direction_mode"],
        reference_direction_mode=architecture["reference_direction_mode"],
    )
    model.load_state_dict(state, strict=True)
    return model.to(dtype=dtype).eval()


def prepare_replay_inputs(bundle: dict, num_samples: int, seed: int, regulator_mass: float):
    available = len(bundle["masks"])
    if num_samples > available:
        raise ValueError(f"requested {num_samples} samples, replay contains {available}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    all_types = bundle["generated_jet_attrs"][:, :5].argmax(dim=-1)
    present_types = all_types.unique(sorted=True)
    base_count, remainder = divmod(num_samples, len(present_types))
    selected_indices = []
    for position, class_id in enumerate(present_types.tolist()):
        class_count = base_count + int(position < remainder)
        if class_count == 0:
            continue
        candidates = (all_types == class_id).nonzero(as_tuple=False).flatten()
        if class_count > len(candidates):
            raise ValueError(
                f"requested {class_count} class-{class_id} samples, replay contains "
                f"{len(candidates)}"
            )
        order = torch.randperm(len(candidates), generator=generator)[:class_count]
        selected_indices.append(candidates[order])
    indices = torch.cat(selected_indices)

    attrs = bundle["generated_jet_attrs"][indices]
    mask = bundle["masks"][indices]
    state = project_to_shell(bundle["internal_prior"][indices] * mask.unsqueeze(-1), regulator_mass)
    final_scale = float(bundle["metadata"]["final_scale"])
    conditions = torch.cat(
        (
            attrs[:, :5],
            attrs[:, 8:9].clamp(max=mask.shape[1]),
            scale_condition_pt(attrs[:, 6], final_scale).unsqueeze(-1),
            (attrs[:, 7] / final_scale).unsqueeze(-1),
        ),
        dim=-1,
    )
    references = build_reference_vectors(
        attrs[:, 5], attrs[:, 6], final_scale, torch.device("cpu"),
        jet_phi=bundle["jet_phi"][indices], jet_mass=attrs[:, 7],
    )
    global_types = attrs[:, :5].argmax(dim=-1)
    return state, conditions, mask, references, global_types


def prepare_jetnet_inputs(data_dir: str, num_samples: int, seed: int,
                          regulator_mass: float, final_scale: float):
    """Reproduce the repository's held-out JetNet split from the local HDF5 files."""
    jet_names = ("g", "q", "t")
    paths = [Path(data_dir) / f"{name}.hdf5" for name in jet_names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing local JetNet files: {missing}")

    lengths = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            lengths.append(len(handle["particle_features"]))
    total = sum(lengths)
    valid_start = int(0.7 * total)
    split_permutation = np.random.default_rng(42).permutation(total)[valid_start:]

    base_count, remainder = divmod(num_samples, len(jet_names))
    selection_generator = torch.Generator(device="cpu").manual_seed(seed)
    particle_batches = []
    jet_batches = []
    offset = 0
    for class_id, (path, length) in enumerate(zip(paths, lengths)):
        class_count = base_count + int(class_id < remainder)
        class_rows = split_permutation[
            (split_permutation >= offset) & (split_permutation < offset + length)
        ] - offset
        if class_count > len(class_rows):
            raise ValueError(
                f"requested {class_count} held-out {jet_names[class_id]} jets, "
                f"split contains {len(class_rows)}"
            )
        order = torch.randperm(len(class_rows), generator=selection_generator)[:class_count]
        # h5py requires monotonically increasing fancy indices.
        selected_rows = np.sort(class_rows[order.numpy()])
        with h5py.File(path, "r") as handle:
            particles = torch.from_numpy(handle["particle_features"][selected_rows, :30])
            raw_jets = torch.from_numpy(handle["jet_features"][selected_rows])
        # Raw HDF5 jet features are (pT, eta, mass, n_particles); training consumes
        # (eta, pT, mass, n_particles, global_type).
        jet_features = torch.stack(
            (
                raw_jets[:, 1],
                raw_jets[:, 0],
                raw_jets[:, 2],
                raw_jets[:, 3].clamp(max=30),
                torch.full_like(raw_jets[:, 0], class_id),
            ),
            dim=-1,
        )
        particle_batches.append(particles)
        jet_batches.append(jet_features)
        offset += length

    particles = torch.cat(particle_batches)
    jets = torch.cat(jet_batches)
    jet_phi = deterministic_jet_phi(len(jets), seed=seed)
    cartesian = transform_rel_particle_coordinates_to_cartesian(
        (particles, jets), jet_phi=jet_phi
    )
    mask = cartesian[..., 4]
    state = project_to_shell(
        (cartesian[..., :4] / final_scale) * mask.unsqueeze(-1), regulator_mass
    )
    global_types = jets[:, 4].long()
    conditions = torch.cat(
        (
            one_hot_enc_jet_type(global_types),
            jets[:, 3:4],
            scale_condition_pt(jets[:, 1], final_scale).unsqueeze(-1),
            (jets[:, 2] / final_scale).unsqueeze(-1),
        ),
        dim=-1,
    )
    references = build_reference_vectors(
        jets[:, 0], jets[:, 1], final_scale, torch.device("cpu"),
        jet_phi=jet_phi, jet_mass=jets[:, 2],
    )
    return state, conditions, mask, references, global_types


def rotation_4x4(axis: str, angle: float) -> torch.Tensor:
    """Spatial rotation embedded in (E, px, py, pz) coordinates."""
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = torch.eye(4, dtype=torch.float64)
    first, second = {
        "x": (2, 3),
        "y": (3, 1),
        "z": (1, 2),
    }[axis]
    transform[first, first] = cosine
    transform[first, second] = -sine
    transform[second, first] = sine
    transform[second, second] = cosine
    return transform


def boost_4x4(axis: str, beta: float) -> torch.Tensor:
    """Lorentz boost parameterized by beta=v/c in (E, px, py, pz) coordinates."""
    if not 0 <= beta < 1:
        raise ValueError(f"beta must satisfy 0 <= beta < 1, got {beta}")
    rapidity = math.atanh(beta)
    cosine_h, sine_h = math.cosh(rapidity), math.sinh(rapidity)
    transform = torch.eye(4, dtype=torch.float64)
    spatial = {"x": 1, "y": 2, "z": 3}[axis]
    transform[0, 0] = cosine_h
    transform[0, spatial] = sine_h
    transform[spatial, 0] = sine_h
    transform[spatial, spatial] = cosine_h
    return transform


def apply_transform(vectors: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return vectors.to(torch.float64) @ transform.T


@torch.inference_mode()
def sweep_model(model: torch.nn.Module, state: torch.Tensor, time: torch.Tensor,
                conditions: torch.Tensor, mask: torch.Tensor, references: torch.Tensor,
                transforms: list[torch.Tensor], *,
                transform_references: bool = True) -> list[dict[str, float]]:
    baseline = model(state, time, conditions, mask, ref_vectors=references)
    records = []
    expanded_mask = mask.unsqueeze(-1).to(torch.float64)
    for transform in transforms:
        expected = apply_transform(baseline, transform) * expanded_mask
        transformed_state = apply_transform(state, transform) * expanded_mask
        transformed_references = (
            apply_transform(references, transform) if transform_references else references
        )
        observed = model(
            transformed_state, time, conditions, mask, ref_vectors=transformed_references
        ) * expanded_mask
        residual = observed.to(torch.float64) - expected

        numerator = torch.sqrt(residual.square().sum(dim=(1, 2)))
        denominator = torch.sqrt(expected.square().sum(dim=(1, 2))).clamp_min(1e-30)
        relative_l2 = numerator / denominator
        max_absolute = residual.abs().amax(dim=(1, 2))
        records.append({
            "num_samples": len(relative_l2),
            "relative_l2_mean": float(relative_l2.mean()),
            "relative_l2_p10": float(relative_l2.quantile(0.10)),
            "relative_l2_p90": float(relative_l2.quantile(0.90)),
            "relative_l2_max": float(relative_l2.max()),
            "max_absolute_mean": float(max_absolute.mean()),
            "max_absolute_max": float(max_absolute.max()),
        })
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def plot_results(output_dir: Path, records: list[dict]) -> None:
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
    for transform_name, x_key, xlabel, stem in (
        ("boost", "beta", r"Boost $\beta=v/c$", "boost_equivariance"),
        ("rotation", "angle", r"Rotation angle $\theta$", "rotation_equivariance"),
        (
            "rotation_fixed_axis", "angle", r"Rotation angle $\theta$",
            "rotation_fixed_axis",
        ),
    ):
        plot_records = [row for row in records if row["transform"] == transform_name]
        positive_values = [
            value
            for row in plot_records
            for value in (row["relative_l2_p10"], row["relative_l2_p90"])
            if value > 0
        ]
        y_min = 10 ** math.floor(math.log10(min(positive_values) / 3))
        y_max = 10 ** math.ceil(math.log10(max(positive_values) * 3))
        fig, axis = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
        for dtype_name in sorted({row["dtype"] for row in records}):
            selected = [
                row for row in records
                if row["transform"] == transform_name and row["dtype"] == dtype_name
            ]
            x = [row[x_key] for row in selected]
            mean = [max(row["relative_l2_mean"], 1e-18) for row in selected]
            # Lambda=I is exactly zero by construction and is not informative on a log axis.
            if x and x[0] == 0:
                mean[0] = math.nan
            style = DTYPE_STYLES[dtype_name]
            axis.plot(
                x, mean, color="#222222", linewidth=1.65,
                linestyle=style["linestyle"], marker=style["marker"],
                markevery=max(1, len(x) // 8), markersize=3.4,
                markerfacecolor="white", markeredgecolor="#222222",
                markeredgewidth=0.8, label=dtype_name,
            )
        axis.set_yscale("log")
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"Relative equivariance residual")
        if transform_name.startswith("rotation") and math.isclose(max(x), 2 * math.pi):
            axis.set_xticks(
                [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi],
                [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
            )
        sns.despine(ax=axis)
        axis.tick_params(direction="out", length=3, width=0.7)
        axis.legend(loc="best", ncol=1, fontsize=8, handlelength=2.8)
        fig.savefig(output_dir / f"{stem}.png", dpi=240)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.num_points < 2:
        raise ValueError("num-samples must be positive and num-points must be at least 2")
    if not 0 <= args.max_beta < 1:
        raise ValueError("max-beta must satisfy 0 <= max-beta < 1")
    if args.max_angle <= 0:
        raise ValueError("max-angle must be positive")

    dtype_names = [name.strip() for name in args.dtypes.split(",") if name.strip()]
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    unknown = set(dtype_names) - set(dtype_map)
    if unknown:
        raise ValueError(f"unsupported dtypes: {sorted(unknown)}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint_architecture(checkpoint)
    if args.jetnet_data_dir:
        embedded_scale = checkpoint.get("config", {}).get("final_scale")
        final_scale = args.final_scale if args.final_scale is not None else embedded_scale
        if final_scale is None:
            raise ValueError(
                "JetNet input requires --final-scale when the checkpoint does not embed it"
            )
        state, conditions, mask, references, jet_types = prepare_jetnet_inputs(
            args.jetnet_data_dir, args.num_samples, args.seed,
            architecture["regulator_mass"], float(final_scale),
        )
        input_source = {
            "kind": "held-out JetNet validation split",
            "path": str(Path(args.jetnet_data_dir).resolve()),
            "final_scale": float(final_scale),
            "split_seed": 42,
            "split_fraction": [0.7, 0.3, 0.0],
        }
    else:
        bundle = torch.load(args.replay_bundle, map_location="cpu", weights_only=False)
        state, conditions, mask, references, jet_types = prepare_replay_inputs(
            bundle, args.num_samples, args.seed, architecture["regulator_mass"]
        )
        input_source = {
            "kind": "saved replay prior",
            "path": str(Path(args.replay_bundle).resolve()),
            "final_scale": float(bundle["metadata"]["final_scale"]),
        }
    time = torch.full((len(state),), args.time, dtype=torch.float64)
    betas = torch.linspace(0.0, args.max_beta, args.num_points, dtype=torch.float64).tolist()
    angles = torch.linspace(0.0, args.max_angle, args.num_points, dtype=torch.float64).tolist()

    all_records = []
    for dtype_name in dtype_names:
        print(f"Evaluating {dtype_name} arithmetic ...", flush=True)
        model = load_model(checkpoint, dtype_map[dtype_name], use_ema=not args.raw_weights)
        boost_records = sweep_model(
            model, state, time, conditions, mask, references,
            [boost_4x4(args.boost_axis, beta) for beta in betas],
        )
        rotation_records = sweep_model(
            model, state, time, conditions, mask, references,
            [rotation_4x4(args.rotation_axis, angle) for angle in angles],
        )
        fixed_axis_records = sweep_model(
            model, state, time, conditions, mask, references,
            [rotation_4x4(args.rotation_axis, angle) for angle in angles],
            transform_references=False,
        )
        for beta, metrics in zip(betas, boost_records):
            all_records.append({
                "transform": "boost", "dtype": dtype_name, "beta": beta,
                "rapidity": math.atanh(beta), "angle": "", **metrics,
            })
        for angle, metrics in zip(angles, rotation_records):
            all_records.append({
                "transform": "rotation", "dtype": dtype_name, "beta": "",
                "rapidity": "", "angle": angle, **metrics,
            })
        for angle, metrics in zip(angles, fixed_axis_records):
            all_records.append({
                "transform": "rotation_fixed_axis", "dtype": dtype_name, "beta": "",
                "rapidity": "", "angle": angle, **metrics,
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "equivariance_sweep.csv", all_records)
    plot_results(output_dir, all_records)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_source": input_source,
        "weight_view": "raw" if args.raw_weights else "EMA if available, otherwise raw",
        "architecture": architecture,
        "num_samples": args.num_samples,
        "time": args.time,
        "boost_axis": args.boost_axis,
        "rotation_axis": args.rotation_axis,
        "fixed_axis_control": (
            "particles rotated while both reference vectors are held fixed; "
            "nonzero residual is expected"
        ),
        "max_beta": args.max_beta,
        "max_angle": args.max_angle,
        "dtypes": dtype_names,
        "aggregation": "arithmetic mean over a class-balanced pooled sample; per-jet p10-p90 retained in CSV",
        "samples_by_jet_type": {
            {0: "g", 1: "q", 2: "t"}.get(class_id, f"class_{class_id}"):
            int((jet_types == class_id).sum())
            for class_id in jet_types.unique(sorted=True).tolist()
        },
        "max_relative_l2": {
            dtype_name: max(
                row["relative_l2_max"] for row in all_records if row["dtype"] == dtype_name
            ) for dtype_name in dtype_names
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote equivariance sweep to {output_dir}")


if __name__ == "__main__":
    main()
