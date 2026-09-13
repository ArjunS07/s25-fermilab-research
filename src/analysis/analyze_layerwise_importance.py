#!/usr/bin/env python3
"""Checkpoint-only depth/width diagnostics for the published H model.

The script follows a common replay trajectory and measures:

* each block's residual updates to invariant state ``h`` and auxiliary vectors ``y``;
* final velocity sensitivity to single-block gates and cumulative depth truncation; and
* the covariance spectrum/effective rank of each block's real-particle ``h`` state.

These are screening diagnostics, not substitutes for retraining or FPND.  In
particular, a checkpoint head was trained on six-block representations, so a
post-hoc depth gate measures dependence/robustness rather than the best quality
that a shallower model can learn from scratch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from models.lorentznet_flow import build_lorentznet
from util.data.jet_attributes import NUM_CLASSES
from util.geometry.conditioning import scale_condition_pt
from util.geometry.coordinates import build_reference_vectors
from util.geometry.mass_shell import project_to_shell


JET_TYPES = ("g", "q", "t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-per-class", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--integration-steps", type=int, default=64)
    parser.add_argument(
        "--snapshot-steps", default="0,8,16,24,32,40,48,56,63",
        help="Comma-separated zero-based states at which to run interventions.",
    )
    parser.add_argument(
        "--guidance-weights", default="g=0,q=0.25,t=0.5",
        help="Per-class internal CFG weights used for velocity and trajectory tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_guidance(raw: str) -> dict[str, float]:
    result = {}
    for item in raw.split(","):
        name, value = item.split("=", 1)
        result[name.strip()] = float(value)
    if set(result) != set(JET_TYPES):
        raise ValueError(f"guidance weights must define exactly {JET_TYPES}; got {result}")
    return result


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


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint_architecture(checkpoint)
    model = build_lorentznet(
        NUM_CLASSES,
        num_layers=architecture["n_layers"],
        hidden_dim=architecture["n_hidden"],
        regulator_mass=architecture["regulator_mass"],
        particle_direction_mode=architecture["particle_direction_mode"],
        reference_direction_mode=architecture["reference_direction_mode"],
    ).to(device)
    state = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint, architecture


def select_balanced_indices(attrs: torch.Tensor, max_per_class: int, seed: int) -> torch.Tensor:
    global_types = attrs[:, :5].argmax(dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    for global_type in range(3):
        candidates = (global_types == global_type).nonzero(as_tuple=False).flatten()
        if len(candidates) < max_per_class:
            raise ValueError(
                f"replay has only {len(candidates)} class-{global_type} jets; "
                f"requested {max_per_class}"
            )
        order = torch.randperm(len(candidates), generator=generator)[:max_per_class]
        selected.append(candidates[order])
    return torch.cat(selected)


def prepare_inputs(bundle: dict, indices: torch.Tensor, final_scale: float,
                   regulator_mass: float, device: torch.device):
    attrs = bundle["generated_jet_attrs"][indices].to(device)
    mask = bundle["masks"][indices].to(device)
    state = bundle["internal_prior"][indices].to(device)
    state = project_to_shell(state * mask.unsqueeze(-1), regulator_mass)
    global_types = attrs[:, :5].argmax(dim=-1)
    if not torch.all((global_types >= 0) & (global_types <= 2)):
        raise ValueError("layer audit currently expects a balanced g/q/t replay bundle")
    cond = torch.cat(
        (
            attrs[:, :5],
            attrs[:, 8:9].clamp(max=mask.shape[1]),
            scale_condition_pt(attrs[:, 6], final_scale).unsqueeze(-1),
            (attrs[:, 7] / final_scale).unsqueeze(-1),
        ),
        dim=-1,
    )
    refs = build_reference_vectors(
        attrs[:, 5], attrs[:, 6], final_scale, device,
        jet_phi=bundle["jet_phi"][indices].to(device), jet_mass=attrs[:, 7],
    )
    return state, cond, mask, refs, global_types


def masked_rms(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value).to(value.dtype)
    denominator = expanded.sum().clamp_min(1)
    return float(torch.sqrt((value.square() * expanded).sum() / denominator).cpu())


def masked_cosine(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.unsqueeze(-1).expand_as(left).to(left.dtype)
    left = left * expanded
    right = right * expanded
    numerator = (left * right).sum()
    denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1e-30)
    return float((numerator / denominator).cpu())


class ResidualRecorder:
    def __init__(self, blocks):
        self.records = {}
        self.handles = [
            block.register_forward_hook(self._hook(layer))
            for layer, block in enumerate(blocks)
        ]

    def _hook(self, layer: int):
        def hook(_module, args, output):
            h0, y0, mask = args[:3]
            h1, y1 = output
            self.records[layer] = {
                "h0": h0.detach(), "h1": h1.detach(),
                "y0": y0.detach(), "y1": y1.detach(),
                "mask": mask.detach(),
            }
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def block_gates(blocks, gates: dict[int, float]):
    handles = []
    for layer, alpha in gates.items():
        def hook(_module, args, output, alpha=float(alpha)):
            h0, y0 = args[:2]
            h1, y1 = output
            return h0 + alpha * (h1 - h0), y0 + alpha * (y1 - y0)
        handles.append(blocks[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def interventions(n_layers: int) -> list[tuple[str, str, dict[int, float]]]:
    specs = []
    for layer in range(n_layers):
        specs.append((f"drop_layer_{layer + 1}", "single_drop", {layer: 0.0}))
        specs.append((f"half_layer_{layer + 1}", "single_half", {layer: 0.5}))
    for retained_depth in range(1, n_layers):
        specs.append((
            f"prefix_depth_{retained_depth}", "prefix",
            {layer: 0.0 for layer in range(retained_depth, n_layers)},
        ))
    if n_layers >= 6:
        specs.extend((
            ("keep_alternating_135", "alternating", {layer: 0.0 for layer in (1, 3, 5)}),
            ("keep_alternating_246", "alternating", {layer: 0.0 for layer in (0, 2, 4)}),
        ))
    return specs


def class_velocity(model, state, time_value, cond, mask, refs, weight):
    batch_t = torch.full(
        (state.shape[0],), float(time_value), device=state.device,
        dtype=next(model.parameters()).dtype,
    )
    velocity = model(state, batch_t, cond, mask, refs)
    if weight > 0:
        unconditional = model(state, batch_t, model.make_null_cond(cond), mask, refs)
        velocity = velocity + weight * (velocity - unconditional)
    return velocity


def update_covariance(accumulator: dict, values: torch.Tensor, mask: torch.Tensor):
    real_values = values[mask.bool()].detach().to(torch.float64)
    accumulator["count"] += int(real_values.shape[0])
    accumulator["sum"] += real_values.sum(dim=0).cpu()
    accumulator["gram"] += (real_values.T @ real_values).cpu()


def spectrum_summary(accumulator: dict) -> dict:
    count = accumulator["count"]
    mean = accumulator["sum"] / count
    covariance = accumulator["gram"] / count - torch.outer(mean, mean)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    fractions = eigenvalues / total
    cumulative = fractions.cumsum(0)
    def rank_at(threshold: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(threshold)).item() + 1)
    participation = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    return {
        "samples": count,
        "participation_ratio": float(participation),
        "rank_90": rank_at(0.90),
        "rank_95": rank_at(0.95),
        "rank_99": rank_at(0.99),
        "variance_explained_top64": float(cumulative[min(63, len(cumulative) - 1)]),
        "eigenvalues": [float(value) for value in eigenvalues],
    }


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output_dir: Path, residual_rows: list[dict], sensitivity_rows: list[dict],
                 n_layers: int):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for class_name in JET_TYPES:
        for metric, axis, label in (
            ("h_relative_rms", axes[0], r"RMS($\Delta h$) / RMS($h$)"),
            ("y_relative_rms", axes[1], r"RMS($\Delta y$) / RMS($y$)"),
        ):
            values = []
            for layer in range(1, n_layers + 1):
                subset = [
                    row[metric] for row in residual_rows
                    if row["jet_class"] == class_name and row["layer"] == layer
                ]
                values.append(sum(subset) / len(subset))
            axis.plot(range(1, n_layers + 1), values, marker="o", label=class_name)
            axis.set_xlabel("Block")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
    for class_name in JET_TYPES:
        values = []
        for layer in range(1, n_layers + 1):
            subset = [
                row["relative_velocity_rms_error"] for row in sensitivity_rows
                if row["jet_class"] == class_name
                and row["intervention"] == f"drop_layer_{layer}"
            ]
            values.append(sum(subset) / len(subset))
        axes[2].plot(range(1, n_layers + 1), values, marker="o", label=class_name)
    axes[2].set_xlabel("Disabled block")
    axes[2].set_ylabel("Relative velocity RMS error")
    axes[2].grid(alpha=0.3)
    axes[2].legend(title="Jet class")
    fig.suptitle("Frozen-checkpoint layerwise screening")
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_importance.png", dpi=200)
    fig.savefig(output_dir / "layerwise_importance.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_per_class < 1 or args.batch_size < 1 or args.integration_steps < 1:
        raise ValueError("sample counts, batch size, and integration steps must be positive")
    snapshot_steps = sorted({int(value) for value in args.snapshot_steps.split(",")})
    if not snapshot_steps or snapshot_steps[0] < 0 or snapshot_steps[-1] >= args.integration_steps:
        raise ValueError("snapshot steps must lie in [0, integration_steps)")
    guidance = parse_guidance(args.guidance_weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, checkpoint, architecture = load_model(args.checkpoint, device)
    bundle = torch.load(args.replay_bundle, map_location="cpu", weights_only=False)
    metadata = bundle.get("metadata", {})
    final_scale = float(metadata["final_scale"])
    if int(metadata["num_particles"]) != 30:
        raise ValueError("this experiment is specified for JetNet-30")
    indices = select_balanced_indices(
        bundle["generated_jet_attrs"], args.max_per_class, args.seed
    )
    state, cond, mask, refs, class_ids = prepare_inputs(
        bundle, indices, final_scale, architecture["regulator_mass"], device
    )
    blocks = model.lorentznet_backbone.blocks
    specs = interventions(len(blocks))
    width = architecture["n_hidden"]
    covariance = {
        layer: {
            "count": 0,
            "sum": torch.zeros(width, dtype=torch.float64),
            "gram": torch.zeros(width, width, dtype=torch.float64),
        }
        for layer in range(len(blocks))
    }
    residual_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    times = torch.linspace(
        0, 0.99999, args.integration_steps + 1, device=device, dtype=torch.float64
    )

    with torch.inference_mode():
        for step in range(args.integration_steps):
            if step in snapshot_steps:
                for class_id, class_name in enumerate(JET_TYPES):
                    class_mask = class_ids == class_id
                    x_c, cond_c = state[class_mask], cond[class_mask]
                    mask_c, refs_c = mask[class_mask], refs[class_mask]
                    for start in range(0, len(x_c), args.batch_size):
                        selection = slice(start, start + args.batch_size)
                        xb, cb = x_c[selection], cond_c[selection]
                        mb, rb = mask_c[selection], refs_c[selection]
                        recorder = ResidualRecorder(blocks)
                        batch_t = torch.full(
                            (xb.shape[0],), float(times[step]), device=device,
                            dtype=next(model.parameters()).dtype,
                        )
                        model(xb, batch_t, cb, mb, rb)
                        recorder.close()
                        baseline = class_velocity(
                            model, xb, times[step], cb, mb, rb, guidance[class_name]
                        )
                        # Residual/rank diagnostics use the conditional branch. Gate
                        # sensitivity below evaluates the complete guided velocity.
                        for layer, record in recorder.records.items():
                            h_rms = masked_rms(record["h0"], record["mask"])
                            y_rms = masked_rms(record["y0"], record["mask"])
                            h_delta = masked_rms(record["h1"] - record["h0"], record["mask"])
                            y_delta = masked_rms(record["y1"] - record["y0"], record["mask"])
                            residual_rows.append({
                                "snapshot_step": step,
                                "time": float(times[step]),
                                "jet_class": class_name,
                                "layer": layer + 1,
                                "h_state_rms": h_rms,
                                "h_delta_rms": h_delta,
                                "h_relative_rms": h_delta / max(h_rms, 1e-30),
                                "y_state_rms": y_rms,
                                "y_delta_rms": y_delta,
                                "y_relative_rms": y_delta / max(y_rms, 1e-30),
                            })
                            update_covariance(covariance[layer], record["h1"], record["mask"])
                        baseline_rms = masked_rms(baseline, mb)
                        for name, kind, gates in specs:
                            with block_gates(blocks, gates):
                                changed = class_velocity(
                                    model, xb, times[step], cb, mb, rb,
                                    guidance[class_name],
                                )
                            changed_rms = masked_rms(changed, mb)
                            sensitivity_rows.append({
                                "snapshot_step": step,
                                "time": float(times[step]),
                                "jet_class": class_name,
                                "intervention": name,
                                "kind": kind,
                                "gates": json.dumps(gates, sort_keys=True),
                                "baseline_velocity_rms": baseline_rms,
                                "changed_velocity_rms": changed_rms,
                                "relative_velocity_rms_error": (
                                    masked_rms(changed - baseline, mb)
                                    / max(baseline_rms, 1e-30)
                                ),
                                "velocity_cosine": masked_cosine(baseline, changed, mb),
                            })

            # Advance only the unmodified reference trajectory.
            next_state = state.clone()
            for class_id, class_name in enumerate(JET_TYPES):
                class_mask = class_ids == class_id
                next_state[class_mask] = model.step_hyperbolic(
                    state[class_mask], cond[class_mask], mask[class_mask],
                    times[step], times[step + 1],
                    use_cfg=guidance[class_name] > 0,
                    guidance_weight=guidance[class_name],
                    ref_vectors=refs[class_mask],
                )
            if not torch.isfinite(next_state).all():
                raise FloatingPointError(f"non-finite reference trajectory at step {step}")
            state = next_state

    write_csv(output_dir / "layer_residuals.csv", residual_rows)
    write_csv(output_dir / "field_sensitivity.csv", sensitivity_rows)
    spectra = {
        str(layer + 1): spectrum_summary(accumulator)
        for layer, accumulator in covariance.items()
    }
    summary = {
        "device": str(device),
        "architecture": architecture,
        "checkpoint_step": checkpoint.get("global_optimizer_step"),
        "checkpoint_sha256": sha256(args.checkpoint),
        "replay_sha256": sha256(args.replay_bundle),
        "samples_per_class": args.max_per_class,
        "integration_steps": args.integration_steps,
        "snapshot_steps": snapshot_steps,
        "guidance_weights": guidance,
        "interventions": [
            {"name": name, "kind": kind, "gates": gates}
            for name, kind, gates in specs
        ],
        "hidden_state_spectra": spectra,
        "interpretation_warning": (
            "Post-hoc gates measure checkpoint dependence, not the attainable FPND "
            "of a shallower model retrained from scratch."
        ),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    plot_summary(output_dir, residual_rows, sensitivity_rows, len(blocks))
    print(json.dumps({
        "output_dir": str(output_dir),
        "architecture": architecture,
        "samples_per_class": args.max_per_class,
        "snapshots": snapshot_steps,
    }, indent=2))


if __name__ == "__main__":
    main()
