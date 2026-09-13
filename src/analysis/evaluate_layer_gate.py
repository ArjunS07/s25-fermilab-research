#!/usr/bin/env python3
"""Run a full common-replay FPND evaluation under a frozen block-gate intervention."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from analysis.analyze_layerwise_importance import block_gates, load_model, sha256
from data import get_data_path
from generate_samples import generate_samples
from models.stage1 import get_model_pth_path
from util.data import jet_attributes
from util.metrics.metrics import run_save_metrics


def parse_gate(raw: str, n_layers: int) -> dict[int, float]:
    """Parse one-based ``layer=alpha`` pairs into zero-based block gates."""
    if raw.strip().lower() in {"", "none", "baseline"}:
        return {}
    gates = {}
    for item in raw.split(","):
        layer_raw, alpha_raw = item.split("=", 1)
        layer = int(layer_raw) - 1
        alpha = float(alpha_raw)
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"layer {layer + 1} is outside 1..{n_layers}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("gate alpha must lie in [0, 1]")
        gates[layer] = alpha
    return gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--replay-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate", default="baseline")
    parser.add_argument("--integration-steps", type=int, default=64)
    parser.add_argument("--guidance-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-class", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, checkpoint, architecture = load_model(args.checkpoint, device)
    gates = parse_gate(args.gate, architecture["n_layers"])
    bundle = torch.load(args.replay_bundle, map_location="cpu", weights_only=False)
    metadata = bundle["metadata"]
    expected_samples = 3 * args.samples_per_class
    if int(metadata["n_samples"]) != expected_samples:
        raise ValueError(
            f"replay contains {metadata['n_samples']} jets; expected {expected_samples}"
        )
    final_scale = float(metadata["final_scale"])

    stage1 = jet_attributes.load_model(
        model_path=get_model_pth_path(args.source_run)
    ).to(device)
    stage1.eval()
    with block_gates(model.lorentznet_backbone.blocks, gates):
        samples, generated_types, generated_pt, priors, generated_eta = generate_samples(
            model=model,
            jet_attr_model=stage1,
            device=device,
            root_output_path=str(output_dir),
            max_particles_per_jet=30,
            final_scale=final_scale,
            integration_steps=args.integration_steps,
            integration_end_time=0.99999,
            n_samples=expected_samples,
            batch_size=args.batch_size,
            jet_types=("g", "q", "t"),
            samples_per_jet_type=args.samples_per_class,
            use_cfg=args.guidance_weight > 0,
            cfg_guidance_weight=args.guidance_weight,
            regulator_mass=architecture["regulator_mass"],
            prior_dist="axis_aligned_lognormal",
            replay_bundle_path=args.replay_bundle,
        )
    torch.save(samples.cpu(), output_dir / "samples.pt")
    torch.save(priors.cpu(), output_dir / "prior_samples.pt")

    with open(Path(get_data_path(args.source_run)) / "x_test.pkl", "rb") as handle:
        x_test = pickle.load(handle)
    metrics = run_save_metrics(
        X_test=x_test,
        jet_types=("g", "q", "t"),
        gen_samples=samples,
        output_path=str(output_dir),
        device=device,
        gen_jet_types=generated_types,
        gen_pt_cond=generated_pt,
        gen_jet_eta=generated_eta,
        prior_samples=priors,
        stratify_by_class=True,
    )
    summary = {
        "architecture": architecture,
        "checkpoint_step": checkpoint.get("global_optimizer_step"),
        "checkpoint_sha256": sha256(args.checkpoint),
        "replay_sha256": sha256(args.replay_bundle),
        "gate_argument": args.gate,
        "gates_zero_based": gates,
        "integration_steps": args.integration_steps,
        "guidance_weight": args.guidance_weight,
        "samples_per_class": args.samples_per_class,
        "metrics": metrics,
        "interpretation_warning": (
            "Post-hoc gates measure checkpoint dependence; they do not estimate the "
            "best attainable quality after retraining a shallower architecture."
        ),
    }
    with (output_dir / "layer_gate_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=lambda value: value.item())
    print({key: value for key, value in metrics.items() if key.startswith("fpnd_")})


if __name__ == "__main__":
    main()
