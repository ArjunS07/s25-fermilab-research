#!/usr/bin/env python3
"""
Standalone LEFTJeN inference script.

Loads a trained checkpoint and runs any combination of:
  1. Vector field visualisation  (--vf_mode cfg | nocfg | both | none)
  2. Sample generation           (skip with --skip_samples)
  3. Metric calculation          (skip with --skip_metrics)

Each stage is wrapped in try/except.  Samples are always saved as samples.pt
before metrics are attempted.

Example usage:
    python infer.py --config configs/infer-30-icp-hyperbolic-compare.yaml \\
        --set paths.checkpoint_path=/mnt/data/output/train/models/latest_checkpoint.pth \\
        --set paths.output_path=/mnt/data/output
"""

import os
import pickle
import traceback

import numpy as np
import torch

from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.jet_attributes import NUM_CLASSES
from jet_attr_model import get_model_pth_path
from util.file_management import make_clear_folder
from util.viz import generate_model_vector_field
from util.metrics import run_save_metrics
from generate_samples import generate_samples
from data import get_data_path
from config import InferRunConfig, build_config, parse_config_cli, infer_config_to_namespace

MAX_N_PARTICLES = 150
NUM_PARTICLE_FEATURES = 4


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    config_path, overrides = parse_config_cli()
    cfg = build_config(InferRunConfig, config_path, overrides)
    args = infer_config_to_namespace(cfg)
    if not args.checkpoint_path:
        raise ValueError("paths.checkpoint_path must be set in the config")
    if not args.output_path:
        raise ValueError("paths.output_path must be set in the config")
    return args


# ── Helpers ────────────────────────────────────────────────────────────────────

_ARCH_KEYS = ("n_hidden", "n_layers", "use_residual", "use_reference_vectors",
              "use_node_scalars", "use_adaln", "use_attention", "use_hyperbolic")


def _resolve_architecture(args, ckpt):
    """If the checkpoint carries a self-describing `full_config` (written by
    train.py's config path), use its model architecture and warn if it
    disagrees with any values the user explicitly passed. Falls back to
    `args` unchanged for older checkpoints without `full_config`."""
    if not isinstance(ckpt, dict):
        return args
    full_config = ckpt.get("full_config")
    if not full_config:
        return args

    ckpt_model = full_config.get("model", {})
    mism = {k: (getattr(args, k, None), ckpt_model[k])
            for k in _ARCH_KEYS if k in ckpt_model and getattr(args, k, None) != ckpt_model[k]}
    if mism:
        print(f"WARNING: CLI/config architecture flags differ from checkpoint: {mism}. "
              f"Using checkpoint's architecture (loaded weights would otherwise not match).")
    for k in _ARCH_KEYS:
        if k in ckpt_model:
            setattr(args, k, ckpt_model[k])
    print("Loaded model architecture from checkpoint config.")
    return args


def _load_main_model(checkpoint_path, n_hidden, n_layers, use_residual,
                     num_particles, device, use_reference_vectors=False, use_node_scalars=False,
                     use_adaln=False, use_attention=False, preloaded_ckpt=None):
    model = LEFTJeN(
        max_num_jet_types=NUM_CLASSES,
        max_particles=num_particles,
        num_layers=n_layers,
        hidden_dim=n_hidden,
        use_residual_update=use_residual,
        include_pt=True,
        use_reference_vectors=use_reference_vectors,
        use_node_scalars=use_node_scalars,
        use_adaln=use_adaln,
        use_attention=use_attention,
    ).to(device)

    ckpt = preloaded_ckpt if preloaded_ckpt is not None else torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}) from {checkpoint_path}")
    else:
        # Raw state dict saved with torch.save(model.state_dict(), ...)
        model.load_state_dict(ckpt)
        print(f"Loaded raw state dict from {checkpoint_path}")

    model.eval()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = args.out_dir or os.path.join(args.output_path, "inference")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Inference outputs → {out_dir}")

    # ── Load shared inputs ─────────────────────────────────────────────────────
    data_path = get_data_path(args.output_path)
    with open(os.path.join(data_path, "x_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    print(f"Test set: {len(X_test)} jets")

    scale_path = os.path.join(args.output_path, "train", "scale.txt")
    with open(scale_path) as f:
        final_scale = float(f.read().strip())
    print(f"Scale: {final_scale:.6f}")

    jet_attr_model = jet_attributes.load_model(
        model_path=get_model_pth_path(args.output_path)
    ).to(device)
    jet_attr_model.eval()
    print("Loaded jet attribute model")

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    args = _resolve_architecture(args, ckpt)

    model = _load_main_model(
        checkpoint_path=args.checkpoint_path,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        use_residual=args.use_residual,
        num_particles=args.num_particles,
        device=device,
        use_reference_vectors=args.use_reference_vectors,
        use_node_scalars=args.use_node_scalars,
        use_adaln=args.use_adaln,
        use_attention=args.use_attention,
        preloaded_ckpt=ckpt,
    )

    # ── Stage 1: Vector field visualisation ───────────────────────────────────
    run_cfg   = args.vf_mode in ("cfg",   "both")
    run_nocfg = args.vf_mode in ("nocfg", "both")

    if run_cfg:
        try:
            vf_cfg_dir = os.path.join(out_dir, "vf_viz_cfg")
            make_clear_folder(vf_cfg_dir)
            print("\n=== Vector field (CFG) ===")
            generate_model_vector_field(
                out_dir=vf_cfg_dir,
                final_model=model,
                jet_attr_model=jet_attr_model,
                X_test=X_test,
                scale=final_scale,
                n_jet_types=len(args.jet_types),
                n_particles_per_jet=args.num_particles,
                n_features_per_particle=NUM_PARTICLE_FEATURES,
                n_viz_samples=args.n_viz_samples,
                integration_steps=args.integration_steps,
                use_cfg=True,
                cfg_guidance_weight=args.cfg_guidance_weight,
                use_hyperbolic=args.use_hyperbolic,
            )
            print("CFG vector field done.")
        except Exception as e:
            print(f"\n[ERROR] CFG vector field failed: {e}")
            traceback.print_exc()

    if run_nocfg:
        try:
            vf_nocfg_dir = os.path.join(out_dir, "vf_viz_nocfg")
            make_clear_folder(vf_nocfg_dir)
            print("\n=== Vector field (no CFG) ===")
            generate_model_vector_field(
                out_dir=vf_nocfg_dir,
                final_model=model,
                jet_attr_model=jet_attr_model,
                X_test=X_test,
                scale=final_scale,
                n_jet_types=len(args.jet_types),
                n_particles_per_jet=args.num_particles,
                n_features_per_particle=NUM_PARTICLE_FEATURES,
                n_viz_samples=args.n_viz_samples,
                integration_steps=args.integration_steps,
                use_cfg=False,
                use_hyperbolic=args.use_hyperbolic,
            )
            print("No-CFG vector field done.")
        except Exception as e:
            print(f"\n[ERROR] No-CFG vector field failed: {e}")
            traceback.print_exc()

    # ── Stage 2: Sample generation ─────────────────────────────────────────────
    samples = None
    if not args.skip_samples:
        try:
            print("\n=== Sample generation ===")
            samples = generate_samples(
                model=model,
                jet_attr_model=jet_attr_model,
                root_output_path=out_dir,
                max_particles_per_jet=args.num_particles,
                final_scale=final_scale,
                integration_steps=args.integration_steps,
                n_samples=args.n_samples,
                n_jet_types=len(args.jet_types),
                device=device,
                batch_size=args.batch_size,
                use_cfg=False,
                use_hyperbolic=args.use_hyperbolic,
                use_reference_vectors=args.use_reference_vectors,
                sampler=args.sampler,
            )
            print(f"Sample generation done. Shape: {samples.shape}")
            pt_path = os.path.join(out_dir, "samples.pt")
            torch.save(samples.cpu(), pt_path)
            print(f"Saved samples → {pt_path}")
        except Exception as e:
            print(f"\n[ERROR] Sample generation failed: {e}")
            traceback.print_exc()

    # ── Stage 3: Metric calculation ────────────────────────────────────────────
    if not args.skip_metrics:
        if samples is None:
            print("\n[WARN] No samples available — skipping metrics.")
        else:
            try:
                print("\n=== Metric calculation ===")
                run_save_metrics(
                    X_test=X_test,
                    jet_types=args.jet_types,
                    gen_samples=samples,
                    output_path=out_dir,
                    device=device,
                )
                print("Metrics done.")
            except Exception as e:
                print(f"\n[ERROR] Metric calculation failed: {e}")
                traceback.print_exc()

    print(f"\nAll stages complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
