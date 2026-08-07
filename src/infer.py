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

import json
import os
import pickle
import random
import subprocess
import traceback
import hashlib

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
from config import (InferRunConfig, build_config, parse_config_cli, infer_config_to_namespace,
                    generation_controls_from_namespace)
from util.checkpoint_config import resolve_architecture

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

def _resolve_architecture(args, ckpt):
    """If the checkpoint carries a self-describing `full_config` (written by
    train.py's config path), use its model architecture and warn if it
    disagrees with any values the user explicitly passed. Falls back to
    `args` unchanged for older checkpoints without `full_config`."""
    args, mism = resolve_architecture(args, ckpt)
    if mism:
        print(f"WARNING: CLI/config architecture flags differ from checkpoint: {mism}. "
              f"Using checkpoint's architecture (loaded weights would otherwise not match).")
    if isinstance(ckpt, dict) and ckpt.get("full_config"):
        print("Loaded model architecture from checkpoint config.")
    return args


def _load_main_model(checkpoint_path, n_hidden, n_layers, num_particles, device,
                     use_reference_vectors=True, include_mass_condition=True,
                     regulator_mass=0.5, preloaded_ckpt=None, use_ema_weights=False):
    model = LEFTJeN(
        max_num_jet_types=NUM_CLASSES,
        num_layers=n_layers,
        hidden_dim=n_hidden,
        include_pt=True,
        use_reference_vectors=use_reference_vectors,
        include_mass_condition=include_mass_condition,
        regulator_mass=regulator_mass,
    ).to(device)

    ckpt = (preloaded_ckpt if preloaded_ckpt is not None else
            torch.load(checkpoint_path, map_location=device, weights_only=False))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_key = "ema_state_dict" if use_ema_weights else "model_state_dict"
        if state_key not in ckpt:
            raise ValueError(f"checkpoint does not contain requested {state_key}")
        model.load_state_dict(ckpt[state_key])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}) from {checkpoint_path}")
        print(f"Weight view: {'EMA' if use_ema_weights else 'raw'}")
    else:
        # Raw state dict saved with torch.save(model.state_dict(), ...)
        model.load_state_dict(ckpt)
        print(f"Loaded raw state dict from {checkpoint_path}")

    model.eval()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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

    # Full training checkpoints contain optimizer/config/RNG state (including
    # NumPy objects), so PyTorch 2.6+'s weights-only default cannot load them.
    # Checkpoints are provenance-pinned, first-party artifacts in this workflow.
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    args = _resolve_architecture(args, ckpt)

    model = _load_main_model(
        checkpoint_path=args.checkpoint_path,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        num_particles=args.num_particles,
        device=device,
        use_reference_vectors=args.use_reference_vectors,
        include_mass_condition=args.include_mass_condition,
        regulator_mass=args.regulator_mass,
        preloaded_ckpt=ckpt,
        use_ema_weights=args.use_ema_weights,
    )

    # ── Stage 1: Vector field visualisation ───────────────────────────────────
    run_cfg   = args.vf_mode in ("cfg",   "both")
    run_nocfg = args.vf_mode in ("nocfg", "both")

    stage_status = {}
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
                use_reference_vectors=args.use_reference_vectors,
            )
            print("CFG vector field done.")
            stage_status["vector_field_cfg"] = {"status": "ok"}
        except Exception as e:
            stage_status["vector_field_cfg"] = {"status": "warning", "error": str(e)}
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
                use_reference_vectors=args.use_reference_vectors,
            )
            print("No-CFG vector field done.")
            stage_status["vector_field_nocfg"] = {"status": "ok"}
        except Exception as e:
            stage_status["vector_field_nocfg"] = {"status": "warning", "error": str(e)}
            print(f"\n[ERROR] No-CFG vector field failed: {e}")
            traceback.print_exc()

    # ── Stage 2: Sample generation ─────────────────────────────────────────────
    samples = None
    gen_jet_types = None
    gen_pt_cond = None
    prior_samples = None
    if not args.skip_samples:
        try:
            print("\n=== Sample generation ===")
            samples, gen_jet_types, gen_pt_cond, prior_samples, gen_jet_eta = generate_samples(
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
                replay_bundle_path=args.replay_bundle_path,
                **generation_controls_from_namespace(args),
            )
            print(f"Sample generation done. Shape: {samples.shape}")
            pt_path = os.path.join(out_dir, "samples.pt")
            torch.save(samples.cpu(), pt_path)
            print(f"Saved samples → {pt_path}")
            prior_path = os.path.join(out_dir, "prior_samples.pt")
            torch.save(prior_samples.cpu(), prior_path)
            print(f"Saved exact integration prior → {prior_path}")
            bundle_path = os.path.join(out_dir, "replay_bundle.pt")
            if os.path.exists(bundle_path):
                print(f"Saved exact replay bundle → {bundle_path}")
            stage_status["samples"] = {"status": "ok"}
        except Exception as e:
            stage_status["samples"] = {"status": "failed", "error": str(e)}
            print(f"\n[ERROR] Sample generation failed: {e}")
            traceback.print_exc()

    # ── Stage 3: Metric calculation ────────────────────────────────────────────
    eval_info = None
    if not args.skip_metrics:
        if samples is None:
            stage_status["metrics"] = {"status": "failed", "error": "no samples available"}
            print("\n[WARN] No samples available — skipping metrics.")
        else:
            try:
                print("\n=== Metric calculation ===")
                eval_info = run_save_metrics(
                    X_test=X_test,
                    jet_types=args.jet_types,
                    gen_samples=samples,
                    output_path=out_dir,
                    device=device,
                    gen_jet_types=gen_jet_types,
                    gen_pt_cond=gen_pt_cond,
                    gen_jet_eta=gen_jet_eta,
                    prior_samples=prior_samples,
                )
                print("Metrics done.")
                stage_status["metrics"] = {"status": "ok"}
            except Exception as e:
                stage_status["metrics"] = {"status": "failed", "error": str(e)}
                print(f"\n[ERROR] Metric calculation failed: {e}")
                traceback.print_exc()

    # ── Stage 4: Write summary.json ───────────────────────────────────────────
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = None

    # Read source run's training summary for final_loss and full_config.
    train_summary_path = os.path.join(args.output_path, "train", "summary.json")
    train_summary = {}
    if os.path.exists(train_summary_path):
        with open(train_summary_path) as f:
            train_summary = json.load(f)

    def _sha256(path):
        if not path or not os.path.isfile(path):
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    stage1_path = get_model_pth_path(args.output_path)
    resolved_replay = (args.replay_bundle_path or os.path.join(out_dir, "replay_bundle.pt"))

    summary = {
        "final_loss": train_summary.get("final_loss"),
        "prior_dist": args.prior_dist,
        "generation": {
            "seed": args.seed,
            "use_cfg": args.use_cfg,
            "use_ema_weights": args.use_ema_weights,
            "cfg_guidance_weight": args.cfg_guidance_weight,
            "integration_steps": args.integration_steps,
            "integration_end_time": args.integration_end_time,
            "sampler": args.sampler,
            "mass_shell_max_step_rapidity": args.mass_shell_max_step_rapidity,
            "mass_shell_max_substeps": args.mass_shell_max_substeps,
            "replay_bundle_path": args.replay_bundle_path,
        },
        "git_commit": git_commit,
        "stage_status": stage_status,
        "provenance": {
            "checkpoint_path": os.path.abspath(args.checkpoint_path),
            "checkpoint_sha256": _sha256(args.checkpoint_path),
            "stage1_path": os.path.abspath(stage1_path),
            "stage1_sha256": _sha256(stage1_path),
            "replay_bundle_path": os.path.abspath(resolved_replay),
            "replay_bundle_sha256": _sha256(resolved_replay),
        },
        "config": train_summary.get("config"),
        "full_config": train_summary.get("full_config"),
        "metrics": eval_info,
    }

    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"Summary written → {summary_path}")

    failed_required = [name for name in ("samples", "metrics")
                       if stage_status.get(name, {}).get("status") == "failed"]
    if failed_required:
        raise RuntimeError("requested inference stages failed: " + ", ".join(failed_required))

    print(f"\nAll stages complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
