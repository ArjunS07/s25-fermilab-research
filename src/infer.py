#!/usr/bin/env python3
"""
Standalone LEFTJeN inference script.

Loads a trained checkpoint, generates samples, and calculates metrics.

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

from models.lorentznet_flow import build_lorentznet
from util.data import jet_attributes
from util.data.jet_attributes import NUM_CLASSES
from models.stage1 import get_model_pth_path
from util.metrics.metrics import run_save_metrics
from generate_samples import generate_samples
from data import get_data_path
from config import (InferRunConfig, build_config, parse_config_cli,
                    generation_controls_from_config)
from util.infra.checkpoint_config import resolve_architecture

MAX_N_PARTICLES = 150


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    config_path, overrides = parse_config_cli()
    cfg = build_config(InferRunConfig, config_path, overrides)
    if not cfg.paths.checkpoint_path:
        raise ValueError("paths.checkpoint_path must be set in the config")
    if not cfg.paths.output_path:
        raise ValueError("paths.output_path must be set in the config")
    return cfg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_architecture(cfg, ckpt):
    """If the checkpoint carries a self-describing `full_config` (written by train.py), use
    its model architecture (returning an updated cfg) and warn on any disagreement. Returns
    cfg unchanged for older raw checkpoints without `full_config`."""
    model_cfg, mism = resolve_architecture(cfg.model, ckpt)
    if mism:
        print(f"WARNING: config architecture differs from checkpoint: {mism}. "
              f"Using checkpoint's architecture (loaded weights would otherwise not match).")
    if isinstance(ckpt, dict) and ckpt.get("full_config"):
        print("Loaded model architecture from checkpoint config.")
    return cfg.model_copy(update={"model": model_cfg})


def _load_main_model(cfg, device, preloaded_ckpt=None):
    model = build_lorentznet(
        NUM_CLASSES,
        num_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.n_hidden,
        regulator_mass=cfg.model.regulator_mass,
    ).to(device)

    ckpt = (preloaded_ckpt if preloaded_ckpt is not None else
            torch.load(cfg.paths.checkpoint_path, map_location=device, weights_only=False))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_key = "ema_state_dict" if cfg.inference.use_ema_weights else "model_state_dict"
        if state_key not in ckpt:
            raise ValueError(f"checkpoint does not contain requested {state_key}")
        model.load_state_dict(ckpt[state_key])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}) from {cfg.paths.checkpoint_path}")
        print(f"Weight view: {'EMA' if cfg.inference.use_ema_weights else 'raw'}")
    else:
        # Raw state dict saved with torch.save(model.state_dict(), ...)
        model.load_state_dict(ckpt)
        print(f"Loaded raw state dict from {cfg.paths.checkpoint_path}")

    model.eval()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()
    random.seed(cfg.inference.seed)
    np.random.seed(cfg.inference.seed)
    torch.manual_seed(cfg.inference.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.inference.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = cfg.paths.out_dir or os.path.join(cfg.paths.output_path, "inference")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Inference outputs → {out_dir}")

    # ── Load shared inputs ─────────────────────────────────────────────────────
    data_path = get_data_path(cfg.paths.output_path)
    with open(os.path.join(data_path, "x_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    print(f"Test set: {len(X_test)} jets")

    scale_path = os.path.join(cfg.paths.output_path, "train", "scale.txt")
    with open(scale_path) as f:
        final_scale = float(f.read().strip())
    print(f"Scale: {final_scale:.6f}")

    jet_attr_model = jet_attributes.load_model(
        model_path=get_model_pth_path(cfg.paths.output_path)
    ).to(device)
    jet_attr_model.eval()
    print("Loaded jet attribute model")

    # Full training checkpoints contain optimizer/config/RNG state (including
    # NumPy objects), so PyTorch 2.6+'s weights-only default cannot load them.
    # Checkpoints are provenance-pinned, first-party artifacts in this workflow.
    ckpt = torch.load(cfg.paths.checkpoint_path, map_location=device, weights_only=False)
    cfg = _resolve_architecture(cfg, ckpt)

    model = _load_main_model(cfg, device, preloaded_ckpt=ckpt)

    stage_status = {}
    # ── Stage 1: Sample generation ─────────────────────────────────────────────
    samples = None
    gen_jet_types = None
    gen_pt_cond = None
    prior_samples = None
    if cfg.inference.skip_samples and cfg.paths.replay_samples_path:
        try:
            print("\n=== Loading saved samples for metrics-only replay ===")
            samples = torch.load(cfg.paths.replay_samples_path, map_location=device, weights_only=False)
            bundle_path = cfg.paths.replay_bundle_path
            if not bundle_path:
                raise ValueError("replay_samples_path requires replay_bundle_path")
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            attrs = bundle["generated_jet_attrs"]
            global_types = attrs[:, :5].argmax(dim=-1)
            gen_jet_types = jet_attributes.local_jet_type_indices(global_types, cfg.data.jet_types)
            # ``generated_jet_attrs[:, 6]`` is the physical jet pT retained by
            # generate_samples; the network-only condition is its scaled view.
            gen_pt_cond = attrs[:, 6]
            gen_jet_eta = attrs[:, 5]
            prior_path = cfg.paths.replay_prior_samples_path
            if prior_path and os.path.isfile(prior_path):
                prior_samples = torch.load(prior_path, map_location=device, weights_only=False)
            stage_status["samples"] = {"status": "ok", "source": "replay"}
            print(f"Loaded saved samples: {samples.shape}")
        except Exception as e:
            stage_status["samples"] = {"status": "failed", "error": str(e)}
            print(f"\n[ERROR] Saved sample replay failed: {e}")
            traceback.print_exc()
    elif not cfg.inference.skip_samples:
        try:
            print("\n=== Sample generation ===")
            samples, gen_jet_types, gen_pt_cond, prior_samples, gen_jet_eta = generate_samples(
                model=model,
                jet_attr_model=jet_attr_model,
                root_output_path=out_dir,
                max_particles_per_jet=cfg.data.num_particles,
                final_scale=final_scale,
                integration_steps=cfg.inference.integration_steps,
                n_samples=cfg.inference.n_samples,
                jet_types=cfg.data.jet_types,
                samples_per_jet_type=cfg.inference.samples_per_jet_type,
                device=device,
                batch_size=cfg.inference.batch_size,
                replay_bundle_path=cfg.paths.replay_bundle_path,
                **generation_controls_from_config(cfg, cfg.inference.prior_dist),
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

    # ── Stage 2: Metric calculation ────────────────────────────────────────────
    eval_info = None
    if not cfg.inference.skip_metrics:
        if samples is None:
            stage_status["metrics"] = {"status": "failed", "error": "no samples available"}
            print("\n[WARN] No samples available — skipping metrics.")
        else:
            try:
                print("\n=== Metric calculation ===")
                eval_info = run_save_metrics(
                    X_test=X_test,
                    jet_types=cfg.data.jet_types,
                    gen_samples=samples,
                    output_path=out_dir,
                    device=device,
                    gen_jet_types=gen_jet_types,
                    gen_pt_cond=gen_pt_cond,
                    gen_jet_eta=gen_jet_eta,
                    prior_samples=prior_samples,
                    stratify_by_class=cfg.inference.stratify_metrics_by_class,
                )
                print("Metrics done.")
                stage_status["metrics"] = {"status": "ok"}
            except Exception as e:
                stage_status["metrics"] = {"status": "failed", "error": str(e)}
                print(f"\n[ERROR] Metric calculation failed: {e}")
                traceback.print_exc()

    # ── Stage 3: Write summary.json ───────────────────────────────────────────
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = None

    # Read source run's training summary for final_loss and full_config.
    train_summary_path = os.path.join(cfg.paths.output_path, "train", "summary.json")
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

    stage1_path = get_model_pth_path(cfg.paths.output_path)
    resolved_replay = (cfg.paths.replay_bundle_path or os.path.join(out_dir, "replay_bundle.pt"))
    generation_diagnostics = {}
    generation_diagnostics_path = os.path.join(out_dir, "generation_diagnostics.json")
    if os.path.isfile(generation_diagnostics_path):
        with open(generation_diagnostics_path) as handle:
            generation_diagnostics = json.load(handle)

    summary = {
        "final_loss": train_summary.get("final_loss"),
        "prior_dist": cfg.inference.prior_dist,
        "generation": {
            "seed": cfg.inference.seed,
            "use_cfg": cfg.inference.use_cfg,
            "use_ema_weights": cfg.inference.use_ema_weights,
            "cfg_guidance_weight": cfg.inference.cfg_guidance_weight,
            "integration_steps": cfg.inference.integration_steps,
            "integration_end_time": cfg.inference.integration_end_time,
            "replay_bundle_path": cfg.paths.replay_bundle_path,
            "sampling_seconds": generation_diagnostics.get("sampling_seconds"),
            "sampler_failures": generation_diagnostics.get("n_failed"),
        },
        "git_commit": git_commit,
        "stage_status": stage_status,
        "provenance": {
            "checkpoint_path": os.path.abspath(cfg.paths.checkpoint_path),
            "checkpoint_sha256": _sha256(cfg.paths.checkpoint_path),
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
