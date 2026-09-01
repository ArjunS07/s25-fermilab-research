"""Deterministic intermediate sampler probes for the training lifecycle."""

import json
import os
import random

import numpy as np
import torch

from generate_samples import generate_samples
from models.stage1 import get_model_pth_path
from util.data import jet_attributes
from util.infra.rng import capture_rng_state


def run_stability_probe(*, optimizer_step, epoch, minibatch, probe_steps, model,
                        optimizer, scheduler, ema, losses, cfg, device,
                        model_output_path, train_loader, run_config, full_config,
                        schedule_definition, final_scale, jet_attr_model):
    """Run a deterministic EMA sampler probe without perturbing training state."""
    if optimizer_step not in probe_steps:
        return jet_attr_model
    probe_dir = f"{model_output_path}/stability_probes/step_{optimizer_step:06d}"
    os.makedirs(probe_dir, exist_ok=True)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    try:
        if cfg.training.stability_probe_save_checkpoints:
            probe_at_epoch_end = minibatch is not None and minibatch + 1 >= len(train_loader)
            probe_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "global_optimizer_step": optimizer_step,
                "config": run_config,
                "full_config": full_config,
                "rng_state": capture_rng_state(),
                "schedule_definition": schedule_definition,
                "resume_epoch": max(0, (epoch + 1 if probe_at_epoch_end else epoch)
                                    if epoch is not None else 0),
                "resume_minibatch": (0 if probe_at_epoch_end else
                                     minibatch + 1 if minibatch is not None else 0),
            }
            if cfg.training.use_cosine_lr:
                probe_ckpt["scheduler_state_dict"] = scheduler.state_dict()
            if ema is not None:
                probe_ckpt["ema_state_dict"] = ema.state_dict()
            torch.save(probe_ckpt, f"{probe_dir}/checkpoint.pth")
        if ema is not None:
            ema.copy_to(model)
        random.seed(cfg.inference.seed)
        np.random.seed(cfg.inference.seed)
        torch.manual_seed(cfg.inference.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.inference.seed)
        if jet_attr_model is None:
            jet_attr_model = jet_attributes.load_model(
                model_path=get_model_pth_path(cfg.paths.output_path)
            ).to(device)
        probe_outputs = generate_samples(
            model=model,
            jet_attr_model=jet_attr_model,
            root_output_path=probe_dir,
            max_particles_per_jet=cfg.data.num_particles,
            final_scale=final_scale,
            integration_steps=cfg.inference.stability_probe_integration_steps,
            n_samples=cfg.inference.stability_probe_samples,
            jet_types=cfg.data.jet_types,
            samples_per_jet_type=None,
            device=device,
            batch_size=min(cfg.training.batch_size, cfg.inference.stability_probe_samples),
            use_cfg=cfg.inference.use_cfg,
            cfg_guidance_weight=cfg.inference.cfg_guidance_weight,
            regulator_mass=cfg.model.regulator_mass,
            integration_end_time=cfg.inference.integration_end_time,
            prior_dist=cfg.training.prior_dist,
        )
        del probe_outputs
        with open(f"{probe_dir}/generation_diagnostics.json") as handle:
            diagnostics = json.load(handle)
        n_unstable = int(diagnostics.get("n_unstable", diagnostics["n_failed"]))
        summary = {
            "optimizer_step": optimizer_step,
            "n_total": int(diagnostics["n_total"]),
            "n_unstable": n_unstable,
            "invalid_fraction": n_unstable / max(int(diagnostics["n_total"]), 1),
            "n_crossed_max_abs_1e6": int(diagnostics["n_crossed_max_abs_1e6"]),
            "failure_reason_counts": diagnostics.get("failure_reason_counts", {}),
        }
        with open(f"{probe_dir}/probe_summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        print(
            f"Stability probe step={optimizer_step}: unstable={n_unstable}/{summary['n_total']} "
            f"reasons={summary['failure_reason_counts']}"
        )
    finally:
        model.load_state_dict(raw_state, strict=True)
        model.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return jet_attr_model
