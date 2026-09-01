"""Final checkpointing, sampling, qualification, and reporting for training."""

import json
import os
import random

import numpy as np
import pandas as pd
import torch

from generate_samples import generate_samples
from models.stage1 import get_model_pth_path
from util.data import jet_attributes
from util.infra.checkpoint_config import build_checkpoint
from util.infra.rng import capture_rng_state
from util.metrics.metrics import run_save_metrics
from util.metrics.qualification import loss_improvement_summary


def finalize_training(*, cfg, model, optimizer, scheduler, ema, losses,
                      last_completed_epoch, global_optimizer_step, run_config,
                      full_config, schedule_definition, next_resume_epoch,
                      next_resume_minibatch, model_output_path, final_scale,
                      training_seconds, x_test, device, max_n_particles,
                      jet_attr_model):
    """Persist the final state, evaluate generated samples, and write run summaries."""
    torch.save(model.state_dict(), f"{model_output_path}/models/final_model.pth")
    final_ckpt = build_checkpoint(
        model_state=model.state_dict(), epoch=last_completed_epoch,
        global_optimizer_step=global_optimizer_step, losses=losses,
        run_config=run_config, full_config=full_config,
        optimizer_state=optimizer.state_dict(), rng_state=capture_rng_state(),
        scheduler_state=scheduler.state_dict() if cfg.training.use_cosine_lr else None,
        ema_state=ema.state_dict() if ema is not None else None,
        extra={"schedule_definition": schedule_definition,
               "resume_epoch": next_resume_epoch,
               "resume_minibatch": next_resume_minibatch},
    )
    torch.save(final_ckpt, f"{model_output_path}/models/final_checkpoint.pth")

    if ema is not None:
        ema.copy_to(model)
        torch.save(model.state_dict(), f"{model_output_path}/models/ema_model.pth")
        print("Using EMA weights for sample generation and metrics.")

    jet_attr_model = (
        jet_attr_model if jet_attr_model is not None
        else jet_attributes.load_model(model_path=get_model_pth_path(cfg.paths.output_path)).to(device)
    )
    gen_jet_types = None
    gen_pt_cond = None
    try:
        random.seed(cfg.inference.seed)
        np.random.seed(cfg.inference.seed)
        torch.manual_seed(cfg.inference.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.inference.seed)
        samples, gen_jet_types, gen_pt_cond, prior_samples, gen_jet_eta = generate_samples(
            model=model, jet_attr_model=jet_attr_model, root_output_path=model_output_path,
            max_particles_per_jet=cfg.data.num_particles, final_scale=final_scale,
            integration_steps=cfg.inference.integration_steps,
            n_samples=cfg.inference.n_samples, jet_types=cfg.data.jet_types,
            samples_per_jet_type=cfg.inference.samples_per_jet_type, device=device,
            batch_size=cfg.training.batch_size if cfg.data.num_particles < max_n_particles else 16,
            use_cfg=cfg.inference.use_cfg,
            cfg_guidance_weight=cfg.inference.cfg_guidance_weight,
            regulator_mass=cfg.model.regulator_mass,
            integration_end_time=cfg.inference.integration_end_time,
            prior_dist=cfg.training.prior_dist,
        )
    except Exception as exc:
        print(f"Error occurred while generating samples: {exc}")
        with open(f"{model_output_path}/error_log.txt", "a") as handle:
            handle.write(f"Error occurred while generating samples: {exc}\n")
        raise SystemExit(1) from exc

    try:
        torch.save(samples[:10000].cpu(), f"{model_output_path}/samples_subset.pt")
        torch.save(prior_samples[:10000].cpu(), f"{model_output_path}/prior_samples_subset.pt")
    except Exception as exc:
        print(f"Error saving sample subset: {exc}")

    generation_diagnostics = None
    invalid_fraction = None
    if (cfg.inference.max_invalid_fraction is not None
            or cfg.inference.warn_invalid_fraction is not None
            or cfg.training.qualification_min_loss_improvement is not None):
        diagnostics_path = f"{model_output_path}/generation_diagnostics.json"
        if not os.path.isfile(diagnostics_path):
            raise RuntimeError("inference.max_invalid_fraction requires generation diagnostics")
        with open(diagnostics_path) as handle:
            generation_diagnostics = json.load(handle)
        n_total = int(generation_diagnostics["n_total"])
        n_unstable = int(generation_diagnostics.get(
            "n_unstable", generation_diagnostics["n_failed"]
        ))
        invalid_fraction = n_unstable / max(n_total, 1)

    qualification_errors = []
    qualification_warnings = []
    qualification_summary = None
    if cfg.training.max_optimizer_steps is not None:
        optimizer_frame = pd.read_csv(f"{model_output_path}/optimizer_steps.csv")
        loss_summary = loss_improvement_summary(optimizer_frame["loss"].to_numpy())
        loss_improvement = loss_summary["loss_improvement_fraction"]
        losses_finite = loss_summary["losses_finite"]
        gradients_finite = bool(optimizer_frame["gradients_finite"].astype(bool).all())
        completed_required_steps = global_optimizer_step == cfg.training.max_optimizer_steps
        no_explosions = bool(
            generation_diagnostics is not None
            and int(generation_diagnostics["n_crossed_max_abs_1e6"]) == 0
        )
        qualification_summary = {
            "global_optimizer_step": global_optimizer_step,
            "required_optimizer_steps": cfg.training.max_optimizer_steps,
            "completed_required_steps": completed_required_steps,
            "loss_window": loss_summary["loss_window"],
            "first_loss_median": loss_summary["first_loss_median"],
            "final_loss_median": loss_summary["final_loss_median"],
            "loss_improvement_fraction": loss_improvement,
            "minimum_loss_improvement_fraction": cfg.training.qualification_min_loss_improvement,
            "losses_finite": losses_finite,
            "gradients_finite": gradients_finite,
            "invalid_fraction": invalid_fraction,
            "warn_invalid_fraction": cfg.inference.warn_invalid_fraction,
            "max_invalid_fraction": cfg.inference.max_invalid_fraction,
            "no_explosions": no_explosions,
        }
        if not completed_required_steps:
            qualification_errors.append("did not reach required optimizer-step budget")
        if not losses_finite:
            qualification_errors.append("optimizer loss contains non-finite values")
        if not gradients_finite:
            qualification_errors.append("gradient norm contains non-finite values")
        if (cfg.training.qualification_min_loss_improvement is not None
                and loss_improvement < cfg.training.qualification_min_loss_improvement):
            qualification_errors.append(
                f"loss improvement {loss_improvement:.6f} is below "
                f"{cfg.training.qualification_min_loss_improvement:.6f}"
            )
        if not no_explosions:
            qualification_errors.append("one or more trajectories crossed |x| > 1e6")

    if (cfg.inference.max_invalid_fraction is not None and invalid_fraction is not None
            and invalid_fraction > cfg.inference.max_invalid_fraction):
        qualification_errors.append(
            f"invalid_fraction={invalid_fraction:.6f} exceeds "
            f"max_invalid_fraction={cfg.inference.max_invalid_fraction:.6f}"
        )
    if (cfg.inference.warn_invalid_fraction is not None and invalid_fraction is not None
            and invalid_fraction > cfg.inference.warn_invalid_fraction):
        warning = (
            f"invalid_fraction={invalid_fraction:.6f} exceeds warning threshold "
            f"{cfg.inference.warn_invalid_fraction:.6f}"
        )
        qualification_warnings.append(warning)
        print(f"WARNING: {warning}; continuing to physics metrics")
    if qualification_summary is not None:
        qualification_summary["passed"] = not qualification_errors
        qualification_summary["errors"] = qualification_errors
        qualification_summary["warnings"] = qualification_warnings
        with open(f"{model_output_path}/qualification_summary.json", "w") as handle:
            json.dump(qualification_summary, handle, indent=2)

    eval_info = {}
    if not cfg.inference.skip_metrics:
        try:
            eval_info = run_save_metrics(
                X_test=x_test, jet_types=cfg.data.jet_types, gen_samples=samples,
                output_path=model_output_path, device=device, gen_jet_types=gen_jet_types,
                gen_pt_cond=gen_pt_cond, gen_jet_eta=gen_jet_eta, prior_samples=prior_samples,
            ) or {}
        except Exception as exc:
            print(f"Error occurred while running/saving metrics: {exc}")
            with open(f"{model_output_path}/error_log.txt", "a") as handle:
                handle.write(f"Error occurred while running/saving metrics: {exc}\n")

    try:
        git_commit = None
        git_commit_path = f"{cfg.paths.output_path}/git_commit.txt"
        if os.path.exists(git_commit_path):
            with open(git_commit_path) as handle:
                git_commit = handle.read().strip()
        summary = {
            "final_loss": losses[-1] if losses else None,
            "num_epochs": len(losses),
            "global_optimizer_step": global_optimizer_step,
            "qualification": qualification_summary,
            "git_commit": git_commit,
            "n_parameters": sum(param.numel() for param in model.parameters()),
            "train_seconds": training_seconds,
            "flow_geometry": "mass_shell",
            "regulator_mass": cfg.model.regulator_mass,
            "effective_prior_dist": cfg.training.prior_dist,
            "effective_coupling": {
                "coupling": cfg.training.coupling,
                "fresh_noise_per_step": True,
                "regulator_mass": cfg.model.regulator_mass,
            },
            "generation": {
                "seed": cfg.inference.seed,
                "use_cfg": bool(cfg.inference.use_cfg),
                "cfg_guidance_weight": cfg.inference.cfg_guidance_weight,
                "integration_steps": cfg.inference.integration_steps,
                "sampling_seconds": (
                    generation_diagnostics.get("sampling_seconds")
                    if generation_diagnostics is not None else None
                ),
                "sampler_failures": (
                    generation_diagnostics.get("n_failed")
                    if generation_diagnostics is not None else None
                ),
            },
            "config": run_config,
            "full_config": full_config,
            "metrics": {key: eval_info.get(key) for key in (
                "w1m", "w1p", "w1efp", "fpd", "cov_mmd",
                "frac_negative_energy", "frac_spacelike", "msq_median",
                "n_generated_invalid", "frac_generated_invalid",
                "n_generated_finite_max_abs_gt_1e6",
                "isotropy_ks_costheta", "isotropy_ks_costheta_p",
                "isotropy_ks_phi", "isotropy_ks_phi_p",
                *(f"fpnd_{jet_type}" for jet_type in cfg.data.jet_types),
            )},
        }

        def json_default(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if hasattr(value, "item"):
                return value.item()
            if hasattr(value, "__float__"):
                return float(value)
            return str(value)

        with open(f"{model_output_path}/summary.json", "w") as handle:
            json.dump(summary, handle, indent=2, default=json_default)
    except Exception as exc:
        print(f"Error writing summary.json: {exc}")
    if qualification_errors and cfg.inference.fail_on_qualification_error:
        raise RuntimeError("qualification gate failed: " + "; ".join(qualification_errors))
