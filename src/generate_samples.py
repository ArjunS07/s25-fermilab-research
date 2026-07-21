import seaborn as sns
import torch
import matplotlib.pyplot as plt

from jetnet.utils import EtaPhiPtE_to_cartesian

from models.LEFT_JeN import LEFTJeN
from util import jet_attributes
from util.file_management import make_clear_folder
from util.distributions import gen_initial_distribution
from util.hyperbolic import to_poincare_ball, from_poincare_ball
from util.coordinates import build_reference_vectors

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")

features = [r"e_c", r"$p_x$", r"$p_y$", r"$p_z$"]


def _resilient_mass_shell_step(model, state, cond, masks, t_start, t_end, **kwargs):
    """Integrate a batch while isolating trajectories that exceed adaptive safety limits."""
    try:
        stepped = model.step_hyperbolic(
            y_t=state, jet_conditions=cond, mask=masks, t_start=t_start, t_end=t_end,
            hyperbolic_model='mass_shell', **kwargs)
        if torch.isfinite(stepped).all():
            return stepped, torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
        raise FloatingPointError("mass-shell step returned non-finite state")
    except FloatingPointError:
        if state.shape[0] == 1:
            failed = torch.full_like(state, float('nan'))
            return failed, torch.ones(1, dtype=torch.bool, device=state.device)
        middle = state.shape[0] // 2
        outputs, failures = [], []
        for slc in (slice(0, middle), slice(middle, state.shape[0])):
            refs = kwargs.get("ref_vectors")
            child_kwargs = dict(kwargs)
            if refs is not None:
                child_kwargs["ref_vectors"] = refs[slc]
            child, child_failed = _resilient_mass_shell_step(
                model, state[slc], cond[slc], masks[slc], t_start, t_end, **child_kwargs)
            outputs.append(child)
            failures.append(child_failed)
        return torch.cat(outputs), torch.cat(failures)


def generate_samples(
        model: LEFTJeN,
        jet_attr_model,
        device,
        root_output_path,
        max_particles_per_jet,
        final_scale,
        integration_steps,
        integration_end_time,
        n_samples,
        batch_size,
        n_jet_types=3,
        use_cfg=False,
        cfg_guidance_weight=2.0,
        use_hyperbolic=False,
        hyperbolic_c=1.0,
        hyperbolic_model='poincare',
        regulator_mass=0.5,
        use_reference_vectors=False,
        backbone='legacy',
        include_mass_condition=False,
        sampler='euler',
        prior_dist='isotropic_com',
        mass_shell_max_step_rapidity=None,
        mass_shell_max_substeps=64,
):


    if use_hyperbolic and hyperbolic_model == 'mass_shell' and sampler != 'euler':
        raise ValueError(
            f"mass-shell integration supports sampler='euler'; got {sampler!r}. "
            "Use mass_shell_max_step_rapidity for adaptive Euler integration."
        )

    # make folder
    make_clear_folder(f"{root_output_path}/samples")

    all_samples = []
    all_prior_samples = []
    all_pt_cond = []      # conditioning pT for each jet (normalized)
    all_jet_types = []    # per-jet class index (argmax of one-hot)
    all_failure_steps = []
    all_explosion_steps = []
    all_generated_jet_attrs = []

    with torch.no_grad():
        model.eval()
        jet_attr_model.eval()

        times = torch.linspace(0, integration_end_time, integration_steps + 1).to(device)

        for start_idx in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start_idx)

            generated_jet_attrs, _ = jet_attributes.generate_jets(
                jet_attr_model, device, n_jet_types=n_jet_types, num_jets=current_batch_size
            )
            # generated_jet_attrs layout: [one_hot(5), eta, pt, mass, n_particles]
            jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
            gen_pt = generated_jet_attrs[:, 6].to(device)   # normalized pT from NF
            gen_mass = generated_jet_attrs[:, 7].to(device)
            gen_n_particles = generated_jet_attrs[:, -1].long().to(device)
            gen_n_particles = gen_n_particles.clamp(max=max_particles_per_jet)

            # Build jet_features for priors that need jet-axis info.
            jet_features = None
            jet_phi = (2 * torch.pi) * torch.rand(current_batch_size, device=device)
            if prior_dist in ('axis_aligned', 'axis_aligned_per_jet', 'jet_ref_frame'):
                gen_eta = generated_jet_attrs[:, 5].to(device)
                gen_pt_prior = gen_pt
                jet_features = torch.stack([gen_eta, gen_pt_prior], dim=-1)

            x = gen_initial_distribution(
                batch_size=current_batch_size,
                num_particles=max_particles_per_jet,
                prior_dist=prior_dist,
                jet_features=jet_features,
                jet_phi=jet_phi,
                device=device,
                model_scale=final_scale,
            )
            x = x.to(device)

            masks = jet_attributes.generate_masks(
                gen_n_particles,
                max_particles_per_jet=max_particles_per_jet,
                device=device
            )
            # Preserve the exact integration start, including sampled attributes,
            # multiplicity mask, orientation, and any geometry-specific projection.
            prior_x = x * masks.unsqueeze(-1)
            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                from util.mass_shell import project_to_shell
                prior_x = project_to_shell(prior_x, regulator_mass) * masks.unsqueeze(-1)
            cond_pt = gen_pt / final_scale if backbone == 'tangent_attention' else gen_pt
            condition_parts = [
                jet_one_hot_enc,
                gen_n_particles.unsqueeze(-1).float(),
                cond_pt.unsqueeze(-1),
            ]
            if include_mass_condition:
                condition_parts.append((gen_mass / final_scale).unsqueeze(-1))
            cond = torch.cat(condition_parts, dim=-1).to(device)

            # Reference virtual particles, reconstructed from the sampled jet attributes.
            ref_vectors = None
            if use_reference_vectors:
                gen_eta = generated_jet_attrs[:, 5].to(device)  # jet eta (layout: [onehot(5), eta, pt, mass, n])
                ref_vectors = build_reference_vectors(gen_eta, gen_pt, final_scale, device,
                                                      jet_phi=jet_phi,
                                                      jet_mass=(gen_mass if include_mass_condition else None))

            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                # Integrate the ODE geodesically on the mass shell. The state stays on H_m
                # (Cartesian on-shell 4-vectors), so the final x is used directly.
                from util.mass_shell import project_to_shell
                y = project_to_shell(x * masks.unsqueeze(-1), regulator_mass)
                failure_step = torch.full(
                    (current_batch_size,), -1, dtype=torch.int64, device=device)
                explosion_step = torch.full_like(failure_step, -1)
                for i in range(integration_steps):
                    active = failure_step < 0
                    if not active.any():
                        break
                    refs_active = ref_vectors[active] if ref_vectors is not None else None
                    stepped, failed = _resilient_mass_shell_step(
                        model, y[active], cond[active], masks[active], times[i], times[i + 1],
                        regulator_mass=regulator_mass,
                        use_cfg=use_cfg,
                        guidance_weight=cfg_guidance_weight,
                        ref_vectors=refs_active,
                        max_step_rapidity=mass_shell_max_step_rapidity,
                        max_substeps=mass_shell_max_substeps,
                    )
                    y = y.clone()
                    y[active] = stepped
                    active_indices = active.nonzero(as_tuple=False).flatten()
                    if failed.any():
                        failure_step[active_indices[failed]] = i
                    finite = torch.isfinite(stepped).all(dim=(1, 2))
                    max_abs = stepped.abs().amax(dim=(1, 2))
                    new_explosive = finite & (max_abs > 1e6)
                    unset = explosion_step[active_indices] < 0
                    if (new_explosive & unset).any():
                        explosion_step[active_indices[new_explosive & unset]] = i
                x = y
            elif use_hyperbolic:
                y = to_poincare_ball(x, c=hyperbolic_c)
                for i in range(integration_steps):
                    y = model.step_hyperbolic(
                        y_t=y,
                        jet_conditions=cond,
                        mask=masks,
                        t_start=times[i],
                        t_end=times[i + 1],
                        c=hyperbolic_c,
                        use_cfg=use_cfg,
                        guidance_weight=cfg_guidance_weight,
                        ref_vectors=ref_vectors,
                    )
                x = from_poincare_ball(y, c=hyperbolic_c)
            else:
                for i in range(integration_steps):
                    x = model.step(x, cond, masks, times[i], times[i + 1], method=sampler,
                                   use_cfg=use_cfg, guidance_weight=cfg_guidance_weight,
                                   ref_vectors=ref_vectors)

            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                from util.mass_shell import massless_energy_view
                # H_m is only the regularized transport manifold.  Publish the physical
                # massless endpoint while preserving the integrated spatial momentum.
                shell_x = project_to_shell(x, regulator_mass)
                x = massless_energy_view(shell_x, masks)
                prior_x = massless_energy_view(prior_x, masks)
            else:
                x = x.clamp(-50, 50)
            scaled_x = final_scale * x * masks.unsqueeze(-1)
            # torch.save(scaled_x, f"{root_output_path}/samples/batch_{start_idx//batch_size:04d}.pt")

            all_samples.append(scaled_x.cpu())
            all_prior_samples.append((final_scale * prior_x).cpu())
            all_pt_cond.append(gen_pt.cpu())
            all_jet_types.append(jet_one_hot_enc.argmax(dim=-1).cpu())
            all_generated_jet_attrs.append(generated_jet_attrs.cpu())
            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                all_failure_steps.append(failure_step.cpu())
                all_explosion_steps.append(explosion_step.cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_samples_cat   = torch.cat(all_samples, dim=0)
    all_prior_samples_cat = torch.cat(all_prior_samples, dim=0)
    all_pt_cond_cat   = torch.cat(all_pt_cond, dim=0)
    all_jet_types_cat = torch.cat(all_jet_types, dim=0)

    if all_failure_steps:
        import json
        failures = torch.cat(all_failure_steps)
        explosions = torch.cat(all_explosion_steps)
        generated_attrs = torch.cat(all_generated_jet_attrs)
        prior_max_abs = all_prior_samples_cat.abs().amax(dim=(1, 2))
        endpoint_max_abs = all_samples_cat.abs().amax(dim=(1, 2))
        noteworthy = (failures >= 0) | (explosions >= 0)
        def quantiles(values):
            q = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64)
            return {name: float(value) for name, value in zip(
                ("p50", "p90", "p99", "p999", "max"),
                torch.quantile(values.to(torch.float64), q))}
        trajectory_records = []
        for idx in noteworthy.nonzero(as_tuple=False).flatten().tolist():
            attrs = generated_attrs[idx]
            trajectory_records.append({
                "index": idx,
                "failure_step": int(failures[idx]),
                "explosion_step": int(explosions[idx]),
                "jet_type": int(attrs[:5].argmax()),
                "eta": float(attrs[5]),
                "pt_condition": float(attrs[6]),
                "mass_condition": float(attrs[7]),
                "n_particles": int(attrs[-1]),
                "prior_max_abs": float(prior_max_abs[idx]),
                "endpoint_max_abs": float(endpoint_max_abs[idx]),
            })
        generation_report = {
            "n_total": int(failures.numel()),
            "n_failed": int((failures >= 0).sum()),
            "n_crossed_max_abs_1e6": int((explosions >= 0).sum()),
            "first_failure_step_counts": {
                str(int(step)): int((failures == step).sum())
                for step in torch.unique(failures[failures >= 0]).tolist()
            },
            "first_explosion_step_counts": {
                str(int(step)): int((explosions == step).sum())
                for step in torch.unique(explosions[explosions >= 0]).tolist()
            },
            "integration_steps": int(integration_steps),
            "integration_end_time": float(integration_end_time),
            "max_step_rapidity": mass_shell_max_step_rapidity,
            "max_substeps": int(mass_shell_max_substeps),
            "all_trajectory_quantiles": {
                "abs_eta": quantiles(generated_attrs[:, 5].abs()),
                "pt_condition": quantiles(generated_attrs[:, 6]),
                "mass_condition": quantiles(generated_attrs[:, 7]),
                "n_particles": quantiles(generated_attrs[:, -1]),
                "prior_max_abs": quantiles(prior_max_abs),
            },
            "noteworthy_trajectories": trajectory_records,
        }
        with open(f"{root_output_path}/generation_diagnostics.json", "w") as handle:
            json.dump(generation_report, handle, indent=2)

    return all_samples_cat.to(device), all_jet_types_cat, all_pt_cond_cat, all_prior_samples_cat
