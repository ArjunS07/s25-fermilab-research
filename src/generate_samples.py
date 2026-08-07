import seaborn as sns
import torch
import matplotlib.pyplot as plt
import os

from jetnet.utils import EtaPhiPtE_to_cartesian

from models.mass_shell_gnn import LEFTJeN
from util.data import jet_attributes
from util.infra.file_management import make_clear_folder
from util.data.distributions import gen_initial_distribution
from util.geometry.coordinates import build_reference_vectors
from util.geometry.conditioning import scale_condition_pt

plt.rc("mathtext", fontset="cm")
sns.set_style("whitegrid")

features = [r"e_c", r"$p_x$", r"$p_y$", r"$p_z$"]


def _resilient_mass_shell_step(model, state, cond, masks, t_start, t_end, **kwargs):
    """Integrate a batch while isolating trajectories that exceed adaptive safety limits."""
    try:
        stepped, telemetry = model.step_hyperbolic(
            y_t=state, jet_conditions=cond, mask=masks, t_start=t_start, t_end=t_end,
            hyperbolic_model='mass_shell', return_diagnostics=True, **kwargs)
        if torch.isfinite(stepped).all():
            return (
                stepped,
                torch.zeros(state.shape[0], dtype=torch.bool, device=state.device),
                [dict(telemetry) for _ in range(state.shape[0])],
                [None] * state.shape[0],
            )
        raise FloatingPointError("mass-shell step returned non-finite state")
    except FloatingPointError as exc:
        from util.geometry.mass_shell import MassShellIntegrationError
        failure_record = (
            exc.as_dict() if isinstance(exc, MassShellIntegrationError)
            else {"reason": "floating_point_error", "message": str(exc)}
        )
        if state.shape[0] == 1:
            failed = torch.full_like(state, float('nan'))
            return (
                failed,
                torch.ones(1, dtype=torch.bool, device=state.device),
                [None],
                [failure_record],
            )
        middle = state.shape[0] // 2
        outputs, failures, telemetry_records, failure_records = [], [], [], []
        for slc in (slice(0, middle), slice(middle, state.shape[0])):
            refs = kwargs.get("ref_vectors")
            child_kwargs = dict(kwargs)
            if refs is not None:
                child_kwargs["ref_vectors"] = refs[slc]
            child, child_failed, child_telemetry, child_failures = _resilient_mass_shell_step(
                model, state[slc], cond[slc], masks[slc], t_start, t_end, **child_kwargs)
            outputs.append(child)
            failures.append(child_failed)
            telemetry_records.extend(child_telemetry)
            failure_records.extend(child_failures)
        return (torch.cat(outputs), torch.cat(failures), telemetry_records, failure_records)


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
        use_hyperbolic=True,
        hyperbolic_model='mass_shell',
        regulator_mass=0.5,
        use_reference_vectors=False,
        include_mass_condition=False,
        sampler='euler',
        prior_dist='isotropic_com',
        mass_shell_max_step_rapidity=None,
        mass_shell_max_substeps=64,
        replay_bundle_path=None,
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
    all_pt_cond = []      # conditioning pT for each jet (physical GeV)
    all_gen_eta = []      # conditioning jet eta for each jet (for the canonical FPND axis)
    all_jet_types = []    # per-jet class index (argmax of one-hot)
    all_failure_steps = []
    all_explosion_steps = []
    all_generated_jet_attrs = []
    all_jet_phi = []
    all_masks = []
    all_internal_prior = []
    all_failure_records = []
    all_step_telemetry = []

    replay_bundle = None
    if replay_bundle_path:
        # Replay bundles are deliberately restricted to tensors and primitive metadata.
        replay_bundle = torch.load(replay_bundle_path, map_location="cpu", weights_only=False)
        if replay_bundle.get("format_version") != 1:
            raise ValueError("unsupported replay bundle format")
        metadata = replay_bundle.get("metadata", {})
        expected = {
            "n_samples": int(n_samples),
            "num_particles": int(max_particles_per_jet),
            "prior_dist": prior_dist,
            "final_scale": float(final_scale),
            "use_hyperbolic": bool(use_hyperbolic),
            "hyperbolic_model": hyperbolic_model,
            "regulator_mass": float(regulator_mass),
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"incompatible replay bundle metadata: {mismatches}")

    with torch.no_grad():
        model.eval()
        jet_attr_model.eval()

        times = torch.linspace(0, integration_end_time, integration_steps + 1).to(device)

        for start_idx in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start_idx)

            if replay_bundle is None:
                generated_jet_attrs, _ = jet_attributes.generate_jets(
                    jet_attr_model, device, n_jet_types=n_jet_types,
                    num_jets=current_batch_size
                )
            else:
                generated_jet_attrs = replay_bundle["generated_jet_attrs"][
                    start_idx:start_idx + current_batch_size
                ].to(device)
            # generated_jet_attrs layout: [one_hot(5), eta, pt, mass, n_particles]
            jet_one_hot_enc = generated_jet_attrs[:, :5].to(device)
            gen_pt = generated_jet_attrs[:, 6].to(device)   # normalized pT from NF
            gen_mass = generated_jet_attrs[:, 7].to(device)
            gen_n_particles = generated_jet_attrs[:, -1].long().to(device)
            gen_n_particles = gen_n_particles.clamp(max=max_particles_per_jet)

            # Build jet_features for priors that need jet-axis info.
            jet_features = None
            jet_phi = (
                (2 * torch.pi) * torch.rand(current_batch_size, device=device)
                if replay_bundle is None else
                replay_bundle["jet_phi"][start_idx:start_idx + current_batch_size].to(device)
            )
            if prior_dist in ('axis_aligned', 'axis_aligned_per_jet', 'jet_ref_frame'):
                gen_eta = generated_jet_attrs[:, 5].to(device)
                gen_pt_prior = gen_pt
                jet_features = torch.stack([gen_eta, gen_pt_prior], dim=-1)

            if replay_bundle is None:
                x = gen_initial_distribution(
                    batch_size=current_batch_size,
                    num_particles=max_particles_per_jet,
                    prior_dist=prior_dist,
                    jet_features=jet_features,
                    jet_phi=jet_phi,
                    device=device,
                    model_scale=final_scale,
                ).to(device)

            masks = jet_attributes.generate_masks(
                gen_n_particles,
                max_particles_per_jet=max_particles_per_jet,
                device=device
            )
            if replay_bundle is not None:
                saved_masks = replay_bundle["masks"][
                    start_idx:start_idx + current_batch_size
                ].to(device)
                if not torch.equal(masks.cpu(), saved_masks.cpu()):
                    raise ValueError("replay bundle mask disagrees with Stage-1 multiplicity")
                x = replay_bundle["internal_prior"][
                    start_idx:start_idx + current_batch_size
                ].to(device)
            # Preserve the exact integration start, including sampled attributes,
            # multiplicity mask, orientation, and any geometry-specific projection.
            prior_x = x * masks.unsqueeze(-1)
            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                from util.geometry.mass_shell import project_to_shell
                if replay_bundle is None:
                    prior_x = project_to_shell(prior_x, regulator_mass) * masks.unsqueeze(-1)
                else:
                    prior_x = x * masks.unsqueeze(-1)
            cond_pt = scale_condition_pt(gen_pt, final_scale)
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
                from util.geometry.mass_shell import project_to_shell
                y = project_to_shell(x * masks.unsqueeze(-1), regulator_mass)
                failure_step = torch.full(
                    (current_batch_size,), -1, dtype=torch.int64, device=device)
                explosion_step = torch.full_like(failure_step, -1)
                failure_records = [None] * current_batch_size
                step_telemetry = [[] for _ in range(current_batch_size)]
                for i in range(integration_steps):
                    active = failure_step < 0
                    if not active.any():
                        break
                    refs_active = ref_vectors[active] if ref_vectors is not None else None
                    stepped, failed, telemetry_records, step_failure_records = _resilient_mass_shell_step(
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
                    for local_idx, global_idx in enumerate(active_indices.tolist()):
                        telemetry = telemetry_records[local_idx]
                        if telemetry is not None:
                            step_telemetry[global_idx].append({
                                "integration_step": i,
                                **telemetry,
                            })
                    if failed.any():
                        failure_step[active_indices[failed]] = i
                        for local_idx in failed.nonzero(as_tuple=False).flatten().tolist():
                            global_idx = int(active_indices[local_idx])
                            failure_records[global_idx] = {
                                "integration_step": i,
                                **(step_failure_records[local_idx] or {
                                    "reason": "unknown", "message": "missing failure record"
                                }),
                            }
                    finite = torch.isfinite(stepped).all(dim=(1, 2))
                    max_abs = stepped.abs().amax(dim=(1, 2))
                    new_explosive = finite & (max_abs > 1e6)
                    unset = explosion_step[active_indices] < 0
                    if (new_explosive & unset).any():
                        explosion_step[active_indices[new_explosive & unset]] = i
                x = y

            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                from util.geometry.mass_shell import massless_energy_view
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
            all_gen_eta.append(generated_jet_attrs[:, 5].cpu())
            all_jet_types.append(jet_one_hot_enc.argmax(dim=-1).cpu())
            all_generated_jet_attrs.append(generated_jet_attrs.cpu())
            all_jet_phi.append(jet_phi.cpu())
            all_masks.append(masks.cpu())
            all_internal_prior.append(prior_x.cpu())
            if use_hyperbolic and hyperbolic_model == 'mass_shell':
                all_failure_steps.append(failure_step.cpu())
                all_explosion_steps.append(explosion_step.cpu())
                all_failure_records.extend(failure_records)
                all_step_telemetry.extend(step_telemetry)

    all_samples_cat   = torch.cat(all_samples, dim=0)
    all_prior_samples_cat = torch.cat(all_prior_samples, dim=0)
    all_pt_cond_cat   = torch.cat(all_pt_cond, dim=0)
    all_gen_eta_cat   = torch.cat(all_gen_eta, dim=0)
    all_jet_types_cat = torch.cat(all_jet_types, dim=0)

    if replay_bundle is None:
        bundle = {
            "format_version": 1,
            "metadata": {
                "n_samples": int(n_samples),
                "num_particles": int(max_particles_per_jet),
                "prior_dist": prior_dist,
                "final_scale": float(final_scale),
                "use_hyperbolic": bool(use_hyperbolic),
                "hyperbolic_model": hyperbolic_model,
                "regulator_mass": float(regulator_mass),
            },
            "generated_jet_attrs": torch.cat(all_generated_jet_attrs),
            "jet_phi": torch.cat(all_jet_phi),
            "masks": torch.cat(all_masks),
            "internal_prior": torch.cat(all_internal_prior),
        }
        torch.save(bundle, os.path.join(root_output_path, "replay_bundle.pt"))

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
        def telemetry_quantiles(field):
            values = [
                float(record[field])
                for trajectory in all_step_telemetry
                for record in trajectory
                if record.get(field) is not None
            ]
            if not values:
                return {name: None for name in ("p50", "p90", "p99", "p999", "max")}
            return quantiles(torch.tensor(values, dtype=torch.float64))
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
                "failure": all_failure_records[idx],
                "step_telemetry": all_step_telemetry[idx],
            })
        failure_reason_counts = {}
        for record in all_failure_records:
            if record is not None:
                reason = record.get("reason", "unknown")
                failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
        generation_report = {
            "n_total": int(failures.numel()),
            "n_failed": int((failures >= 0).sum()),
            "n_crossed_max_abs_1e6": int((explosions >= 0).sum()),
            "n_unstable": int(((failures >= 0) | (explosions >= 0)).sum()),
            "first_failure_step_counts": {
                str(int(step)): int((failures == step).sum())
                for step in torch.unique(failures[failures >= 0]).tolist()
            },
            "first_explosion_step_counts": {
                str(int(step)): int((explosions == step).sum())
                for step in torch.unique(explosions[explosions >= 0]).tolist()
            },
            "failure_reason_counts": failure_reason_counts,
            "integration_telemetry_quantiles": {
                "substeps": telemetry_quantiles("substeps"),
                "max_tangent_norm": telemetry_quantiles("max_tangent_norm"),
                "max_step_rapidity": telemetry_quantiles("max_step_rapidity"),
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

    return (all_samples_cat.to(device), all_jet_types_cat, all_pt_cond_cat,
            all_prior_samples_cat, all_gen_eta_cat)
